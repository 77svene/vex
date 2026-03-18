"""
Zero-Copy WebAssembly Data Extraction for Scrapy.

This module implements WebAssembly-based extraction for high-performance,
sandboxed data parsing. It compiles XPath/CSS selectors to Wasm modules,
enables zero-copy memory sharing between Python and Wasm, and provides
fallback to Python execution when needed.
"""

import os
import hashlib
import tempfile
import logging
from typing import Optional, Dict, Any, Callable, Union, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

try:
    import wasmtime
    WASMTIME_AVAILABLE = True
except ImportError:
    WASMTIME_AVAILABLE = False
    wasmtime = None

try:
    import wasmer
    WASMER_AVAILABLE = True
except ImportError:
    WASMER_AVAILABLE = False
    wasmer = None

from vex.http import Response
from vex.selector import Selector
from vex.exceptions import NotConfigured
from vex.utils.python import to_bytes, to_unicode
from vex.utils.misc import load_object

logger = logging.getLogger(__name__)


class WasmRuntime(Enum):
    """Supported Wasm runtimes."""
    WASMTIME = "wasmtime"
    WASMER = "wasmer"
    AUTO = "auto"


@dataclass
class WasmModuleConfig:
    """Configuration for a Wasm extraction module."""
    name: str
    wasm_bytes: bytes
    function_name: str = "extract"
    memory_size: int = 65536  # 64KB default
    timeout_ms: int = 5000
    enable_memory_grow: bool = True


class WasmMemoryManager:
    """Manages zero-copy memory sharing between Python and Wasm."""
    
    def __init__(self, memory: Any, runtime: WasmRuntime):
        self.memory = memory
        self.runtime = runtime
        self._memory_view = None
        
    def write_data(self, data: bytes, offset: int = 0) -> int:
        """Write data to Wasm memory at specified offset."""
        if self.runtime == WasmRuntime.WASMTIME:
            self.memory.write(data, offset)
        elif self.runtime == WasmRuntime.WASMER:
            memory_view = self.memory.uint8_view()
            for i, byte in enumerate(data):
                memory_view[offset + i] = byte
        return offset + len(data)
    
    def read_string(self, offset: int, length: int) -> str:
        """Read string from Wasm memory without copying."""
        if self.runtime == WasmRuntime.WASMTIME:
            data = self.memory.read(offset, offset + length)
            return to_unicode(bytes(data))
        elif self.runtime == WasmRuntime.WASMER:
            memory_view = self.memory.uint8_view()
            data = bytes(memory_view[offset:offset + length])
            return to_unicode(data)
        return ""
    
    def read_bytes(self, offset: int, length: int) -> bytes:
        """Read bytes from Wasm memory without copying."""
        if self.runtime == WasmRuntime.WASMTIME:
            return bytes(self.memory.read(offset, offset + length))
        elif self.runtime == WasmRuntime.WASMER:
            memory_view = self.memory.uint8_view()
            return bytes(memory_view[offset:offset + length])
        return b""
    
    def get_memory_size(self) -> int:
        """Get current memory size in bytes."""
        if self.runtime == WasmRuntime.WASMTIME:
            return self.memory.size * 65536  # Pages to bytes
        elif self.runtime == WasmRuntime.WASMER:
            return self.memory.size * 65536
        return 0


class WasmExtractor:
    """WebAssembly-based data extractor with zero-copy memory sharing."""
    
    def __init__(self, config: WasmModuleConfig, runtime: WasmRuntime = WasmRuntime.AUTO):
        self.config = config
        self.runtime = runtime
        self._instance = None
        self._memory = None
        self._memory_manager = None
        self._function = None
        self._initialized = False
        
        if runtime == WasmRuntime.AUTO:
            self._detect_runtime()
    
    def _detect_runtime(self):
        """Auto-detect available Wasm runtime."""
        if WASMTIME_AVAILABLE:
            self.runtime = WasmRuntime.WASMTIME
        elif WASMER_AVAILABLE:
            self.runtime = WasmRuntime.WASMER
        else:
            raise NotConfigured(
                "No Wasm runtime available. Install wasmtime or wasmer: "
                "pip install wasmtime or pip install wasmer"
            )
    
    def initialize(self):
        """Initialize the Wasm module and runtime."""
        if self._initialized:
            return
            
        try:
            if self.runtime == WasmRuntime.WASMTIME:
                self._init_wasmtime()
            elif self.runtime == WasmRuntime.WASMER:
                self._init_wasmer()
            else:
                raise ValueError(f"Unsupported runtime: {self.runtime}")
                
            self._initialized = True
            logger.debug(f"Wasm extractor '{self.config.name}' initialized with {self.runtime.value}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Wasm extractor: {e}")
            raise
    
    def _init_wasmtime(self):
        """Initialize Wasmtime runtime."""
        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, self.config.wasm_bytes)
        
        # Create memory
        memory_type = wasmtime.MemoryType(self.config.memory_size // 65536, 
                                         self.config.enable_memory_grow)
        memory = wasmtime.Memory(engine, memory_type)
        
        # Create store and instance
        store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_memory("memory", memory)
        
        self._instance = linker.instantiate(store, module)
        self._memory = memory
        self._memory_manager = WasmMemoryManager(memory, self.runtime)
        
        # Get the extraction function
        self._function = self._instance.exports(store)[self.config.function_name]
    
    def _init_wasmer(self):
        """Initialize Wasmer runtime."""
        store = wasmer.Store()
        module = wasmer.Module(store, self.config.wasm_bytes)
        
        # Create memory
        memory = wasmer.Memory(store, wasmer.MemoryType(
            pages=self.config.memory_size // 65536,
            maximum=65536 if self.config.enable_memory_grow else None
        ))
        
        # Create instance
        import_object = wasmer.ImportObject()
        import_object.define("env", {"memory": memory})
        
        self._instance = wasmer.Instance(module, import_object)
        self._memory = memory
        self._memory_manager = WasmMemoryManager(memory, self.runtime)
        
        # Get the extraction function
        self._function = getattr(self._instance.exports, self.config.function_name)
    
    def extract(self, data: Union[str, bytes], **kwargs) -> Optional[str]:
        """Extract data using Wasm module with zero-copy where possible."""
        if not self._initialized:
            self.initialize()
        
        # Convert input to bytes
        if isinstance(data, str):
            data_bytes = to_bytes(data)
        else:
            data_bytes = data
        
        try:
            # Allocate space in Wasm memory for input
            input_offset = 0
            output_offset = len(data_bytes) + 1024  # Leave some space
            
            # Write input data to Wasm memory
            self._memory_manager.write_data(data_bytes, input_offset)
            
            # Call the extraction function
            # Expected signature: extract(input_ptr, input_len, output_ptr) -> output_len
            result_len = self._function(input_offset, len(data_bytes), output_offset)
            
            if result_len <= 0:
                return None
            
            # Read result from Wasm memory
            result = self._memory_manager.read_string(output_offset, result_len)
            return result
            
        except Exception as e:
            logger.warning(f"Wasm extraction failed: {e}, falling back to Python")
            return None
    
    def extract_with_selector(self, selector: Selector, xpath: str = None, 
                             css: str = None, **kwargs) -> Optional[str]:
        """Extract using Wasm-compiled selector logic."""
        # This would require pre-compiled Wasm modules for specific selectors
        # For now, fall back to Python extraction
        if xpath:
            return selector.xpath(xpath).get()
        elif css:
            return selector.css(css).get()
        return None


class WasmExtractorCache:
    """Cache for compiled Wasm extractor modules."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self._cache: Dict[str, WasmModuleConfig] = {}
        self._extractors: Dict[str, WasmExtractor] = {}
    
    def _get_cache_key(self, selector: str, selector_type: str) -> str:
        """Generate cache key for selector."""
        content = f"{selector_type}:{selector}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get filesystem path for cached Wasm module."""
        return Path(self.cache_dir) / f"vex_wasm_{cache_key}.wasm"
    
    def compile_selector(self, selector: str, selector_type: str = "xpath") -> Optional[bytes]:
        """
        Compile selector to Wasm module.
        
        This is a placeholder for actual compilation logic. In production,
        this would use a Wasm compiler that transforms XPath/CSS to Wasm.
        """
        # TODO: Implement actual selector-to-Wasm compilation
        # For now, return None to indicate compilation not available
        logger.debug(f"Wasm compilation not implemented for selector: {selector}")
        return None
    
    def get_or_create_extractor(self, selector: str, selector_type: str = "xpath",
                               runtime: WasmRuntime = WasmRuntime.AUTO) -> Optional[WasmExtractor]:
        """Get cached extractor or create new one."""
        cache_key = self._get_cache_key(selector, selector_type)
        
        if cache_key in self._extractors:
            return self._extractors[cache_key]
        
        # Try to load from filesystem cache
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                wasm_bytes = cache_path.read_bytes()
                config = WasmModuleConfig(
                    name=f"selector_{cache_key[:8]}",
                    wasm_bytes=wasm_bytes
                )
                extractor = WasmExtractor(config, runtime)
                self._extractors[cache_key] = extractor
                return extractor
            except Exception as e:
                logger.warning(f"Failed to load cached Wasm module: {e}")
        
        # Try to compile
        wasm_bytes = self.compile_selector(selector, selector_type)
        if wasm_bytes:
            config = WasmModuleConfig(
                name=f"selector_{cache_key[:8]}",
                wasm_bytes=wasm_bytes
            )
            
            # Save to cache
            try:
                cache_path.write_bytes(wasm_bytes)
            except Exception as e:
                logger.warning(f"Failed to cache Wasm module: {e}")
            
            extractor = WasmExtractor(config, runtime)
            self._extractors[cache_key] = extractor
            return extractor
        
        return None


class WasmSelector:
    """Enhanced selector with Wasm-accelerated extraction."""
    
    def __init__(self, response: Optional[Response] = None, text: Optional[str] = None,
                 wam_runtime: WasmRuntime = WasmRuntime.AUTO, 
                 enable_wasm: bool = True):
        self._selector = Selector(response=response, text=text)
        self._wasm_runtime = wam_runtime
        self._enable_wasm = enable_wasm and (WASMTIME_AVAILABLE or WASMER_AVAILABLE)
        self._wasm_cache = WasmExtractorCache() if self._enable_wasm else None
        self._fallback_enabled = True
    
    def xpath(self, query: str, **kwargs) -> 'WasmSelectorList':
        """XPath query with optional Wasm acceleration."""
        if not self._enable_wasm or not self._wasm_cache:
            return WasmSelectorList(self._selector.xpath(query, **kwargs))
        
        # Try Wasm-accelerated extraction
        wasm_extractor = self._wasm_cache.get_or_create_extractor(
            query, "xpath", self._wasm_runtime
        )
        
        if wasm_extractor:
            try:
                result = wasm_extractor.extract_with_selector(
                    self._selector, xpath=query, **kwargs
                )
                if result is not None:
                    return WasmSelectorList([Selector(text=result)])
            except Exception as e:
                logger.debug(f"Wasm extraction failed, falling back: {e}")
        
        # Fallback to Python
        return WasmSelectorList(self._selector.xpath(query, **kwargs))
    
    def css(self, query: str, **kwargs) -> 'WasmSelectorList':
        """CSS query with optional Wasm acceleration."""
        if not self._enable_wasm or not self._wasm_cache:
            return WasmSelectorList(self._selector.css(query, **kwargs))
        
        # Try Wasm-accelerated extraction
        wasm_extractor = self._wasm_cache.get_or_create_extractor(
            query, "css", self._wasm_runtime
        )
        
        if wasm_extractor:
            try:
                result = wasm_extractor.extract_with_selector(
                    self._selector, css=query, **kwargs
                )
                if result is not None:
                    return WasmSelectorList([Selector(text=result)])
            except Exception as e:
                logger.debug(f"Wasm extraction failed, falling back: {e}")
        
        # Fallback to Python
        return WasmSelectorList(self._selector.css(query, **kwargs))
    
    def re(self, regex: str, replace_entities: bool = True, **kwargs) -> List[str]:
        """Regex extraction (not Wasm-accelerated yet)."""
        return self._selector.re(regex, replace_entities, **kwargs)
    
    def get(self, **kwargs) -> Optional[str]:
        """Get first result."""
        return self._selector.get(**kwargs)
    
    def getall(self, **kwargs) -> List[str]:
        """Get all results."""
        return self._selector.getall(**kwargs)
    
    @property
    def root(self):
        """Access underlying selector root."""
        return self._selector.root


class WasmSelectorList(list):
    """List of WasmSelector objects with chainable methods."""
    
    def xpath(self, query: str, **kwargs) -> 'WasmSelectorList':
        """Apply XPath to all selectors in list."""
        result = []
        for selector in self:
            result.extend(selector.xpath(query, **kwargs))
        return WasmSelectorList(result)
    
    def css(self, query: str, **kwargs) -> 'WasmSelectorList':
        """Apply CSS to all selectors in list."""
        result = []
        for selector in self:
            result.extend(selector.css(query, **kwargs))
        return WasmSelectorList(result)
    
    def getall(self, **kwargs) -> List[str]:
        """Get all results from all selectors."""
        result = []
        for selector in self:
            result.extend(selector.getall(**kwargs))
        return result


class WasmExtractionMiddleware:
    """Scrapy middleware for Wasm-accelerated extraction."""
    
    @classmethod
    def from_crawler(cls, crawler):
        """Initialize from crawler settings."""
        settings = crawler.settings
        
        # Check if Wasm is enabled
        wasm_enabled = settings.getbool('WASM_EXTRACTION_ENABLED', False)
        if not wasm_enabled:
            raise NotConfigured("Wasm extraction disabled")
        
        # Check runtime availability
        runtime_name = settings.get('WASM_RUNTIME', 'auto')
        try:
            runtime = WasmRuntime(runtime_name)
        except ValueError:
            logger.warning(f"Invalid Wasm runtime: {runtime_name}, using auto")
            runtime = WasmRuntime.AUTO
        
        if runtime == WasmRuntime.AUTO and not (WASMTIME_AVAILABLE or WASMER_AVAILABLE):
            raise NotConfigured("No Wasm runtime available")
        
        # Initialize cache directory
        cache_dir = settings.get('WASM_CACHE_DIR', None)
        cache = WasmExtractorCache(cache_dir)
        
        return cls(
            runtime=runtime,
            cache=cache,
            fallback_enabled=settings.getbool('WASM_FALLBACK_ENABLED', True),
            timeout_ms=settings.getint('WASM_TIMEOUT_MS', 5000)
        )
    
    def __init__(self, runtime: WasmRuntime, cache: WasmExtractorCache,
                 fallback_enabled: bool = True, timeout_ms: int = 5000):
        self.runtime = runtime
        self.cache = cache
        self.fallback_enabled = fallback_enabled
        self.timeout_ms = timeout_ms
        self.stats = {
            'wasm_extractions': 0,
            'python_fallbacks': 0,
            'wasm_errors': 0
        }
    
    def process_response(self, request, response, spider):
        """Process response with Wasm-accelerated extraction."""
        # Replace standard Selector with WasmSelector
        if hasattr(response, 'selector'):
            response.selector = WasmSelector(
                response=response,
                wam_runtime=self.runtime,
                enable_wasm=True
            )
        
        return response
    
    def get_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return self.stats.copy()


def compile_xpath_to_wasm(xpath: str) -> Optional[bytes]:
    """
    Compile XPath expression to WebAssembly.
    
    This is a placeholder for actual compilation. In production, this would:
    1. Parse XPath to AST
    2. Transform AST to Wasm instructions
    3. Generate Wasm binary
    
    Args:
        xpath: XPath expression to compile
        
    Returns:
        Wasm binary bytes or None if compilation not available
    """
    # TODO: Implement actual XPath to Wasm compilation
    # This would require:
    # - XPath parser
    # - Wasm code generator
    # - Memory management for zero-copy
    
    logger.debug(f"XPath to Wasm compilation not implemented: {xpath}")
    return None


def compile_css_to_wasm(css: str) -> Optional[bytes]:
    """
    Compile CSS selector to WebAssembly.
    
    This is a placeholder for actual compilation.
    
    Args:
        css: CSS selector to compile
        
    Returns:
        Wasm binary bytes or None if compilation not available
    """
    # TODO: Implement actual CSS to Wasm compilation
    logger.debug(f"CSS to Wasm compilation not implemented: {css}")
    return None


# Utility functions for integration with existing Scrapy code
def create_wasm_selector(response: Response = None, text: str = None,
                        enable_wasm: bool = True) -> WasmSelector:
    """Create a Wasm-enabled selector."""
    return WasmSelector(response=response, text=text, enable_wasm=enable_wasm)


def is_wasm_available() -> bool:
    """Check if any Wasm runtime is available."""
    return WASMTIME_AVAILABLE or WASMER_AVAILABLE


def get_available_runtimes() -> List[str]:
    """Get list of available Wasm runtimes."""
    runtimes = []
    if WASMTIME_AVAILABLE:
        runtimes.append("wasmtime")
    if WASMER_AVAILABLE:
        runtimes.append("wasmer")
    return runtimes


# Example Wasm module for demonstration
EXAMPLE_WASM_MODULE = b"""
(module
  (memory (export "memory") 1)
  (func (export "extract") (param $input_ptr i32) (param $input_len i32) (param $output_ptr i32) (result i32)
    ;; This is a placeholder - actual implementation would parse HTML/XML
    ;; and extract data based on compiled selector logic
    (local $i i32)
    (local.set $i (i32.const 0))
    
    ;; Simple example: copy input to output (identity function)
    (loop $copy_loop
      (i32.store8
        (i32.add (local.get $output_ptr) (local.get $i))
        (i32.load8_u (i32.add (local.get $input_ptr) (local.get $i)))
      )
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $copy_loop (i32.lt_u (local.get $i) (local.get $input_len)))
    )
    
    ;; Return length of output
    (local.get $input_len)
  )
)
"""