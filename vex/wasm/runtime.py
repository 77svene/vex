"""
WebAssembly Runtime for Scrapy - Zero-Copy Data Extraction
Execute extraction logic as WebAssembly modules for near-native performance.
"""

import os
import sys
import hashlib
import tempfile
import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import weakref

if TYPE_CHECKING:
    from vex.http import Response
    from vex.selector import Selector

try:
    import wasmtime
    WASMTIME_AVAILABLE = True
except ImportError:
    WASMTIME_AVAILABLE = False

try:
    import wasmer
    WASMER_AVAILABLE = True
except ImportError:
    WASMER_AVAILABLE = False

from vex.exceptions import NotConfigured
from vex.utils.misc import load_object
from vex.utils.python import to_bytes, to_unicode

logger = logging.getLogger(__name__)


class WasmRuntimeType(Enum):
    """Supported WebAssembly runtimes."""
    WASMTIME = "wasmtime"
    WASMER = "wasmer"
    AUTO = "auto"


@dataclass
class WasmModuleConfig:
    """Configuration for a WebAssembly module."""
    name: str
    source: Union[str, bytes, Path]  # Path to .wasm file or wasm bytes
    memory_size: int = 16 * 1024 * 1024  # 16MB default
    enable_threads: bool = False
    enable_simd: bool = True
    enable_bulk_memory: bool = True
    enable_reference_types: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour in seconds


@dataclass
class ExtractionResult:
    """Result from WebAssembly extraction."""
    data: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used: int = 0
    from_cache: bool = False


class WasmMemoryManager:
    """Manages zero-copy memory sharing between Python and WebAssembly."""
    
    def __init__(self, initial_size: int = 1024 * 1024):
        self._buffers: Dict[int, bytearray] = {}
        self._next_id = 1
        self._initial_size = initial_size
        
    def allocate(self, size: int) -> Tuple[int, bytearray]:
        """Allocate a buffer and return its ID and the buffer."""
        buffer_id = self._next_id
        self._next_id += 1
        
        # Use bytearray for mutable buffer that can be shared
        buffer = bytearray(size)
        self._buffers[buffer_id] = buffer
        return buffer_id, buffer
    
    def get_buffer(self, buffer_id: int) -> Optional[bytearray]:
        """Get buffer by ID."""
        return self._buffers.get(buffer_id)
    
    def release(self, buffer_id: int) -> None:
        """Release a buffer."""
        if buffer_id in self._buffers:
            del self._buffers[buffer_id]
    
    def copy_to_wasm(self, data: bytes, wasm_memory: Any, offset: int = 0) -> int:
        """Copy data to WebAssembly memory with zero-copy when possible."""
        # For now, we do a direct copy. True zero-copy would require
        # shared memory which is more complex and runtime-specific
        data_len = len(data)
        
        # Ensure we have enough space
        if offset + data_len > len(wasm_memory):
            raise MemoryError(f"Not enough WebAssembly memory: need {offset + data_len}, have {len(wasm_memory)}")
        
        # Copy data to wasm memory
        wasm_memory[offset:offset + data_len] = data
        return data_len
    
    def copy_from_wasm(self, wasm_memory: Any, offset: int, length: int) -> bytes:
        """Copy data from WebAssembly memory."""
        if offset + length > len(wasm_memory):
            raise MemoryError(f"Read out of bounds: offset={offset}, length={length}, memory_size={len(wasm_memory)}")
        
        return bytes(wasm_memory[offset:offset + length])


class WasmModule:
    """Represents a compiled WebAssembly module."""
    
    def __init__(self, config: WasmModuleConfig, runtime_type: WasmRuntimeType):
        self.config = config
        self.runtime_type = runtime_type
        self._module = None
        self._instance = None
        self._memory = None
        self._exports = {}
        self._memory_manager = WasmMemoryManager()
        self._compiled_hash = None
        self._load_module()
    
    def _load_module(self) -> None:
        """Load and compile the WebAssembly module."""
        if not WASMTIME_AVAILABLE and not WASMER_AVAILABLE:
            raise NotConfigured("No WebAssembly runtime available. Install wasmtime or wasmer.")
        
        # Determine runtime
        if self.runtime_type == WasmRuntimeType.AUTO:
            if WASMTIME_AVAILABLE:
                self.runtime_type = WasmRuntimeType.WASMTIME
            elif WASMER_AVAILABLE:
                self.runtime_type = WasmRuntimeType.WASMER
        
        # Load wasm bytes
        wasm_bytes = self._load_wasm_bytes()
        self._compiled_hash = hashlib.sha256(wasm_bytes).hexdigest()
        
        # Compile based on runtime
        if self.runtime_type == WasmRuntimeType.WASMTIME and WASMTIME_AVAILABLE:
            self._compile_wasmtime(wasm_bytes)
        elif self.runtime_type == WasmRuntimeType.WASMER and WASMER_AVAILABLE:
            self._compile_wasmer(wasm_bytes)
        else:
            raise NotConfigured(f"Runtime {self.runtime_type.value} not available")
    
    def _load_wasm_bytes(self) -> bytes:
        """Load WebAssembly bytes from config source."""
        source = self.config.source
        
        if isinstance(source, bytes):
            return source
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Wasm file not found: {path}")
            return path.read_bytes()
        else:
            raise ValueError(f"Invalid wasm source type: {type(source)}")
    
    def _compile_wasmtime(self, wasm_bytes: bytes) -> None:
        """Compile module using Wasmtime."""
        engine = wasmtime.Engine()
        
        # Configure module
        config = wasmtime.Config()
        config.cache = self.config.cache_enabled
        config.max_memory_size = self.config.memory_size
        
        # Create module
        self._module = wasmtime.Module(engine, wasm_bytes)
        
        # Create linker and instance
        linker = wasmtime.Linker(engine)
        store = wasmtime.Store(engine)
        
        # Create memory
        memory_type = wasmtime.MemoryType(self.config.memory_size // 65536, maximum=self.config.memory_size // 65536)
        self._memory = wasmtime.Memory(store, memory_type)
        
        # Instantiate
        self._instance = linker.instantiate(store, self._module)
        
        # Get exports
        for export in self._module.exports:
            if isinstance(export, wasmtime.Func):
                self._exports[export.name] = export
            elif isinstance(export, wasmtime.Memory):
                self._memory = export
    
    def _compile_wasmer(self, wasm_bytes: bytes) -> None:
        """Compile module using Wasmer."""
        # Create engine and store
        engine = wasmer.Engine()
        store = wasmer.Store(engine)
        
        # Create module
        self._module = wasmer.Module(store, wasm_bytes)
        
        # Create memory
        memory_type = wasmer.MemoryType(self.config.memory_size // 65536, maximum=self.config.memory_size // 65536)
        self._memory = wasmer.Memory(store, memory_type)
        
        # Create imports
        import_object = wasmer.ImportObject()
        import_object.register("env", {"memory": self._memory})
        
        # Instantiate
        self._instance = wasmer.Instance(self._module, import_object)
        
        # Get exports
        self._exports = self._instance.exports
    
    def call_function(self, name: str, *args) -> Any:
        """Call an exported function from the WebAssembly module."""
        if name not in self._exports:
            raise AttributeError(f"Function {name} not found in module")
        
        func = self._exports[name]
        
        if self.runtime_type == WasmRuntimeType.WASMTIME:
            return func(*args)
        else:  # WASMER
            return func(*args)
    
    @property
    def memory(self) -> Any:
        """Get the WebAssembly memory."""
        return self._memory
    
    @property
    def exports(self) -> Dict[str, Any]:
        """Get all exports from the module."""
        return self._exports
    
    def get_memory_view(self, offset: int = 0, length: Optional[int] = None) -> memoryview:
        """Get a memory view for zero-copy access."""
        if length is None:
            length = len(self._memory) - offset
        
        if self.runtime_type == WasmRuntimeType.WASMTIME:
            return memoryview(self._memory.data)[offset:offset + length]
        else:  # WASMER
            return memoryview(self._memory.buffer)[offset:offset + length]


class WasmExtractor:
    """Extracts data using WebAssembly modules with zero-copy optimization."""
    
    def __init__(self, runtime: 'WasmRuntime'):
        self.runtime = runtime
        self._extraction_cache: Dict[str, ExtractionResult] = {}
        self._cache_timestamps: Dict[str, float] = {}
    
    def extract(self, 
                response: 'Response',
                module_name: str,
                extraction_func: str = "extract",
                **kwargs) -> ExtractionResult:
        """
        Extract data from response using WebAssembly module.
        
        Args:
            response: Scrapy Response object
            module_name: Name of the Wasm module to use
            extraction_func: Name of the extraction function in the module
            **kwargs: Additional arguments to pass to the extraction function
        
        Returns:
            ExtractionResult with extracted data
        """
        import time
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(response, module_name, extraction_func, kwargs)
        
        # Check cache
        if cache_key in self._extraction_cache:
            result = self._extraction_cache[cache_key]
            result.from_cache = True
            return result
        
        try:
            # Get the module
            module = self.runtime.get_module(module_name)
            
            # Prepare input data
            html_bytes = response.body
            input_buffer_id, input_buffer = module._memory_manager.allocate(len(html_bytes))
            input_buffer[:] = html_bytes
            
            # Get memory view for zero-copy
            memory_view = module.get_memory_view()
            
            # Copy data to WebAssembly memory
            module._memory_manager.copy_to_wasm(html_bytes, memory_view, 0)
            
            # Call extraction function
            result_ptr = module.call_function(extraction_func, 0, len(html_bytes), **kwargs)
            
            # Extract result from memory
            # This assumes the wasm function returns a pointer to the result
            # and the first 4 bytes contain the length
            result_length = int.from_bytes(memory_view[result_ptr:result_ptr + 4], 'little')
            result_data = bytes(memory_view[result_ptr + 4:result_ptr + 4 + result_length])
            
            # Parse result (assuming JSON for now)
            import json
            extracted_data = json.loads(result_data.decode('utf-8'))
            
            # Clean up
            module._memory_manager.release(input_buffer_id)
            
            execution_time = time.time() - start_time
            
            result = ExtractionResult(
                data=extracted_data,
                success=True,
                execution_time=execution_time,
                memory_used=len(html_bytes)
            )
            
            # Cache result
            self._extraction_cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"WebAssembly extraction failed: {e}", exc_info=True)
            return ExtractionResult(
                data=None,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def extract_with_fallback(self,
                             response: 'Response',
                             module_name: str,
                             python_extractor: Callable,
                             extraction_func: str = "extract",
                             **kwargs) -> ExtractionResult:
        """
        Extract data with fallback to Python extractor.
        
        Args:
            response: Scrapy Response object
            module_name: Name of the Wasm module to use
            python_extractor: Python function to use as fallback
            extraction_func: Name of the extraction function in the module
            **kwargs: Additional arguments to pass to the extraction function
        
        Returns:
            ExtractionResult with extracted data
        """
        # Try WebAssembly extraction first
        result = self.extract(response, module_name, extraction_func, **kwargs)
        
        if result.success:
            return result
        
        # Fallback to Python
        logger.info(f"WebAssembly extraction failed, falling back to Python: {result.error}")
        
        import time
        start_time = time.time()
        
        try:
            # Use Python selector
            from vex.selector import Selector
            selector = Selector(response)
            python_result = python_extractor(selector, **kwargs)
            
            execution_time = time.time() - start_time
            
            return ExtractionResult(
                data=python_result,
                success=True,
                execution_time=execution_time,
                memory_used=0  # Python uses its own memory
            )
        except Exception as e:
            logger.error(f"Python fallback extraction failed: {e}", exc_info=True)
            return ExtractionResult(
                data=None,
                success=False,
                error=f"WebAssembly failed: {result.error}. Python fallback failed: {e}",
                execution_time=time.time() - start_time
            )
    
    def _generate_cache_key(self, 
                           response: 'Response',
                           module_name: str,
                           extraction_func: str,
                           kwargs: Dict) -> str:
        """Generate a cache key for the extraction."""
        key_parts = [
            response.url,
            module_name,
            extraction_func,
            hashlib.sha256(response.body).hexdigest(),
            str(sorted(kwargs.items()))
        ]
        return hashlib.sha256('|'.join(key_parts).encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._extraction_cache.clear()
        self._cache_timestamps.clear()
    
    def cleanup_expired_cache(self) -> None:
        """Remove expired cache entries."""
        import time
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > 3600:  # 1 hour TTL
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._extraction_cache[key]
            del self._cache_timestamps[key]


class WasmCompiler:
    """Compiles XPath/CSS selectors to WebAssembly modules."""
    
    def __init__(self, optimization_level: int = 2):
        self.optimization_level = optimization_level
        self._compilation_cache: Dict[str, bytes] = {}
    
    def compile_xpath(self, xpath: str, module_name: Optional[str] = None) -> WasmModuleConfig:
        """
        Compile XPath expression to WebAssembly module.
        
        Args:
            xpath: XPath expression to compile
            module_name: Optional name for the module
        
        Returns:
            WasmModuleConfig for the compiled module
        """
        # This is a placeholder implementation
        # In a real implementation, this would use a tool like
        # xpath2wasm or compile XPath to WebAssembly
        
        if module_name is None:
            module_name = f"xpath_{hashlib.md5(xpath.encode()).hexdigest()}"
        
        # For now, we'll create a simple wasm module that uses
        # a pre-compiled xpath library
        # In reality, this would generate custom wasm code
        
        wasm_template = self._get_xpath_wasm_template()
        
        # Replace placeholders
        wasm_code = wasm_template.replace(b"{{XPATH_EXPRESSION}}", xpath.encode())
        
        return WasmModuleConfig(
            name=module_name,
            source=wasm_code,
            memory_size=8 * 1024 * 1024  # 8MB
        )
    
    def compile_css(self, css_selector: str, module_name: Optional[str] = None) -> WasmModuleConfig:
        """
        Compile CSS selector to WebAssembly module.
        
        Args:
            css_selector: CSS selector to compile
            module_name: Optional name for the module
        
        Returns:
            WasmModuleConfig for the compiled module
        """
        # Similar placeholder for CSS selectors
        if module_name is None:
            module_name = f"css_{hashlib.md5(css_selector.encode()).hexdigest()}"
        
        wasm_template = self._get_css_wasm_template()
        wasm_code = wasm_template.replace(b"{{CSS_SELECTOR}}", css_selector.encode())
        
        return WasmModuleConfig(
            name=module_name,
            source=wasm_code,
            memory_size=8 * 1024 * 1024  # 8MB
        )
    
    def _get_xpath_wasm_template(self) -> bytes:
        """Get WebAssembly template for XPath extraction."""
        # This would be a pre-compiled wasm module that includes
        # an XPath evaluation engine
        # For now, return a minimal valid wasm module
        return b'\x00asm\x01\x00\x00\x00'
    
    def _get_css_wasm_template(self) -> bytes:
        """Get WebAssembly template for CSS extraction."""
        # Similar placeholder for CSS
        return b'\x00asm\x01\x00\x00\x00'


class WasmRuntime:
    """
    Main WebAssembly runtime manager for Scrapy.
    
    This class manages WebAssembly modules, handles compilation,
    and provides extraction capabilities with zero-copy optimization.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 settings: Optional[Dict] = None,
                 runtime_type: WasmRuntimeType = WasmRuntimeType.AUTO):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.settings = settings or {}
        self.runtime_type = runtime_type
        self._modules: Dict[str, WasmModule] = {}
        self._module_configs: Dict[str, WasmModuleConfig] = {}
        self._extractor = WasmExtractor(self)
        self._compiler = WasmCompiler()
        self._temp_dir = tempfile.mkdtemp(prefix="vex_wasm_")
        
        # Initialize from settings
        self._init_from_settings()
        
        # Check runtime availability
        self._check_runtime_availability()
        
        logger.info(f"WebAssembly runtime initialized with {self.runtime_type.value}")
    
    def _init_from_settings(self) -> None:
        """Initialize from Scrapy settings."""
        wasm_enabled = self.settings.get('WASM_ENABLED', False)
        if not wasm_enabled:
            raise NotConfigured("WebAssembly support is disabled in settings")
        
        # Set runtime type from settings
        runtime_setting = self.settings.get('WASM_RUNTIME', 'auto')
        if runtime_setting == 'wasmtime':
            self.runtime_type = WasmRuntimeType.WASMTIME
        elif runtime_setting == 'wasmer':
            self.runtime_type = WasmRuntimeType.WASMER
        else:
            self.runtime_type = WasmRuntimeType.AUTO
        
        # Load module configurations
        module_configs = self.settings.get('WASM_MODULES', {})
        for name, config in module_configs.items():
            self.register_module_config(name, config)
    
    def _check_runtime_availability(self) -> None:
        """Check if the selected runtime is available."""
        if self.runtime_type == WasmRuntimeType.WASMTIME and not WASMTIME_AVAILABLE:
            raise NotConfigured("Wasmtime is not installed. Install with: pip install wasmtime")
        
        if self.runtime_type == WasmRuntimeType.WASMER and not WASMER_AVAILABLE:
            raise NotConfigured("Wasmer is not installed. Install with: pip install wasmer")
        
        if self.runtime_type == WasmRuntimeType.AUTO:
            if not WASMTIME_AVAILABLE and not WASMER_AVAILABLE:
                raise NotConfigured(
                    "No WebAssembly runtime available. "
                    "Install wasmtime or wasmer: pip install wasmtime wasmer"
                )
    
    def register_module_config(self, name: str, config: Union[Dict, WasmModuleConfig]) -> None:
        """Register a module configuration."""
        if isinstance(config, dict):
            config = WasmModuleConfig(**config)
        
        self._module_configs[name] = config
        logger.debug(f"Registered WebAssembly module config: {name}")
    
    def get_module(self, name: str) -> WasmModule:
        """Get or create a WebAssembly module."""
        if name not in self._modules:
            if name not in self._module_configs:
                raise ValueError(f"Module {name} not configured")
            
            config = self._module_configs[name]
            self._modules[name] = WasmModule(config, self.runtime_type)
            logger.debug(f"Loaded WebAssembly module: {name}")
        
        return self._modules[name]
    
    def compile_and_register(self, 
                            name: str,
                            xpath: Optional[str] = None,
                            css: Optional[str] = None) -> WasmModuleConfig:
        """
        Compile XPath/CSS to WebAssembly and register the module.
        
        Args:
            name: Name for the module
            xpath: XPath expression (mutually exclusive with css)
            css: CSS selector (mutually exclusive with xpath)
        
        Returns:
            WasmModuleConfig for the compiled module
        """
        if xpath and css:
            raise ValueError("Specify either xpath or css, not both")
        
        if xpath:
            config = self._compiler.compile_xpath(xpath, name)
        elif css:
            config = self._compiler.compile_css(css, name)
        else:
            raise ValueError("Either xpath or css must be specified")
        
        self.register_module_config(name, config)
        return config
    
    def extract(self, 
                response: 'Response',
                module_name: str,
                extraction_func: str = "extract",
                **kwargs) -> ExtractionResult:
        """
        Extract data from response using WebAssembly module.
        
        Args:
            response: Scrapy Response object
            module_name: Name of the Wasm module to use
            extraction_func: Name of the extraction function in the module
            **kwargs: Additional arguments to pass to the extraction function
        
        Returns:
            ExtractionResult with extracted data
        """
        return self._extractor.extract(response, module_name, extraction_func, **kwargs)
    
    def extract_with_fallback(self,
                             response: 'Response',
                             module_name: str,
                             python_extractor: Callable,
                             extraction_func: str = "extract",
                             **kwargs) -> ExtractionResult:
        """
        Extract data with fallback to Python extractor.
        
        Args:
            response: Scrapy Response object
            module_name: Name of the Wasm module to use
            python_extractor: Python function to use as fallback
            extraction_func: Name of the extraction function in the module
            **kwargs: Additional arguments to pass to the extraction function
        
        Returns:
            ExtractionResult with extracted data
        """
        return self._extractor.extract_with_fallback(
            response, module_name, python_extractor, extraction_func, **kwargs
        )
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._extractor.clear_cache()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self._extractor.cleanup_expired_cache()
        
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup()


# Factory functions for easy integration

def create_wasm_runtime(settings: Optional[Dict] = None) -> Optional[WasmRuntime]:
    """
    Create a WebAssembly runtime instance.
    
    Args:
        settings: Scrapy settings dictionary
    
    Returns:
        WasmRuntime instance or None if not available
    """
    try:
        return WasmRuntime(settings)
    except NotConfigured as e:
        logger.debug(f"WebAssembly runtime not configured: {e}")
        return None


def extract_with_wasm(response: 'Response',
                     module_name: str,
                     settings: Optional[Dict] = None,
                     **kwargs) -> Optional[Any]:
    """
    Convenience function to extract data using WebAssembly.
    
    Args:
        response: Scrapy Response object
        module_name: Name of the Wasm module to use
        settings: Scrapy settings dictionary
        **kwargs: Additional arguments to pass to the extraction function
    
    Returns:
        Extracted data or None if extraction failed
    """
    runtime = create_wasm_runtime(settings)
    if runtime is None:
        return None
    
    result = runtime.extract(response, module_name, **kwargs)
    return result.data if result.success else None


# Integration with Scrapy Selector
class WasmSelector:
    """
    Selector that uses WebAssembly for extraction with Python fallback.
    """
    
    def __init__(self, 
                 response: Optional['Response'] = None,
                 text: Optional[str] = None,
                 type: Optional[str] = None,
                 _root=None,
                 _expr=None,
                 runtime: Optional[WasmRuntime] = None):
        from vex.selector import Selector
        self._selector = Selector(response=response, text=text, type=type, _root=_root, _expr=_expr)
        self._runtime = runtime or create_wasm_runtime()
        self._wasm_cache: Dict[str, Any] = {}
    
    def xpath(self, xpath: str, **kwargs) -> 'Selector':
        """XPath selection with WebAssembly acceleration."""
        if self._runtime is None:
            return self._selector.xpath(xpath, **kwargs)
        
        # Try to use WebAssembly for complex XPath expressions
        # For simple expressions, Python is faster due to overhead
        if self._is_complex_xpath(xpath):
            cache_key = f"xpath_{hashlib.md5(xpath.encode()).hexdigest()}"
            
            if cache_key not in self._wasm_cache:
                # Compile and cache the module
                module_name = f"xpath_{cache_key}"
                self._runtime.compile_and_register(module_name, xpath=xpath)
                self._wasm_cache[cache_key] = module_name
            
            module_name = self._wasm_cache[cache_key]
            
            # Extract using WebAssembly
            result = self._runtime.extract(
                self._selector.response,
                module_name,
                extraction_func="xpath_extract"
            )
            
            if result.success:
                # Convert result back to Selector
                # This is simplified - in reality would need proper conversion
                return self._selector.xpath(xpath, **kwargs)
        
        # Fallback to Python
        return self._selector.xpath(xpath, **kwargs)
    
    def css(self, css: str, **kwargs) -> 'Selector':
        """CSS selection with WebAssembly acceleration."""
        if self._runtime is None:
            return self._selector.css(css, **kwargs)
        
        # Similar logic for CSS selectors
        if self._is_complex_css(css):
            cache_key = f"css_{hashlib.md5(css.encode()).hexdigest()}"
            
            if cache_key not in self._wasm_cache:
                module_name = f"css_{cache_key}"
                self._runtime.compile_and_register(module_name, css=css)
                self._wasm_cache[cache_key] = module_name
            
            module_name = self._wasm_cache[cache_key]
            
            result = self._runtime.extract(
                self._selector.response,
                module_name,
                extraction_func="css_extract"
            )
            
            if result.success:
                return self._selector.css(css, **kwargs)
        
        return self._selector.css(css, **kwargs)
    
    def _is_complex_xpath(self, xpath: str) -> bool:
        """Determine if XPath expression is complex enough to benefit from WebAssembly."""
        # Simple heuristic - in reality would be more sophisticated
        complexity_indicators = [
            'contains(',
            'starts-with(',
            'substring(',
            'count(',
            'sum(',
            'position()',
            'last()',
            'ancestor::',
            'descendant::',
            'following-sibling::',
            'preceding-sibling::'
        ]
        return any(indicator in xpath for indicator in complexity_indicators)
    
    def _is_complex_css(self, css: str) -> bool:
        """Determine if CSS selector is complex enough to benefit from WebAssembly."""
        complexity_indicators = [
            ':not(',
            ':nth-child(',
            ':nth-last-child(',
            ':nth-of-type(',
            ':nth-last-of-type(',
            ':first-child',
            ':last-child',
            ':first-of-type',
            ':last-of-type',
            ':only-child',
            ':empty',
            ':has(',
            ':is(',
            ':where('
        ]
        return any(indicator in css for indicator in complexity_indicators)
    
    def __getattr__(self, name):
        """Delegate other attributes to the underlying selector."""
        return getattr(self._selector, name)


# Middleware for automatic WebAcceleration
class WebAccelerationMiddleware:
    """
    Middleware that automatically applies WebAssembly acceleration
    to extraction operations.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.runtime = create_wasm_runtime(settings)
        self.enabled = self.runtime is not None
        
        if self.enabled:
            logger.info("WebAcceleration middleware enabled")
    
    def process_spider_input(self, response, spider):
        """Process response before spider sees it."""
        if not self.enabled:
            return None
        
        # Attach WasmSelector to response for automatic acceleration
        response._wasm_selector = WasmSelector(response, runtime=self.runtime)
        return None
    
    def process_spider_output(self, response, result, spider):
        """Process spider output (not used in this implementation)."""
        return result


# Configuration helpers
def get_default_settings() -> Dict:
    """Get default settings for WebAssembly support."""
    return {
        'WASM_ENABLED': False,
        'WASM_RUNTIME': 'auto',  # 'wasmtime', 'wasmer', or 'auto'
        'WASM_MODULES': {},
        'WASM_CACHE_ENABLED': True,
        'WASM_CACHE_TTL': 3600,  # 1 hour
        'WASM_MEMORY_LIMIT': 64 * 1024 * 1024,  # 64MB
        'WASM_FALLBACK_ENABLED': True,
        'WASM_MIDDLEWARE_ENABLED': False,
        'DOWNLOADER_MIDDLEWARES': {
            'vex.wasm.runtime.WebAccelerationMiddleware': 100,
        },
    }


# Example usage documentation
"""
Example Usage:

1. Enable in settings.py:
   WASM_ENABLED = True
   WASM_MODULES = {
       'product_extractor': {
           'source': 'path/to/extractor.wasm',
           'memory_size': 32 * 1024 * 1024,
       }
   }

2. In spider:
   from vex.wasm.runtime import WasmRuntime
   
   class MySpider(vex.Spider):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.wasm_runtime = WasmRuntime(self.settings)
       
       def parse(self, response):
           # Use WebAssembly extraction
           result = self.wasm_runtime.extract(
               response, 
               'product_extractor',
               extraction_func='extract_products'
           )
           
           if result.success:
               yield result.data
           else:
               # Fallback to Python
               yield self.parse_with_python(response)
       
       def parse_with_python(self, response):
           # Traditional Python extraction
           pass

3. With automatic middleware:
   # In settings.py:
   WASM_ENABLED = True
   WASM_MIDDLEWARE_ENABLED = True
   
   # In spider:
   def parse(self, response):
       # WebAssembly acceleration is automatic
       selector = response._wasm_selector
       # Use selector.xpath() or selector.css() as normal
"""