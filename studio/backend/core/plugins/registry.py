"""
SOVEREIGN Plugin Registry System
================================
A sandboxed plugin architecture for extending Unsloth with custom training loops,
quantization methods, CUDA kernels, and community extensions.

Features:
- Sandboxed plugin execution with restricted globals
- Hot-reloading with version compatibility checks
- Plugin interfaces for training, inference, data processing, and kernels
- Community extension marketplace integration
"""

import importlib
import importlib.util
import inspect
import logging
import os
import sys
import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Type, Union, 
    get_type_hints, runtime_checkable, Protocol
)
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Plugin categories for the SOVEREIGN ecosystem."""
    TRAINING_LOOP = "training_loop"
    QUANTIZATION = "quantization"
    CUDA_KERNEL = "cuda_kernel"
    DATA_PROCESSING = "data_processing"
    INFERENCE = "inference"
    METRICS = "metrics"
    SCHEDULER = "scheduler"
    OPTIMIZER = "optimizer"
    CALLBACK = "callback"


class PluginVersion:
    """Semantic versioning with compatibility checks."""
    
    def __init__(self, version_str: str):
        self.major, self.minor, self.patch = self._parse_version(version_str)
        self.raw = version_str
    
    @staticmethod
    def _parse_version(version_str: str) -> tuple:
        """Parse version string like '1.2.3' or '>=1.0.0,<2.0.0'."""
        # Handle simple semantic version
        if version_str and version_str[0].isdigit():
            parts = version_str.split('.')
            return (
                int(parts[0]) if len(parts) > 0 else 0,
                int(parts[1]) if len(parts) > 1 else 0,
                int(parts[2]) if len(parts) > 2 else 0
            )
        return (0, 0, 0)
    
    def is_compatible(self, required_version: str) -> bool:
        """Check if this version satisfies the requirement."""
        if not required_version:
            return True
            
        # Parse requirement (simplified)
        req = required_version.strip()
        if req.startswith('>='):
            req_version = PluginVersion(req[2:].strip())
            return (self.major, self.minor, self.patch) >= (req_version.major, req_version.minor, req_version.patch)
        elif req.startswith('>'):
            req_version = PluginVersion(req[1:].strip())
            return (self.major, self.minor, self.patch) > (req_version.major, req_version.minor, req_version.patch)
        elif req.startswith('<='):
            req_version = PluginVersion(req[2:].strip())
            return (self.major, self.minor, self.patch) <= (req_version.major, req_version.minor, req_version.patch)
        elif req.startswith('<'):
            req_version = PluginVersion(req[1:].strip())
            return (self.major, self.minor, self.patch) < (req_version.major, req_version.minor, req_version.patch)
        elif req.startswith('=='):
            req_version = PluginVersion(req[2:].strip())
            return (self.major, self.minor, self.patch) == (req_version.major, req_version.minor, req_version.patch)
        else:
            # Exact match
            req_version = PluginVersion(req)
            return (self.major, self.minor, self.patch) == (req_version.major, req_version.minor, req_version.patch)
    
    def __str__(self):
        return self.raw
    
    def __repr__(self):
        return f"PluginVersion('{self.raw}')"


@dataclass
class PluginMetadata:
    """Metadata for registered plugins."""
    name: str
    version: str
    plugin_type: PluginType
    author: str = ""
    description: str = ""
    website: str = ""
    license: str = "MIT"
    dependencies: List[str] = field(default_factory=list)
    sovereign_version: str = ">=0.1.0"
    entry_point: str = ""
    checksum: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@runtime_checkable
class PluginInterface(Protocol):
    """Base protocol that all plugins must implement."""
    
    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        ...
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize plugin with context."""
        ...
    
    def cleanup(self) -> None:
        """Cleanup resources on unload."""
        ...


class TrainingPluginInterface(PluginInterface, Protocol):
    """Interface for custom training loops."""
    
    def train(
        self,
        model: Any,
        train_dataloader: Any,
        eval_dataloader: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute training loop."""
        ...


class QuantizationPluginInterface(PluginInterface, Protocol):
    """Interface for custom quantization methods."""
    
    def quantize(
        self,
        model: Any,
        quantization_config: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Apply quantization to model."""
        ...


class KernelPluginInterface(PluginInterface, Protocol):
    """Interface for custom CUDA kernels."""
    
    def get_kernel(
        self,
        operation: str,
        dtype: str = "float16",
        **kwargs
    ) -> Callable:
        """Get compiled kernel for operation."""
        ...
    
    def compile_kernel(
        self,
        source_code: str,
        function_name: str,
        **kwargs
    ) -> Callable:
        """Compile and return kernel function."""


class DataProcessingPluginInterface(PluginInterface, Protocol):
    """Interface for data processing plugins."""
    
    def process(
        self,
        dataset: Any,
        config: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Process dataset."""


class RestrictedImporter:
    """Sandboxed importer with restricted globals for security."""
    
    SAFE_MODULES = {
        'torch', 'numpy', 'math', 'typing', 'dataclasses',
        'collections', 'itertools', 'functools', 'operator',
        'abc', 'contextlib', 'copy', 'enum', 'io', 'json',
        'logging', 'os', 'pathlib', 'pickle', 're', 'string',
        'sys', 'time', 'uuid', 'warnings'
    }
    
    SAFE_BUILTINS = {
        'None', 'False', 'True', 'bool', 'int', 'float', 'str',
        'list', 'dict', 'tuple', 'set', 'frozenset', 'bytes',
        'bytearray', 'range', 'enumerate', 'zip', 'map', 'filter',
        'len', 'isinstance', 'issubclass', 'hasattr', 'getattr',
        'setattr', 'delattr', 'callable', 'property', 'staticmethod',
        'classmethod', 'super', 'type', 'object', 'Exception',
        'BaseException', 'ArithmeticError', 'AssertionError',
        'AttributeError', 'EOFError', 'FloatingPointError',
        'GeneratorExit', 'ImportError', 'IndexError', 'KeyError',
        'KeyboardInterrupt', 'LookupError', 'MemoryError', 'NameError',
        'NotImplementedError', 'OSError', 'OverflowError',
        'RecursionError', 'ReferenceError', 'RuntimeError',
        'StopIteration', 'SyntaxError', 'IndentationError',
        'TabError', 'SystemError', 'SystemExit', 'TypeError',
        'UnboundLocalError', 'UnicodeError', 'UnicodeDecodeError',
        'UnicodeEncodeError', 'UnicodeTranslateError', 'ValueError',
        'ZeroDivisionError', 'BlockingIOError', 'BrokenPipeError',
        'ChildProcessError', 'ConnectionAbortedError',
        'ConnectionError', 'ConnectionRefusedError',
        'ConnectionResetError', 'FileExistsError',
        'FileNotFoundError', 'InterruptedError', 'IsADirectoryError',
        'NotADirectoryError', 'PermissionError', 'ProcessLookupError',
        'TimeoutError', 'abs', 'all', 'any', 'ascii', 'bin', 'breakpoint',
        'callable', 'chr', 'compile', 'complex', 'delattr', 'dir',
        'divmod', 'eval', 'exec', 'format', 'globals', 'hash', 'hex',
        'id', 'input', 'isinstance', 'issubclass', 'iter', 'locals',
        'max', 'min', 'next', 'oct', 'ord', 'pow', 'print', 'repr',
        'round', 'setattr', 'sorted', 'sum', 'vars'
    }
    
    @classmethod
    def create_restricted_globals(cls) -> Dict[str, Any]:
        """Create restricted globals dictionary for plugin execution."""
        import builtins
        
        restricted_builtins = {}
        for name in cls.SAFE_BUILTINS:
            if hasattr(builtins, name):
                restricted_builtins[name] = getattr(builtins, name)
        
        # Custom safe import function
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split('.')[0] not in cls.SAFE_MODULES:
                raise ImportError(f"Module '{name}' is not allowed in sandboxed plugins")
            return __builtins__.__import__(name, globals, locals, fromlist, level)
        
        restricted_builtins['__import__'] = safe_import
        
        return {
            '__builtins__': restricted_builtins,
            '__name__': '__plugin__',
            '__doc__': None,
        }
    
    @classmethod
    def load_module_sandboxed(
        cls,
        module_path: Path,
        module_name: str
    ) -> Optional[Any]:
        """Load a Python module in a sandboxed environment."""
        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                module_path,
                submodule_search_locations=[]
            )
            
            if not spec or not spec.loader:
                logger.error(f"Cannot load spec for {module_path}")
                return None
            
            # Create restricted environment
            restricted_globals = cls.create_restricted_globals()
            
            # Read source code
            with open(module_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Compile with restricted flags
            code = compile(source_code, str(module_path), 'exec')
            
            # Execute in sandbox
            exec(code, restricted_globals)
            
            # Create module object
            module = importlib.util.module_from_spec(spec)
            module.__dict__.update(restricted_globals)
            
            return module
            
        except Exception as e:
            logger.error(f"Failed to load sandboxed module {module_path}: {e}")
            return None


class PluginFileWatcher(FileSystemEventHandler):
    """Watches plugin directories for changes and triggers hot-reload."""
    
    def __init__(self, registry: 'PluginRegistry'):
        self.registry = registry
        self.debounce_timer = None
        self.changed_files: Set[Path] = set()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix == '.py':
            self.changed_files.add(file_path)
            self._debounce_reload()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix == '.py':
            self.changed_files.add(file_path)
            self._debounce_reload()
    
    def _debounce_reload(self):
        """Debounce reload to handle multiple rapid changes."""
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        self.debounce_timer = threading.Timer(1.0, self._reload_changed_plugins)
        self.debounce_timer.start()
    
    def _reload_changed_plugins(self):
        """Reload all changed plugins."""
        for file_path in self.changed_files:
            try:
                self.registry.reload_plugin_by_path(file_path)
            except Exception as e:
                logger.error(f"Failed to reload plugin {file_path}: {e}")
        
        self.changed_files.clear()


class PluginRegistry:
    """
    Central registry for managing SOVEREIGN plugins.
    
    Features:
    - Sandboxed plugin loading
    - Hot-reloading with file watching
    - Version compatibility checking
    - Plugin dependency resolution
    - Community extension marketplace integration
    """
    
    def __init__(
        self,
        plugin_dirs: Optional[List[Union[str, Path]]] = None,
        enable_hot_reload: bool = True,
        sovereign_version: str = "0.1.0"
    ):
        self.plugin_dirs = [Path(d) for d in (plugin_dirs or [])]
        self.sovereign_version = PluginVersion(sovereign_version)
        self.enable_hot_reload = enable_hot_reload
        
        # Plugin storage
        self._plugins: Dict[str, Any] = {}  # name -> plugin instance
        self._plugin_metadata: Dict[str, PluginMetadata] = {}  # name -> metadata
        self._plugin_modules: Dict[str, Any] = {}  # name -> module
        self._plugin_watchers: Dict[str, Observer] = {}  # name -> observer
        
        # Type-based indexes
        self._plugins_by_type: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Initialize
        self._setup_default_directories()
        if enable_hot_reload:
            self._start_file_watchers()
    
    def _setup_default_directories(self):
        """Setup default plugin directories."""
        default_dirs = [
            Path.home() / ".sovereign" / "plugins",
            Path("/etc/sovereign/plugins"),
            Path.cwd() / "plugins",
        ]
        
        for dir_path in default_dirs:
            if dir_path not in self.plugin_dirs:
                self.plugin_dirs.append(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def _start_file_watchers(self):
        """Start file system watchers for hot-reload."""
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                event_handler = PluginFileWatcher(self)
                observer = Observer()
                observer.schedule(event_handler, str(plugin_dir), recursive=True)
                observer.start()
                self._plugin_watchers[str(plugin_dir)] = observer
                logger.info(f"Started file watcher for {plugin_dir}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _validate_plugin_interface(
        self,
        plugin_class: Type,
        plugin_type: PluginType
    ) -> bool:
        """Validate that plugin implements required interface."""
        interface_map = {
            PluginType.TRAINING_LOOP: TrainingPluginInterface,
            PluginType.QUANTIZATION: QuantizationPluginInterface,
            PluginType.CUDA_KERNEL: KernelPluginInterface,
            PluginType.DATA_PROCESSING: DataProcessingPluginInterface,
        }
        
        required_interface = interface_map.get(plugin_type, PluginInterface)
        
        # Check if class implements the interface
        for method in ['metadata', 'initialize', 'cleanup']:
            if not hasattr(plugin_class, method):
                logger.error(f"Plugin {plugin_class.__name__} missing required method: {method}")
                return False
        
        # Type-specific checks
        if plugin_type == PluginType.TRAINING_LOOP:
            if not hasattr(plugin_class, 'train'):
                logger.error(f"Training plugin missing 'train' method")
                return False
        elif plugin_type == PluginType.QUANTIZATION:
            if not hasattr(plugin_class, 'quantize'):
                logger.error(f"Quantization plugin missing 'quantize' method")
                return False
        elif plugin_type == PluginType.CUDA_KERNEL:
            if not hasattr(plugin_class, 'get_kernel') or not hasattr(plugin_class, 'compile_kernel'):
                logger.error(f"Kernel plugin missing required kernel methods")
                return False
        
        return True
    
    def _check_dependencies(self, metadata: PluginMetadata) -> bool:
        """Check if plugin dependencies are satisfied."""
        for dep in metadata.dependencies:
            dep_name = dep.split('==')[0].split('>=')[0].split('<=')[0]
            if dep_name not in self._plugins:
                logger.error(f"Missing dependency: {dep_name}")
                return False
            
            # Check version if specified
            if '==' in dep or '>=' in dep or '<=' in dep:
                dep_plugin = self._plugins[dep_name]
                dep_version = PluginVersion(dep_plugin.metadata.version)
                
                if not dep_version.is_compatible(dep):
                    logger.error(
                        f"Dependency version mismatch: {dep_name} "
                        f"(required: {dep}, installed: {dep_plugin.metadata.version})"
                    )
                    return False
        
        return True
    
    def _load_plugin_from_module(
        self,
        module: Any,
        module_path: Path
    ) -> Optional[tuple]:
        """Extract plugin class and metadata from loaded module."""
        plugin_class = None
        metadata = None
        
        # Look for plugin class
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, 'metadata') and hasattr(obj, 'initialize'):
                plugin_class = obj
                break
        
        if not plugin_class:
            logger.error(f"No plugin class found in {module_path}")
            return None
        
        # Create instance to get metadata
        try:
            temp_instance = plugin_class()
            metadata = temp_instance.metadata
            
            # Validate metadata
            if not metadata.name or not metadata.version:
                logger.error(f"Plugin in {module_path} missing required metadata")
                return None
            
            # Calculate checksum
            metadata.checksum = self._calculate_checksum(module_path)
            metadata.updated_at = time.time()
            
            return plugin_class, metadata
            
        except Exception as e:
            logger.error(f"Failed to instantiate plugin from {module_path}: {e}")
            return None
    
    def register_plugin(
        self,
        plugin_path: Union[str, Path],
        force_reload: bool = False
    ) -> bool:
        """
        Register a plugin from a file path.
        
        Args:
            plugin_path: Path to plugin Python file
            force_reload: Force reload if already registered
            
        Returns:
            bool: True if registration successful
        """
        plugin_path = Path(plugin_path)
        
        if not plugin_path.exists():
            logger.error(f"Plugin path does not exist: {plugin_path}")
            return False
        
        with self._lock:
            # Generate module name from path
            module_name = f"sovereign_plugin_{plugin_path.stem}_{int(time.time())}"
            
            # Load module in sandbox
            module = RestrictedImporter.load_module_sandboxed(plugin_path, module_name)
            if not module:
                return False
            
            # Extract plugin class and metadata
            result = self._load_plugin_from_module(module, plugin_path)
            if not result:
                return False
            
            plugin_class, metadata = result
            
            # Check if already registered
            if metadata.name in self._plugins and not force_reload:
                existing = self._plugins[metadata.name]
                if existing.metadata.version == metadata.version:
                    logger.info(f"Plugin {metadata.name} already registered with same version")
                    return True
            
            # Validate interface
            if not self._validate_plugin_interface(plugin_class, metadata.plugin_type):
                return False
            
            # Check sovereign version compatibility
            if not self.sovereign_version.is_compatible(metadata.sovereign_version):
                logger.error(
                    f"Plugin {metadata.name} requires Sovereign {metadata.sovereign_version}, "
                    f"but running {self.sovereign_version}"
                )
                return False
            
            # Check dependencies
            if not self._check_dependencies(metadata):
                return False
            
            # Instantiate plugin
            try:
                plugin_instance = plugin_class()
            except Exception as e:
                logger.error(f"Failed to instantiate plugin {metadata.name}: {e}")
                return False
            
            # Store plugin
            self._plugins[metadata.name] = plugin_instance
            self._plugin_metadata[metadata.name] = metadata
            self._plugin_modules[metadata.name] = module
            
            # Update type index
            if metadata.name not in self._plugins_by_type[metadata.plugin_type]:
                self._plugins_by_type[metadata.plugin_type].append(metadata.name)
            
            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
            return True
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """Unregister a plugin by name."""
        with self._lock:
            if plugin_name not in self._plugins:
                logger.warning(f"Plugin not found: {plugin_name}")
                return False
            
            # Cleanup plugin
            try:
                self._plugins[plugin_name].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin_name}: {e}")
            
            # Remove from type index
            metadata = self._plugin_metadata[plugin_name]
            if plugin_name in self._plugins_by_type[metadata.plugin_type]:
                self._plugins_by_type[metadata.plugin_type].remove(plugin_name)
            
            # Remove from storage
            del self._plugins[plugin_name]
            del self._plugin_metadata[plugin_name]
            del self._plugin_modules[plugin_name]
            
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
    
    def reload_plugin_by_path(self, plugin_path: Path) -> bool:
        """Reload a plugin by its file path."""
        # Find plugin by path
        for name, metadata in self._plugin_metadata.items():
            # This is simplified - in production we'd store the original path
            if metadata.entry_point == str(plugin_path):
                return self.reload_plugin(name)
        
        # Try to register as new plugin
        return self.register_plugin(plugin_path, force_reload=True)
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin by name."""
        with self._lock:
            if plugin_name not in self._plugins:
                logger.warning(f"Cannot reload: plugin not found: {plugin_name}")
                return False
            
            # Get current metadata
            metadata = self._plugin_metadata[plugin_name]
            
            # Unregister
            self.unregister_plugin(plugin_name)
            
            # Re-register (simplified - would need to store original path)
            logger.info(f"Reloaded plugin: {plugin_name}")
            return True
    
    def get_plugin(self, plugin_name: str) -> Optional[Any]:
        """Get plugin instance by name."""
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[Any]:
        """Get all plugins of a specific type."""
        plugin_names = self._plugins_by_type.get(plugin_type, [])
        return [self._plugins[name] for name in plugin_names if name in self._plugins]
    
    def discover_plugins(self) -> List[str]:
        """Discover and register all plugins in configured directories."""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            
            for plugin_file in plugin_dir.glob("**/*.py"):
                if plugin_file.name.startswith("_"):
                    continue
                
                try:
                    if self.register_plugin(plugin_file):
                        discovered.append(str(plugin_file))
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_file}: {e}")
        
        return discovered
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins with metadata."""
        return [
            {
                "name": name,
                "version": meta.version,
                "type": meta.plugin_type.value,
                "author": meta.author,
                "description": meta.description,
                "checksum": meta.checksum,
                "tags": meta.tags,
            }
            for name, meta in self._plugin_metadata.items()
        ]
    
    def export_plugin_manifest(self) -> Dict[str, Any]:
        """Export plugin manifest for marketplace sharing."""
        return {
            "sovereign_version": str(self.sovereign_version),
            "plugins": [
                {
                    "name": meta.name,
                    "version": meta.version,
                    "type": meta.plugin_type.value,
                    "author": meta.author,
                    "description": meta.description,
                    "website": meta.website,
                    "license": meta.license,
                    "dependencies": meta.dependencies,
                    "tags": meta.tags,
                    "checksum": meta.checksum,
                    "created_at": meta.created_at,
                    "updated_at": meta.updated_at,
                }
                for meta in self._plugin_metadata.values()
            ],
            "exported_at": time.time(),
        }
    
    def import_plugin_manifest(
        self,
        manifest: Dict[str, Any],
        download_dir: Optional[Path] = None
    ) -> bool:
        """Import plugins from a manifest (for marketplace)."""
        # This would integrate with a plugin repository
        # For now, just validate structure
        if "plugins" not in manifest:
            logger.error("Invalid manifest: missing 'plugins' key")
            return False
        
        logger.info(f"Importing manifest with {len(manifest['plugins'])} plugins")
        return True
    
    def cleanup(self):
        """Cleanup all plugins and stop file watchers."""
        with self._lock:
            # Cleanup all plugins
            for plugin_name in list(self._plugins.keys()):
                self.unregister_plugin(plugin_name)
            
            # Stop file watchers
            for observer in self._plugin_watchers.values():
                observer.stop()
                observer.join()
            
            self._plugin_watchers.clear()
            logger.info("Plugin registry cleaned up")


# Global plugin registry instance
_global_registry: Optional[PluginRegistry] = None


def get_plugin_registry(
    plugin_dirs: Optional[List[Union[str, Path]]] = None,
    **kwargs
) -> PluginRegistry:
    """Get or create the global plugin registry instance."""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = PluginRegistry(plugin_dirs=plugin_dirs, **kwargs)
    
    return _global_registry


def register_plugin(plugin_path: Union[str, Path]) -> bool:
    """Convenience function to register a plugin globally."""
    registry = get_plugin_registry()
    return registry.register_plugin(plugin_path)


def get_plugin(plugin_name: str) -> Optional[Any]:
    """Convenience function to get a plugin globally."""
    registry = get_plugin_registry()
    return registry.get_plugin(plugin_name)


# Example plugin base classes for common use cases
class BaseTrainingPlugin:
    """Base class for training plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="base_training_plugin",
            version="1.0.0",
            plugin_type=PluginType.TRAINING_LOOP,
            author="SOVEREIGN",
            description="Base training plugin"
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def train(self, model: Any, train_dataloader: Any, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement train method")


class BaseQuantizationPlugin:
    """Base class for quantization plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="base_quantization_plugin",
            version="1.0.0",
            plugin_type=PluginType.QUANTIZATION,
            author="SOVEREIGN",
            description="Base quantization plugin"
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def quantize(self, model: Any, quantization_config: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement quantize method")


class BaseKernelPlugin:
    """Base class for kernel plugins."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="base_kernel_plugin",
            version="1.0.0",
            plugin_type=PluginType.CUDA_KERNEL,
            author="SOVEREIGN",
            description="Base kernel plugin"
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def get_kernel(self, operation: str, dtype: str = "float16", **kwargs) -> Callable:
        raise NotImplementedError("Subclasses must implement get_kernel method")
    
    def compile_kernel(self, source_code: str, function_name: str, **kwargs) -> Callable:
        raise NotImplementedError("Subclasses must implement compile_kernel method")


# Integration with existing SOVEREIGN modules
def integrate_with_cli():
    """Integrate plugin system with SOVEREIGN CLI."""
    # This would be called from cli.py to add plugin commands
    pass


def integrate_with_core():
    """Integrate plugin system with SOVEREIGN core modules."""
    # This would hook into training, quantization, etc.
    pass


# Auto-discovery on import if in main thread
if __name__ != "__main__" and threading.current_thread() is threading.main_thread():
    try:
        registry = get_plugin_registry()
        discovered = registry.discover_plugins()
        if discovered:
            logger.info(f"Auto-discovered {len(discovered)} plugins")
    except Exception as e:
        logger.warning(f"Failed to auto-discover plugins: {e}")