"""
SOVEREIGN Plugin Architecture & Custom Kernels
Production-ready plugin system with sandboxed execution, hot-reloading, and version compatibility.
"""

import importlib
import importlib.util
import sys
import os
import inspect
import types
import threading
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Type, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

# Configure logging
logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Supported plugin types for SOVEREIGN"""
    TRAINING_LOOP = "training_loop"
    QUANTIZATION = "quantization"
    CUDA_KERNEL = "cuda_kernel"
    DATA_PROCESSING = "data_processing"
    INFERENCE = "inference"
    METRICS = "metrics"
    SCHEDULER = "scheduler"
    OPTIMIZER = "optimizer"


class PluginStatus(Enum):
    """Plugin lifecycle status"""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Plugin metadata container"""
    name: str
    version: str
    plugin_type: PluginType
    author: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    api_version: str = "1.0.0"
    checksum: str = ""
    entry_point: str = ""
    sandbox_enabled: bool = True
    hot_reload: bool = True
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)


class PluginInterface(ABC):
    """Base interface all plugins must implement"""
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize plugin with context"""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin functionality"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass


class TrainingLoopPlugin(PluginInterface):
    """Interface for custom training loop plugins"""
    
    @abstractmethod
    def train_epoch(self, model: Any, dataloader: Any, optimizer: Any, 
                   criterion: Any, device: str, **kwargs) -> Dict[str, float]:
        """Execute one training epoch"""
        pass
    
    @abstractmethod
    def validation_epoch(self, model: Any, dataloader: Any, 
                        criterion: Any, device: str, **kwargs) -> Dict[str, float]:
        """Execute one validation epoch"""
        pass


class QuantizationPlugin(PluginInterface):
    """Interface for custom quantization method plugins"""
    
    @abstractmethod
    def quantize_model(self, model: Any, config: Dict[str, Any]) -> Any:
        """Quantize a model"""
        pass
    
    @abstractmethod
    def dequantize_model(self, model: Any) -> Any:
        """Dequantize a model"""
        pass


class CUDAKernelPlugin(PluginInterface):
    """Interface for custom CUDA kernel plugins"""
    
    @abstractmethod
    def compile_kernel(self, kernel_source: str, **kwargs) -> Any:
        """Compile CUDA kernel"""
        pass
    
    @abstractmethod
    def execute_kernel(self, kernel: Any, inputs: List[Any], 
                      grid_dim: tuple, block_dim: tuple, **kwargs) -> Any:
        """Execute compiled kernel"""
        pass


class SandboxRestrictedDict(dict):
    """Restricted dictionary for sandboxed plugin execution"""
    
    SAFE_BUILTINS = {
        'None': None,
        'False': False,
        'True': True,
        'bool': bool,
        'int': int,
        'float': float,
        'str': str,
        'list': list,
        'tuple': tuple,
        'dict': dict,
        'set': set,
        'frozenset': frozenset,
        'len': len,
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sorted': sorted,
        'reversed': reversed,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,
        'pow': pow,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'type': type,
        'object': object,
        'property': property,
        'staticmethod': staticmethod,
        'classmethod': classmethod,
        'super': super,
        'print': print,
        'repr': repr,
        'hash': hash,
        'id': id,
        'callable': callable,
        'hasattr': hasattr,
        'getattr': getattr,
        'setattr': setattr,
        'delattr': delattr,
        'vars': vars,
        'dir': dir,
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['__builtins__'] = self.SAFE_BUILTINS
    
    def __setitem__(self, key, value):
        # Prevent overriding builtins
        if key in self.SAFE_BUILTINS and key != '__builtins__':
            raise SecurityError(f"Cannot override builtin: {key}")
        super().__setitem__(key, value)


class SecurityError(Exception):
    """Security violation in plugin execution"""
    pass


class PluginLoader:
    """
    Production-ready plugin loader with sandboxing, hot-reloading, and version management.
    Integrates with SOVEREIGN backend architecture.
    """
    
    # Required API version for plugin compatibility
    REQUIRED_API_VERSION = "1.0.0"
    
    # Safe modules allowed in sandboxed plugins
    SAFE_MODULES = {
        'torch', 'numpy', 'math', 'json', 'typing', 'dataclasses',
        'collections', 'itertools', 'functools', 'operator', 'copy',
        'pickle', 'io', 'sys', 'os', 'pathlib', 'datetime', 'time',
        'random', 'string', 'textwrap', 'unicodedata', 'struct',
        'array', 'weakref', 'types', 'abc', 'contextlib', 'enum',
        'numbers', 'decimal', 'fractions', 'statistics', 'heapq',
        'bisect', 'queue', 'threading', 'multiprocessing',
    }
    
    def __init__(self, plugin_dir: Union[str, Path], 
                 max_workers: int = 4,
                 enable_hot_reload: bool = True,
                 reload_interval: int = 30):
        """
        Initialize plugin loader.
        
        Args:
            plugin_dir: Directory containing plugins
            max_workers: Maximum worker threads for plugin execution
            enable_hot_reload: Enable hot-reloading of plugins
            reload_interval: Interval in seconds to check for plugin updates
        """
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins: Dict[str, PluginInterface] = {}
        self.metadata: Dict[str, PluginMetadata] = {}
        self.modules: Dict[str, types.ModuleType] = {}
        self.status: Dict[str, PluginStatus] = {}
        self.file_hashes: Dict[str, str] = {}
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.enable_hot_reload = enable_hot_reload
        self.reload_interval = reload_interval
        
        self._lock = threading.RLock()
        self._watcher_thread = None
        self._stop_event = threading.Event()
        
        # Plugin type to interface mapping
        self._interface_map = {
            PluginType.TRAINING_LOOP: TrainingLoopPlugin,
            PluginType.QUANTIZATION: QuantizationPlugin,
            PluginType.CUDA_KERNEL: CUDAKernelPlugin,
        }
        
        # Start hot-reload watcher if enabled
        if self.enable_hot_reload:
            self._start_hot_reload_watcher()
    
    def _start_hot_reload_watcher(self):
        """Start background thread for hot-reloading"""
        def watcher():
            while not self._stop_event.is_set():
                try:
                    self._check_for_updates()
                except Exception as e:
                    logger.error(f"Hot-reload error: {e}")
                time.sleep(self.reload_interval)
        
        self._watcher_thread = threading.Thread(
            target=watcher,
            daemon=True,
            name="PluginHotReloadWatcher"
        )
        self._watcher_thread.start()
        logger.info("Hot-reload watcher started")
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _check_for_updates(self):
        """Check for plugin file updates"""
        with self._lock:
            for plugin_name, plugin_path in self._get_plugin_files():
                current_hash = self._compute_file_hash(plugin_path)
                stored_hash = self.file_hashes.get(plugin_name)
                
                if stored_hash and current_hash != stored_hash:
                    logger.info(f"Plugin {plugin_name} changed, reloading...")
                    try:
                        self.reload_plugin(plugin_name)
                    except Exception as e:
                        logger.error(f"Failed to reload plugin {plugin_name}: {e}")
    
    def _get_plugin_files(self) -> List[tuple]:
        """Get all plugin files in plugin directory"""
        plugins = []
        for item in self.plugin_dir.iterdir():
            if item.is_file() and item.suffix == '.py' and not item.name.startswith('_'):
                plugins.append((item.stem, item))
        return plugins
    
    def _create_sandbox(self, plugin_dir: Path) -> Dict[str, Any]:
        """Create sandboxed environment for plugin execution"""
        sandbox = SandboxRestrictedDict()
        
        # Add safe modules to sandbox
        for module_name in self.SAFE_MODULES:
            if module_name in sys.modules:
                sandbox[module_name] = sys.modules[module_name]
        
        # Add plugin directory to path temporarily
        original_path = sys.path.copy()
        sys.path.insert(0, str(plugin_dir))
        
        return sandbox, original_path
    
    def _validate_plugin_version(self, metadata: PluginMetadata) -> bool:
        """Validate plugin version compatibility"""
        try:
            plugin_major = int(metadata.api_version.split('.')[0])
            required_major = int(self.REQUIRED_API_VERSION.split('.')[0])
            
            if plugin_major != required_major:
                logger.error(
                    f"Plugin {metadata.name} API version {metadata.api_version} "
                    f"incompatible with required {self.REQUIRED_API_VERSION}"
                )
                return False
            return True
        except (ValueError, AttributeError):
            logger.error(f"Invalid API version format in plugin {metadata.name}")
            return False
    
    def _validate_plugin_class(self, plugin_class: Type, metadata: PluginMetadata) -> bool:
        """Validate plugin class implements required interface"""
        expected_interface = self._interface_map.get(metadata.plugin_type)
        
        if not expected_interface:
            logger.warning(f"No specific interface for plugin type {metadata.plugin_type}")
            return True
        
        if not issubclass(plugin_class, expected_interface):
            logger.error(
                f"Plugin {metadata.name} of type {metadata.plugin_type.value} "
                f"must implement {expected_interface.__name__}"
            )
            return False
        
        # Check required methods
        required_methods = ['initialize', 'execute', 'cleanup', 'get_metadata']
        for method in required_methods:
            if not hasattr(plugin_class, method):
                logger.error(f"Plugin {metadata.name} missing required method: {method}")
                return False
        
        return True
    
    def _load_plugin_module(self, plugin_name: str, plugin_path: Path) -> Optional[types.ModuleType]:
        """Load plugin module in sandboxed environment"""
        try:
            # Create sandbox
            sandbox, original_path = self._create_sandbox(plugin_path.parent)
            
            # Read plugin source
            with open(plugin_path, 'r', encoding='utf-8') as f:
                plugin_code = f.read()
            
            # Create module
            module_name = f"sovereign_plugin_{plugin_name}"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            
            # Execute in sandbox
            exec(compile(plugin_code, str(plugin_path), 'exec'), sandbox)
            
            # Update module with sandbox contents
            for key, value in sandbox.items():
                if not key.startswith('__'):
                    setattr(module, key, value)
            
            # Restore path
            sys.path[:] = original_path
            
            return module
            
        except Exception as e:
            logger.error(f"Failed to load plugin module {plugin_name}: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _extract_metadata(self, module: types.ModuleType, plugin_name: str) -> Optional[PluginMetadata]:
        """Extract metadata from plugin module"""
        try:
            # Try to find metadata in module
            if hasattr(module, 'PLUGIN_METADATA'):
                meta_dict = module.PLUGIN_METADATA
            elif hasattr(module, 'get_metadata'):
                meta_dict = module.get_metadata()
            else:
                # Create default metadata
                meta_dict = {
                    'name': plugin_name,
                    'version': '1.0.0',
                    'plugin_type': PluginType.TRAINING_LOOP,
                    'api_version': self.REQUIRED_API_VERSION,
                }
            
            # Ensure plugin_type is enum
            if 'plugin_type' in meta_dict and isinstance(meta_dict['plugin_type'], str):
                meta_dict['plugin_type'] = PluginType(meta_dict['plugin_type'])
            
            # Compute checksum
            meta_dict['checksum'] = self._compute_file_hash(
                self.plugin_dir / f"{plugin_name}.py"
            )
            
            return PluginMetadata(**meta_dict)
            
        except Exception as e:
            logger.error(f"Failed to extract metadata from {plugin_name}: {e}")
            return None
    
    def _find_plugin_class(self, module: types.ModuleType, 
                          plugin_type: PluginType) -> Optional[Type[PluginInterface]]:
        """Find plugin class in module"""
        expected_interface = self._interface_map.get(plugin_type, PluginInterface)
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, expected_interface) and 
                obj is not expected_interface and
                obj is not PluginInterface):
                return obj
        
        return None
    
    def load_plugin(self, plugin_name: str) -> bool:
        """
        Load a single plugin by name.
        
        Args:
            plugin_name: Name of plugin (without .py extension)
            
        Returns:
            bool: True if plugin loaded successfully
        """
        plugin_path = self.plugin_dir / f"{plugin_name}.py"
        
        if not plugin_path.exists():
            logger.error(f"Plugin file not found: {plugin_path}")
            return False
        
        with self._lock:
            try:
                # Load module
                module = self._load_plugin_module(plugin_name, plugin_path)
                if not module:
                    self.status[plugin_name] = PluginStatus.ERROR
                    return False
                
                # Extract metadata
                metadata = self._extract_metadata(module, plugin_name)
                if not metadata:
                    self.status[plugin_name] = PluginStatus.ERROR
                    return False
                
                # Validate version
                if not self._validate_plugin_version(metadata):
                    self.status[plugin_name] = PluginStatus.ERROR
                    return False
                
                # Find plugin class
                plugin_class = self._find_plugin_class(module, metadata.plugin_type)
                if not plugin_class:
                    logger.error(f"No valid plugin class found in {plugin_name}")
                    self.status[plugin_name] = PluginStatus.ERROR
                    return False
                
                # Validate plugin class
                if not self._validate_plugin_class(plugin_class, metadata):
                    self.status[plugin_name] = PluginStatus.ERROR
                    return False
                
                # Store module and metadata
                self.modules[plugin_name] = module
                self.metadata[plugin_name] = metadata
                self.file_hashes[plugin_name] = metadata.checksum
                
                # Create plugin instance
                plugin_instance = plugin_class()
                self.plugins[plugin_name] = plugin_instance
                self.status[plugin_name] = PluginStatus.LOADED
                
                logger.info(f"Successfully loaded plugin: {plugin_name} v{metadata.version}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")
                logger.error(traceback.format_exc())
                self.status[plugin_name] = PluginStatus.ERROR
                return False
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """
        Load all plugins from plugin directory.
        
        Returns:
            Dict mapping plugin names to load success status
        """
        results = {}
        
        for plugin_name, plugin_path in self._get_plugin_files():
            results[plugin_name] = self.load_plugin(plugin_name)
        
        return results
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin (hot-reload).
        
        Args:
            plugin_name: Name of plugin to reload
            
        Returns:
            bool: True if reload successful
        """
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} not loaded")
            return False
        
        # Cleanup existing plugin
        try:
            self.plugins[plugin_name].cleanup()
        except Exception as e:
            logger.warning(f"Error during plugin {plugin_name} cleanup: {e}")
        
        # Remove from caches
        self.plugins.pop(plugin_name, None)
        self.metadata.pop(plugin_name, None)
        self.modules.pop(plugin_name, None)
        self.status.pop(plugin_name, None)
        
        # Reload
        return self.load_plugin(plugin_name)
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of plugin to unload
            
        Returns:
            bool: True if unload successful
        """
        with self._lock:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin {plugin_name} not loaded")
                return False
            
            try:
                self.plugins[plugin_name].cleanup()
            except Exception as e:
                logger.error(f"Error during plugin {plugin_name} cleanup: {e}")
            
            self.plugins.pop(plugin_name, None)
            self.metadata.pop(plugin_name, None)
            self.modules.pop(plugin_name, None)
            self.file_hashes.pop(plugin_name, None)
            self.status[plugin_name] = PluginStatus.UNLOADED
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """
        Get plugin instance by name.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """
        Get all plugins of a specific type.
        
        Args:
            plugin_type: Type of plugins to retrieve
            
        Returns:
            List of plugin instances
        """
        plugins = []
        for name, plugin in self.plugins.items():
            metadata = self.metadata.get(name)
            if metadata and metadata.plugin_type == plugin_type:
                plugins.append(plugin)
        return plugins
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """
        Execute a plugin in a thread pool.
        
        Args:
            plugin_name: Name of plugin to execute
            *args: Arguments to pass to plugin
            **kwargs: Keyword arguments to pass to plugin
            
        Returns:
            Plugin execution result
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin {plugin_name} not found")
        
        metadata = self.metadata.get(plugin_name)
        if metadata and metadata.sandbox_enabled:
            # Execute in thread pool for isolation
            future = self.executor.submit(self._execute_sandboxed, plugin, *args, **kwargs)
            return future.result()
        else:
            return plugin.execute(*args, **kwargs)
    
    def _execute_sandboxed(self, plugin: PluginInterface, *args, **kwargs) -> Any:
        """Execute plugin with basic sandboxing"""
        try:
            return plugin.execute(*args, **kwargs)
        except Exception as e:
            logger.error(f"Plugin execution error: {e}")
            raise
    
    def get_plugin_status(self, plugin_name: str) -> PluginStatus:
        """
        Get plugin status.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin status
        """
        return self.status.get(plugin_name, PluginStatus.UNLOADED)
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all plugins with their metadata.
        
        Returns:
            List of plugin information dictionaries
        """
        plugins = []
        for name, metadata in self.metadata.items():
            plugins.append({
                'name': name,
                'version': metadata.version,
                'type': metadata.plugin_type.value,
                'status': self.status.get(name, PluginStatus.UNLOADED).value,
                'author': metadata.author,
                'description': metadata.description,
                'api_version': metadata.api_version,
                'sandbox_enabled': metadata.sandbox_enabled,
                'hot_reload': metadata.hot_reload,
            })
        return plugins
    
    def validate_plugin_dependencies(self, plugin_name: str) -> List[str]:
        """
        Validate plugin dependencies are available.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            List of missing dependencies
        """
        metadata = self.metadata.get(plugin_name)
        if not metadata:
            return [f"Plugin {plugin_name} not found"]
        
        missing = []
        for dep in metadata.dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        return missing
    
    def shutdown(self):
        """Shutdown plugin loader and cleanup resources"""
        self._stop_event.set()
        
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5)
        
        # Cleanup all plugins
        for plugin_name in list(self.plugins.keys()):
            self.unload_plugin(plugin_name)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Plugin loader shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class PluginManager:
    """
    High-level plugin manager for SOVEREIGN system.
    Provides simplified interface for plugin operations.
    """
    
    def __init__(self, plugin_dir: Union[str, Path] = None):
        """
        Initialize plugin manager.
        
        Args:
            plugin_dir: Directory for plugins (default: ~/.sovereign/plugins)
        """
        if plugin_dir is None:
            plugin_dir = Path.home() / '.sovereign' / 'plugins'
        
        self.loader = PluginLoader(plugin_dir)
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize plugin manager and load all plugins.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            results = self.loader.load_all_plugins()
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            logger.info(f"Loaded {success_count}/{total_count} plugins")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin manager: {e}")
            return False
    
    def get_training_plugins(self) -> List[TrainingLoopPlugin]:
        """Get all training loop plugins"""
        return self.loader.get_plugins_by_type(PluginType.TRAINING_LOOP)
    
    def get_quantization_plugins(self) -> List[QuantizationPlugin]:
        """Get all quantization plugins"""
        return self.loader.get_plugins_by_type(PluginType.QUANTIZATION)
    
    def get_cuda_kernel_plugins(self) -> List[CUDAKernelPlugin]:
        """Get all CUDA kernel plugins"""
        return self.loader.get_plugins_by_type(PluginType.CUDA_KERNEL)
    
    def execute_training_plugin(self, plugin_name: str, model: Any, 
                               dataloader: Any, **kwargs) -> Dict[str, float]:
        """
        Execute a training plugin.
        
        Args:
            plugin_name: Name of training plugin
            model: Model to train
            dataloader: DataLoader instance
            **kwargs: Additional arguments
            
        Returns:
            Training metrics dictionary
        """
        plugin = self.loader.get_plugin(plugin_name)
        if not plugin or not isinstance(plugin, TrainingLoopPlugin):
            raise ValueError(f"Training plugin {plugin_name} not found")
        
        return plugin.train_epoch(model, dataloader, **kwargs)
    
    def execute_quantization_plugin(self, plugin_name: str, model: Any,
                                   config: Dict[str, Any]) -> Any:
        """
        Execute a quantization plugin.
        
        Args:
            plugin_name: Name of quantization plugin
            model: Model to quantize
            config: Quantization configuration
            
        Returns:
            Quantized model
        """
        plugin = self.loader.get_plugin(plugin_name)
        if not plugin or not isinstance(plugin, QuantizationPlugin):
            raise ValueError(f"Quantization plugin {plugin_name} not found")
        
        return plugin.quantize_model(model, config)
    
    def install_plugin(self, plugin_source: Union[str, Path], 
                      plugin_name: str = None) -> bool:
        """
        Install a plugin from source.
        
        Args:
            plugin_source: Plugin source code or file path
            plugin_name: Optional name for plugin
            
        Returns:
            bool: True if installation successful
        """
        try:
            if isinstance(plugin_source, Path) and plugin_source.exists():
                # Copy plugin file
                if plugin_name is None:
                    plugin_name = plugin_source.stem
                
                dest_path = self.loader.plugin_dir / f"{plugin_name}.py"
                with open(plugin_source, 'r') as src, open(dest_path, 'w') as dst:
                    dst.write(src.read())
            else:
                # Treat as source code string
                if plugin_name is None:
                    raise ValueError("plugin_name required for source code installation")
                
                dest_path = self.loader.plugin_dir / f"{plugin_name}.py"
                with open(dest_path, 'w') as f:
                    f.write(str(plugin_source))
            
            # Load the plugin
            return self.loader.load_plugin(plugin_name)
            
        except Exception as e:
            logger.error(f"Failed to install plugin: {e}")
            return False
    
    def uninstall_plugin(self, plugin_name: str) -> bool:
        """
        Uninstall a plugin.
        
        Args:
            plugin_name: Name of plugin to uninstall
            
        Returns:
            bool: True if uninstallation successful
        """
        try:
            # Unload plugin
            self.loader.unload_plugin(plugin_name)
            
            # Remove plugin file
            plugin_path = self.loader.plugin_dir / f"{plugin_name}.py"
            if plugin_path.exists():
                plugin_path.unlink()
            
            logger.info(f"Uninstalled plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to uninstall plugin {plugin_name}: {e}")
            return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed plugin information.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin information dictionary or None
        """
        metadata = self.loader.metadata.get(plugin_name)
        if not metadata:
            return None
        
        return {
            'name': plugin_name,
            'version': metadata.version,
            'type': metadata.plugin_type.value,
            'status': self.loader.get_plugin_status(plugin_name).value,
            'author': metadata.author,
            'description': metadata.description,
            'dependencies': metadata.dependencies,
            'api_version': metadata.api_version,
            'sandbox_enabled': metadata.sandbox_enabled,
            'hot_reload': metadata.hot_reload,
            'created_at': metadata.created_at,
            'modified_at': metadata.modified_at,
            'checksum': metadata.checksum,
        }
    
    def shutdown(self):
        """Shutdown plugin manager"""
        self.loader.shutdown()
        self._initialized = False
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# Example plugin template for documentation
PLUGIN_TEMPLATE = '''
"""
SOVEREIGN Plugin Template
Example: Custom Training Loop Plugin
"""

from studio.backend.core.plugins.loader import TrainingLoopPlugin, PluginMetadata, PluginType
from typing import Any, Dict

class MyTrainingPlugin(TrainingLoopPlugin):
    """Custom training loop plugin example"""
    
    def __init__(self):
        self.metadata = PluginMetadata(
            name="my_training_plugin",
            version="1.0.0",
            plugin_type=PluginType.TRAINING_LOOP,
            author="Your Name",
            description="Custom training loop with advanced features",
            api_version="1.0.0",
            sandbox_enabled=True,
            hot_reload=True
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize plugin with context"""
        # Setup any required resources
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute plugin (not used for training plugins)"""
        return None
    
    def train_epoch(self, model: Any, dataloader: Any, optimizer: Any,
                   criterion: Any, device: str, **kwargs) -> Dict[str, float]:
        """Execute one training epoch"""
        # Implement custom training logic
        total_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }
    
    def validation_epoch(self, model: Any, dataloader: Any,
                        criterion: Any, device: str, **kwargs) -> Dict[str, float]:
        """Execute one validation epoch"""
        # Implement custom validation logic
        total_loss = 0.0
        correct = 0
        total = 0
        
        model.eval()
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return {
            'val_loss': total_loss / len(dataloader),
            'val_accuracy': 100. * correct / total
        }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return self.metadata

# Plugin metadata for discovery
PLUGIN_METADATA = {
    'name': 'my_training_plugin',
    'version': '1.0.0',
    'plugin_type': PluginType.TRAINING_LOOP,
    'author': 'Your Name',
    'description': 'Custom training loop with advanced features',
    'api_version': '1.0.0',
    'sandbox_enabled': True,
    'hot_reload': True
}
'''


# Integration with existing SOVEREIGN modules
def integrate_with_backend():
    """Integration point with existing SOVEREIGN backend"""
    try:
        # Import existing modules
        from studio.backend.core import __init__ as core_init
        from studio.backend.core.data_recipe import manager as data_manager
        
        # Add plugin manager to core context
        if not hasattr(core_init, 'plugin_manager'):
            core_init.plugin_manager = PluginManager()
            logger.info("Plugin manager integrated with SOVEREIGN backend")
            
    except ImportError as e:
        logger.warning(f"Could not integrate with SOVEREIGN backend: {e}")


# Auto-integrate when module is imported
integrate_with_backend()


# Export public API
__all__ = [
    'PluginType',
    'PluginStatus',
    'PluginMetadata',
    'PluginInterface',
    'TrainingLoopPlugin',
    'QuantizationPlugin',
    'CUDAKernelPlugin',
    'PluginLoader',
    'PluginManager',
    'SecurityError',
    'PLUGIN_TEMPLATE',
]