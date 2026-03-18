import importlib
import importlib.util
import sys
import types
import hashlib
import logging
import inspect
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import ast
import textwrap
import tempfile
import os
import json

logger = logging.getLogger(__name__)

# Plugin API version for compatibility checking
PLUGIN_API_VERSION = "1.0.0"

class PluginType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    QUANTIZATION = "quantization"
    KERNEL = "kernel"
    DATA_PROCESSING = "data_processing"
    CUSTOM = "custom"

@dataclass
class PluginMetadata:
    """Metadata for plugin versioning and compatibility."""
    name: str
    version: str
    plugin_type: PluginType
    api_version: str = PLUGIN_API_VERSION
    author: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    entry_point: str = "main"
    checksum: Optional[str] = None
    
    def is_compatible(self) -> bool:
        """Check if plugin is compatible with current API version."""
        try:
            current_major, current_minor, _ = PLUGIN_API_VERSION.split('.')
            plugin_major, plugin_minor, _ = self.api_version.split('.')
            return current_major == plugin_major and int(current_minor) >= int(plugin_minor)
        except (ValueError, AttributeError):
            return False

class PluginInterface(ABC):
    """Base interface all plugins must implement."""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the plugin with context."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    def get_version(self) -> str:
        """Return plugin version."""
        return self.get_metadata().version

class TrainingPluginInterface(PluginInterface):
    """Interface for custom training loops."""
    
    @abstractmethod
    def create_training_loop(self, model: Any, optimizer: Any, **kwargs) -> Callable:
        """Create a custom training loop function."""
        pass
    
    @abstractmethod
    def get_supported_optimizers(self) -> List[str]:
        """Return list of supported optimizer names."""
        pass

class QuantizationPluginInterface(PluginInterface):
    """Interface for custom quantization methods."""
    
    @abstractmethod
    def quantize_model(self, model: Any, config: Dict[str, Any]) -> Any:
        """Quantize a model with given configuration."""
        pass
    
    @abstractmethod
    def get_quantization_methods(self) -> List[str]:
        """Return list of supported quantization methods."""
        pass

class KernelPluginInterface(PluginInterface):
    """Interface for custom CUDA kernels."""
    
    @abstractmethod
    def compile_kernel(self, source_code: str, kernel_name: str) -> Any:
        """Compile a CUDA kernel from source code."""
        pass
    
    @abstractmethod
    def execute_kernel(self, kernel: Any, inputs: List[Any], **kwargs) -> Any:
        """Execute a compiled kernel."""
        pass
    
    @abstractmethod
    def get_supported_architectures(self) -> List[str]:
        """Return list of supported GPU architectures."""
        pass

class DataProcessingPluginInterface(PluginInterface):
    """Interface for data processing plugins."""
    
    @abstractmethod
    def process_dataset(self, dataset: Any, config: Dict[str, Any]) -> Any:
        """Process a dataset with given configuration."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported data formats."""
        pass

class PluginSandbox:
    """Sandbox environment for loading and executing plugins safely."""
    
    def __init__(self, allowed_modules: Optional[List[str]] = None):
        self.allowed_modules = allowed_modules or [
            'torch', 'numpy', 'math', 'json', 'typing', 'dataclasses',
            'collections', 'itertools', 'functools', 'operator'
        ]
        self.restricted_builtins = self._create_restricted_builtins()
    
    def _create_restricted_builtins(self) -> Dict[str, Any]:
        """Create restricted builtins for sandboxed execution."""
        import builtins
        
        safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
            'chr', 'complex', 'dict', 'dir', 'divmod', 'enumerate',
            'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
            'hasattr', 'hash', 'hex', 'id', 'int', 'isinstance', 'issubclass',
            'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'object',
            'oct', 'ord', 'pow', 'print', 'property', 'range', 'repr',
            'reversed', 'round', 'set', 'setattr', 'slice', 'sorted',
            'str', 'sum', 'super', 'tuple', 'type', 'vars', 'zip'
        }
        
        restricted = {}
        for name in safe_builtins:
            if hasattr(builtins, name):
                restricted[name] = getattr(builtins, name)
        
        # Add restricted import function
        restricted['__import__'] = self._restricted_import
        
        return restricted
    
    def _restricted_import(self, name: str, globals=None, locals=None, fromlist=(), level=0):
        """Restricted import function that only allows whitelisted modules."""
        if name not in self.allowed_modules:
            raise ImportError(f"Module '{name}' is not allowed in plugin sandbox")
        
        return importlib.import_module(name)
    
    def create_sandbox_globals(self, extra_globals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create globals dictionary for sandboxed execution."""
        sandbox_globals = {
            '__builtins__': self.restricted_builtins,
            '__name__': '__plugin__',
            '__doc__': None,
        }
        
        if extra_globals:
            sandbox_globals.update(extra_globals)
        
        return sandbox_globals

class PluginValidator:
    """Validates plugin code for security and correctness."""
    
    @staticmethod
    def validate_plugin_code(code: str) -> List[str]:
        """Validate plugin code for security issues."""
        errors = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax error: {e}"]
        
        # Check for dangerous patterns
        dangerous_patterns = [
            ('exec', 'Use of exec() is not allowed'),
            ('eval', 'Use of eval() is not allowed'),
            ('compile', 'Use of compile() is not allowed'),
            ('__import__', 'Direct use of __import__ is not allowed'),
            ('open', 'File operations are restricted in plugins'),
            ('os.system', 'System commands are not allowed'),
            ('subprocess', 'Subprocess calls are not allowed'),
        ]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in [p[0] for p in dangerous_patterns]:
                        for pattern, msg in dangerous_patterns:
                            if node.func.id == pattern:
                                errors.append(msg)
                
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['system', 'popen', 'remove', 'unlink']:
                        errors.append(f"System operation '{node.func.attr}' is not allowed")
        
        # Check for restricted imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] not in PluginSandbox().allowed_modules:
                        errors.append(f"Import of '{alias.name}' is not allowed")
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in PluginSandbox().allowed_modules:
                    errors.append(f"Import from '{node.module}' is not allowed")
        
        return errors
    
    @staticmethod
    def validate_plugin_structure(code: str) -> List[str]:
        """Validate that plugin has required structure."""
        errors = []
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax error: {e}"]
        
        # Check for plugin class
        plugin_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and 'PluginInterface' in base.id:
                        plugin_classes.append(node.name)
        
        if not plugin_classes:
            errors.append("No plugin class found that inherits from PluginInterface")
        
        # Check for required methods
        required_methods = ['get_metadata', 'initialize', 'cleanup']
        for class_node in ast.iter_child_nodes(tree):
            if isinstance(class_node, ast.ClassDef) and class_node.name in plugin_classes:
                methods = {node.name for node in ast.iter_child_nodes(class_node) 
                          if isinstance(node, ast.FunctionDef)}
                
                for method in required_methods:
                    if method not in methods:
                        errors.append(f"Plugin class missing required method: {method}")
        
        return errors

class PluginLoader:
    """Loads plugins from files or code strings with sandboxing."""
    
    def __init__(self, sandbox: Optional[PluginSandbox] = None):
        self.sandbox = sandbox or PluginSandbox()
        self.validator = PluginValidator()
    
    def load_from_file(self, file_path: Union[str, Path], 
                      plugin_name: Optional[str] = None) -> Optional[Type[PluginInterface]]:
        """Load a plugin from a Python file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"Plugin file not found: {file_path}")
            return None
        
        try:
            code = file_path.read_text(encoding='utf-8')
            return self.load_from_code(code, plugin_name or file_path.stem)
        except Exception as e:
            logger.error(f"Failed to load plugin from {file_path}: {e}")
            return None
    
    def load_from_code(self, code: str, plugin_name: str) -> Optional[Type[PluginInterface]]:
        """Load a plugin from code string."""
        # Validate code
        security_errors = self.validator.validate_plugin_code(code)
        structure_errors = self.validator.validate_plugin_structure(code)
        
        all_errors = security_errors + structure_errors
        if all_errors:
            logger.error(f"Plugin validation failed for {plugin_name}: {all_errors}")
            return None
        
        # Create module
        module = types.ModuleType(f"plugin_{plugin_name}")
        module.__file__ = f"<plugin:{plugin_name}>"
        
        # Execute in sandbox
        sandbox_globals = self.sandbox.create_sandbox_globals({
            '__name__': module.__name__,
            '__file__': module.__file__,
        })
        
        try:
            exec(compile(code, module.__file__, 'exec'), sandbox_globals)
        except Exception as e:
            logger.error(f"Failed to execute plugin code for {plugin_name}: {e}")
            return None
        
        # Extract plugin class
        plugin_class = None
        for name, obj in sandbox_globals.items():
            if (isinstance(obj, type) and 
                issubclass(obj, PluginInterface) and 
                obj is not PluginInterface):
                plugin_class = obj
                break
        
        if not plugin_class:
            logger.error(f"No plugin class found in {plugin_name}")
            return None
        
        # Verify it's the right type
        if not any(issubclass(plugin_class, interface) for interface in [
            TrainingPluginInterface,
            QuantizationPluginInterface,
            KernelPluginInterface,
            DataProcessingPluginInterface
        ]):
            logger.warning(f"Plugin {plugin_class.__name__} doesn't implement a specific interface")
        
        return plugin_class
    
    def load_from_module(self, module_path: str, 
                        class_name: Optional[str] = None) -> Optional[Type[PluginInterface]]:
        """Load a plugin from an installed module."""
        try:
            module = importlib.import_module(module_path)
            
            if class_name:
                plugin_class = getattr(module, class_name, None)
            else:
                # Find first PluginInterface subclass
                plugin_class = None
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, PluginInterface) and 
                        obj is not PluginInterface):
                        plugin_class = obj
                        break
            
            if plugin_class:
                return plugin_class
            else:
                logger.error(f"No plugin class found in module {module_path}")
                return None
                
        except ImportError as e:
            logger.error(f"Failed to import plugin module {module_path}: {e}")
            return None

class PluginInstance:
    """Wrapper for a plugin instance with lifecycle management."""
    
    def __init__(self, plugin_class: Type[PluginInterface], context: Dict[str, Any]):
        self.plugin_class = plugin_class
        self.context = context
        self.instance: Optional[PluginInterface] = None
        self.initialized = False
        self.load_time = time.time()
        self.last_used = time.time()
        self.usage_count = 0
        self.error_count = 0
        
        # Create instance
        try:
            self.instance = plugin_class()
        except Exception as e:
            logger.error(f"Failed to instantiate plugin {plugin_class.__name__}: {e}")
            self.instance = None
    
    def initialize(self) -> bool:
        """Initialize the plugin instance."""
        if not self.instance:
            return False
        
        try:
            self.initialized = self.instance.initialize(self.context)
            if self.initialized:
                logger.info(f"Plugin {self.instance.get_metadata().name} initialized successfully")
            else:
                logger.warning(f"Plugin {self.instance.get_metadata().name} initialization returned False")
            return self.initialized
        except Exception as e:
            logger.error(f"Failed to initialize plugin: {e}")
            self.error_count += 1
            return False
    
    def use(self, method_name: str, *args, **kwargs) -> Any:
        """Use a plugin method with error handling."""
        if not self.instance or not self.initialized:
            raise RuntimeError("Plugin not initialized")
        
        try:
            method = getattr(self.instance, method_name)
            result = method(*args, **kwargs)
            self.last_used = time.time()
            self.usage_count += 1
            return result
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in plugin method {method_name}: {e}")
            raise
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if self.instance and self.initialized:
            try:
                self.instance.cleanup()
                self.initialized = False
                logger.info(f"Plugin {self.instance.get_metadata().name} cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up plugin: {e}")
    
    def get_metadata(self) -> Optional[PluginMetadata]:
        """Get plugin metadata."""
        if self.instance:
            return self.instance.get_metadata()
        return None

class PluginManager:
    """Manages plugin lifecycle, discovery, and hot-reloading."""
    
    def __init__(self, plugin_dirs: Optional[List[Union[str, Path]]] = None,
                 context: Optional[Dict[str, Any]] = None):
        self.plugin_dirs = [Path(d) for d in (plugin_dirs or [])]
        self.context = context or {}
        self.plugins: Dict[str, PluginInstance] = {}
        self.plugin_classes: Dict[str, Type[PluginInterface]] = {}
        self.loader = PluginLoader()
        self.watcher_thread: Optional[threading.Thread] = None
        self.watcher_running = False
        self.file_checksums: Dict[str, str] = {}
        
        # Add default plugin directory if it exists
        default_dir = Path(__file__).parent.parent.parent.parent / "plugins"
        if default_dir.exists() and default_dir not in self.plugin_dirs:
            self.plugin_dirs.append(default_dir)
    
    def discover_plugins(self) -> List[Path]:
        """Discover plugin files in configured directories."""
        plugin_files = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                logger.warning(f"Plugin directory does not exist: {plugin_dir}")
                continue
            
            for file_path in plugin_dir.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue  # Skip private files
                
                plugin_files.append(file_path)
        
        return plugin_files
    
    def load_plugin(self, file_path: Union[str, Path], 
                   plugin_name: Optional[str] = None) -> bool:
        """Load a single plugin from file."""
        file_path = Path(file_path)
        name = plugin_name or file_path.stem
        
        # Calculate checksum for hot-reloading
        try:
            code = file_path.read_text(encoding='utf-8')
            checksum = hashlib.md5(code.encode()).hexdigest()
            self.file_checksums[str(file_path)] = checksum
        except Exception as e:
            logger.error(f"Failed to read plugin file {file_path}: {e}")
            return False
        
        # Load plugin class
        plugin_class = self.loader.load_from_file(file_path, name)
        if not plugin_class:
            return False
        
        # Store plugin class
        self.plugin_classes[name] = plugin_class
        
        # Create and initialize instance
        plugin_instance = PluginInstance(plugin_class, self.context)
        if plugin_instance.initialize():
            self.plugins[name] = plugin_instance
            logger.info(f"Loaded plugin: {name} (v{plugin_instance.get_metadata().version})")
            return True
        else:
            logger.error(f"Failed to initialize plugin: {name}")
            return False
    
    def load_all_plugins(self) -> Dict[str, bool]:
        """Load all discovered plugins."""
        results = {}
        plugin_files = self.discover_plugins()
        
        for file_path in plugin_files:
            try:
                success = self.load_plugin(file_path)
                results[file_path.stem] = success
            except Exception as e:
                logger.error(f"Failed to load plugin {file_path}: {e}")
                results[file_path.stem] = False
        
        return results
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Hot-reload a specific plugin."""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin not found: {plugin_name}")
            return False
        
        # Find the plugin file
        plugin_file = None
        for file_path in self.discover_plugins():
            if file_path.stem == plugin_name:
                plugin_file = file_path
                break
        
        if not plugin_file:
            logger.error(f"Plugin file not found for: {plugin_name}")
            return False
        
        # Check if file has changed
        try:
            current_code = plugin_file.read_text(encoding='utf-8')
            current_checksum = hashlib.md5(current_code.encode()).hexdigest()
            
            if self.file_checksums.get(str(plugin_file)) == current_checksum:
                logger.info(f"Plugin {plugin_name} has not changed, skipping reload")
                return True
        except Exception as e:
            logger.error(f"Failed to check plugin file: {e}")
            return False
        
        # Clean up old instance
        old_instance = self.plugins[plugin_name]
        old_instance.cleanup()
        
        # Reload
        logger.info(f"Reloading plugin: {plugin_name}")
        return self.load_plugin(plugin_file, plugin_name)
    
    def start_hot_reload(self, interval: float = 5.0) -> None:
        """Start hot-reload watcher thread."""
        if self.watcher_running:
            logger.warning("Hot-reload watcher already running")
            return
        
        self.watcher_running = True
        
        def watcher():
            while self.watcher_running:
                try:
                    self._check_for_changes()
                except Exception as e:
                    logger.error(f"Error in hot-reload watcher: {e}")
                time.sleep(interval)
        
        self.watcher_thread = threading.Thread(target=watcher, daemon=True)
        self.watcher_thread.start()
        logger.info(f"Started hot-reload watcher (interval: {interval}s)")
    
    def stop_hot_reload(self) -> None:
        """Stop hot-reload watcher thread."""
        self.watcher_running = False
        if self.watcher_thread:
            self.watcher_thread.join(timeout=2.0)
            self.watcher_thread = None
        logger.info("Stopped hot-reload watcher")
    
    def _check_for_changes(self) -> None:
        """Check for plugin file changes and reload if needed."""
        for file_path in self.discover_plugins():
            try:
                current_code = file_path.read_text(encoding='utf-8')
                current_checksum = hashlib.md5(current_code.encode()).hexdigest()
                stored_checksum = self.file_checksums.get(str(file_path))
                
                if stored_checksum and stored_checksum != current_checksum:
                    logger.info(f"Detected change in plugin: {file_path.stem}")
                    self.reload_plugin(file_path.stem)
            except Exception as e:
                logger.error(f"Error checking plugin file {file_path}: {e}")
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInstance]:
        """Get a plugin instance by name."""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInstance]:
        """Get all plugins of a specific type."""
        result = []
        for plugin in self.plugins.values():
            metadata = plugin.get_metadata()
            if metadata and metadata.plugin_type == plugin_type:
                result.append(plugin)
        return result
    
    def execute_plugin_method(self, plugin_name: str, method_name: str, 
                             *args, **kwargs) -> Any:
        """Execute a method on a specific plugin."""
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        return plugin.use(method_name, *args, **kwargs)
    
    def cleanup_all(self) -> None:
        """Clean up all plugins."""
        for plugin_name, plugin in list(self.plugins.items()):
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin_name}: {e}")
        
        self.plugins.clear()
        self.plugin_classes.clear()
        self.file_checksums.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plugin manager statistics."""
        return {
            "total_plugins": len(self.plugins),
            "plugin_types": {
                ptype.value: len(self.get_plugins_by_type(ptype))
                for ptype in PluginType
            },
            "total_usage": sum(p.usage_count for p in self.plugins.values()),
            "total_errors": sum(p.error_count for p in self.plugins.values()),
            "hot_reload_running": self.watcher_running,
        }
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins with their metadata."""
        plugins_list = []
        
        for name, plugin in self.plugins.items():
            metadata = plugin.get_metadata()
            if metadata:
                plugins_list.append({
                    "name": name,
                    "version": metadata.version,
                    "type": metadata.plugin_type.value,
                    "author": metadata.author,
                    "description": metadata.description,
                    "initialized": plugin.initialized,
                    "usage_count": plugin.usage_count,
                    "error_count": plugin.error_count,
                    "last_used": plugin.last_used,
                })
        
        return plugins_list

# Example plugin implementations for testing
class ExampleTrainingPlugin(TrainingPluginInterface):
    """Example training plugin implementation."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_training",
            version="1.0.0",
            plugin_type=PluginType.TRAINING,
            author="Unsoth Team",
            description="Example training loop plugin"
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        logger.info("Example training plugin initialized")
        return True
    
    def cleanup(self) -> None:
        logger.info("Example training plugin cleaned up")
    
    def create_training_loop(self, model: Any, optimizer: Any, **kwargs) -> Callable:
        def training_loop(data_loader, epochs=1):
            logger.info(f"Running custom training loop for {epochs} epochs")
            # Custom training logic would go here
            return {"status": "completed", "epochs": epochs}
        return training_loop
    
    def get_supported_optimizers(self) -> List[str]:
        return ["adam", "sgd", "adamw"]

class ExampleQuantizationPlugin(QuantizationPluginInterface):
    """Example quantization plugin implementation."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_quantization",
            version="1.0.0",
            plugin_type=PluginType.QUANTIZATION,
            author="Unsoth Team",
            description="Example quantization plugin"
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        logger.info("Example quantization plugin initialized")
        return True
    
    def cleanup(self) -> None:
        logger.info("Example quantization plugin cleaned up")
    
    def quantize_model(self, model: Any, config: Dict[str, Any]) -> Any:
        logger.info(f"Quantizing model with config: {config}")
        # Quantization logic would go here
        return model
    
    def get_quantization_methods(self) -> List[str]:
        return ["int8", "fp16", "dynamic"]

# Integration with existing Unsoth codebase
def create_plugin_manager(plugin_dirs: Optional[List[Union[str, Path]]] = None,
                         context: Optional[Dict[str, Any]] = None) -> PluginManager:
    """Factory function to create a plugin manager with default configuration."""
    
    # Default plugin directories
    default_dirs = []
    
    # Add user home directory plugins
    home_plugins = Path.home() / ".vex" / "plugins"
    if home_plugins.exists():
        default_dirs.append(home_plugins)
    
    # Add current working directory plugins
    cwd_plugins = Path.cwd() / "plugins"
    if cwd_plugins.exists():
        default_dirs.append(cwd_plugins)
    
    # Add provided directories
    if plugin_dirs:
        default_dirs.extend(plugin_dirs)
    
    # Create context with common utilities
    if context is None:
        context = {}
    
    # Add common utilities to context
    context.update({
        "logger": logger,
        "torch": None,  # Will be imported if available
        "numpy": None,  # Will be imported if available
    })
    
    # Try to import torch and numpy
    try:
        import torch
        context["torch"] = torch
    except ImportError:
        pass
    
    try:
        import numpy as np
        context["numpy"] = np
    except ImportError:
        pass
    
    return PluginManager(default_dirs, context)

# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None

def get_plugin_manager() -> PluginManager:
    """Get or create the global plugin manager instance."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = create_plugin_manager()
    return _plugin_manager

def set_plugin_manager(manager: PluginManager) -> None:
    """Set the global plugin manager instance."""
    global _plugin_manager
    _plugin_manager = manager

# Decorator for registering plugins
def plugin(plugin_type: PluginType, 
          name: Optional[str] = None,
          version: str = "1.0.0",
          author: str = "",
          description: str = ""):
    """Decorator to register a class as a plugin."""
    def decorator(cls):
        # Add metadata as class attribute
        cls._plugin_metadata = PluginMetadata(
            name=name or cls.__name__,
            version=version,
            plugin_type=plugin_type,
            author=author,
            description=description
        )
        
        # Ensure it inherits from PluginInterface
        if not issubclass(cls, PluginInterface):
            # Dynamically create a new class that inherits from both
            class PluginClass(cls, PluginInterface):
                def get_metadata(self):
                    return getattr(self, '_plugin_metadata', None)
                
                def initialize(self, context):
                    if hasattr(super(), 'initialize'):
                        return super().initialize(context)
                    return True
                
                def cleanup(self):
                    if hasattr(super(), 'cleanup'):
                        super().cleanup()
            
            PluginClass.__name__ = cls.__name__
            PluginClass.__qualname__ = cls.__qualname__
            return PluginClass
        
        return cls
    
    return decorator

# Context manager for plugin execution
class PluginExecutionContext:
    """Context manager for executing code with plugins."""
    
    def __init__(self, plugin_manager: Optional[PluginManager] = None):
        self.plugin_manager = plugin_manager or get_plugin_manager()
        self.original_manager = None
    
    def __enter__(self):
        self.original_manager = _plugin_manager
        set_plugin_manager(self.plugin_manager)
        return self.plugin_manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_manager:
            set_plugin_manager(self.original_manager)
        return False

# Utility functions for common plugin operations
def load_plugin_from_string(code: str, plugin_name: str = "dynamic_plugin") -> Optional[Type[PluginInterface]]:
    """Load a plugin from a string of code."""
    loader = PluginLoader()
    return loader.load_from_code(code, plugin_name)

def create_plugin_instance(plugin_class: Type[PluginInterface], 
                          context: Optional[Dict[str, Any]] = None) -> Optional[PluginInstance]:
    """Create a plugin instance from a class."""
    if context is None:
        context = get_plugin_manager().context
    
    instance = PluginInstance(plugin_class, context)
    if instance.initialize():
        return instance
    return None

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create plugin manager
    manager = create_plugin_manager()
    
    # Start hot-reload
    manager.start_hot_reload(interval=10.0)
    
    try:
        # Load all plugins
        results = manager.load_all_plugins()
        print(f"Loaded plugins: {results}")
        
        # List plugins
        plugins = manager.list_plugins()
        print(f"Available plugins: {plugins}")
        
        # Get stats
        stats = manager.get_stats()
        print(f"Plugin stats: {stats}")
        
        # Example: Execute a plugin method if available
        if "example_training" in manager.plugins:
            result = manager.execute_plugin_method(
                "example_training",
                "create_training_loop",
                model=None,
                optimizer=None
            )
            print(f"Training loop result: {result}")
    
    finally:
        # Clean up
        manager.stop_hot_reload()
        manager.cleanup_all()