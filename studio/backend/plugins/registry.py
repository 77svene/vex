"""
SOVEREIGN Plugin Registry System
Extensible architecture for custom models, data processors, and training strategies.
"""

import importlib
import importlib.metadata
import importlib.util
import inspect
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported by the registry."""
    MODEL = "model"
    DATA_PROCESSOR = "data_processor"
    TRAINING_STRATEGY = "training_strategy"
    EXTENSION = "extension"


@dataclass
class PluginMetadata:
    """Metadata for a registered plugin."""
    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    entry_point: Optional[str] = None
    module_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


class BasePlugin(ABC):
    """Base class for all plugins."""
    
    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """Return the type of this plugin."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this plugin."""
        pass
    
    @property
    def version(self) -> str:
        """Return the version of this plugin."""
        return "1.0.0"
    
    @property
    def description(self) -> str:
        """Return a description of this plugin."""
        return ""
    
    @property
    def author(self) -> str:
        """Return the author of this plugin."""
        return ""
    
    @property
    def dependencies(self) -> List[str]:
        """Return list of plugin dependencies."""
        return []
    
    @property
    def tags(self) -> List[str]:
        """Return tags for this plugin."""
        return []
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the plugin with optional configuration."""
        pass
    
    def validate(self) -> bool:
        """Validate that the plugin is properly configured."""
        return True


class ModelPlugin(BasePlugin):
    """Base class for model architecture plugins."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.MODEL
    
    @abstractmethod
    def create_model(self, **kwargs) -> Any:
        """Create and return a model instance."""
        pass
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Return default model configuration."""
        pass
    
    def supports_architecture(self, architecture_name: str) -> bool:
        """Check if this plugin supports a given architecture."""
        return False


class DataProcessorPlugin(BasePlugin):
    """Base class for data processing plugins."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.DATA_PROCESSOR
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process and return transformed data."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported data formats."""
        return []
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data before processing."""
        return True


class TrainingStrategyPlugin(BasePlugin):
    """Base class for training strategy plugins."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.TRAINING_STRATEGY
    
    @abstractmethod
    def configure_training(self, **kwargs) -> Dict[str, Any]:
        """Configure and return training parameters."""
        pass
    
    @abstractmethod
    def get_optimizer(self, **kwargs) -> Any:
        """Return configured optimizer."""
        pass
    
    @abstractmethod
    def get_scheduler(self, **kwargs) -> Any:
        """Return configured learning rate scheduler."""
        pass
    
    def supports_distributed(self) -> bool:
        """Check if this strategy supports distributed training."""
        return False


class ExtensionPlugin(BasePlugin):
    """Base class for general extension plugins."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.EXTENSION
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Any:
        """Execute the extension with given context."""
        pass
    
    @abstractmethod
    def get_hooks(self) -> List[str]:
        """Return list of hooks this extension registers for."""
        return []


class PluginFileWatcher(FileSystemEventHandler):
    """Watches for changes in plugin directories."""
    
    def __init__(self, registry: 'PluginRegistry'):
        self.registry = registry
        self.debounce_seconds = 2.0
        self.last_modified = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        src_path = event.src_path
        if not src_path.endswith('.py'):
            return
        
        current_time = importlib.util._bootstrap._get_supported_file_loaders()
        if src_path in self.last_modified:
            if current_time - self.last_modified[src_path] < self.debounce_seconds:
                return
        
        self.last_modified[src_path] = current_time
        logger.info(f"Plugin file modified: {src_path}")
        
        # Schedule reload
        import threading
        timer = threading.Timer(self.debounce_seconds, self._reload_plugin, [src_path])
        timer.start()
    
    def _reload_plugin(self, file_path: str):
        """Reload a specific plugin file."""
        try:
            self.registry.reload_plugin_by_path(file_path)
        except Exception as e:
            logger.error(f"Failed to reload plugin {file_path}: {e}")


class PluginRegistry:
    """
    Central registry for managing SOVEREIGN plugins.
    Supports discovery via entry points and direct registration.
    """
    
    _instance = None
    ENTRY_POINT_GROUP = "sovereign.plugins"
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PluginRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._plugins: Dict[str, BasePlugin] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        self._plugin_classes: Dict[PluginType, Set[Type[BasePlugin]]] = {
            plugin_type: set() for plugin_type in PluginType
        }
        self._watched_paths: Set[str] = set()
        self._observer: Optional[Observer] = None
        self._hot_reload_enabled = False
        
        # Plugin directories
        self.plugin_dirs = [
            Path.home() / ".sovereign" / "plugins",
            Path(__file__).parent.parent.parent.parent / "plugins",
            Path.cwd() / "plugins",
        ]
        
        self._initialized = True
    
    def discover_plugins(self, force_reload: bool = False) -> None:
        """
        Discover plugins from entry points and plugin directories.
        
        Args:
            force_reload: If True, reload already discovered plugins
        """
        if force_reload:
            self._plugins.clear()
            self._metadata.clear()
            for plugin_type in self._plugin_classes:
                self._plugin_classes[plugin_type].clear()
        
        # Discover from entry points
        self._discover_entry_points()
        
        # Discover from plugin directories
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists() and plugin_dir.is_dir():
                self._discover_directory(plugin_dir)
        
        logger.info(f"Discovered {len(self._plugins)} plugins")
    
    def _discover_entry_points(self) -> None:
        """Discover plugins registered via setuptools entry points."""
        try:
            # Python 3.9+ uses importlib.metadata
            if hasattr(importlib.metadata, 'entry_points'):
                eps = importlib.metadata.entry_points()
                if hasattr(eps, 'select'):
                    # Python 3.10+
                    plugin_eps = eps.select(group=self.ENTRY_POINT_GROUP)
                else:
                    # Python 3.9
                    plugin_eps = eps.get(self.ENTRY_POINT_GROUP, [])
            else:
                # Fallback to pkg_resources
                import pkg_resources
                plugin_eps = pkg_resources.iter_entry_points(self.ENTRY_POINT_GROUP)
            
            for entry_point in plugin_eps:
                try:
                    plugin_class = entry_point.load()
                    if self._is_valid_plugin_class(plugin_class):
                        self._register_plugin_class(plugin_class, entry_point=str(entry_point))
                except Exception as e:
                    logger.warning(f"Failed to load plugin from entry point {entry_point}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to discover entry points: {e}")
    
    def _discover_directory(self, directory: Path) -> None:
        """Discover plugins in a directory."""
        for item in directory.iterdir():
            if item.is_file() and item.suffix == '.py':
                self._load_plugin_file(item)
            elif item.is_dir() and (item / '__init__.py').exists():
                self._load_plugin_package(item)
    
    def _load_plugin_file(self, file_path: Path) -> None:
        """Load plugins from a Python file."""
        try:
            module_name = f"sovereign_plugin_{file_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                # Find plugin classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if self._is_valid_plugin_class(obj):
                        self._register_plugin_class(obj, module_path=str(file_path))
        
        except Exception as e:
            logger.warning(f"Failed to load plugin file {file_path}: {e}")
    
    def _load_plugin_package(self, package_path: Path) -> None:
        """Load plugins from a Python package."""
        try:
            package_name = f"sovereign_plugin_{package_path.name}"
            sys.path.insert(0, str(package_path.parent))
            
            module = importlib.import_module(package_path.name)
            
            # Find plugin classes in the package
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if self._is_valid_plugin_class(obj):
                    self._register_plugin_class(obj, module_path=str(package_path))
            
            sys.path.pop(0)
        
        except Exception as e:
            logger.warning(f"Failed to load plugin package {package_path}: {e}")
    
    def _is_valid_plugin_class(self, cls: Type) -> bool:
        """Check if a class is a valid plugin class."""
        if not inspect.isclass(cls):
            return False
        
        # Check if it's a subclass of BasePlugin (but not BasePlugin itself)
        if not issubclass(cls, BasePlugin) or cls == BasePlugin:
            return False
        
        # Check if it's abstract (has unimplemented abstract methods)
        if inspect.isabstract(cls):
            return False
        
        return True
    
    def _register_plugin_class(
        self, 
        plugin_class: Type[BasePlugin], 
        entry_point: Optional[str] = None,
        module_path: Optional[str] = None
    ) -> None:
        """Register a plugin class."""
        try:
            # Create instance to get metadata
            instance = plugin_class()
            plugin_name = instance.name
            
            if plugin_name in self._plugins:
                logger.warning(f"Plugin '{plugin_name}' already registered, skipping")
                return
            
            # Create metadata
            metadata = PluginMetadata(
                name=plugin_name,
                version=instance.version,
                plugin_type=instance.plugin_type,
                description=instance.description,
                author=instance.author,
                entry_point=entry_point,
                module_path=module_path,
                dependencies=instance.dependencies,
                tags=instance.tags
            )
            
            # Register plugin
            self._plugins[plugin_name] = instance
            self._metadata[plugin_name] = metadata
            self._plugin_classes[instance.plugin_type].add(plugin_class)
            
            logger.info(f"Registered plugin: {plugin_name} ({instance.plugin_type.value})")
        
        except Exception as e:
            logger.error(f"Failed to register plugin class {plugin_class}: {e}")
    
    def register_plugin(self, plugin: BasePlugin) -> None:
        """
        Manually register a plugin instance.
        
        Args:
            plugin: Plugin instance to register
        """
        plugin_name = plugin.name
        
        if plugin_name in self._plugins:
            logger.warning(f"Plugin '{plugin_name}' already registered, overwriting")
        
        metadata = PluginMetadata(
            name=plugin_name,
            version=plugin.version,
            plugin_type=plugin.plugin_type,
            description=plugin.description,
            author=plugin.author,
            dependencies=plugin.dependencies,
            tags=plugin.tags
        )
        
        self._plugins[plugin_name] = plugin
        self._metadata[plugin_name] = metadata
        self._plugin_classes[plugin.plugin_type].add(type(plugin))
        
        logger.info(f"Manually registered plugin: {plugin_name}")
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self._plugins.get(name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all plugins of a specific type."""
        return [
            plugin for plugin in self._plugins.values()
            if plugin.plugin_type == plugin_type
        ]
    
    def get_model_plugins(self) -> List[ModelPlugin]:
        """Get all model plugins."""
        return self.get_plugins_by_type(PluginType.MODEL)
    
    def get_data_processor_plugins(self) -> List[DataProcessorPlugin]:
        """Get all data processor plugins."""
        return self.get_plugins_by_type(PluginType.DATA_PROCESSOR)
    
    def get_training_strategy_plugins(self) -> List[TrainingStrategyPlugin]:
        """Get all training strategy plugins."""
        return self.get_plugins_by_type(PluginType.TRAINING_STRATEGY)
    
    def get_extension_plugins(self) -> List[ExtensionPlugin]:
        """Get all extension plugins."""
        return self.get_plugins_by_type(PluginType.EXTENSION)
    
    def get_plugin_metadata(self, name: str) -> Optional[PluginMetadata]:
        """Get metadata for a plugin."""
        return self._metadata.get(name)
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all registered plugins with their metadata."""
        return list(self._metadata.values())
    
    def enable_hot_reload(self, watch_dirs: Optional[List[Path]] = None) -> None:
        """
        Enable hot reloading of plugins.
        
        Args:
            watch_dirs: Additional directories to watch for changes
        """
        if self._hot_reload_enabled:
            return
        
        if watch_dirs:
            for watch_dir in watch_dirs:
                if watch_dir.exists():
                    self.plugin_dirs.append(watch_dir)
        
        # Start file watcher
        self._observer = Observer()
        event_handler = PluginFileWatcher(self)
        
        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                self._observer.schedule(event_handler, str(plugin_dir), recursive=True)
                self._watched_paths.add(str(plugin_dir))
        
        self._observer.start()
        self._hot_reload_enabled = True
        logger.info("Hot reloading enabled for plugins")
    
    def disable_hot_reload(self) -> None:
        """Disable hot reloading of plugins."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        
        self._hot_reload_enabled = False
        self._watched_paths.clear()
        logger.info("Hot reloading disabled")
    
    def reload_plugin_by_path(self, file_path: str) -> None:
        """Reload a plugin by its file path."""
        # Find plugins from this path
        plugins_to_reload = [
            name for name, metadata in self._metadata.items()
            if metadata.module_path == file_path
        ]
        
        for plugin_name in plugins_to_reload:
            self._reload_plugin(plugin_name)
    
    def _reload_plugin(self, plugin_name: str) -> None:
        """Reload a specific plugin."""
        if plugin_name not in self._plugins:
            return
        
        old_plugin = self._plugins[plugin_name]
        plugin_class = type(old_plugin)
        
        try:
            # Reload the module
            module = inspect.getmodule(plugin_class)
            if module:
                importlib.reload(module)
                
                # Re-register the plugin
                new_plugin = plugin_class()
                self._plugins[plugin_name] = new_plugin
                
                # Update metadata
                self._metadata[plugin_name] = PluginMetadata(
                    name=new_plugin.name,
                    version=new_plugin.version,
                    plugin_type=new_plugin.plugin_type,
                    description=new_plugin.description,
                    author=new_plugin.author,
                    dependencies=new_plugin.dependencies,
                    tags=new_plugin.tags
                )
                
                logger.info(f"Reloaded plugin: {plugin_name}")
        
        except Exception as e:
            logger.error(f"Failed to reload plugin {plugin_name}: {e}")
            # Restore old plugin
            self._plugins[plugin_name] = old_plugin
    
    def unload_plugin(self, name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            name: Name of the plugin to unload
            
        Returns:
            True if plugin was unloaded, False if not found
        """
        if name not in self._plugins:
            return False
        
        plugin = self._plugins[name]
        plugin_type = plugin.plugin_type
        
        # Remove from registry
        del self._plugins[name]
        del self._metadata[name]
        
        # Remove from plugin classes
        plugin_class = type(plugin)
        if plugin_class in self._plugin_classes[plugin_type]:
            self._plugin_classes[plugin_type].remove(plugin_class)
        
        logger.info(f"Unloaded plugin: {name}")
        return True
    
    def validate_plugin(self, name: str) -> bool:
        """Validate a plugin."""
        plugin = self.get_plugin(name)
        if not plugin:
            return False
        
        return plugin.validate()
    
    def get_plugin_dependencies(self, name: str) -> List[str]:
        """Get dependencies for a plugin."""
        metadata = self.get_plugin_metadata(name)
        if metadata:
            return metadata.dependencies
        return []
    
    def check_dependencies(self, name: str) -> Dict[str, bool]:
        """Check if all dependencies for a plugin are satisfied."""
        dependencies = self.get_plugin_dependencies(name)
        results = {}
        
        for dep in dependencies:
            # Check if dependency is another plugin
            if dep in self._plugins:
                results[dep] = True
            else:
                # Check if it's a Python package
                try:
                    importlib.import_module(dep)
                    results[dep] = True
                except ImportError:
                    results[dep] = False
        
        return results


# Global registry instance
plugin_registry = PluginRegistry()


# Convenience functions
def get_plugin(name: str) -> Optional[BasePlugin]:
    """Get a plugin by name."""
    return plugin_registry.get_plugin(name)


def get_model_plugins() -> List[ModelPlugin]:
    """Get all model plugins."""
    return plugin_registry.get_model_plugins()


def get_data_processor_plugins() -> List[DataProcessorPlugin]:
    """Get all data processor plugins."""
    return plugin_registry.get_data_processor_plugins()


def get_training_strategy_plugins() -> List[TrainingStrategyPlugin]:
    """Get all training strategy plugins."""
    return plugin_registry.get_training_strategy_plugins()


def get_extension_plugins() -> List[ExtensionPlugin]:
    """Get all extension plugins."""
    return plugin_registry.get_extension_plugins()


def discover_plugins(force_reload: bool = False) -> None:
    """Discover plugins from all sources."""
    plugin_registry.discover_plugins(force_reload)


def enable_hot_reload(watch_dirs: Optional[List[Path]] = None) -> None:
    """Enable hot reloading of plugins."""
    plugin_registry.enable_hot_reload(watch_dirs)


def disable_hot_reload() -> None:
    """Disable hot reloading of plugins."""
    plugin_registry.disable_hot_reload()


# Example plugin implementations (for reference)
class ExampleModelPlugin(ModelPlugin):
    """Example model plugin implementation."""
    
    @property
    def name(self) -> str:
        return "example_model"
    
    @property
    def description(self) -> str:
        return "An example model plugin for demonstration"
    
    def create_model(self, **kwargs) -> Any:
        # Implementation would create and return a model
        return None
    
    def get_model_config(self) -> Dict[str, Any]:
        return {"hidden_size": 768, "num_layers": 12}


class ExampleDataProcessorPlugin(DataProcessorPlugin):
    """Example data processor plugin implementation."""
    
    @property
    def name(self) -> str:
        return "example_processor"
    
    @property
    def description(self) -> str:
        return "An example data processor plugin"
    
    def process(self, data: Any, **kwargs) -> Any:
        # Implementation would process data
        return data
    
    def get_supported_formats(self) -> List[str]:
        return ["json", "csv", "parquet"]


class ExampleTrainingStrategyPlugin(TrainingStrategyPlugin):
    """Example training strategy plugin implementation."""
    
    @property
    def name(self) -> str:
        return "example_strategy"
    
    @property
    def description(self) -> str:
        return "An example training strategy plugin"
    
    def configure_training(self, **kwargs) -> Dict[str, Any]:
        return {
            "learning_rate": 1e-4,
            "batch_size": 32,
            "epochs": 10
        }
    
    def get_optimizer(self, **kwargs) -> Any:
        # Implementation would return optimizer
        return None
    
    def get_scheduler(self, **kwargs) -> Any:
        # Implementation would return scheduler
        return None


# Auto-discover plugins on import
discover_plugins()