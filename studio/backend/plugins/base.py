"""
Unsloth Studio Plugin System
Extensible architecture for custom models, data processors, and training strategies.
"""

import abc
import importlib
import importlib.metadata
import inspect
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

from studio.backend.core.data_recipe.jobs.types import DataRecipe, TrainingJob

logger = logging.getLogger(__name__)

# Type variables for generic plugin types
T = TypeVar("T")
PluginType = TypeVar("PluginType", bound="BasePlugin")


class PluginCategory(Enum):
    """Categories of plugins supported by the system."""
    MODEL = "model"
    DATA_PROCESSOR = "data_processor"
    TRAINING_STRATEGY = "training_strategy"
    METRIC = "metric"
    CALLBACK = "callback"
    EXPORT = "export"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    category: PluginCategory
    description: str = ""
    author: str = ""
    homepage: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0  # Higher priority plugins are preferred


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginLoadError(PluginError):
    """Raised when a plugin fails to load."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin is not found."""
    pass


class BasePlugin(abc.ABC):
    """
    Abstract base class for all plugins.
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    # Class-level metadata (can be overridden by subclasses)
    _metadata: Optional[PluginMetadata] = None
    
    def __init__(self):
        self._initialized = False
        self._lock = threading.RLock()
    
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """Get plugin metadata."""
        if cls._metadata is None:
            # Generate metadata from class attributes if not explicitly set
            name = getattr(cls, "PLUGIN_NAME", cls.__name__)
            version = getattr(cls, "PLUGIN_VERSION", "1.0.0")
            category = getattr(cls, "PLUGIN_CATEGORY", PluginCategory.MODEL)
            description = getattr(cls, "__doc__", "").strip() if cls.__doc__ else ""
            
            cls._metadata = PluginMetadata(
                name=name,
                version=version,
                category=category,
                description=description,
                author=getattr(cls, "PLUGIN_AUTHOR", ""),
                homepage=getattr(cls, "PLUGIN_HOMEPAGE", ""),
                dependencies=getattr(cls, "PLUGIN_DEPENDENCIES", []),
                tags=getattr(cls, "PLUGIN_TAGS", []),
            )
        return cls._metadata
    
    @abc.abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    @abc.abstractmethod
    def validate(self) -> bool:
        """Validate that the plugin is properly configured."""
        pass
    
    def teardown(self) -> None:
        """Clean up resources when plugin is unloaded."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.teardown()


class BaseModelPlugin(BasePlugin):
    """
    Base class for custom model architecture plugins.
    
    Plugins should implement model creation, configuration, and any custom forward passes.
    """
    
    PLUGIN_CATEGORY = PluginCategory.MODEL
    
    @abc.abstractmethod
    def create_model(self, config: Dict[str, Any]) -> Any:
        """Create and return a model instance."""
        pass
    
    @abc.abstractmethod
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        pass
    
    @abc.abstractmethod
    def list_available_models(self) -> List[str]:
        """List all available model architectures provided by this plugin."""
        pass
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model."""
        return {
            "name": model_name,
            "plugin": self.get_metadata().name,
            "category": self.get_metadata().category.value,
        }


class BaseDataProcessorPlugin(BasePlugin):
    """
    Base class for data processing plugins.
    
    Plugins should implement data loading, transformation, and preprocessing.
    """
    
    PLUGIN_CATEGORY = PluginCategory.DATA_PROCESSOR
    
    @abc.abstractmethod
    def process_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Process input data according to configuration."""
        pass
    
    @abc.abstractmethod
    def get_supported_formats(self) -> List[str]:
        """List supported data formats."""
        pass
    
    @abc.abstractmethod
    def create_data_recipe(self, source: str, **kwargs) -> DataRecipe:
        """Create a data recipe from a data source."""
        pass
    
    def validate_data(self, data: Any) -> bool:
        """Validate that data can be processed by this processor."""
        return True


class BaseTrainingStrategyPlugin(BasePlugin):
    """
    Base class for training strategy plugins.
    
    Plugins should implement custom training loops, optimizers, schedulers, etc.
    """
    
    PLUGIN_CATEGORY = PluginCategory.TRAINING_STRATEGY
    
    @abc.abstractmethod
    def create_training_job(
        self,
        model: Any,
        data: Any,
        config: Dict[str, Any],
    ) -> TrainingJob:
        """Create a training job with this strategy."""
        pass
    
    @abc.abstractmethod
    def get_optimizer(self, model_parameters: Any, config: Dict[str, Any]) -> Any:
        """Get optimizer for training."""
        pass
    
    @abc.abstractmethod
    def get_scheduler(self, optimizer: Any, config: Dict[str, Any]) -> Any:
        """Get learning rate scheduler."""
        pass
    
    @abc.abstractmethod
    def get_loss_function(self, config: Dict[str, Any]) -> Any:
        """Get loss function for training."""
        pass


class BaseMetricPlugin(BasePlugin):
    """Base class for metric plugins."""
    
    PLUGIN_CATEGORY = PluginCategory.METRIC
    
    @abc.abstractmethod
    def compute_metric(self, predictions: Any, targets: Any, **kwargs) -> float:
        """Compute a metric value."""
        pass
    
    @abc.abstractmethod
    def get_metric_name(self) -> str:
        """Get the name of this metric."""
        pass


class BaseCallbackPlugin(BasePlugin):
    """Base class for callback plugins."""
    
    PLUGIN_CATEGORY = PluginCategory.CALLBACK
    
    @abc.abstractmethod
    def on_train_begin(self, **kwargs) -> None:
        """Called at the beginning of training."""
        pass
    
    @abc.abstractmethod
    def on_train_end(self, **kwargs) -> None:
        """Called at the end of training."""
        pass
    
    @abc.abstractmethod
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Called at the beginning of an epoch."""
        pass
    
    @abc.abstractmethod
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any], **kwargs) -> None:
        """Called at the end of an epoch."""
        pass


class BaseExportPlugin(BasePlugin):
    """Base class for model export plugins."""
    
    PLUGIN_CATEGORY = PluginCategory.EXPORT
    
    @abc.abstractmethod
    def export_model(self, model: Any, path: Path, format: str, **kwargs) -> Path:
        """Export a model to a specific format."""
        pass
    
    @abc.abstractmethod
    def get_supported_formats(self) -> List[str]:
        """List supported export formats."""
        pass


# Plugin discovery and management

class PluginRegistry:
    """Registry for managing loaded plugins."""
    
    def __init__(self):
        self._plugins: Dict[PluginCategory, Dict[str, Type[BasePlugin]]] = {
            category: {} for category in PluginCategory
        }
        self._instances: Dict[str, BasePlugin] = {}
        self._lock = threading.RLock()
    
    def register_plugin(
        self,
        plugin_class: Type[BasePlugin],
        name: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Register a plugin class."""
        metadata = plugin_class.get_metadata()
        plugin_name = name or metadata.name
        category = metadata.category
        
        with self._lock:
            if not force and plugin_name in self._plugins[category]:
                existing = self._plugins[category][plugin_name]
                if existing != plugin_class:
                    raise PluginLoadError(
                        f"Plugin '{plugin_name}' already registered in category {category.value}"
                    )
            
            self._plugins[category][plugin_name] = plugin_class
            logger.info(f"Registered plugin: {plugin_name} ({category.value})")
    
    def unregister_plugin(self, name: str, category: Optional[PluginCategory] = None) -> None:
        """Unregister a plugin."""
        with self._lock:
            if category:
                if name in self._plugins[category]:
                    del self._plugins[category][name]
                    # Also remove instance if exists
                    instance_key = f"{category.value}:{name}"
                    if instance_key in self._instances:
                        self._instances[instance_key].teardown()
                        del self._instances[instance_key]
            else:
                # Remove from all categories
                for cat in PluginCategory:
                    if name in self._plugins[cat]:
                        del self._plugins[cat][name]
                        instance_key = f"{cat.value}:{name}"
                        if instance_key in self._instances:
                            self._instances[instance_key].teardown()
                            del self._instances[instance_key]
    
    def get_plugin_class(
        self,
        name: str,
        category: Optional[PluginCategory] = None,
    ) -> Optional[Type[BasePlugin]]:
        """Get a plugin class by name and optional category."""
        with self._lock:
            if category:
                return self._plugins[category].get(name)
            
            # Search all categories
            for cat_plugins in self._plugins.values():
                if name in cat_plugins:
                    return cat_plugins[name]
        return None
    
    def get_plugin_instance(
        self,
        name: str,
        category: Optional[PluginCategory] = None,
        **init_kwargs,
    ) -> Optional[BasePlugin]:
        """Get or create a plugin instance."""
        plugin_class = self.get_plugin_class(name, category)
        if not plugin_class:
            return None
        
        with self._lock:
            # Determine the actual category
            actual_category = category or plugin_class.get_metadata().category
            instance_key = f"{actual_category.value}:{name}"
            
            if instance_key not in self._instances:
                try:
                    instance = plugin_class()
                    instance.initialize(**init_kwargs)
                    if not instance.validate():
                        raise PluginError(f"Plugin {name} validation failed")
                    self._instances[instance_key] = instance
                except Exception as e:
                    logger.error(f"Failed to initialize plugin {name}: {e}")
                    raise PluginLoadError(f"Failed to initialize plugin {name}: {e}")
            
            return self._instances[instance_key]
    
    def list_plugins(
        self,
        category: Optional[PluginCategory] = None,
        enabled_only: bool = True,
    ) -> List[PluginMetadata]:
        """List all registered plugins."""
        plugins = []
        with self._lock:
            categories = [category] if category else list(PluginCategory)
            for cat in categories:
                for name, plugin_class in self._plugins[cat].items():
                    metadata = plugin_class.get_metadata()
                    if not enabled_only or metadata.enabled:
                        plugins.append(metadata)
        
        # Sort by priority (higher first) then by name
        plugins.sort(key=lambda p: (-p.priority, p.name))
        return plugins
    
    def clear(self) -> None:
        """Clear all plugins and instances."""
        with self._lock:
            for instance in self._instances.values():
                try:
                    instance.teardown()
                except Exception as e:
                    logger.warning(f"Error during plugin teardown: {e}")
            
            self._plugins = {category: {} for category in PluginCategory}
            self._instances.clear()


class PluginDiscovery:
    """Discovers plugins from various sources."""
    
    ENTRY_POINT_GROUP = "vex.plugins"
    
    @classmethod
    def discover_entry_points(cls) -> List[Type[BasePlugin]]:
        """Discover plugins registered via entry points."""
        plugins = []
        
        try:
            # Try importlib.metadata first (Python 3.8+)
            try:
                from importlib.metadata import entry_points
                eps = entry_points()
                if hasattr(eps, 'select'):  # Python 3.10+
                    plugin_eps = eps.select(group=cls.ENTRY_POINT_GROUP)
                else:  # Python 3.8-3.9
                    plugin_eps = eps.get(cls.ENTRY_POINT_GROUP, [])
            except (ImportError, AttributeError):
                # Fallback to pkg_resources for older Python
                import pkg_resources
                plugin_eps = pkg_resources.iter_entry_points(cls.ENTRY_POINT_GROUP)
            
            for ep in plugin_eps:
                try:
                    plugin_class = ep.load()
                    if inspect.isclass(plugin_class) and issubclass(plugin_class, BasePlugin):
                        plugins.append(plugin_class)
                        logger.debug(f"Discovered plugin via entry point: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin from entry point {ep.name}: {e}")
        
        except Exception as e:
            logger.warning(f"Failed to discover entry points: {e}")
        
        return plugins
    
    @classmethod
    def discover_directory(cls, directory: Path) -> List[Type[BasePlugin]]:
        """Discover plugins in a directory."""
        plugins = []
        
        if not directory.exists() or not directory.is_dir():
            return plugins
        
        # Add directory to Python path temporarily
        sys.path.insert(0, str(directory))
        
        try:
            for py_file in directory.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                module_name = py_file.stem
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find all plugin classes in the module
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (issubclass(obj, BasePlugin) and 
                            obj != BasePlugin and
                            not inspect.isabstract(obj)):
                            plugins.append(obj)
                            logger.debug(f"Discovered plugin in {directory}: {name}")
                
                except Exception as e:
                    logger.warning(f"Failed to load plugin from {py_file}: {e}")
        
        finally:
            # Remove directory from path
            if str(directory) in sys.path:
                sys.path.remove(str(directory))
        
        return plugins
    
    @classmethod
    def discover_module(cls, module_name: str) -> List[Type[BasePlugin]]:
        """Discover plugins in a Python module."""
        plugins = []
        
        try:
            module = importlib.import_module(module_name)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BasePlugin) and 
                    obj != BasePlugin and
                    not inspect.isabstract(obj)):
                    plugins.append(obj)
                    logger.debug(f"Discovered plugin in module {module_name}: {name}")
        
        except Exception as e:
            logger.warning(f"Failed to discover plugins in module {module_name}: {e}")
        
        return plugins


class PluginHotReloader:
    """Hot-reloads plugins when files change."""
    
    def __init__(self, plugin_manager: "PluginManager"):
        self.plugin_manager = plugin_manager
        self.watched_paths: Set[Path] = set()
        self.observer: Optional[Observer] = None
        self._lock = threading.RLock()
        
        if not WATCHDOG_AVAILABLE:
            logger.warning(
                "Watchdog not installed. Hot-reloading disabled. "
                "Install with: pip install watchdog"
            )
    
    def watch_directory(self, directory: Path) -> None:
        """Start watching a directory for changes."""
        if not WATCHDOG_AVAILABLE:
            return
        
        with self._lock:
            directory = Path(directory).resolve()
            if directory in self.watched_paths:
                return
            
            self.watched_paths.add(directory)
            
            if self.observer is None:
                self.observer = Observer()
                self.observer.start()
            
            event_handler = PluginFileHandler(self.plugin_manager, directory)
            self.observer.schedule(event_handler, str(directory), recursive=True)
            logger.info(f"Watching directory for plugin changes: {directory}")
    
    def unwatch_directory(self, directory: Path) -> None:
        """Stop watching a directory."""
        if not WATCHDOG_AVAILABLE or self.observer is None:
            return
        
        with self._lock:
            directory = Path(directory).resolve()
            if directory not in self.watched_paths:
                return
            
            self.watched_paths.remove(directory)
            
            # Note: watchdog doesn't have a clean way to unschedule a specific path
            # We'll need to restart the observer if we want to remove a path
            if not self.watched_paths:
                self.observer.stop()
                self.observer = None
    
    def stop(self) -> None:
        """Stop all watching."""
        if self.observer:
            self.observer.stop()
            self.observer = None
        self.watched_paths.clear()


if WATCHDOG_AVAILABLE:
    class PluginFileHandler(FileSystemEventHandler):
        """Handles file system events for plugin hot-reloading."""
        
        def __init__(self, plugin_manager: "PluginManager", base_path: Path):
            self.plugin_manager = plugin_manager
            self.base_path = base_path
            self._last_reload: Dict[str, float] = {}
            self._reload_delay = 1.0  # seconds
        
        def on_modified(self, event):
            if event.is_directory:
                return
            
            src_path = Path(event.src_path)
            
            # Only handle Python files
            if src_path.suffix != ".py":
                return
            
            # Debounce reloads
            current_time = time.time()
            file_key = str(src_path)
            
            if (file_key in self._last_reload and 
                current_time - self._last_reload[file_key] < self._reload_delay):
                return
            
            self._last_reload[file_key] = current_time
            
            # Determine module name from path
            try:
                rel_path = src_path.relative_to(self.base_path)
                module_parts = list(rel_path.with_suffix("").parts)
                
                # Convert path to module name
                module_name = ".".join(module_parts)
                
                logger.info(f"Detected change in {src_path}, reloading plugins...")
                self.plugin_manager.reload_plugins_from_module(module_name)
            
            except Exception as e:
                logger.error(f"Failed to reload plugin from {src_path}: {e}")


class PluginManager:
    """
    Main plugin manager for the Unsloth Studio system.
    
    Discovers, loads, and manages plugins for models, data processing, and training.
    """
    
    _instance: Optional["PluginManager"] = None
    _instance_lock = threading.RLock()
    
    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.registry = PluginRegistry()
        self.hot_reloader = PluginHotReloader(self)
        self._plugin_dirs: List[Path] = []
        self._loaded_modules: Set[str] = set()
        self._lock = threading.RLock()
        self._initialized = True
        
        # Register built-in example plugins
        self._register_example_plugins()
        
        logger.info("PluginManager initialized")
    
    def _register_example_plugins(self) -> None:
        """Register example plugins for demonstration."""
        # These would be actual plugin implementations
        # For now, we'll just log that they would be registered
        logger.debug("Example plugins would be registered here")
    
    def add_plugin_directory(self, directory: Union[str, Path]) -> None:
        """Add a directory to search for plugins."""
        directory = Path(directory).resolve()
        
        with self._lock:
            if directory not in self._plugin_dirs:
                self._plugin_dirs.append(directory)
                
                # Discover plugins in this directory
                self._discover_from_directory(directory)
                
                # Start watching for changes if hot-reload is enabled
                self.hot_reloader.watch_directory(directory)
    
    def _discover_from_directory(self, directory: Path) -> None:
        """Discover and register plugins from a directory."""
        plugins = PluginDiscovery.discover_directory(directory)
        for plugin_class in plugins:
            try:
                self.registry.register_plugin(plugin_class)
            except PluginLoadError as e:
                logger.warning(f"Failed to register plugin {plugin_class.__name__}: {e}")
    
    def discover_all(self) -> None:
        """Discover plugins from all configured sources."""
        # Discover from entry points
        entry_point_plugins = PluginDiscovery.discover_entry_points()
        for plugin_class in entry_point_plugins:
            try:
                self.registry.register_plugin(plugin_class)
            except PluginLoadError as e:
                logger.warning(f"Failed to register entry point plugin: {e}")
        
        # Discover from configured directories
        for directory in self._plugin_dirs:
            self._discover_from_directory(directory)
        
        # Discover from known modules
        known_modules = [
            "studio.backend.plugins.models",
            "studio.backend.plugins.processors",
            "studio.backend.plugins.strategies",
        ]
        
        for module_name in known_modules:
            plugins = PluginDiscovery.discover_module(module_name)
            for plugin_class in plugins:
                try:
                    self.registry.register_plugin(plugin_class)
                except PluginLoadError as e:
                    logger.warning(f"Failed to register plugin from {module_name}: {e}")
    
    def get_plugin(
        self,
        name: str,
        category: Optional[PluginCategory] = None,
        **init_kwargs,
    ) -> Optional[BasePlugin]:
        """Get a plugin instance by name."""
        return self.registry.get_plugin_instance(name, category, **init_kwargs)
    
    def get_model_plugin(self, name: str, **kwargs) -> Optional[BaseModelPlugin]:
        """Get a model plugin by name."""
        plugin = self.get_plugin(name, PluginCategory.MODEL, **kwargs)
        return plugin if isinstance(plugin, BaseModelPlugin) else None
    
    def get_data_processor_plugin(
        self, name: str, **kwargs
    ) -> Optional[BaseDataProcessorPlugin]:
        """Get a data processor plugin by name."""
        plugin = self.get_plugin(name, PluginCategory.DATA_PROCESSOR, **kwargs)
        return plugin if isinstance(plugin, BaseDataProcessorPlugin) else None
    
    def get_training_strategy_plugin(
        self, name: str, **kwargs
    ) -> Optional[BaseTrainingStrategyPlugin]:
        """Get a training strategy plugin by name."""
        plugin = self.get_plugin(name, PluginCategory.TRAINING_STRATEGY, **kwargs)
        return plugin if isinstance(plugin, BaseTrainingStrategyPlugin) else None
    
    def list_plugins(
        self,
        category: Optional[PluginCategory] = None,
        enabled_only: bool = True,
    ) -> List[PluginMetadata]:
        """List all available plugins."""
        return self.registry.list_plugins(category, enabled_only)
    
    def reload_plugins_from_module(self, module_name: str) -> None:
        """Reload plugins from a specific module."""
        with self._lock:
            try:
                # Reload the module
                if module_name in sys.modules:
                    module = importlib.reload(sys.modules[module_name])
                else:
                    module = importlib.import_module(module_name)
                
                # Find and re-register plugin classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BasePlugin) and 
                        obj != BasePlugin and
                        not inspect.isabstract(obj)):
                        
                        # Unregister old version if exists
                        metadata = obj.get_metadata()
                        self.registry.unregister_plugin(
                            metadata.name, metadata.category
                        )
                        
                        # Register new version
                        self.registry.register_plugin(obj)
                        logger.info(f"Reloaded plugin: {metadata.name}")
                
                self._loaded_modules.add(module_name)
            
            except Exception as e:
                logger.error(f"Failed to reload plugins from {module_name}: {e}")
    
    def enable_plugin(self, name: str, category: Optional[PluginCategory] = None) -> bool:
        """Enable a plugin."""
        plugin_class = self.registry.get_plugin_class(name, category)
        if plugin_class:
            plugin_class.get_metadata().enabled = True
            return True
        return False
    
    def disable_plugin(self, name: str, category: Optional[PluginCategory] = None) -> bool:
        """Disable a plugin."""
        plugin_class = self.registry.get_plugin_class(name, category)
        if plugin_class:
            plugin_class.get_metadata().enabled = False
            return True
        return False
    
    def cleanup(self) -> None:
        """Clean up all plugins and stop hot-reloading."""
        self.hot_reloader.stop()
        self.registry.clear()
        self._plugin_dirs.clear()
        self._loaded_modules.clear()
        logger.info("PluginManager cleaned up")
    
    def __del__(self):
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass


# Convenience functions for easy access

def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    return PluginManager()


def discover_plugins() -> None:
    """Discover all available plugins."""
    manager = get_plugin_manager()
    manager.discover_all()


def get_plugin(name: str, category: Optional[PluginCategory] = None, **kwargs) -> Optional[BasePlugin]:
    """Get a plugin instance by name."""
    manager = get_plugin_manager()
    return manager.get_plugin(name, category, **kwargs)


def list_plugins(category: Optional[PluginCategory] = None) -> List[PluginMetadata]:
    """List all available plugins."""
    manager = get_plugin_manager()
    return manager.list_plugins(category)


# Decorator for easy plugin registration

def plugin(
    name: Optional[str] = None,
    category: Optional[PluginCategory] = None,
    version: str = "1.0.0",
    description: str = "",
    **metadata_kwargs,
):
    """
    Decorator to register a class as a plugin.
    
    Example:
        @plugin(name="llama", category=PluginCategory.MODEL)
        class LlamaModelPlugin(BaseModelPlugin):
            ...
    """
    def decorator(cls: Type[PluginType]) -> Type[PluginType]:
        # Set metadata
        metadata = PluginMetadata(
            name=name or cls.__name__,
            version=version,
            category=category or getattr(cls, "PLUGIN_CATEGORY", PluginCategory.MODEL),
            description=description or cls.__doc__ or "",
            **metadata_kwargs,
        )
        cls._metadata = metadata
        
        # Auto-register if plugin manager exists
        try:
            manager = get_plugin_manager()
            manager.registry.register_plugin(cls)
        except Exception as e:
            logger.debug(f"Could not auto-register plugin {cls.__name__}: {e}")
        
        return cls
    
    return decorator


# Example plugin implementations (stubs for demonstration)

@plugin(name="example_llama", category=PluginCategory.MODEL, description="Example LLaMA model plugin")
class ExampleLlamaModelPlugin(BaseModelPlugin):
    """Example implementation of a model plugin for LLaMA architecture."""
    
    def initialize(self, **kwargs) -> None:
        self.config = kwargs
        logger.info("ExampleLlamaModelPlugin initialized")
    
    def validate(self) -> bool:
        return True
    
    def create_model(self, config: Dict[str, Any]) -> Any:
        # This would create an actual model in a real implementation
        logger.info(f"Creating LLaMA model with config: {config}")
        return {"type": "llama", "config": config}
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        return {"hidden_size": 4096, "num_layers": 32}
    
    def list_available_models(self) -> List[str]:
        return ["llama-7b", "llama-13b", "llama-70b"]


@plugin(name="example_hf_processor", category=PluginCategory.DATA_PROCESSOR)
class ExampleHFDataProcessorPlugin(BaseDataProcessorPlugin):
    """Example Hugging Face data processor plugin."""
    
    def initialize(self, **kwargs) -> None:
        self.tokenizer = None
        logger.info("ExampleHFDataProcessorPlugin initialized")
    
    def validate(self) -> bool:
        return True
    
    def process_data(self, data: Any, config: Dict[str, Any]) -> Any:
        # This would process data in a real implementation
        logger.info(f"Processing data with config: {config}")
        return {"processed": True, "data": data}
    
    def get_supported_formats(self) -> List[str]:
        return ["json", "csv", "parquet", "huggingface_dataset"]
    
    def create_data_recipe(self, source: str, **kwargs) -> DataRecipe:
        # This would create an actual DataRecipe in a real implementation
        from studio.backend.core.data_recipe.jobs.types import DataRecipe
        return DataRecipe(
            name=f"recipe_from_{source}",
            source=source,
            transformations=[],
        )


# Integration with existing codebase

def integrate_with_data_recipe_system():
    """
    Integrate plugin system with existing data recipe system.
    
    This function should be called during application startup to ensure
    plugins can extend the data recipe system.
    """
    from studio.backend.core.data_recipe.jobs.manager import DataRecipeManager
    
    manager = get_plugin_manager()
    
    # Register data processor plugins with the data recipe system
    data_processors = manager.list_plugins(PluginCategory.DATA_PROCESSOR)
    logger.info(f"Found {len(data_processors)} data processor plugins")
    
    # This would integrate with the actual DataRecipeManager
    # For now, just log the integration
    logger.info("Plugin system integrated with data recipe system")


def integrate_with_training_system():
    """
    Integrate plugin system with training system.
    
    This function should be called during application startup to ensure
    plugins can extend the training system.
    """
    manager = get_plugin_manager()
    
    # Get available training strategies
    strategies = manager.list_plugins(PluginCategory.TRAINING_STRATEGY)
    logger.info(f"Found {len(strategies)} training strategy plugins")
    
    # Get available metrics
    metrics = manager.list_plugins(PluginCategory.METRIC)
    logger.info(f"Found {len(metrics)} metric plugins")
    
    # Get available callbacks
    callbacks = manager.list_plugins(PluginCategory.CALLBACK)
    logger.info(f"Found {len(callbacks)} callback plugins")


# Auto-discovery on module import
def _auto_discover():
    """Auto-discover plugins when module is imported."""
    try:
        discover_plugins()
    except Exception as e:
        logger.warning(f"Auto-discovery failed: {e}")


# Run auto-discovery when module is imported
_auto_discover()