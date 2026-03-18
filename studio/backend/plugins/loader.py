"""
studio/backend/plugins/loader.py
Plugin System for Custom Models & Extensions
"""

import sys
import importlib
import importlib.metadata
import importlib.util
import logging
import inspect
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Types of plugins supported by the system."""
    MODEL = "model"
    DATA_PROCESSOR = "data_processor"
    TRAINING_STRATEGY = "training_strategy"
    CUSTOM = "custom"


class PluginStatus(Enum):
    """Plugin status states."""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    plugin_type: PluginType
    description: str = ""
    author: str = ""
    entry_point: str = ""
    dependencies: List[str] = field(default_factory=list)
    vex_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    checksum: Optional[str] = None


class PluginBase(ABC):
    """Base class for all plugins."""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self.status = PluginStatus.DISCOVERED
        self.loaded_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self._lock = threading.RLock()
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate plugin configuration and dependencies."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities and features."""
        pass
    
    def activate(self) -> bool:
        """Activate the plugin for use."""
        with self._lock:
            if self.status != PluginStatus.LOADED:
                return False
            self.status = PluginStatus.ACTIVE
            return True
    
    def deactivate(self) -> bool:
        """Deactivate the plugin."""
        with self._lock:
            if self.status != PluginStatus.ACTIVE:
                return False
            self.status = PluginStatus.LOADED
            return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "type": self.metadata.plugin_type.value,
            "status": self.status.value,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "error": self.error,
            "capabilities": self.get_capabilities()
        }


class ModelPlugin(PluginBase):
    """Base class for model architecture plugins."""
    
    @abstractmethod
    def create_model(self, config: Dict[str, Any]) -> Any:
        """Create and return a model instance."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        pass
    
    @abstractmethod
    def supports_inference(self) -> bool:
        """Check if model supports inference."""
        pass
    
    @abstractmethod
    def supports_training(self) -> bool:
        """Check if model supports training."""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_inference": self.supports_inference(),
            "supports_training": self.supports_training(),
            "model_info": self.get_model_info()
        }


class DataProcessorPlugin(PluginBase):
    """Base class for data processor plugins."""
    
    @abstractmethod
    def process(self, data: Any, config: Dict[str, Any]) -> Any:
        """Process data according to configuration."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get list of supported data formats."""
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """Validate input data."""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supported_formats": self.get_supported_formats(),
            "supports_validation": True
        }


class TrainingStrategyPlugin(PluginBase):
    """Base class for training strategy plugins."""
    
    @abstractmethod
    def create_trainer(self, model: Any, config: Dict[str, Any]) -> Any:
        """Create and return a trainer instance."""
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for this strategy."""
        pass
    
    @abstractmethod
    def supports_distributed(self) -> bool:
        """Check if strategy supports distributed training."""
        pass
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "supports_distributed": self.supports_distributed(),
            "default_hyperparameters": self.get_hyperparameters()
        }


class PluginLoadError(Exception):
    """Exception raised when plugin loading fails."""
    pass


class PluginRegistry:
    """Registry for managing discovered and loaded plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, PluginBase] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self._lock = threading.RLock()
    
    def register(self, plugin: PluginBase) -> bool:
        """Register a plugin in the registry."""
        with self._lock:
            if plugin.metadata.name in self._plugins:
                logger.warning(f"Plugin {plugin.metadata.name} already registered")
                return False
            
            self._plugins[plugin.metadata.name] = plugin
            self._plugins_by_type[plugin.metadata.plugin_type].append(plugin.metadata.name)
            logger.info(f"Registered plugin: {plugin.metadata.name}")
            return True
    
    def unregister(self, plugin_name: str) -> bool:
        """Unregister a plugin from the registry."""
        with self._lock:
            if plugin_name not in self._plugins:
                return False
            
            plugin = self._plugins[plugin_name]
            self._plugins_by_type[plugin.metadata.plugin_type].remove(plugin_name)
            del self._plugins[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name}")
            return True
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """Get a plugin by name."""
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginBase]:
        """Get all plugins of a specific type."""
        return [self._plugins[name] for name in self._plugins_by_type.get(plugin_type, [])]
    
    def get_all_plugins(self) -> List[PluginBase]:
        """Get all registered plugins."""
        return list(self._plugins.values())
    
    def clear(self):
        """Clear all plugins from the registry."""
        with self._lock:
            self._plugins.clear()
            for plugin_type in self._plugins_by_type:
                self._plugins_by_type[plugin_type].clear()


class PluginLoader:
    """
    Main plugin loader for discovering, loading, and managing plugins.
    
    Supports:
    - Plugin discovery via entry points
    - Hot-reloading of plugins
    - Multiple plugin types (models, data processors, training strategies)
    - Plugin validation and dependency checking
    - Dynamic plugin loading from directories
    """
    
    # Entry point groups for different plugin types
    ENTRY_POINT_GROUPS = {
        PluginType.MODEL: "vex.model_plugins",
        PluginType.DATA_PROCESSOR: "vex.data_processor_plugins",
        PluginType.TRAINING_STRATEGY: "vex.training_strategy_plugins",
        PluginType.CUSTOM: "vex.custom_plugins"
    }
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.registry = PluginRegistry()
        self.plugin_dirs = plugin_dirs or []
        self._watched_files: Dict[str, float] = {}  # file_path -> last_modified
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_watching = threading.Event()
        self._lock = threading.RLock()
        
        # Add default plugin directories
        default_dirs = [
            Path.home() / ".vex" / "plugins",
            Path("/etc/vex/plugins"),
            Path("plugins")  # Relative to current directory
        ]
        self.plugin_dirs.extend(str(d) for d in default_dirs if d.exists())
        
        logger.info(f"Plugin loader initialized with directories: {self.plugin_dirs}")
    
    def discover_plugins(self) -> List[PluginMetadata]:
        """
        Discover available plugins from entry points and plugin directories.
        
        Returns:
            List of discovered plugin metadata
        """
        discovered = []
        
        # Discover from entry points
        discovered.extend(self._discover_entry_point_plugins())
        
        # Discover from plugin directories
        discovered.extend(self._discover_directory_plugins())
        
        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered
    
    def _discover_entry_point_plugins(self) -> List[PluginMetadata]:
        """Discover plugins from Python entry points."""
        discovered = []
        
        try:
            # Try importlib.metadata (Python 3.8+)
            for plugin_type, group in self.ENTRY_POINT_GROUPS.items():
                try:
                    eps = importlib.metadata.entry_points()
                    # Handle different Python versions
                    if hasattr(eps, 'select'):
                        # Python 3.10+
                        entries = eps.select(group=group)
                    elif isinstance(eps, dict):
                        # Python 3.8-3.9
                        entries = eps.get(group, [])
                    else:
                        entries = []
                    
                    for entry_point in entries:
                        try:
                            metadata = self._extract_metadata_from_entry_point(entry_point, plugin_type)
                            if metadata:
                                discovered.append(metadata)
                        except Exception as e:
                            logger.error(f"Failed to extract metadata from entry point {entry_point}: {e}")
                except Exception as e:
                    logger.debug(f"No entry points found for group {group}: {e}")
        
        except Exception as e:
            logger.error(f"Error discovering entry point plugins: {e}")
        
        return discovered
    
    def _extract_metadata_from_entry_point(self, entry_point, plugin_type: PluginType) -> Optional[PluginMetadata]:
        """Extract metadata from an entry point."""
        try:
            # Try to load the entry point to get metadata
            # This is a lightweight load - just enough to get metadata
            module_name = entry_point.module if hasattr(entry_point, 'module') else entry_point.value.split(':')[0]
            
            # Calculate checksum of the module
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                with open(spec.origin, 'rb') as f:
                    checksum = hashlib.md5(f.read()).hexdigest()
            else:
                checksum = None
            
            # Create metadata
            return PluginMetadata(
                name=entry_point.name,
                version="1.0.0",  # Default version, can be overridden by plugin
                plugin_type=plugin_type,
                description=f"Plugin loaded from entry point: {entry_point}",
                entry_point=str(entry_point),
                checksum=checksum
            )
        except Exception as e:
            logger.error(f"Failed to extract metadata from entry point {entry_point}: {e}")
            return None
    
    def _discover_directory_plugins(self) -> List[PluginMetadata]:
        """Discover plugins from plugin directories."""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
            
            # Look for plugin manifest files
            for manifest_file in plugin_path.glob("**/plugin.json"):
                try:
                    metadata = self._load_plugin_manifest(manifest_file)
                    if metadata:
                        discovered.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to load plugin manifest {manifest_file}: {e}")
            
            # Also look for Python files that might be plugins
            for py_file in plugin_path.glob("**/*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                try:
                    metadata = self._extract_metadata_from_file(py_file)
                    if metadata:
                        discovered.append(metadata)
                except Exception as e:
                    logger.error(f"Failed to extract metadata from {py_file}: {e}")
        
        return discovered
    
    def _load_plugin_manifest(self, manifest_path: Path) -> Optional[PluginMetadata]:
        """Load plugin metadata from a manifest file."""
        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            required = ['name', 'version', 'type']
            if not all(field in data for field in required):
                logger.warning(f"Plugin manifest {manifest_path} missing required fields")
                return None
            
            # Map type string to PluginType
            type_map = {
                'model': PluginType.MODEL,
                'data_processor': PluginType.DATA_PROCESSOR,
                'training_strategy': PluginType.TRAINING_STRATEGY,
                'custom': PluginType.CUSTOM
            }
            
            plugin_type = type_map.get(data['type'].lower())
            if not plugin_type:
                logger.warning(f"Unknown plugin type: {data['type']}")
                return None
            
            # Calculate checksum of the plugin directory
            checksum = self._calculate_directory_checksum(manifest_path.parent)
            
            return PluginMetadata(
                name=data['name'],
                version=data['version'],
                plugin_type=plugin_type,
                description=data.get('description', ''),
                author=data.get('author', ''),
                dependencies=data.get('dependencies', []),
                vex_version=data.get('vex_version'),
                checksum=checksum
            )
        except Exception as e:
            logger.error(f"Failed to load manifest {manifest_path}: {e}")
            return None
    
    def _extract_metadata_from_file(self, file_path: Path) -> Optional[PluginMetadata]:
        """Extract metadata from a Python file by parsing docstrings and attributes."""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for plugin metadata in comments or docstrings
            # This is a simple implementation - could be enhanced
            lines = content.split('\n')
            metadata = {}
            
            for line in lines[:50]:  # Check first 50 lines
                line = line.strip()
                if line.startswith('# PLUGIN_'):
                    # Parse plugin metadata from comments
                    parts = line[9:].split(':', 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip().lower(), parts[1].strip()
                        metadata[key] = value
            
            if 'name' not in metadata:
                # Use filename as plugin name
                metadata['name'] = file_path.stem
            
            if 'type' not in metadata:
                # Try to infer type from filename or content
                if 'model' in file_path.stem.lower():
                    metadata['type'] = 'model'
                elif 'processor' in file_path.stem.lower() or 'data' in file_path.stem.lower():
                    metadata['type'] = 'data_processor'
                elif 'strategy' in file_path.stem.lower() or 'train' in file_path.stem.lower():
                    metadata['type'] = 'training_strategy'
                else:
                    metadata['type'] = 'custom'
            
            # Calculate file checksum
            with open(file_path, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()
            
            type_map = {
                'model': PluginType.MODEL,
                'data_processor': PluginType.DATA_PROCESSOR,
                'training_strategy': PluginType.TRAINING_STRATEGY,
                'custom': PluginType.CUSTOM
            }
            
            return PluginMetadata(
                name=metadata.get('name', file_path.stem),
                version=metadata.get('version', '1.0.0'),
                plugin_type=type_map.get(metadata['type'], PluginType.CUSTOM),
                description=metadata.get('description', f'Plugin from {file_path.name}'),
                author=metadata.get('author', ''),
                checksum=checksum
            )
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return None
    
    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate checksum for a directory."""
        hasher = hashlib.md5()
        
        for file_path in sorted(directory.rglob('*')):
            if file_path.is_file() and not file_path.name.startswith('.'):
                try:
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
                except:
                    pass
        
        return hasher.hexdigest()
    
    def load_plugin(self, plugin_name: str, plugin_metadata: Optional[PluginMetadata] = None) -> Optional[PluginBase]:
        """
        Load a specific plugin by name.
        
        Args:
            plugin_name: Name of the plugin to load
            plugin_metadata: Optional metadata if already discovered
            
        Returns:
            Loaded plugin instance or None if failed
        """
        with self._lock:
            # Check if already loaded
            existing = self.registry.get_plugin(plugin_name)
            if existing and existing.status in [PluginStatus.LOADED, PluginStatus.ACTIVE]:
                logger.info(f"Plugin {plugin_name} already loaded")
                return existing
            
            # Discover if metadata not provided
            if not plugin_metadata:
                discovered = self.discover_plugins()
                plugin_metadata = next((p for p in discovered if p.name == plugin_name), None)
            
            if not plugin_metadata:
                logger.error(f"Plugin {plugin_name} not found")
                return None
            
            try:
                # Load the plugin based on type
                plugin = self._load_plugin_by_type(plugin_metadata)
                
                if plugin:
                    # Initialize the plugin
                    if plugin.initialize():
                        plugin.status = PluginStatus.LOADED
                        plugin.loaded_at = datetime.now()
                        
                        # Register in registry
                        self.registry.register(plugin)
                        
                        # Start watching for changes if hot-reload enabled
                        self._watch_plugin(plugin)
                        
                        logger.info(f"Successfully loaded plugin: {plugin_name}")
                        return plugin
                    else:
                        plugin.status = PluginStatus.ERROR
                        plugin.error = "Initialization failed"
                        logger.error(f"Failed to initialize plugin: {plugin_name}")
                
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_name}: {e}")
                if plugin:
                    plugin.status = PluginStatus.ERROR
                    plugin.error = str(e)
        
        return None
    
    def _load_plugin_by_type(self, metadata: PluginMetadata) -> Optional[PluginBase]:
        """Load a plugin based on its type."""
        # This is a factory method that would create the appropriate plugin type
        # For now, we'll create a generic plugin wrapper
        # In a real implementation, this would load the actual plugin class
        
        class GenericPlugin(PluginBase):
            """Generic plugin wrapper for demonstration."""
            
            def initialize(self) -> bool:
                # In a real implementation, this would import and initialize the plugin
                return True
            
            def validate(self) -> bool:
                return True
            
            def get_capabilities(self) -> Dict[str, Any]:
                return {"generic": True}
        
        return GenericPlugin(metadata)
    
    def load_all_plugins(self, plugin_type: Optional[PluginType] = None) -> List[PluginBase]:
        """
        Load all discovered plugins, optionally filtered by type.
        
        Args:
            plugin_type: Optional filter for plugin type
            
        Returns:
            List of loaded plugins
        """
        loaded_plugins = []
        discovered = self.discover_plugins()
        
        for metadata in discovered:
            if plugin_type and metadata.plugin_type != plugin_type:
                continue
            
            plugin = self.load_plugin(metadata.name, metadata)
            if plugin:
                loaded_plugins.append(plugin)
        
        return loaded_plugins
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if successfully unloaded
        """
        with self._lock:
            plugin = self.registry.get_plugin(plugin_name)
            if not plugin:
                return False
            
            # Stop watching
            self._stop_watching_plugin(plugin)
            
            # Deactivate if active
            if plugin.status == PluginStatus.ACTIVE:
                plugin.deactivate()
            
            # Unregister
            self.registry.unregister(plugin_name)
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
    
    def reload_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """
        Reload a plugin (useful for hot-reloading).
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            Reloaded plugin instance or None if failed
        """
        logger.info(f"Reloading plugin: {plugin_name}")
        
        # Unload first
        self.unload_plugin(plugin_name)
        
        # Reload
        return self.load_plugin(plugin_name)
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """Get a loaded plugin by name."""
        return self.registry.get_plugin(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginBase]:
        """Get all loaded plugins of a specific type."""
        return self.registry.get_plugins_by_type(plugin_type)
    
    def get_all_plugins(self) -> List[PluginBase]:
        """Get all loaded plugins."""
        return self.registry.get_all_plugins()
    
    def activate_plugin(self, plugin_name: str) -> bool:
        """Activate a loaded plugin."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.activate()
        return False
    
    def deactivate_plugin(self, plugin_name: str) -> bool:
        """Deactivate an active plugin."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.deactivate()
        return False
    
    def enable_hot_reload(self, check_interval: float = 5.0):
        """
        Enable hot-reloading of plugins.
        
        Args:
            check_interval: Interval in seconds to check for changes
        """
        if self._watch_thread and self._watch_thread.is_alive():
            logger.warning("Hot-reload already enabled")
            return
        
        self._stop_watching.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_for_changes,
            args=(check_interval,),
            daemon=True,
            name="PluginHotReload"
        )
        self._watch_thread.start()
        logger.info(f"Hot-reload enabled with interval: {check_interval}s")
    
    def disable_hot_reload(self):
        """Disable hot-reloading of plugins."""
        if self._watch_thread:
            self._stop_watching.set()
            self._watch_thread.join(timeout=10)
            self._watch_thread = None
            logger.info("Hot-reload disabled")
    
    def _watch_plugin(self, plugin: PluginBase):
        """Start watching a plugin for changes."""
        # In a real implementation, this would track the plugin's source files
        pass
    
    def _stop_watching_plugin(self, plugin: PluginBase):
        """Stop watching a plugin for changes."""
        # In a real implementation, this would remove the plugin from watch list
        pass
    
    def _watch_for_changes(self, check_interval: float):
        """Watch for plugin file changes and trigger reloads."""
        while not self._stop_watching.is_set():
            try:
                # Check each loaded plugin for changes
                for plugin in self.registry.get_all_plugins():
                    if self._check_plugin_changed(plugin):
                        logger.info(f"Detected change in plugin {plugin.metadata.name}, reloading...")
                        self.reload_plugin(plugin.metadata.name)
            except Exception as e:
                logger.error(f"Error in hot-reload watcher: {e}")
            
            # Wait for next check
            self._stop_watching.wait(check_interval)
    
    def _check_plugin_changed(self, plugin: PluginBase) -> bool:
        """Check if a plugin's source files have changed."""
        # This is a simplified implementation
        # In reality, you'd check file modification times or checksums
        return False
    
    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a plugin."""
        plugin = self.get_plugin(plugin_name)
        if plugin:
            return plugin.get_info()
        return None
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
        """
        List all plugins with their information.
        
        Args:
            plugin_type: Optional filter by plugin type
            
        Returns:
            List of plugin information dictionaries
        """
        plugins = self.get_all_plugins()
        
        if plugin_type:
            plugins = [p for p in plugins if p.metadata.plugin_type == plugin_type]
        
        return [plugin.get_info() for plugin in plugins]
    
    def validate_dependencies(self, plugin_name: str) -> Dict[str, Any]:
        """
        Validate a plugin's dependencies.
        
        Args:
            plugin_name: Name of the plugin to validate
            
        Returns:
            Validation result with missing/available dependencies
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            return {"valid": False, "error": "Plugin not found"}
        
        missing_deps = []
        available_deps = []
        
        for dep in plugin.metadata.dependencies:
            try:
                importlib.import_module(dep)
                available_deps.append(dep)
            except ImportError:
                missing_deps.append(dep)
        
        return {
            "valid": len(missing_deps) == 0,
            "missing_dependencies": missing_deps,
            "available_dependencies": available_deps
        }


# Global plugin loader instance
_plugin_loader: Optional[PluginLoader] = None
_loader_lock = threading.Lock()


def get_plugin_loader(plugin_dirs: Optional[List[str]] = None) -> PluginLoader:
    """
    Get the global plugin loader instance (singleton pattern).
    
    Args:
        plugin_dirs: Optional list of plugin directories
        
    Returns:
        Global PluginLoader instance
    """
    global _plugin_loader
    
    with _loader_lock:
        if _plugin_loader is None:
            _plugin_loader = PluginLoader(plugin_dirs)
        return _plugin_loader


def reset_plugin_loader():
    """Reset the global plugin loader (useful for testing)."""
    global _plugin_loader
    
    with _loader_lock:
        if _plugin_loader:
            _plugin_loader.disable_hot_reload()
            _plugin_loader.registry.clear()
        _plugin_loader = None


# Example plugin implementations for demonstration
class ExampleModelPlugin(ModelPlugin):
    """Example model plugin for demonstration."""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="example_model",
            version="1.0.0",
            plugin_type=PluginType.MODEL,
            description="Example model plugin for demonstration"
        )
        super().__init__(metadata)
    
    def initialize(self) -> bool:
        logger.info("Initializing example model plugin")
        return True
    
    def validate(self) -> bool:
        return True
    
    def create_model(self, config: Dict[str, Any]) -> Any:
        # This would create an actual model in a real implementation
        return {"type": "example_model", "config": config}
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "architecture": "ExampleModel",
            "parameters": "100M",
            "supports_flash_attention": True
        }
    
    def supports_inference(self) -> bool:
        return True
    
    def supports_training(self) -> bool:
        return True


class ExampleDataProcessorPlugin(DataProcessorPlugin):
    """Example data processor plugin for demonstration."""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="example_processor",
            version="1.0.0",
            plugin_type=PluginType.DATA_PROCESSOR,
            description="Example data processor plugin"
        )
        super().__init__(metadata)
    
    def initialize(self) -> bool:
        logger.info("Initializing example data processor plugin")
        return True
    
    def validate(self) -> bool:
        return True
    
    def process(self, data: Any, config: Dict[str, Any]) -> Any:
        # This would process data in a real implementation
        return {"processed": True, "data": data, "config": config}
    
    def get_supported_formats(self) -> List[str]:
        return ["json", "csv", "parquet"]
    
    def validate_data(self, data: Any) -> bool:
        return True


class ExampleTrainingStrategyPlugin(TrainingStrategyPlugin):
    """Example training strategy plugin for demonstration."""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="example_strategy",
            version="1.0.0",
            plugin_type=PluginType.TRAINING_STRATEGY,
            description="Example training strategy plugin"
        )
        super().__init__(metadata)
    
    def initialize(self) -> bool:
        logger.info("Initializing example training strategy plugin")
        return True
    
    def validate(self) -> bool:
        return True
    
    def create_trainer(self, model: Any, config: Dict[str, Any]) -> Any:
        # This would create a trainer in a real implementation
        return {"type": "example_trainer", "model": model, "config": config}
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        return {
            "learning_rate": 2e-5,
            "batch_size": 8,
            "num_epochs": 3,
            "warmup_steps": 100
        }
    
    def supports_distributed(self) -> bool:
        return True


# Convenience functions for common operations
def load_plugin(plugin_name: str) -> Optional[PluginBase]:
    """Load a plugin by name using the global loader."""
    return get_plugin_loader().load_plugin(plugin_name)


def get_plugin(plugin_name: str) -> Optional[PluginBase]:
    """Get a loaded plugin by name using the global loader."""
    return get_plugin_loader().get_plugin(plugin_name)


def list_plugins(plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
    """List all plugins using the global loader."""
    return get_plugin_loader().list_plugins(plugin_type)


def enable_hot_reload(check_interval: float = 5.0):
    """Enable hot-reload using the global loader."""
    get_plugin_loader().enable_hot_reload(check_interval)


# Integration with existing vex modules
def integrate_with_core():
    """Integrate plugin system with existing vex core modules."""
    try:
        # Import existing modules to ensure compatibility
        from studio.backend.core.data_recipe.huggingface import HuggingFaceDataRecipe
        from studio.backend.core.data_recipe.jobs.manager import JobManager
        
        logger.info("Plugin system integrated with existing vex core modules")
        
        # Register example plugins for demonstration
        loader = get_plugin_loader()
        
        # These would be registered via entry points in a real deployment
        example_model = ExampleModelPlugin()
        example_processor = ExampleDataProcessorPlugin()
        example_strategy = ExampleTrainingStrategyPlugin()
        
        # Initialize and register
        for plugin in [example_model, example_processor, example_strategy]:
            if plugin.initialize():
                plugin.status = PluginStatus.LOADED
                plugin.loaded_at = datetime.now()
                loader.registry.register(plugin)
        
        logger.info(f"Registered {len([example_model, example_processor, example_strategy])} example plugins")
        
    except ImportError as e:
        logger.warning(f"Could not integrate with all core modules: {e}")


# Auto-integrate when module is imported
integrate_with_core()