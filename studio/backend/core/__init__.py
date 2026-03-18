# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unified core module for Unsloth backend

Imports are LAZY (via __getattr__) so that training subprocesses can
import core.training.worker without pulling in heavy ML dependencies
like vex, transformers, or torch before the version activation
code has a chance to run.

Plugin system for custom models, data processors, and training strategies.
"""

import sys
from pathlib import Path
import importlib.metadata
import importlib.util
import logging
from typing import Dict, List, Optional, Type, Any, Callable
from abc import ABC, abstractmethod

# Ensure the backend directory is on sys.path so that bare "from utils.*"
# imports used throughout the backend work when core is imported as a package
# (e.g. from the CLI: "from studio.backend.core import ModelConfig").
_backend_dir = str(Path(__file__).resolve().parent.parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

logger = logging.getLogger(__name__)

# Plugin System Base Classes
class ModelPlugin(ABC):
    """Base class for custom model architecture plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the model architecture."""
        pass
    
    @property
    @abstractmethod
    def supported_tasks(self) -> List[str]:
        """List of tasks this model supports (e.g., ['text-generation', 'text-classification'])."""
        pass
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> Any:
        """Load and return the model instance."""
        pass
    
    @abstractmethod
    def get_tokenizer(self, model_path: str, **kwargs) -> Any:
        """Load and return the tokenizer for this model."""
        pass
    
    def get_training_config(self, **kwargs) -> Dict[str, Any]:
        """Return default training configuration for this model."""
        return {}


class DataProcessorPlugin(ABC):
    """Base class for custom data processor plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the data processor."""
        pass
    
    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """Process and return the transformed data."""
        pass
    
    def validate(self, data: Any) -> bool:
        """Validate if data is compatible with this processor."""
        return True


class TrainingStrategyPlugin(ABC):
    """Base class for custom training strategy plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the training strategy."""
        pass
    
    @abstractmethod
    def configure_training(self, model: Any, **kwargs) -> Dict[str, Any]:
        """Configure and return training parameters."""
        pass
    
    @abstractmethod
    def get_optimizer(self, model_parameters, **kwargs) -> Any:
        """Return configured optimizer."""
        pass
    
    @abstractmethod
    def get_scheduler(self, optimizer: Any, **kwargs) -> Any:
        """Return configured learning rate scheduler."""
        pass


# Plugin Registry
class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        self._model_plugins: Dict[str, Type[ModelPlugin]] = {}
        self._data_processor_plugins: Dict[str, Type[DataProcessorPlugin]] = {}
        self._training_strategy_plugins: Dict[str, Type[TrainingStrategyPlugin]] = {}
        self._entry_point_group = "vex.plugins"
        self._loaded = False
    
    def _discover_entry_points(self) -> Dict[str, List[Any]]:
        """Discover plugins via entry points."""
        plugins = {
            "model": [],
            "data_processor": [],
            "training_strategy": []
        }
        
        try:
            eps = importlib.metadata.entry_points()
            
            # Handle different Python versions for entry_points
            if hasattr(eps, 'select'):
                # Python 3.10+
                plugin_eps = eps.select(group=self._entry_point_group)
            else:
                # Python 3.9 and below
                plugin_eps = eps.get(self._entry_point_group, [])
            
            for ep in plugin_eps:
                try:
                    plugin_class = ep.load()
                    
                    if issubclass(plugin_class, ModelPlugin):
                        plugins["model"].append(plugin_class)
                    elif issubclass(plugin_class, DataProcessorPlugin):
                        plugins["data_processor"].append(plugin_class)
                    elif issubclass(plugin_class, TrainingStrategyPlugin):
                        plugins["training_strategy"].append(plugin_class)
                        
                except Exception as e:
                    logger.warning(f"Failed to load plugin {ep.name}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to discover plugins via entry points: {e}")
        
        return plugins
    
    def _discover_file_plugins(self, plugin_dir: Optional[Path] = None) -> Dict[str, List[Any]]:
        """Discover plugins from Python files in a directory."""
        plugins = {
            "model": [],
            "data_processor": [],
            "training_strategy": []
        }
        
        if plugin_dir is None:
            # Default plugin directory
            plugin_dir = Path(__file__).parent.parent / "plugins"
        
        if not plugin_dir.exists():
            return plugins
        
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
                
            try:
                module_name = f"vex_plugin_{plugin_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    # Look for plugin classes in the module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        if isinstance(attr, type):
                            if issubclass(attr, ModelPlugin) and attr is not ModelPlugin:
                                plugins["model"].append(attr)
                            elif issubclass(attr, DataProcessorPlugin) and attr is not DataProcessorPlugin:
                                plugins["data_processor"].append(attr)
                            elif issubclass(attr, TrainingStrategyPlugin) and attr is not TrainingStrategyPlugin:
                                plugins["training_strategy"].append(attr)
                                
            except Exception as e:
                logger.warning(f"Failed to load plugin file {plugin_file}: {e}")
        
        return plugins
    
    def load_plugins(self, plugin_dir: Optional[Path] = None) -> None:
        """Load all available plugins."""
        if self._loaded:
            return
        
        # Discover from entry points
        entry_plugins = self._discover_entry_points()
        
        # Discover from plugin directory
        file_plugins = self._discover_file_plugins(plugin_dir)
        
        # Register all discovered plugins
        for plugin_class in entry_plugins["model"] + file_plugins["model"]:
            self.register_model_plugin(plugin_class)
        
        for plugin_class in entry_plugins["data_processor"] + file_plugins["data_processor"]:
            self.register_data_processor_plugin(plugin_class)
        
        for plugin_class in entry_plugins["training_strategy"] + file_plugins["training_strategy"]:
            self.register_training_strategy_plugin(plugin_class)
        
        self._loaded = True
        logger.info(f"Loaded {len(self._model_plugins)} model plugins, "
                   f"{len(self._data_processor_plugins)} data processor plugins, "
                   f"{len(self._training_strategy_plugins)} training strategy plugins")
    
    def register_model_plugin(self, plugin_class: Type[ModelPlugin]) -> None:
        """Register a model plugin."""
        if not issubclass(plugin_class, ModelPlugin):
            raise TypeError("Plugin must be a subclass of ModelPlugin")
        
        instance = plugin_class()
        self._model_plugins[instance.name] = plugin_class
        logger.debug(f"Registered model plugin: {instance.name}")
    
    def register_data_processor_plugin(self, plugin_class: Type[DataProcessorPlugin]) -> None:
        """Register a data processor plugin."""
        if not issubclass(plugin_class, DataProcessorPlugin):
            raise TypeError("Plugin must be a subclass of DataProcessorPlugin")
        
        instance = plugin_class()
        self._data_processor_plugins[instance.name] = plugin_class
        logger.debug(f"Registered data processor plugin: {instance.name}")
    
    def register_training_strategy_plugin(self, plugin_class: Type[TrainingStrategyPlugin]) -> None:
        """Register a training strategy plugin."""
        if not issubclass(plugin_class, TrainingStrategyPlugin):
            raise TypeError("Plugin must be a subclass of TrainingStrategyPlugin")
        
        instance = plugin_class()
        self._training_strategy_plugins[instance.name] = plugin_class
        logger.debug(f"Registered training strategy plugin: {instance.name}")
    
    def get_model_plugin(self, name: str) -> Optional[Type[ModelPlugin]]:
        """Get a model plugin by name."""
        return self._model_plugins.get(name)
    
    def get_data_processor_plugin(self, name: str) -> Optional[Type[DataProcessorPlugin]]:
        """Get a data processor plugin by name."""
        return self._data_processor_plugins.get(name)
    
    def get_training_strategy_plugin(self, name: str) -> Optional[Type[TrainingStrategyPlugin]]:
        """Get a training strategy plugin by name."""
        return self._training_strategy_plugins.get(name)
    
    def list_model_plugins(self) -> List[str]:
        """List all registered model plugin names."""
        return list(self._model_plugins.keys())
    
    def list_data_processor_plugins(self) -> List[str]:
        """List all registered data processor plugin names."""
        return list(self._data_processor_plugins.keys())
    
    def list_training_strategy_plugins(self) -> List[str]:
        """List all registered training strategy plugin names."""
        return list(self._training_strategy_plugins.keys())
    
    def reload_plugins(self, plugin_dir: Optional[Path] = None) -> None:
        """Hot-reload all plugins."""
        self._model_plugins.clear()
        self._data_processor_plugins.clear()
        self._training_strategy_plugins.clear()
        self._loaded = False
        self.load_plugins(plugin_dir)
        logger.info("Plugins reloaded")


# Global plugin registry instance
_plugin_registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    return _plugin_registry


# Example plugins (for demonstration)
class ExampleLlamaModelPlugin(ModelPlugin):
    """Example plugin for LLaMA models."""
    
    @property
    def name(self) -> str:
        return "llama"
    
    @property
    def supported_tasks(self) -> List[str]:
        return ["text-generation", "text-classification"]
    
    def load_model(self, model_path: str, **kwargs) -> Any:
        # This would integrate with actual model loading logic
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    def get_tokenizer(self, model_path: str, **kwargs) -> Any:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    
    def get_training_config(self, **kwargs) -> Dict[str, Any]:
        return {
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
        }


class ExampleAlpacaDataProcessor(DataProcessorPlugin):
    """Example plugin for Alpaca-style data processing."""
    
    @property
    def name(self) -> str:
        return "alpaca"
    
    def process(self, data: Any, **kwargs) -> Any:
        # Example processing logic
        if isinstance(data, list):
            processed = []
            for item in data:
                if isinstance(item, dict):
                    # Convert to instruction format
                    text = f"### Instruction:\n{item.get('instruction', '')}\n\n"
                    if item.get('input'):
                        text += f"### Input:\n{item['input']}\n\n"
                    text += f"### Response:\n{item.get('output', '')}"
                    processed.append(text)
            return processed
        return data
    
    def validate(self, data: Any) -> bool:
        if isinstance(data, list):
            return all(isinstance(item, dict) for item in data)
        return False


class ExampleLoRATrainingStrategy(TrainingStrategyPlugin):
    """Example plugin for LoRA training strategy."""
    
    @property
    def name(self) -> str:
        return "lora"
    
    def configure_training(self, model: Any, **kwargs) -> Dict[str, Any]:
        return {
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        }
    
    def get_optimizer(self, model_parameters, **kwargs) -> Any:
        import torch.optim as optim
        return optim.AdamW(model_parameters, lr=2e-5)
    
    def get_scheduler(self, optimizer: Any, **kwargs) -> Any:
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=1000
        )


# Register example plugins
def _register_example_plugins():
    """Register example plugins with the registry."""
    registry = get_plugin_registry()
    
    try:
        registry.register_model_plugin(ExampleLlamaModelPlugin)
    except Exception as e:
        logger.debug(f"Failed to register example model plugin: {e}")
    
    try:
        registry.register_data_processor_plugin(ExampleAlpacaDataProcessor)
    except Exception as e:
        logger.debug(f"Failed to register example data processor plugin: {e}")
    
    try:
        registry.register_training_strategy_plugin(ExampleLoRATrainingStrategy)
    except Exception as e:
        logger.debug(f"Failed to register example training strategy plugin: {e}")


# Initialize plugin system
def init_plugins(plugin_dir: Optional[Path] = None) -> None:
    """Initialize the plugin system."""
    registry = get_plugin_registry()
    registry.load_plugins(plugin_dir)
    _register_example_plugins()


__all__ = [
    # Inference
    "InferenceBackend",
    "get_inference_backend",
    # Training
    "get_training_backend",
    "TrainingBackend",
    "TrainingProgress",
    # Config
    "ModelConfig",
    "is_vision_model",
    "scan_trained_loras",
    "load_model_defaults",
    "get_base_model_from_lora",
    # Utils
    "format_and_template_dataset",
    "normalize_path",
    "is_local_path",
    "is_model_cached",
    "without_hf_auth",
    "format_error_message",
    "get_gpu_memory_info",
    "log_gpu_memory",
    "get_device",
    "is_apple_silicon",
    "clear_gpu_cache",
    "DeviceType",
    # Plugin System
    "ModelPlugin",
    "DataProcessorPlugin",
    "TrainingStrategyPlugin",
    "PluginRegistry",
    "get_plugin_registry",
    "init_plugins",
]


def __getattr__(name):
    # Inference
    if name in ("InferenceBackend", "get_inference_backend"):
        from .inference import InferenceBackend, get_inference_backend

        globals()["InferenceBackend"] = InferenceBackend
        globals()["get_inference_backend"] = get_inference_backend
        return globals()[name]

    # Training
    if name in ("TrainingBackend", "get_training_backend", "TrainingProgress"):
        from .training import TrainingBackend, get_training_backend, TrainingProgress

        globals()["TrainingBackend"] = TrainingBackend
        globals()["get_training_backend"] = get_training_backend
        globals()["TrainingProgress"] = TrainingProgress
        return globals()[name]

    # Config (from utils.models)
    if name in (
        "is_vision_model",
        "ModelConfig",
        "scan_trained_loras",
        "load_model_defaults",
        "get_base_model_from_lora",
    ):
        from utils.models import (
            is_vision_model,
            ModelConfig,
            scan_trained_loras,
            load_model_defaults,
            get_base_model_from_lora,
        )

        globals()["is_vision_model"] = is_vision_model
        globals()["ModelConfig"] = ModelConfig
        globals()["scan_trained_loras"] = scan_trained_loras
        globals()["load_model_defaults"] = load_model_defaults
        globals()["get_base_model_from_lora"] = get_base_model_from_lora
        return globals()[name]

    # Paths
    if name in ("normalize_path", "is_local_path", "is_model_cached"):
        from utils.paths import normalize_path, is_local_path, is_model_cached

        globals()["normalize_path"] = normalize_path
        globals()["is_local_path"] = is_local_path
        globals()["is_model_cached"] = is_model_cached
        return globals()[name]

    # Utils
    if name in ("without_hf_auth", "format_error_message"):
        from utils.utils import without_hf_auth, format_error_message

        globals()["without_hf_auth"] = without_hf_auth
        globals()["format_error_message"] = format_error_message
        return globals()[name]

    # Hardware
    if name in (
        "get_device",
        "is_apple_silicon",
        "clear_gpu_cache",
        "get_gpu_memory_info",
        "log_gpu_memory",
        "DeviceType",
    ):
        from utils.hardware import (
            get_device,
            is_apple_silicon,
            clear_gpu_cache,
            get_gpu_memory_info,
            log_gpu_memory,
            DeviceType,
        )

        globals()["get_device"] = get_device
        globals()["is_apple_silicon"] = is_apple_silicon
        globals()["clear_gpu_cache"] = clear_gpu_cache
        globals()["get_gpu_memory_info"] = get_gpu_memory_info
        globals()["log_gpu_memory"] = log_gpu_memory
        globals()["DeviceType"] = DeviceType
        return globals()[name]

    # Datasets
    if name == "format_and_template_dataset":
        from utils.datasets import format_and_template_dataset

        globals()["format_and_template_dataset"] = format_and_template_dataset
        return format_and_template_dataset

    # Plugin System (lazy initialization)
    if name in ("ModelPlugin", "DataProcessorPlugin", "TrainingStrategyPlugin", 
                "PluginRegistry", "get_plugin_registry", "init_plugins"):
        # These are already defined in this module, so just return them
        return globals()[name]

    raise AttributeError(f"module 'core' has no attribute {name!r}")


# Initialize plugin system when module is imported
try:
    init_plugins()
except Exception as e:
    logger.warning(f"Failed to initialize plugin system: {e}")