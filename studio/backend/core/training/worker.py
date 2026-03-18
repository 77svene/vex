# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Training subprocess entry point.

Each training job runs in a fresh subprocess (mp.get_context("spawn")).
This gives us a clean Python interpreter with no stale module state —
solving the transformers version-switching problem completely.

Pattern follows core/data_recipe/jobs/worker.py.
"""

from __future__ import annotations

import structlog
from loggers import get_logger
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
import importlib
import importlib.util
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

logger = get_logger(__name__)


@dataclass
class PluginMetadata:
    """Metadata for a loaded plugin."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: str  # "training", "quantization", "kernel", "data_processing"
    entry_point: str
    hash: str
    loaded_at: datetime
    compatibility_version: str = "1.0.0"


class PluginInterface(ABC):
    """Base interface all plugins must implement."""
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def validate_compatibility(self, worker_version: str) -> bool:
        """Check if plugin is compatible with current worker version."""
        pass


class TrainingLoopPlugin(PluginInterface):
    """Interface for custom training loop plugins."""
    
    @abstractmethod
    def create_trainer(self, config: Dict[str, Any], **kwargs) -> Any:
        """Create a trainer instance with custom training loop."""
        pass
    
    @abstractmethod
    def get_supported_algorithms(self) -> List[str]:
        """Return list of supported training algorithms."""
        pass


class QuantizationPlugin(PluginInterface):
    """Interface for custom quantization method plugins."""
    
    @abstractmethod
    def quantize_model(self, model: Any, config: Dict[str, Any]) -> Any:
        """Apply quantization to a model."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported quantization formats."""
        pass


class CUDAPlugin(PluginInterface):
    """Interface for custom CUDA kernel plugins."""
    
    @abstractmethod
    def get_kernel_module(self) -> Any:
        """Return the compiled CUDA kernel module."""
        pass
    
    @abstractmethod
    def get_kernel_functions(self) -> Dict[str, Any]:
        """Return mapping of kernel names to functions."""
        pass


class DataProcessingPlugin(PluginInterface):
    """Interface for data processing plugins."""
    
    @abstractmethod
    def process_dataset(self, dataset: Any, config: Dict[str, Any]) -> Any:
        """Process a dataset with custom logic."""
        pass


class PluginSandbox:
    """Sandboxed environment for loading plugins with restricted globals."""
    
    def __init__(self):
        self.safe_builtins = {
            'None': None,
            'False': False,
            'True': True,
            'bool': bool,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'bytes': bytes,
            'bytearray': bytearray,
            'range': range,
            'slice': slice,
            'property': property,
            'staticmethod': staticmethod,
            'classmethod': classmethod,
            'super': super,
            'object': object,
            'type': type,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'len': len,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'any': any,
            'all': all,
            'min': min,
            'max': max,
            'sum': sum,
            'abs': abs,
            'round': round,
            'pow': pow,
            'hash': hash,
            'id': id,
            'repr': repr,
            'ascii': ascii,
            'ord': ord,
            'chr': chr,
            'bin': bin,
            'oct': oct,
            'hex': hex,
            'format': format,
            'vars': vars,
            'dir': dir,
            'callable': callable,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'delattr': delattr,
        }
        
        self.restricted_modules = {
            'os': {'path': os.path, 'environ': os.environ},
            'sys': {'platform': sys.platform, 'version': sys.version},
            'pathlib': {'Path': Path},
            'typing': {},
            'abc': {'ABC': ABC, 'abstractmethod': abstractmethod},
            'dataclasses': {'dataclass': dataclass},
            'datetime': {'datetime': datetime},
        }
    
    def create_restricted_globals(self) -> Dict[str, Any]:
        """Create restricted globals dictionary for plugin execution."""
        restricted_globals = {
            '__builtins__': self.safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
        }
        
        # Add restricted module access
        for mod_name, allowed_attrs in self.restricted_modules.items():
            try:
                mod = importlib.import_module(mod_name)
                restricted_globals[mod_name] = type('RestrictedModule', (), {
                    attr: getattr(mod, attr) for attr in allowed_attrs 
                    if hasattr(mod, attr)
                })()
            except ImportError:
                continue
        
        return restricted_globals


class PluginRegistry:
    """Registry for managing loaded plugins with hot-reloading support."""
    
    def __init__(self, plugin_dirs: List[Path]):
        self.plugin_dirs = plugin_dirs
        self.plugins: Dict[str, PluginInterface] = {}
        self.metadata: Dict[str, PluginMetadata] = {}
        self.file_hashes: Dict[str, str] = {}
        self.sandbox = PluginSandbox()
        self.worker_version = "1.0.0"  # Should come from config
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a plugin file."""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _load_plugin_module(self, plugin_path: Path) -> Optional[Any]:
        """Load a plugin module in a sandboxed environment."""
        try:
            module_name = f"vex_plugin_{plugin_path.stem}"
            spec = importlib.util.spec_from_file_location(
                module_name, 
                plugin_path,
                submodule_search_locations=[]
            )
            
            if spec is None or spec.loader is None:
                logger.error("Failed to create module spec for %s", plugin_path)
                return None
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute in restricted environment
            restricted_globals = self.sandbox.create_restricted_globals()
            restricted_globals['__file__'] = str(plugin_path)
            
            # Read and compile source with restrictions
            with open(plugin_path, 'r') as f:
                source = f.read()
            
            code = compile(source, str(plugin_path), 'exec')
            exec(code, restricted_globals)
            
            # Copy allowed attributes to module
            for key, value in restricted_globals.items():
                if not key.startswith('__') and key not in ['__builtins__']:
                    setattr(module, key, value)
            
            return module
            
        except Exception as e:
            logger.error("Failed to load plugin %s: %s", plugin_path, e)
            logger.debug(traceback.format_exc())
            return None
    
    def _discover_plugins(self) -> List[Path]:
        """Discover all plugin files in configured directories."""
        plugin_files = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                logger.warning("Plugin directory does not exist: %s", plugin_dir)
                continue
            
            # Look for .py files and packages
            for item in plugin_dir.iterdir():
                if item.is_file() and item.suffix == '.py':
                    plugin_files.append(item)
                elif item.is_dir() and (item / '__init__.py').exists():
                    plugin_files.append(item / '__init__.py')
        
        return plugin_files
    
    def _validate_plugin_interface(self, plugin_obj: Any) -> bool:
        """Validate that plugin implements required interface."""
        required_methods = ['get_metadata', 'validate_compatibility']
        
        for method in required_methods:
            if not hasattr(plugin_obj, method) or not callable(getattr(plugin_obj, method)):
                return False
        
        # Check for plugin type specific methods
        if isinstance(plugin_obj, TrainingLoopPlugin):
            if not hasattr(plugin_obj, 'create_trainer'):
                return False
        elif isinstance(plugin_obj, QuantizationPlugin):
            if not hasattr(plugin_obj, 'quantize_model'):
                return False
        elif isinstance(plugin_obj, CUDAPlugin):
            if not hasattr(plugin_obj, 'get_kernel_module'):
                return False
        elif isinstance(plugin_obj, DataProcessingPlugin):
            if not hasattr(plugin_obj, 'process_dataset'):
                return False
        else:
            return False
        
        return True
    
    def load_plugin(self, plugin_path: Path) -> bool:
        """Load a single plugin from file path."""
        try:
            # Check if plugin has changed
            current_hash = self._calculate_file_hash(plugin_path)
            if plugin_path in self.file_hashes and self.file_hashes[plugin_path] == current_hash:
                logger.debug("Plugin %s unchanged, skipping reload", plugin_path)
                return True
            
            # Load module
            module = self._load_plugin_module(plugin_path)
            if module is None:
                return False
            
            # Find plugin class in module
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, PluginInterface) and 
                    attr is not PluginInterface):
                    plugin_class = attr
                    break
            
            if plugin_class is None:
                logger.error("No plugin class found in %s", plugin_path)
                return False
            
            # Instantiate plugin
            plugin_instance = plugin_class()
            
            # Validate interface
            if not self._validate_plugin_interface(plugin_instance):
                logger.error("Plugin %s does not implement required interface", plugin_path)
                return False
            
            # Get metadata
            metadata = plugin_instance.get_metadata()
            
            # Check compatibility
            if not plugin_instance.validate_compatibility(self.worker_version):
                logger.error("Plugin %s is not compatible with worker version %s", 
                           metadata.name, self.worker_version)
                return False
            
            # Register plugin
            plugin_key = f"{metadata.plugin_type}:{metadata.name}"
            self.plugins[plugin_key] = plugin_instance
            self.metadata[plugin_key] = metadata
            self.file_hashes[plugin_path] = current_hash
            
            logger.info("Loaded plugin: %s v%s by %s", 
                       metadata.name, metadata.version, metadata.author)
            
            return True
            
        except Exception as e:
            logger.error("Failed to load plugin from %s: %s", plugin_path, e)
            logger.debug(traceback.format_exc())
            return False
    
    def load_all_plugins(self) -> None:
        """Load all discovered plugins."""
        plugin_files = self._discover_plugins()
        logger.info("Discovered %d potential plugin files", len(plugin_files))
        
        loaded_count = 0
        for plugin_path in plugin_files:
            if self.load_plugin(plugin_path):
                loaded_count += 1
        
        logger.info("Successfully loaded %d/%d plugins", loaded_count, len(plugin_files))
    
    def reload_plugins(self) -> None:
        """Reload all plugins (hot-reload)."""
        logger.info("Hot-reloading plugins...")
        
        # Clear existing plugins
        self.plugins.clear()
        self.metadata.clear()
        
        # Reload all
        self.load_all_plugins()
    
    def get_plugins_by_type(self, plugin_type: str) -> List[PluginInterface]:
        """Get all plugins of specified type."""
        return [
            plugin for key, plugin in self.plugins.items()
            if self.metadata[key].plugin_type == plugin_type
        ]
    
    def get_plugin(self, plugin_type: str, plugin_name: str) -> Optional[PluginInterface]:
        """Get specific plugin by type and name."""
        key = f"{plugin_type}:{plugin_name}"
        return self.plugins.get(key)


def _activate_transformers_version(model_name: str) -> None:
    """Activate the correct transformers version BEFORE any ML imports.

    If the model needs transformers 5.x, prepend the pre-installed .venv_t5/
    directory to sys.path. Otherwise do nothing (default 4.57.x in .venv/).
    """
    # Ensure backend is on path for utils imports
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.transformers_version import (
        needs_transformers_5,
        _resolve_base_model,
        _ensure_venv_t5_exists,
        _VENV_T5_DIR,
    )

    resolved = _resolve_base_model(model_name)
    if needs_transformers_5(resolved):
        if not _ensure_venv_t5_exists():
            raise RuntimeError(
                f"Cannot activate transformers 5.x: .venv_t5 missing at {_VENV_T5_DIR}"
            )
        if _VENV_T5_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_DIR)
        logger.info("Activated transformers 5.x from %s", _VENV_T5_DIR)
        # Propagate to child subprocesses (e.g. GGUF converter)
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_DIR + (os.pathsep + _pp if _pp else "")
    else:
        logger.info("Using default transformers (4.57.x) for %s", model_name)


def run_training_process(
    *,
    event_queue: Any,
    stop_queue: Any,
    config: dict,
) -> None:
    """Subprocess entrypoint. Fresh Python — no stale module state.

    Args:
        event_queue: mp.Queue for sending progress/status/error events to parent.
        stop_queue: mp.Queue for receiving stop commands from parent.
        config: Training configuration dict with all parameters.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = (
        "ignore"  # Suppress warnings at C-level before imports
    )

    import warnings
    from loggers.config import LogConfig

    if os.getenv("ENVIRONMENT_TYPE", "production") == "production":
        warnings.filterwarnings("ignore")

    LogConfig.setup_logging(
        service_name = "vex-studio-training-worker",
        env = os.getenv("ENVIRONMENT_TYPE", "production"),
    )

    model_name = config["model_name"]

    # ── 1. Activate correct transformers version BEFORE any ML imports ──
    try:
        _activate_transformers_version(model_name)
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to activate transformers version: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 1a. Auto-enable trust_remote_code for vex/* transformers 5.x models ──
    # Some newer architectures (e.g. NemotronH) have config parsing bugs in
    # transformers that require trust_remote_code=True as a workaround.
    # Only auto-enable for vex/* prefixed models (trusted source).
    from utils.transformers_version import needs_transformers_5

    if (
        needs_transformers_5(model_name)
        and model_name.lower().startswith("vex/")
        and not config.get("trust_remote_code", False)
    ):
        config["trust_remote_code"] = True
        logger.info(
            "Auto-enabled trust_remote_code for vex/* transformers 5.x model: %s",
            model_name,
        )

    # ── 1b. Auto-install mamba-ssm for SSM/hybrid models (NemotronH, Falcon-H1) ──
    _SSM_MODEL_SUBSTRINGS = ("nemotron_h", "nemotron-3-nano", "falcon_h1", "falcon-h1")
    if any(sub in model_name.lower() for sub in _SSM_MODEL_SUBSTRINGS):
        try:
            import mamba_ssm  # noqa: F401

            logger.info("mamba-ssm already installed")
        except ImportError:
            logger.info(
                "SSM model detected — installing mamba-ssm and causal-conv1d (this may take several minutes)..."
            )
            _send_status(
                event_queue, "Installing mamba-ssm (first time only, ~7 min)..."
            )
            import subprocess as _sp

            # --no-build-isolation: compile against current torch (no version conflicts)
            # --no-deps: don't pull in torch/transformers/triton (already installed)
            for _pkg in ["causal_conv1d", "mamba_ssm"]:
                _r = _sp.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-build-isolation",
                        "--no-deps",
                        "--no-cache-dir",
                        _pkg,
                    ],
                    stdout = _sp.PIPE,
                    stderr = _sp.STDOUT,
                    text = True,
                )
                if _r.returncode != 0:
                    logger.error("Failed to install %s:\n%s", _pkg, _r.stdout)
                else:
                    logger.info("Installed %s successfully", _pkg)
            logger.info("mamba-ssm installation complete")

    # ── 1c. Set fork start method so dataset.map() can multiprocess ──
    # The parent launched us via spawn (clean process), but the compiled
    # SFTTrainer checks get_start_method() and disables num_proc if not "fork".
    # Linux only: fork is the default start method and is safe here (no CUDA
    # context exists yet). macOS defaults to spawn since Python 3.8 because
    # fork is unsafe with macOS frameworks (Metal/MPS, CoreFoundation) --
    # do NOT override on macOS. Windows has no fork at all.
    if sys.platform == "linux":
        import multiprocessing as _mp

        try:
            _mp.set_start_method("fork", force = True)
        except RuntimeError:
            pass  # Already set

    # ── 1c. On Windows, check Triton availability (must be before import torch) ──
    if sys.platform == "win32":
        try:
            import triton  # noqa: F401

            logger.info("Triton available — torch.compile enabled")
        except ImportError:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            logger.warning(
                "Triton not found on Windows — torch.compile disabled. "
                'Install for better performance: pip install "triton-windows<3.7"'
            )

    # ── 2. Now import ML libraries (fresh in this clean process) ──
    try:
        _send_status(event_queue, "Importing Unsloth...")

        backend_path = str(Path(__file__).resolve().parent.parent.parent)
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from core.training.trainer import UnslothTrainer, TrainingProgress
        from utils.paths import (
            ensure_dir,
            resolve_output_dir,
            resolve_tensorboard_dir,
            datasets_root,
        )

        import transformers

        logger.info("Subprocess loaded transformers %s", transformers.__version__)
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to import ML libraries: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
        )
        return

    # ── 2b. Initialize plugin system ──
    plugin_dirs = [
        Path.home() / ".vex" / "plugins",
        Path("/etc/vex/plugins"),
        Path(backend_path) / "plugins",
    ]
    
    # Add custom plugin directories from config
    if "plugin_dirs" in config:
        for dir_path in config["plugin_dirs"]:
            plugin_dirs.append(Path(dir_path))
    
    plugin_registry = PluginRegistry(plugin_dirs)
    
    try:
        _send_status(event_queue, "Loading plugins...")
        plugin_registry.load_all_plugins()
        
        # Log loaded plugins
        for plugin_key, metadata in plugin_registry.metadata.items():
            logger.info("Loaded plugin: %s v%s (%s)", 
                       metadata.name, metadata.version, metadata.plugin_type)
            
    except Exception as exc:
        logger.warning("Failed to initialize plugin system: %s", exc)
        logger.debug(traceback.format_exc())
        # Continue without plugins
    
    # ── 2c. Check for hot-reload requests ──
    def check_for_reload():
        """Check if plugins should be reloaded."""
        reload_file = Path.home() / ".vex" / "reload_plugins"
        if reload_file.exists():
            try:
                reload_file.unlink()
                plugin_registry.reload_plugins()
                logger.info("Plugins hot-reloaded successfully")
            except Exception as e:
                logger.error("Failed to hot-reload plugins: %s", e)
    
    # ── 3. Setup training configuration ──
    try:
        _send_status(event_queue, "Setting up training...")
        
        # Check for custom training loop plugin
        trainer_class = UnslothTrainer
        training_plugins = plugin_registry.get_plugins_by_type("training")
        
        if training_plugins:
            # Use first available training plugin
            training_plugin = training_plugins[0]
            metadata = training_plugin.get_metadata()
            logger.info("Using custom training loop from plugin: %s", metadata.name)
            
            # Get custom trainer class
            custom_trainer = training_plugin.create_trainer(config)
            if custom_trainer is not None:
                trainer_class = type(custom_trainer)
        
        # Check for custom quantization plugins
        quantization_plugins = plugin_registry.get_plugins_by_type("quantization")
        if quantization_plugins:
            logger.info("Found %d quantization plugins", len(quantization_plugins))
            # Store for later use
            config["_quantization_plugins"] = quantization_plugins
        
        # Check for custom CUDA kernel plugins
        kernel_plugins = plugin_registry.get_plugins_by_type("kernel")
        if kernel_plugins:
            logger.info("Found %d CUDA kernel plugins", len(kernel_plugins))
            # Load kernel modules
            for kernel_plugin in kernel_plugins:
                try:
                    kernel_module = kernel_plugin.get_kernel_module()
                    if kernel_module:
                        # Make kernel functions available
                        kernel_functions = kernel_plugin.get_kernel_functions()
                        for func_name, func in kernel_functions.items():
                            # Register in global namespace for use by trainer
                            globals()[f"custom_kernel_{func_name}"] = func
                        logger.info("Loaded custom CUDA kernels from %s", 
                                   kernel_plugin.get_metadata().name)
                except Exception as e:
                    logger.warning("Failed to load CUDA kernels from plugin: %s", e)
        
        # Check for data processing plugins
        data_plugins = plugin_registry.get_plugins_by_type("data_processing")
        if data_plugins:
            logger.info("Found %d data processing plugins", len(data_plugins))
            config["_data_plugins"] = data_plugins
        
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Failed to setup training configuration: {exc}",
                "stack": traceback.format_exc(limit=20),
                "ts": time.time(),
            }
        )
        return

    # ── 4. Run training ──
    try:
        # Periodically check for plugin reload requests
        import threading
        reload_thread = threading.Thread(
            target=lambda: [check_for_reload() or time.sleep(5) for _ in range(100)],
            daemon=True
        )
        reload_thread.start()
        
        # Create trainer instance
        trainer = trainer_class(
            config=config,
            event_queue=event_queue,
            stop_queue=stop_queue,
            plugin_registry=plugin_registry,
        )
        
        # Run training
        trainer.train()
        
    except Exception as exc:
        event_queue.put(
            {
                "type": "error",
                "error": f"Training failed: {exc}",
                "stack": traceback.format_exc(limit=20),
                "ts": time.time(),
            }
        )
        return
    
    finally:
        # Cleanup
        check_for_reload()  # Final check for reload requests
        
        # Unload plugins
        try:
            for plugin_key, plugin in plugin_registry.plugins.items():
                if hasattr(plugin, 'cleanup'):
                    plugin.cleanup()
        except Exception as e:
            logger.warning("Error during plugin cleanup: %s", e)
    
    event_queue.put(
        {
            "type": "status",
            "status": "completed",
            "ts": time.time(),
        }
    )


def _send_status(queue: Any, message: str) -> None:
    """Helper to send status updates."""
    queue.put(
        {
            "type": "status",
            "status": "running",
            "message": message,
            "ts": time.time(),
        }
    )