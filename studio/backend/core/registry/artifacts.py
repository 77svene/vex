"""
Model Versioning & Experiment Tracking System
Comprehensive model registry with Git-like branching, MLflow integration,
and one-click rollback capabilities.
"""

import os
import json
import shutil
import hashlib
import datetime
import tempfile
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import semver
import yaml

# MLflow integration
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

# Existing imports from codebase
from studio.backend.auth.storage import StorageBackend
from studio.backend.core.data_recipe.jobs.types import JobStatus


class ModelVersionStatus(Enum):
    """Status of a model version in the registry."""
    DRAFT = "draft"
    TRAINING = "training"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ExperimentStatus(Enum):
    """Status of an experiment run."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if key == "custom_metrics":
                    result.update(value)
                else:
                    result[key] = value
        return result


@dataclass
class ModelHyperparameters:
    """Model hyperparameters."""
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    optimizer: Optional[str] = None
    loss_function: Optional[str] = None
    custom_hyperparams: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if key == "custom_hyperparams":
                    result.update(value)
                else:
                    result[key] = value
        return result


@dataclass
class ModelCard:
    """Model card with comprehensive metadata."""
    name: str
    description: str
    version: str
    created_by: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    tags: List[str] = field(default_factory=list)
    intended_use: Optional[str] = None
    limitations: Optional[str] = None
    ethical_considerations: Optional[str] = None
    training_data: Optional[str] = None
    evaluation_data: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    license: Optional[str] = None
    contact: Optional[str] = None
    paper_url: Optional[str] = None
    repository_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO format datetimes."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCard":
        """Create from dictionary, parsing datetime strings."""
        data = data.copy()
        data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.datetime.fromisoformat(data["updated_at"])
        return cls(**data)


@dataclass
class ModelVersion:
    """Represents a specific version of a model in the registry."""
    model_id: str
    version: str
    branch: str
    status: ModelVersionStatus
    created_at: datetime.datetime
    updated_at: datetime.datetime
    created_by: str
    model_path: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    hyperparameters: Optional[ModelHyperparameters] = None
    model_card: Optional[ModelCard] = None
    parent_version: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO format datetimes."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        result["updated_at"] = self.updated_at.isoformat()
        result["status"] = self.status.value
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        if self.hyperparameters:
            result["hyperparameters"] = self.hyperparameters.to_dict()
        if self.model_card:
            result["model_card"] = self.model_card.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary, parsing datetime strings and enums."""
        data = data.copy()
        data["created_at"] = datetime.datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.datetime.fromisoformat(data["updated_at"])
        data["status"] = ModelVersionStatus(data["status"])
        
        if "metrics" in data and data["metrics"]:
            data["metrics"] = ModelMetrics(**data["metrics"])
        if "hyperparameters" in data and data["hyperparameters"]:
            data["hyperparameters"] = ModelHyperparameters(**data["hyperparameters"])
        if "model_card" in data and data["model_card"]:
            data["model_card"] = ModelCard.from_dict(data["model_card"])
        
        return cls(**data)


@dataclass
class ExperimentRun:
    """Represents an experiment run with MLflow integration."""
    experiment_id: str
    run_id: str
    name: str
    status: ExperimentStatus
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    mlflow_run_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO format datetimes."""
        result = asdict(self)
        result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        result["status"] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        """Create from dictionary, parsing datetime strings and enums."""
        data = data.copy()
        data["start_time"] = datetime.datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.datetime.fromisoformat(data["end_time"])
        data["status"] = ExperimentStatus(data["status"])
        return cls(**data)


class ModelDiff:
    """Computes differences between two model versions."""
    
    @staticmethod
    def diff_versions(version_a: ModelVersion, version_b: ModelVersion) -> Dict[str, Any]:
        """Compare two model versions and return differences."""
        diff = {
            "version_diff": version_a.version != version_b.version,
            "branch_diff": version_a.branch != version_b.branch,
            "status_diff": version_a.status != version_b.status,
            "metrics_diff": {},
            "hyperparameters_diff": {},
            "metadata_diff": {},
            "model_card_diff": {}
        }
        
        # Compare metrics
        if version_a.metrics and version_b.metrics:
            metrics_a = version_a.metrics.to_dict()
            metrics_b = version_b.metrics.to_dict()
            all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
            for key in all_keys:
                val_a = metrics_a.get(key)
                val_b = metrics_b.get(key)
                if val_a != val_b:
                    diff["metrics_diff"][key] = {"old": val_a, "new": val_b}
        
        # Compare hyperparameters
        if version_a.hyperparameters and version_b.hyperparameters:
            hyp_a = version_a.hyperparameters.to_dict()
            hyp_b = version_b.hyperparameters.to_dict()
            all_keys = set(hyp_a.keys()) | set(hyp_b.keys())
            for key in all_keys:
                val_a = hyp_a.get(key)
                val_b = hyp_b.get(key)
                if val_a != val_b:
                    diff["hyperparameters_diff"][key] = {"old": val_a, "new": val_b}
        
        # Compare metadata
        all_meta_keys = set(version_a.metadata.keys()) | set(version_b.metadata.keys())
        for key in all_meta_keys:
            val_a = version_a.metadata.get(key)
            val_b = version_b.metadata.get(key)
            if val_a != val_b:
                diff["metadata_diff"][key] = {"old": val_a, "new": val_b}
        
        # Compare model cards
        if version_a.model_card and version_b.model_card:
            card_a = version_a.model_card.to_dict()
            card_b = version_b.model_card.to_dict()
            all_keys = set(card_a.keys()) | set(card_b.keys())
            for key in all_keys:
                if key in ["created_at", "updated_at"]:
                    continue  # Skip timestamp comparisons
                val_a = card_a.get(key)
                val_b = card_b.get(key)
                if val_a != val_b:
                    diff["model_card_diff"][key] = {"old": val_a, "new": val_b}
        
        return diff


class ModelRegistry:
    """
    Comprehensive model registry with Git-like branching and versioning.
    Integrates with MLflow for experiment tracking.
    """
    
    def __init__(self, storage_path: str = "./model_registry", 
                 mlflow_tracking_uri: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            storage_path: Path to store model registry data
            mlflow_tracking_uri: Optional MLflow tracking URI for experiment tracking
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage backend
        self.storage = StorageBackend(str(self.storage_path))
        
        # Initialize MLflow if available and tracking URI provided
        self.mlflow_client = None
        self.mlflow_tracking_uri = mlflow_tracking_uri
        if MLFLOW_AVAILABLE and mlflow_tracking_uri:
            try:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                self.mlflow_client = MlflowClient()
            except Exception as e:
                print(f"Warning: Failed to initialize MLflow: {e}")
        
        # Create directory structure
        self._init_directory_structure()
    
    def _init_directory_structure(self):
        """Initialize the directory structure for the registry."""
        directories = [
            "models",
            "experiments",
            "artifacts",
            "branches",
            "metadata"
        ]
        
        for directory in directories:
            (self.storage_path / directory).mkdir(exist_ok=True)
    
    def _get_model_path(self, model_id: str) -> Path:
        """Get the path for a specific model."""
        return self.storage_path / "models" / model_id
    
    def _get_version_path(self, model_id: str, version: str, branch: str) -> Path:
        """Get the path for a specific model version."""
        return self._get_model_path(model_id) / "branches" / branch / version
    
    def _get_branch_path(self, model_id: str, branch: str) -> Path:
        """Get the path for a specific branch."""
        return self._get_model_path(model_id) / "branches" / branch
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def create_model(self, model_id: str, name: str, description: str, 
                     created_by: str, **kwargs) -> ModelVersion:
        """
        Create a new model in the registry.
        
        Args:
            model_id: Unique identifier for the model
            name: Human-readable name for the model
            description: Description of the model
            created_by: User who created the model
            **kwargs: Additional metadata
            
        Returns:
            Initial model version (v0.1.0 on 'main' branch)
        """
        model_path = self._get_model_path(model_id)
        if model_path.exists():
            raise ValueError(f"Model {model_id} already exists")
        
        # Create model directory structure
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / "branches").mkdir(exist_ok=True)
        (model_path / "artifacts").mkdir(exist_ok=True)
        
        # Create main branch
        main_branch_path = self._get_branch_path(model_id, "main")
        main_branch_path.mkdir(exist_ok=True)
        
        # Create initial version
        now = datetime.datetime.now()
        initial_version = ModelVersion(
            model_id=model_id,
            version="0.1.0",
            branch="main",
            status=ModelVersionStatus.DRAFT,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            model_card=ModelCard(
                name=name,
                description=description,
                version="0.1.0",
                created_by=created_by,
                created_at=now,
                updated_at=now,
                **kwargs
            )
        )
        
        # Save version metadata
        self._save_version_metadata(initial_version)
        
        # Update branch pointer
        self._update_branch_pointer(model_id, "main", "0.1.0")
        
        return initial_version
    
    def _save_version_metadata(self, version: ModelVersion):
        """Save version metadata to storage."""
        version_path = self._get_version_path(
            version.model_id, version.version, version.branch
        )
        version_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = version_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def _update_branch_pointer(self, model_id: str, branch: str, version: str):
        """Update the branch pointer to the specified version."""
        branch_path = self._get_branch_path(model_id, branch)
        pointer_file = branch_path / "HEAD"
        
        with open(pointer_file, "w") as f:
            f.write(version)
    
    def _get_branch_head(self, model_id: str, branch: str) -> str:
        """Get the current HEAD version for a branch."""
        pointer_file = self._get_branch_path(model_id, branch) / "HEAD"
        
        if not pointer_file.exists():
            raise ValueError(f"Branch {branch} does not exist for model {model_id}")
        
        with open(pointer_file, "r") as f:
            return f.read().strip()
    
    def create_branch(self, model_id: str, source_branch: str, 
                      new_branch: str, created_by: str) -> ModelVersion:
        """
        Create a new branch from an existing branch.
        
        Args:
            model_id: Model identifier
            source_branch: Source branch to branch from
            new_branch: Name of the new branch
            created_by: User creating the branch
            
        Returns:
            The version at the branch point
        """
        # Get the current version on source branch
        source_version = self._get_branch_head(model_id, source_branch)
        
        # Create new branch directory
        new_branch_path = self._get_branch_path(model_id, new_branch)
        if new_branch_path.exists():
            raise ValueError(f"Branch {new_branch} already exists")
        
        new_branch_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the version from source branch
        source_version_path = self._get_version_path(
            model_id, source_version, source_branch
        )
        dest_version_path = self._get_version_path(
            model_id, source_version, new_branch
        )
        
        shutil.copytree(source_version_path, dest_version_path)
        
        # Update branch pointer
        self._update_branch_pointer(model_id, new_branch, source_version)
        
        # Load and return the version
        return self.get_version(model_id, source_version, new_branch)
    
    def create_version(self, model_id: str, branch: str, version_type: str = "patch",
                       model_path: Optional[str] = None,
                       metrics: Optional[ModelMetrics] = None,
                       hyperparameters: Optional[ModelHyperparameters] = None,
                       model_card_updates: Optional[Dict[str, Any]] = None,
                       created_by: str = "system",
                       mlflow_run_id: Optional[str] = None,
                       **metadata) -> ModelVersion:
        """
        Create a new version of a model.
        
        Args:
            model_id: Model identifier
            branch: Branch to create version on
            version_type: Type of version increment ('major', 'minor', 'patch')
            model_path: Path to the model file/artifact
            metrics: Model performance metrics
            hyperparameters: Model hyperparameters
            model_card_updates: Updates to the model card
            created_by: User creating the version
            mlflow_run_id: Optional MLflow run ID
            **metadata: Additional metadata
            
        Returns:
            The newly created model version
        """
        # Get current version on branch
        current_version_str = self._get_branch_head(model_id, branch)
        current_version = self.get_version(model_id, current_version_str, branch)
        
        # Calculate new version number
        current_semver = semver.VersionInfo.parse(current_version_str)
        if version_type == "major":
            new_version = str(current_semver.bump_major())
        elif version_type == "minor":
            new_version = str(current_semver.bump_minor())
        else:  # patch
            new_version = str(current_semver.bump_patch())
        
        # Create new version
        now = datetime.datetime.now()
        new_version_obj = ModelVersion(
            model_id=model_id,
            version=new_version,
            branch=branch,
            status=ModelVersionStatus.DRAFT,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            parent_version=current_version_str,
            mlflow_run_id=mlflow_run_id,
            metadata=metadata
        )
        
        # Copy model card from parent and apply updates
        if current_version.model_card:
            new_card = ModelCard(
                name=current_version.model_card.name,
                description=current_version.model_card.description,
                version=new_version,
                created_by=created_by,
                created_at=now,
                updated_at=now,
                tags=current_version.model_card.tags.copy(),
                intended_use=current_version.model_card.intended_use,
                limitations=current_version.model_card.limitations,
                ethical_considerations=current_version.model_card.ethical_considerations,
                training_data=current_version.model_card.training_data,
                evaluation_data=current_version.model_card.evaluation_data,
                citations=current_version.model_card.citations.copy(),
                license=current_version.model_card.license,
                contact=current_version.model_card.contact,
                paper_url=current_version.model_card.paper_url,
                repository_url=current_version.model_card.repository_url
            )
            
            # Apply updates to model card
            if model_card_updates:
                for key, value in model_card_updates.items():
                    if hasattr(new_card, key):
                        setattr(new_card, key, value)
            
            new_version_obj.model_card = new_card
        
        # Set metrics and hyperparameters
        if metrics:
            new_version_obj.metrics = metrics
        if hyperparameters:
            new_version_obj.hyperparameters = hyperparameters
        
        # Copy model artifact if provided
        if model_path and os.path.exists(model_path):
            version_path = self._get_version_path(model_id, new_version, branch)
            artifact_path = version_path / "model"
            artifact_path.mkdir(exist_ok=True)
            
            # Copy model file
            dest_path = artifact_path / os.path.basename(model_path)
            shutil.copy2(model_path, dest_path)
            
            # Calculate and store hash
            file_hash = self._calculate_file_hash(model_path)
            new_version_obj.metadata["model_hash"] = file_hash
            new_version_obj.model_path = str(dest_path)
        
        # Save version metadata
        self._save_version_metadata(new_version_obj)
        
        # Update branch pointer
        self._update_branch_pointer(model_id, branch, new_version)
        
        # Log to MLflow if available
        if self.mlflow_client and mlflow_run_id:
            self._log_to_mlflow(new_version_obj)
        
        return new_version_obj
    
    def _log_to_mlflow(self, version: ModelVersion):
        """Log model version to MLflow."""
        if not self.mlflow_client:
            return
        
        try:
            # Log metrics
            if version.metrics:
                for key, value in version.metrics.to_dict().items():
                    if value is not None:
                        self.mlflow_client.log_metric(
                            version.mlflow_run_id, key, value
                        )
            
            # Log hyperparameters
            if version.hyperparameters:
                for key, value in version.hyperparameters.to_dict().items():
                    if value is not None:
                        self.mlflow_client.log_param(
                            version.mlflow_run_id, key, value
                        )
            
            # Log tags
            tags = {
                "model_id": version.model_id,
                "version": version.version,
                "branch": version.branch,
                "status": version.status.value
            }
            for key, value in tags.items():
                self.mlflow_client.set_tag(version.mlflow_run_id, key, value)
            
            # Log model artifact if path exists
            if version.model_path and os.path.exists(version.model_path):
                self.mlflow_client.log_artifact(
                    version.mlflow_run_id, version.model_path
                )
                
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {e}")
    
    def get_version(self, model_id: str, version: str, branch: str) -> ModelVersion:
        """
        Get a specific model version.
        
        Args:
            model_id: Model identifier
            version: Version string
            branch: Branch name
            
        Returns:
            The requested model version
        """
        version_path = self._get_version_path(model_id, version, branch)
        metadata_file = version_path / "metadata.json"
        
        if not metadata_file.exists():
            raise ValueError(
                f"Version {version} not found for model {model_id} on branch {branch}"
            )
        
        with open(metadata_file, "r") as f:
            data = json.load(f)
        
        return ModelVersion.from_dict(data)
    
    def get_latest_version(self, model_id: str, branch: str) -> ModelVersion:
        """
        Get the latest version on a branch.
        
        Args:
            model_id: Model identifier
            branch: Branch name
            
        Returns:
            The latest model version on the branch
        """
        latest_version = self._get_branch_head(model_id, branch)
        return self.get_version(model_id, latest_version, branch)
    
    def list_versions(self, model_id: str, branch: Optional[str] = None) -> List[ModelVersion]:
        """
        List all versions of a model.
        
        Args:
            model_id: Model identifier
            branch: Optional branch to filter by
            
        Returns:
            List of model versions
        """
        versions = []
        
        if branch:
            branches = [branch]
        else:
            branches_path = self._get_model_path(model_id) / "branches"
            branches = [d.name for d in branches_path.iterdir() if d.is_dir()]
        
        for branch_name in branches:
            branch_path = self._get_branch_path(model_id, branch_name)
            if not branch_path.exists():
                continue
            
            for version_dir in branch_path.iterdir():
                if version_dir.is_dir() and version_dir.name != "HEAD":
                    try:
                        version = self.get_version(model_id, version_dir.name, branch_name)
                        versions.append(version)
                    except Exception:
                        continue
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions
    
    def list_branches(self, model_id: str) -> List[str]:
        """
        List all branches for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of branch names
        """
        branches_path = self._get_model_path(model_id) / "branches"
        if not branches_path.exists():
            return []
        
        return [d.name for d in branches_path.iterdir() if d.is_dir()]
    
    def rollback(self, model_id: str, branch: str, target_version: str) -> ModelVersion:
        """
        Rollback a branch to a specific version.
        
        Args:
            model_id: Model identifier
            branch: Branch to rollback
            target_version: Version to rollback to
            
        Returns:
            The version after rollback
        """
        # Verify target version exists
        target = self.get_version(model_id, target_version, branch)
        
        # Update branch pointer
        self._update_branch_pointer(model_id, branch, target_version)
        
        # Update target version status
        target.status = ModelVersionStatus.DEPLOYED
        target.updated_at = datetime.datetime.now()
        self._save_version_metadata(target)
        
        return target
    
    def update_version_status(self, model_id: str, version: str, 
                              branch: str, status: ModelVersionStatus) -> ModelVersion:
        """
        Update the status of a model version.
        
        Args:
            model_id: Model identifier
            version: Version string
            branch: Branch name
            status: New status
            
        Returns:
            Updated model version
        """
        version_obj = self.get_version(model_id, version, branch)
        version_obj.status = status
        version_obj.updated_at = datetime.datetime.now()
        
        self._save_version_metadata(version_obj)
        return version_obj
    
    def delete_version(self, model_id: str, version: str, branch: str):
        """
        Delete a specific model version.
        
        Args:
            model_id: Model identifier
            version: Version string
            branch: Branch name
        """
        version_path = self._get_version_path(model_id, version, branch)
        
        if not version_path.exists():
            raise ValueError(f"Version {version} not found")
        
        # Check if this is the branch head
        current_head = self._get_branch_head(model_id, branch)
        if current_head == version:
            raise ValueError(
                f"Cannot delete branch head. Rollback or create a new version first."
            )
        
        # Delete the version directory
        shutil.rmtree(version_path)
    
    def diff_versions(self, model_id: str, version_a: str, 
                      version_b: str, branch: str) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            model_id: Model identifier
            version_a: First version string
            version_b: Second version string
            branch: Branch name
            
        Returns:
            Dictionary of differences
        """
        ver_a = self.get_version(model_id, version_a, branch)
        ver_b = self.get_version(model_id, version_b, branch)
        
        return ModelDiff.diff_versions(ver_a, ver_b)
    
    def create_experiment(self, experiment_id: str, name: str, 
                          description: str = "", created_by: str = "system") -> ExperimentRun:
        """
        Create a new experiment run.
        
        Args:
            experiment_id: Unique experiment identifier
            name: Experiment name
            description: Experiment description
            created_by: User creating the experiment
            
        Returns:
            The created experiment run
        """
        now = datetime.datetime.now()
        
        experiment = ExperimentRun(
            experiment_id=experiment_id,
            run_id=f"{experiment_id}_{int(now.timestamp())}",
            name=name,
            status=ExperimentStatus.RUNNING,
            start_time=now,
            tags={"description": description, "created_by": created_by}
        )
        
        # Create experiment directory
        exp_path = self.storage_path / "experiments" / experiment_id
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        exp_file = exp_path / f"{experiment.run_id}.json"
        with open(exp_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)
        
        # Start MLflow run if available
        if self.mlflow_client:
            try:
                mlflow_run = self.mlflow_client.create_run(
                    experiment_id=experiment_id,
                    tags=experiment.tags
                )
                experiment.mlflow_run_id = mlflow_run.info.run_id
                
                # Update experiment file with MLflow run ID
                with open(exp_file, "w") as f:
                    json.dump(experiment.to_dict(), f, indent=2)
                    
            except Exception as e:
                print(f"Warning: Failed to create MLflow run: {e}")
        
        return experiment
    
    def log_experiment_metric(self, experiment_id: str, run_id: str, 
                              metric_name: str, value: float, step: int = 0):
        """
        Log a metric to an experiment run.
        
        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            metric_name: Name of the metric
            value: Metric value
            step: Step number (for time series)
        """
        # Load experiment
        exp_file = self.storage_path / "experiments" / experiment_id / f"{run_id}.json"
        if not exp_file.exists():
            raise ValueError(f"Experiment run {run_id} not found")
        
        with open(exp_file, "r") as f:
            experiment_data = json.load(f)
        
        experiment = ExperimentRun.from_dict(experiment_data)
        experiment.metrics[metric_name] = value
        
        # Save updated experiment
        with open(exp_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)
        
        # Log to MLflow if available
        if self.mlflow_client and experiment.mlflow_run_id:
            try:
                self.mlflow_client.log_metric(
                    experiment.mlflow_run_id, metric_name, value, step=step
                )
            except Exception as e:
                print(f"Warning: Failed to log metric to MLflow: {e}")
    
    def complete_experiment(self, experiment_id: str, run_id: str, 
                            status: ExperimentStatus = ExperimentStatus.COMPLETED):
        """
        Mark an experiment run as completed.
        
        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            status: Final status (default: COMPLETED)
        """
        exp_file = self.storage_path / "experiments" / experiment_id / f"{run_id}.json"
        if not exp_file.exists():
            raise ValueError(f"Experiment run {run_id} not found")
        
        with open(exp_file, "r") as f:
            experiment_data = json.load(f)
        
        experiment = ExperimentRun.from_dict(experiment_data)
        experiment.status = status
        experiment.end_time = datetime.datetime.now()
        
        # Save updated experiment
        with open(exp_file, "w") as f:
            json.dump(experiment.to_dict(), f, indent=2)
        
        # End MLflow run if available
        if self.mlflow_client and experiment.mlflow_run_id:
            try:
                mlflow_status = "FINISHED" if status == ExperimentStatus.COMPLETED else "FAILED"
                self.mlflow_client.set_terminated(experiment.mlflow_run_id, mlflow_status)
            except Exception as e:
                print(f"Warning: Failed to terminate MLflow run: {e}")
    
    def get_experiment(self, experiment_id: str, run_id: str) -> ExperimentRun:
        """
        Get an experiment run.
        
        Args:
            experiment_id: Experiment identifier
            run_id: Run identifier
            
        Returns:
            The experiment run
        """
        exp_file = self.storage_path / "experiments" / experiment_id / f"{run_id}.json"
        if not exp_file.exists():
            raise ValueError(f"Experiment run {run_id} not found")
        
        with open(exp_file, "r") as f:
            experiment_data = json.load(f)
        
        return ExperimentRun.from_dict(experiment_data)
    
    def list_experiments(self, experiment_id: Optional[str] = None) -> List[ExperimentRun]:
        """
        List experiment runs.
        
        Args:
            experiment_id: Optional experiment ID to filter by
            
        Returns:
            List of experiment runs
        """
        experiments = []
        
        if experiment_id:
            exp_dirs = [self.storage_path / "experiments" / experiment_id]
        else:
            exp_base = self.storage_path / "experiments"
            exp_dirs = [d for d in exp_base.iterdir() if d.is_dir()]
        
        for exp_dir in exp_dirs:
            if not exp_dir.exists():
                continue
            
            for exp_file in exp_dir.glob("*.json"):
                try:
                    with open(exp_file, "r") as f:
                        experiment_data = json.load(f)
                    experiments.append(ExperimentRun.from_dict(experiment_data))
                except Exception:
                    continue
        
        # Sort by start time (newest first)
        experiments.sort(key=lambda e: e.start_time, reverse=True)
        return experiments
    
    def export_model(self, model_id: str, version: str, branch: str, 
                     export_path: str, include_artifacts: bool = True):
        """
        Export a model version to a directory.
        
        Args:
            model_id: Model identifier
            version: Version string
            branch: Branch name
            export_path: Path to export to
            include_artifacts: Whether to include model artifacts
        """
        version_obj = self.get_version(model_id, version, branch)
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export metadata
        metadata_file = export_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(version_obj.to_dict(), f, indent=2)
        
        # Export model card as YAML
        if version_obj.model_card:
            card_file = export_dir / "model_card.yaml"
            with open(card_file, "w") as f:
                yaml.dump(version_obj.model_card.to_dict(), f, default_flow_style=False)
        
        # Export model artifacts if requested
        if include_artifacts and version_obj.model_path:
            model_source = Path(version_obj.model_path)
            if model_source.exists():
                model_dest = export_dir / "model"
                model_dest.mkdir(exist_ok=True)
                
                if model_source.is_file():
                    shutil.copy2(model_source, model_dest / model_source.name)
                else:
                    shutil.copytree(model_source, model_dest / "model_files")
    
    def import_model(self, import_path: str, model_id: str, branch: str = "main",
                     created_by: str = "system") -> ModelVersion:
        """
        Import a model from an exported directory.
        
        Args:
            import_path: Path to import from
            model_id: Model identifier for the imported model
            branch: Branch to import to
            created_by: User importing the model
            
        Returns:
            The imported model version
        """
        import_dir = Path(import_path)
        
        # Load metadata
        metadata_file = import_dir / "metadata.json"
        if not metadata_file.exists():
            raise ValueError("No metadata.json found in import directory")
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Create or get model
        try:
            # Try to get existing model
            self.get_latest_version(model_id, branch)
        except ValueError:
            # Model doesn't exist, create it
            model_card_data = metadata.get("model_card", {})
            self.create_model(
                model_id=model_id,
                name=model_card_data.get("name", model_id),
                description=model_card_data.get("description", ""),
                created_by=created_by
            )
        
        # Check for model artifacts
        model_path = None
        model_dir = import_dir / "model"
        if model_dir.exists():
            # Find the first file in the model directory
            for item in model_dir.rglob("*"):
                if item.is_file():
                    model_path = str(item)
                    break
        
        # Create new version
        return self.create_version(
            model_id=model_id,
            branch=branch,
            version_type="patch",
            model_path=model_path,
            created_by=created_by,
            **metadata.get("metadata", {})
        )


# Singleton instance for easy access
_registry_instance = None


def get_registry(storage_path: str = "./model_registry", 
                 mlflow_tracking_uri: Optional[str] = None) -> ModelRegistry:
    """
    Get or create the global model registry instance.
    
    Args:
        storage_path: Path to store model registry data
        mlflow_tracking_uri: Optional MLflow tracking URI
        
    Returns:
        ModelRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(storage_path, mlflow_tracking_uri)
    return _registry_instance


# Convenience functions for common operations
def register_model(model_id: str, name: str, description: str, 
                   created_by: str, **kwargs) -> ModelVersion:
    """Convenience function to register a new model."""
    registry = get_registry()
    return registry.create_model(model_id, name, description, created_by, **kwargs)


def create_model_version(model_id: str, branch: str = "main", 
                         version_type: str = "patch", **kwargs) -> ModelVersion:
    """Convenience function to create a new model version."""
    registry = get_registry()
    return registry.create_version(model_id, branch, version_type, **kwargs)


def rollback_model(model_id: str, branch: str, target_version: str) -> ModelVersion:
    """Convenience function to rollback a model."""
    registry = get_registry()
    return registry.rollback(model_id, branch, target_version)


def compare_model_versions(model_id: str, version_a: str, 
                           version_b: str, branch: str = "main") -> Dict[str, Any]:
    """Convenience function to compare two model versions."""
    registry = get_registry()
    return registry.diff_versions(model_id, version_a, version_b, branch)