"""
Model Versioning & Experiment Tracking for Unsloth Studio
Git-like branching, MLflow integration, one-click rollback, and semantic versioning.
"""

import json
import os
import hashlib
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not installed. Experiment tracking features will be limited.")

logger = logging.getLogger(__name__)


class VersionType(Enum):
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelHyperparameters:
    """Container for model hyperparameters."""
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    epochs: Optional[int] = None
    optimizer: Optional[str] = None
    loss_function: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelHyperparameters':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelCard:
    """Comprehensive model documentation and metadata."""
    model_name: str
    description: str = ""
    author: str = ""
    tags: List[str] = field(default_factory=list)
    framework: str = "pytorch"
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    ethical_considerations: str = ""
    limitations: str = ""
    intended_use: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCard':
        return cls(**data)


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    version_id: str
    version_number: str  # Semantic versioning: major.minor.patch
    branch: str = "main"
    parent_version: Optional[str] = None
    model_path: str = ""
    model_hash: str = ""
    metrics: Optional[ModelMetrics] = None
    hyperparameters: Optional[ModelHyperparameters] = None
    model_card: Optional[ModelCard] = None
    mlflow_run_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.metrics:
            data['metrics'] = self.metrics.to_dict()
        if self.hyperparameters:
            data['hyperparameters'] = self.hyperparameters.to_dict()
        if self.model_card:
            data['model_card'] = self.model_card.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'metrics' in data and data['metrics']:
            data['metrics'] = ModelMetrics.from_dict(data['metrics'])
        if 'hyperparameters' in data and data['hyperparameters']:
            data['hyperparameters'] = ModelHyperparameters.from_dict(data['hyperparameters'])
        if 'model_card' in data and data['model_card']:
            data['model_card'] = ModelCard.from_dict(data['model_card'])
        return cls(**data)


@dataclass
class ModelBranch:
    """Represents a branch in model versioning."""
    name: str
    head_version: str
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'head_version': self.head_version,
            'created_at': self.created_at.isoformat(),
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelBranch':
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class SemanticVersion:
    """Handles semantic versioning for models."""
    
    @staticmethod
    def parse(version_str: str) -> Tuple[int, int, int]:
        """Parse semantic version string into (major, minor, patch)."""
        try:
            parts = version_str.split('.')
            if len(parts) != 3:
                raise ValueError(f"Invalid semantic version: {version_str}")
            return int(parts[0]), int(parts[1]), int(parts[2])
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid semantic version format: {version_str}") from e
    
    @staticmethod
    def increment(version_str: str, version_type: VersionType) -> str:
        """Increment version based on type."""
        major, minor, patch = SemanticVersion.parse(version_str)
        
        if version_type == VersionType.MAJOR:
            return f"{major + 1}.0.0"
        elif version_type == VersionType.MINOR:
            return f"{major}.{minor + 1}.0"
        elif version_type == VersionType.PATCH:
            return f"{major}.{minor}.{patch + 1}"
        else:
            raise ValueError(f"Unknown version type: {version_type}")
    
    @staticmethod
    def compare(version1: str, version2: str) -> int:
        """Compare two semantic versions. Returns -1, 0, or 1."""
        v1 = SemanticVersion.parse(version1)
        v2 = SemanticVersion.parse(version2)
        
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0


class ModelDiff:
    """Handles diffing between model versions."""
    
    @staticmethod
    def diff_versions(version1: ModelVersion, version2: ModelVersion) -> Dict[str, Any]:
        """Generate diff between two model versions."""
        diff = {
            'version_changed': version1.version_number != version2.version_number,
            'branch_changed': version1.branch != version2.branch,
            'metrics_diff': {},
            'hyperparameters_diff': {},
            'metadata_diff': {},
            'model_card_diff': {}
        }
        
        # Compare metrics
        if version1.metrics and version2.metrics:
            v1_metrics = version1.metrics.to_dict()
            v2_metrics = version2.metrics.to_dict()
            diff['metrics_diff'] = ModelDiff._dict_diff(v1_metrics, v2_metrics)
        
        # Compare hyperparameters
        if version1.hyperparameters and version2.hyperparameters:
            v1_hparams = version1.hyperparameters.to_dict()
            v2_hparams = version2.hyperparameters.to_dict()
            diff['hyperparameters_diff'] = ModelDiff._dict_diff(v1_hparams, v2_hparams)
        
        # Compare metadata
        diff['metadata_diff'] = ModelDiff._dict_diff(version1.metadata, version2.metadata)
        
        # Compare model cards
        if version1.model_card and version2.model_card:
            v1_card = version1.model_card.to_dict()
            v2_card = version2.model_card.to_dict()
            diff['model_card_diff'] = ModelDiff._dict_diff(v1_card, v2_card)
        
        return diff
    
    @staticmethod
    def _dict_diff(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate difference between two dictionaries."""
        diff = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            if val1 != val2:
                diff[key] = {
                    'old': val1,
                    'new': val2,
                    'changed': True
                }
        
        return diff


class ExperimentTracker:
    """
    Main class for model versioning and experiment tracking.
    Provides Git-like branching, MLflow integration, and semantic versioning.
    """
    
    def __init__(self, registry_path: str = "model_registry"):
        """
        Initialize the Experiment Tracker.
        
        Args:
            registry_path: Path to store model registry data
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize registry files
        self.versions_file = self.registry_path / "versions.json"
        self.branches_file = self.registry_path / "branches.json"
        self.models_file = self.registry_path / "models.json"
        
        # Load or initialize registry
        self._load_registry()
        
        # Setup MLflow if available
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
    
    def _load_registry(self):
        """Load registry data from files."""
        self.versions: Dict[str, ModelVersion] = {}
        self.branches: Dict[str, ModelBranch] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                versions_data = json.load(f)
                for version_id, data in versions_data.items():
                    self.versions[version_id] = ModelVersion.from_dict(data)
        
        if self.branches_file.exists():
            with open(self.branches_file, 'r') as f:
                branches_data = json.load(f)
                for branch_name, data in branches_data.items():
                    self.branches[branch_name] = ModelBranch.from_dict(data)
        
        if self.models_file.exists():
            with open(self.models_file, 'r') as f:
                self.models = json.load(f)
        
        # Ensure main branch exists
        if "main" not in self.branches:
            self.branches["main"] = ModelBranch(
                name="main",
                head_version="",
                description="Main production branch"
            )
    
    def _save_registry(self):
        """Save registry data to files."""
        # Save versions
        versions_data = {
            version_id: version.to_dict() 
            for version_id, version in self.versions.items()
        }
        with open(self.versions_file, 'w') as f:
            json.dump(versions_data, f, indent=2, default=str)
        
        # Save branches
        branches_data = {
            branch_name: branch.to_dict()
            for branch_name, branch in self.branches.items()
        }
        with open(self.branches_file, 'w') as f:
            json.dump(branches_data, f, indent=2, default=str)
        
        # Save models
        with open(self.models_file, 'w') as f:
            json.dump(self.models, f, indent=2, default=str)
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            # Set MLflow tracking URI (can be configured)
            mlflow.set_tracking_uri(f"file://{self.registry_path / 'mlruns'}")
            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}")
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate hash of model file for integrity checking."""
        if not os.path.exists(model_path):
            return ""
        
        hasher = hashlib.sha256()
        with open(model_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        return str(uuid.uuid4())
    
    def create_model(self, model_name: str, description: str = "", 
                    author: str = "", tags: List[str] = None) -> str:
        """
        Create a new model in the registry.
        
        Args:
            model_name: Name of the model
            description: Model description
            author: Model author
            tags: List of tags
            
        Returns:
            Model ID
        """
        if model_name in self.models:
            raise ValueError(f"Model '{model_name}' already exists")
        
        model_id = hashlib.md5(model_name.encode()).hexdigest()[:8]
        self.models[model_name] = {
            'id': model_id,
            'description': description,
            'author': author,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'latest_version': None
        }
        
        self._save_registry()
        logger.info(f"Created model: {model_name}")
        return model_id
    
    def log_model(self, 
                  model_name: str,
                  model_path: str,
                  metrics: Optional[ModelMetrics] = None,
                  hyperparameters: Optional[ModelHyperparameters] = None,
                  model_card: Optional[ModelCard] = None,
                  branch: str = "main",
                  version_type: VersionType = VersionType.PATCH,
                  metadata: Dict[str, Any] = None,
                  parent_version: Optional[str] = None) -> ModelVersion:
        """
        Log a new model version to the registry.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            metrics: Model performance metrics
            hyperparameters: Model hyperparameters
            model_card: Model documentation
            branch: Branch to log to
            version_type: Type of version increment
            metadata: Additional metadata
            parent_version: Parent version ID (for branching)
            
        Returns:
            Created ModelVersion
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Create it first.")
        
        # Calculate model hash
        model_hash = self._calculate_model_hash(model_path)
        
        # Determine version number
        if parent_version:
            parent = self.versions.get(parent_version)
            if not parent:
                raise ValueError(f"Parent version '{parent_version}' not found")
            version_number = SemanticVersion.increment(parent.version_number, version_type)
        else:
            # Check for existing versions in branch
            branch_versions = [
                v for v in self.versions.values() 
                if v.branch == branch and v.model_path == model_path
            ]
            if branch_versions:
                latest = max(branch_versions, 
                           key=lambda v: SemanticVersion.parse(v.version_number))
                version_number = SemanticVersion.increment(latest.version_number, version_type)
            else:
                version_number = "1.0.0"
        
        # Create model version
        version_id = self._generate_version_id()
        model_version = ModelVersion(
            version_id=version_id,
            version_number=version_number,
            branch=branch,
            parent_version=parent_version,
            model_path=model_path,
            model_hash=model_hash,
            metrics=metrics,
            hyperparameters=hyperparameters,
            model_card=model_card,
            metadata=metadata or {}
        )
        
        # Log to MLflow if available
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"{model_name}_v{version_number}"):
                    # Log parameters
                    if hyperparameters:
                        hparams = hyperparameters.to_dict()
                        mlflow.log_params({k: v for k, v in hparams.items() if v is not None})
                    
                    # Log metrics
                    if metrics:
                        metrics_dict = metrics.to_dict()
                        mlflow.log_metrics({k: v for k, v in metrics_dict.items() if v is not None})
                    
                    # Log model
                    if os.path.exists(model_path):
                        mlflow.log_artifact(model_path)
                    
                    # Log model card as artifact
                    if model_card:
                        card_path = self.registry_path / f"model_cards/{version_id}.json"
                        card_path.parent.mkdir(exist_ok=True)
                        with open(card_path, 'w') as f:
                            json.dump(model_card.to_dict(), f, indent=2)
                        mlflow.log_artifact(str(card_path))
                    
                    model_version.mlflow_run_id = mlflow.active_run().info.run_id
                    
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")
        
        # Store version
        self.versions[version_id] = model_version
        
        # Update branch head
        if branch in self.branches:
            self.branches[branch].head_version = version_id
        
        # Update model's latest version
        self.models[model_name]['latest_version'] = version_id
        
        self._save_registry()
        logger.info(f"Logged model version: {model_name} v{version_number} on branch {branch}")
        
        return model_version
    
    def create_branch(self, branch_name: str, from_version: str, 
                     description: str = "") -> ModelBranch:
        """
        Create a new branch from an existing version.
        
        Args:
            branch_name: Name of the new branch
            from_version: Version ID to branch from
            description: Branch description
            
        Returns:
            Created ModelBranch
        """
        if branch_name in self.branches:
            raise ValueError(f"Branch '{branch_name}' already exists")
        
        if from_version not in self.versions:
            raise ValueError(f"Version '{from_version}' not found")
        
        # Create branch
        branch = ModelBranch(
            name=branch_name,
            head_version=from_version,
            description=description
        )
        
        self.branches[branch_name] = branch
        self._save_registry()
        
        logger.info(f"Created branch '{branch_name}' from version {from_version}")
        return branch
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self.versions.get(version_id)
    
    def get_latest_version(self, model_name: str, branch: str = "main") -> Optional[ModelVersion]:
        """Get the latest version of a model on a specific branch."""
        if model_name not in self.models:
            return None
        
        model_versions = [
            v for v in self.versions.values()
            if v.branch == branch and v.model_path.startswith(model_name)
        ]
        
        if not model_versions:
            return None
        
        return max(model_versions, 
                  key=lambda v: SemanticVersion.parse(v.version_number))
    
    def list_versions(self, model_name: str = None, branch: str = None) -> List[ModelVersion]:
        """List all versions, optionally filtered by model and branch."""
        versions = list(self.versions.values())
        
        if model_name:
            versions = [v for v in versions if model_name in v.model_path]
        
        if branch:
            versions = [v for v in versions if v.branch == branch]
        
        return sorted(versions, 
                     key=lambda v: SemanticVersion.parse(v.version_number), 
                     reverse=True)
    
    def list_branches(self) -> List[ModelBranch]:
        """List all branches."""
        return list(self.branches.values())
    
    def rollback(self, model_name: str, target_version_id: str, 
                branch: str = "main") -> ModelVersion:
        """
        Rollback to a previous version.
        
        Args:
            model_name: Name of the model
            target_version_id: Version ID to rollback to
            branch: Branch to rollback on
            
        Returns:
            The rolled back version (as a new version)
        """
        target_version = self.get_version(target_version_id)
        if not target_version:
            raise ValueError(f"Version '{target_version_id}' not found")
        
        # Create a new version that's a copy of the target version
        rollback_version = self.log_model(
            model_name=model_name,
            model_path=target_version.model_path,
            metrics=target_version.metrics,
            hyperparameters=target_version.hyperparameters,
            model_card=target_version.model_card,
            branch=branch,
            version_type=VersionType.PATCH,
            metadata={
                **target_version.metadata,
                'rollback_from': target_version_id,
                'rollback_timestamp': datetime.now().isoformat()
            },
            parent_version=target_version_id
        )
        
        logger.info(f"Rolled back {model_name} to version {target_version.version_number}")
        return rollback_version
    
    def diff_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Dictionary containing differences
        """
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        return ModelDiff.diff_versions(version1, version2)
    
    def get_version_history(self, model_name: str, branch: str = None) -> List[Dict[str, Any]]:
        """
        Get version history with lineage information.
        
        Args:
            model_name: Name of the model
            branch: Optional branch filter
            
        Returns:
            List of version history entries with lineage
        """
        versions = self.list_versions(model_name, branch)
        history = []
        
        for version in versions:
            history_entry = {
                'version_id': version.version_id,
                'version_number': version.version_number,
                'branch': version.branch,
                'created_at': version.created_at.isoformat(),
                'parent_version': version.parent_version,
                'metrics': version.metrics.to_dict() if version.metrics else {},
                'metadata': version.metadata
            }
            history.append(history_entry)
        
        return history
    
    def export_model_card(self, version_id: str, output_path: str) -> str:
        """
        Export model card as markdown.
        
        Args:
            version_id: Version ID
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        version = self.get_version(version_id)
        if not version or not version.model_card:
            raise ValueError("Version or model card not found")
        
        card = version.model_card
        markdown = f"""# Model Card: {card.model_name}

## Version Information
- **Version**: {version.version_number}
- **Branch**: {version.branch}
- **Created**: {version.created_at.isoformat()}

## Model Details
- **Framework**: {card.framework}
- **Author**: {card.author}
- **Tags**: {', '.join(card.tags)}

## Description
{card.description}

## Intended Use
{card.intended_use}

## Training Data
{json.dumps(card.dataset_info, indent=2) if card.dataset_info else 'Not specified'}

## Training Configuration
{json.dumps(card.training_config, indent=2) if card.training_config else 'Not specified'}

## Performance Metrics
{json.dumps(version.metrics.to_dict(), indent=2) if version.metrics else 'Not specified'}

## Hyperparameters
{json.dumps(version.hyperparameters.to_dict(), indent=2) if version.hyperparameters else 'Not specified'}

## Ethical Considerations
{card.ethical_considerations if card.ethical_considerations else 'Not specified'}

## Limitations
{card.limitations if card.limitations else 'Not specified'}

## Model Hash
{version.model_hash}
"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(markdown)
        
        logger.info(f"Exported model card to {output_path}")
        return str(output_file)
    
    def search_models(self, query: str = None, tags: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search models by query or tags.
        
        Args:
            query: Search query (searches name and description)
            tags: Filter by tags
            
        Returns:
            List of matching models
        """
        results = []
        
        for model_name, model_data in self.models.items():
            # Search by query
            if query:
                if (query.lower() not in model_name.lower() and 
                    query.lower() not in model_data.get('description', '').lower()):
                    continue
            
            # Filter by tags
            if tags:
                model_tags = model_data.get('tags', [])
                if not any(tag in model_tags for tag in tags):
                    continue
            
            results.append({
                'name': model_name,
                **model_data
            })
        
        return results


# Singleton instance for global access
_tracker_instance = None


def get_experiment_tracker(registry_path: str = "model_registry") -> ExperimentTracker:
    """Get or create the global ExperimentTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = ExperimentTracker(registry_path)
    return _tracker_instance


# Convenience functions for common operations
def log_experiment(model_name: str, model_path: str, **kwargs) -> ModelVersion:
    """Convenience function to log an experiment."""
    tracker = get_experiment_tracker()
    return tracker.log_model(model_name, model_path, **kwargs)


def create_experiment_branch(branch_name: str, from_version: str, **kwargs) -> ModelBranch:
    """Convenience function to create a branch."""
    tracker = get_experiment_tracker()
    return tracker.create_branch(branch_name, from_version, **kwargs)


def rollback_experiment(model_name: str, target_version: str, **kwargs) -> ModelVersion:
    """Convenience function for rollback."""
    tracker = get_experiment_tracker()
    return tracker.rollback(model_name, target_version, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker("./test_registry")
    
    # Create a model
    model_name = "vex_llama_7b"
    tracker.create_model(
        model_name=model_name,
        description="Fine-tuned Llama 7B model using Unsloth",
        author="Unsloth Team",
        tags=["llm", "llama", "fine-tuned"]
    )
    
    # Create model card
    model_card = ModelCard(
        model_name=model_name,
        description="A fine-tuned Llama 7B model optimized for instruction following",
        author="Unsloth Team",
        tags=["llm", "instruction-following"],
        framework="pytorch",
        dataset_info={"name": "alpaca", "size": "52k samples"},
        training_config={"epochs": 3, "learning_rate": 2e-5},
        intended_use="Research and educational purposes",
        limitations="May generate biased or harmful content"
    )
    
    # Log a model version
    version = tracker.log_model(
        model_name=model_name,
        model_path="./models/vex_llama_7b_v1.bin",
        metrics=ModelMetrics(accuracy=0.85, loss=0.45),
        hyperparameters=ModelHyperparameters(learning_rate=2e-5, batch_size=4),
        model_card=model_card,
        branch="main",
        version_type=VersionType.MAJOR
    )
    
    print(f"Logged version: {version.version_number}")
    
    # Create a branch
    branch = tracker.create_branch(
        branch_name="experimental",
        from_version=version.version_id,
        description="Experimental improvements"
    )
    
    print(f"Created branch: {branch.name}")
    
    # List versions
    versions = tracker.list_versions(model_name)
    print(f"Total versions: {len(versions)}")
    
    # Export model card
    tracker.export_model_card(version.version_id, "./model_card.md")