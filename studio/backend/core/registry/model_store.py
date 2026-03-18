"""
Model Versioning & Experiment Tracking System for SOVEREIGN
Implements comprehensive model versioning with Git-like branching,
experiment tracking with MLflow integration, and one-click rollback capabilities.
"""

import os
import json
import sqlite3
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import semver
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import StaticPool

# Import existing modules for integration
from studio.backend.auth.authentication import get_current_user
from studio.backend.core.data_recipe.jobs.manager import JobManager

Base = declarative_base()

class VersionStatus(Enum):
    """Status of a model version"""
    DRAFT = "draft"
    TRAINING = "training"
    VALIDATED = "validated"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"

class ExperimentStatus(Enum):
    """Status of an experiment"""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if key == "custom_metrics":
                    result.update(value)
                else:
                    result[key] = value
        return result

@dataclass
class ModelCard:
    """Model card with comprehensive metadata"""
    model_id: str
    version: str
    description: str
    author: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    hyperparameters: Dict[str, Any]
    metrics: ModelMetrics
    training_data: Dict[str, Any]
    intended_use: str
    limitations: List[str]
    ethical_considerations: str
    citations: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    license: str = "Apache-2.0"
    
    def to_markdown(self) -> str:
        """Generate markdown representation of model card"""
        md = f"""# Model Card: {self.model_id} v{self.version}

## Model Details
- **Model ID**: {self.model_id}
- **Version**: {self.version}
- **Author**: {self.author}
- **Created**: {self.created_at.isoformat()}
- **Updated**: {self.updated_at.isoformat()}
- **License**: {self.license}

## Description
{self.description}

## Intended Use
{self.intended_use}

## Training Data
{json.dumps(self.training_data, indent=2)}

## Hyperparameters
{json.dumps(self.hyperparameters, indent=2)}

## Evaluation Metrics
{json.dumps(self.metrics.to_dict(), indent=2)}

## Limitations
{chr(10).join(f'- {limitation}' for limitation in self.limitations)}

## Ethical Considerations
{self.ethical_considerations}

## Citations
{chr(10).join(f'- {citation}' for citation in self.citations)}

## Tags
{', '.join(self.tags)}
"""
        return md

class ModelVersion(Base):
    """SQLAlchemy model for model versions"""
    __tablename__ = 'model_versions'
    
    id = Column(String, primary_key=True)
    model_id = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False)
    branch = Column(String, default="main")
    parent_version = Column(String, nullable=True)
    status = Column(String, default=VersionStatus.DRAFT.value)
    description = Column(Text, default="")
    author = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    hyperparameters = Column(JSON, default={})
    metrics = Column(JSON, default={})
    training_data_hash = Column(String, nullable=True)
    model_artifact_path = Column(String, nullable=True)
    mlflow_run_id = Column(String, nullable=True)
    model_card = Column(JSON, default={})
    tags = Column(JSON, default=[])
    is_production = Column(Boolean, default=False)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique version ID"""
        content = f"{self.model_id}:{self.version}:{self.branch}:{datetime.datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'model_id': self.model_id,
            'version': self.version,
            'branch': self.branch,
            'parent_version': self.parent_version,
            'status': self.status,
            'description': self.description,
            'author': self.author,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'training_data_hash': self.training_data_hash,
            'model_artifact_path': self.model_artifact_path,
            'mlflow_run_id': self.mlflow_run_id,
            'model_card': self.model_card,
            'tags': self.tags,
            'is_production': self.is_production
        }

class Experiment(Base):
    """SQLAlchemy model for experiments"""
    __tablename__ = 'experiments'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, default="")
    model_id = Column(String, nullable=False, index=True)
    version_id = Column(String, nullable=False)
    status = Column(String, default=ExperimentStatus.RUNNING.value)
    hyperparameters = Column(JSON, default={})
    metrics = Column(JSON, default={})
    artifacts = Column(JSON, default={})
    mlflow_experiment_id = Column(String, nullable=True)
    mlflow_run_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    tags = Column(JSON, default=[])
    notes = Column(Text, default="")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique experiment ID"""
        content = f"{self.name}:{self.model_id}:{datetime.datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class ModelStore:
    """
    Comprehensive model versioning and experiment tracking system
    with Git-like branching and MLflow integration.
    """
    
    def __init__(self, db_url: str = "sqlite:///model_registry.db", 
                 mlflow_tracking_uri: Optional[str] = None,
                 storage_path: str = "./model_artifacts"):
        """
        Initialize the ModelStore.
        
        Args:
            db_url: Database connection URL
            mlflow_tracking_uri: MLflow tracking server URI
            storage_path: Path to store model artifacts
        """
        self.db_url = db_url
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.engine = create_engine(
            db_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
        )
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
        # Initialize MLflow
        self.mlflow_client = None
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            self.mlflow_client = MlflowClient()
        
        # Initialize job manager for integration
        self.job_manager = JobManager()
    
    def _get_session(self):
        """Get database session"""
        return self.Session()
    
    def create_model_version(
        self,
        model_id: str,
        version: str,
        branch: str = "main",
        parent_version: Optional[str] = None,
        description: str = "",
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_data_info: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> ModelVersion:
        """
        Create a new model version with semantic versioning.
        
        Args:
            model_id: Unique model identifier
            version: Semantic version (e.g., "1.0.0")
            branch: Branch name (default: "main")
            parent_version: Parent version for branching
            description: Version description
            hyperparameters: Model hyperparameters
            training_data_info: Training data metadata
            tags: Version tags
            
        Returns:
            Created ModelVersion object
        """
        session = self._get_session()
        try:
            # Validate semantic version
            semver.VersionInfo.parse(version)
            
            # Check if version already exists on branch
            existing = session.query(ModelVersion).filter_by(
                model_id=model_id,
                version=version,
                branch=branch
            ).first()
            
            if existing:
                raise ValueError(f"Version {version} already exists on branch {branch}")
            
            # Get current user for author
            try:
                author = get_current_user().username
            except:
                author = "system"
            
            # Create model version
            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                branch=branch,
                parent_version=parent_version,
                description=description,
                author=author,
                hyperparameters=hyperparameters or {},
                tags=tags or []
            )
            
            # Calculate training data hash if provided
            if training_data_info:
                training_data_hash = hashlib.sha256(
                    json.dumps(training_data_info, sort_keys=True).encode()
                ).hexdigest()
                model_version.training_data_hash = training_data_hash
            
            session.add(model_version)
            session.commit()
            
            # Create MLflow experiment if MLflow is configured
            if self.mlflow_client:
                self._create_mlflow_experiment(model_version, training_data_info)
            
            return model_version
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def _create_mlflow_experiment(self, model_version: ModelVersion, 
                                 training_data_info: Optional[Dict[str, Any]] = None):
        """Create MLflow experiment for model version"""
        try:
            # Create or get MLflow experiment
            experiment_name = f"{model_version.model_id}_{model_version.version}_{model_version.branch}"
            experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
            
            if not experiment:
                experiment_id = self.mlflow_client.create_experiment(
                    experiment_name,
                    tags={
                        "model_id": model_version.model_id,
                        "version": model_version.version,
                        "branch": model_version.branch,
                        "author": model_version.author
                    }
                )
            else:
                experiment_id = experiment.experiment_id
            
            # Start MLflow run
            with mlflow.start_run(experiment_id=experiment_id) as run:
                # Log hyperparameters
                if model_version.hyperparameters:
                    mlflow.log_params(model_version.hyperparameters)
                
                # Log training data info
                if training_data_info:
                    mlflow.log_params({f"training_data_{k}": v for k, v in training_data_info.items()})
                
                # Update model version with MLflow run ID
                session = self._get_session()
                try:
                    model_version.mlflow_run_id = run.info.run_id
                    session.merge(model_version)
                    session.commit()
                finally:
                    session.close()
        except Exception as e:
            print(f"Warning: Failed to create MLflow experiment: {e}")
    
    def log_experiment(
        self,
        experiment_name: str,
        model_id: str,
        version_id: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        notes: str = "",
        tags: Optional[List[str]] = None
    ) -> Experiment:
        """
        Log an experiment with metrics and artifacts.
        
        Args:
            experiment_name: Name of the experiment
            model_id: Model identifier
            version_id: Model version ID
            hyperparameters: Experiment hyperparameters
            metrics: Performance metrics
            artifacts: Artifact paths
            notes: Experiment notes
            tags: Experiment tags
            
        Returns:
            Created Experiment object
        """
        session = self._get_session()
        try:
            # Get model version
            model_version = session.query(ModelVersion).filter_by(id=version_id).first()
            if not model_version:
                raise ValueError(f"Model version {version_id} not found")
            
            # Create experiment
            experiment = Experiment(
                name=experiment_name,
                model_id=model_id,
                version_id=version_id,
                hyperparameters=hyperparameters or {},
                metrics=metrics or {},
                artifacts=artifacts or {},
                notes=notes,
                tags=tags or []
            )
            
            # Log to MLflow if configured
            if self.mlflow_client and model_version.mlflow_run_id:
                try:
                    with mlflow.start_run(run_id=model_version.mlflow_run_id):
                        if hyperparameters:
                            mlflow.log_params(hyperparameters)
                        if metrics:
                            mlflow.log_metrics(metrics)
                        if artifacts:
                            for artifact_name, artifact_path in artifacts.items():
                                mlflow.log_artifact(artifact_path, artifact_name)
                except Exception as e:
                    print(f"Warning: Failed to log to MLflow: {e}")
            
            session.add(experiment)
            session.commit()
            
            return experiment
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def update_model_metrics(
        self,
        version_id: str,
        metrics: Dict[str, float],
        status: Optional[VersionStatus] = None
    ) -> ModelVersion:
        """
        Update model metrics and optionally status.
        
        Args:
            version_id: Model version ID
            metrics: Performance metrics
            status: New status
            
        Returns:
            Updated ModelVersion object
        """
        session = self._get_session()
        try:
            model_version = session.query(ModelVersion).filter_by(id=version_id).first()
            if not model_version:
                raise ValueError(f"Model version {version_id} not found")
            
            # Update metrics
            current_metrics = model_version.metrics or {}
            current_metrics.update(metrics)
            model_version.metrics = current_metrics
            
            # Update status if provided
            if status:
                model_version.status = status.value
            
            # Update timestamp
            model_version.updated_at = datetime.datetime.utcnow()
            
            session.merge(model_version)
            session.commit()
            
            # Log to MLflow
            if self.mlflow_client and model_version.mlflow_run_id:
                try:
                    with mlflow.start_run(run_id=model_version.mlflow_run_id):
                        mlflow.log_metrics(metrics)
                except Exception as e:
                    print(f"Warning: Failed to log metrics to MLflow: {e}")
            
            return model_version
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def create_branch(
        self,
        model_id: str,
        source_branch: str,
        new_branch: str,
        version: str
    ) -> ModelVersion:
        """
        Create a new branch from an existing version.
        
        Args:
            model_id: Model identifier
            source_branch: Source branch name
            new_branch: New branch name
            version: Version to branch from
            
        Returns:
            New ModelVersion on the new branch
        """
        session = self._get_session()
        try:
            # Get source version
            source_version = session.query(ModelVersion).filter_by(
                model_id=model_id,
                version=version,
                branch=source_branch
            ).first()
            
            if not source_version:
                raise ValueError(f"Version {version} not found on branch {source_branch}")
            
            # Create new version on new branch
            new_version = ModelVersion(
                model_id=model_id,
                version=version,
                branch=new_branch,
                parent_version=source_version.id,
                description=f"Branched from {source_branch}/{version}",
                author=source_version.author,
                hyperparameters=source_version.hyperparameters,
                metrics=source_version.metrics,
                training_data_hash=source_version.training_data_hash,
                tags=source_version.tags
            )
            
            session.add(new_version)
            session.commit()
            
            return new_version
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def merge_branch(
        self,
        model_id: str,
        source_branch: str,
        target_branch: str,
        version: str,
        merge_strategy: str = "theirs"
    ) -> ModelVersion:
        """
        Merge a branch into another branch.
        
        Args:
            model_id: Model identifier
            source_branch: Source branch name
            target_branch: Target branch name
            version: Version to merge
            merge_strategy: Merge strategy ("theirs", "ours", "manual")
            
        Returns:
            Merged ModelVersion on target branch
        """
        session = self._get_session()
        try:
            # Get source and target versions
            source_version = session.query(ModelVersion).filter_by(
                model_id=model_id,
                version=version,
                branch=source_branch
            ).first()
            
            target_version = session.query(ModelVersion).filter_by(
                model_id=model_id,
                version=version,
                branch=target_branch
            ).first()
            
            if not source_version:
                raise ValueError(f"Version {version} not found on branch {source_branch}")
            
            # Create merged version
            if merge_strategy == "theirs":
                # Use source version's data
                merged_hyperparameters = source_version.hyperparameters
                merged_metrics = source_version.metrics
                merged_tags = source_version.tags
            elif merge_strategy == "ours" and target_version:
                # Keep target version's data
                merged_hyperparameters = target_version.hyperparameters
                merged_metrics = target_version.metrics
                merged_tags = target_version.tags
            else:
                # Manual merge - combine both
                merged_hyperparameters = {}
                if target_version:
                    merged_hyperparameters.update(target_version.hyperparameters)
                merged_hyperparameters.update(source_version.hyperparameters)
                
                merged_metrics = {}
                if target_version:
                    merged_metrics.update(target_version.metrics)
                merged_metrics.update(source_version.metrics)
                
                merged_tags = list(set(
                    (target_version.tags if target_version else []) + 
                    source_version.tags
                ))
            
            # Increment patch version for merge
            current_version = semver.VersionInfo.parse(version)
            merged_version_str = str(current_version.bump_patch())
            
            # Create merged version
            merged_version = ModelVersion(
                model_id=model_id,
                version=merged_version_str,
                branch=target_branch,
                parent_version=target_version.id if target_version else None,
                description=f"Merged from {source_branch}/{version}",
                author=source_version.author,
                hyperparameters=merged_hyperparameters,
                metrics=merged_metrics,
                training_data_hash=source_version.training_data_hash,
                tags=merged_tags
            )
            
            session.add(merged_version)
            session.commit()
            
            return merged_version
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def rollback_version(
        self,
        model_id: str,
        target_version: str,
        branch: str = "main",
        create_new_version: bool = True
    ) -> ModelVersion:
        """
        Rollback to a previous version.
        
        Args:
            model_id: Model identifier
            target_version: Version to rollback to
            branch: Branch name
            create_new_version: If True, create a new version with rollback data
            
        Returns:
            Rolled back ModelVersion
        """
        session = self._get_session()
        try:
            # Get target version
            target = session.query(ModelVersion).filter_by(
                model_id=model_id,
                version=target_version,
                branch=branch
            ).first()
            
            if not target:
                raise ValueError(f"Target version {target_version} not found on branch {branch}")
            
            if create_new_version:
                # Create new version with rollback data
                current_version = session.query(ModelVersion).filter_by(
                    model_id=model_id,
                    branch=branch
                ).order_by(ModelVersion.created_at.desc()).first()
                
                if current_version:
                    current_semver = semver.VersionInfo.parse(current_version.version)
                    new_version_str = str(current_semver.bump_patch())
                else:
                    new_version_str = "1.0.0"
                
                rollback_version = ModelVersion(
                    model_id=model_id,
                    version=new_version_str,
                    branch=branch,
                    parent_version=current_version.id if current_version else None,
                    description=f"Rollback to version {target_version}",
                    author=target.author,
                    hyperparameters=target.hyperparameters,
                    metrics=target.metrics,
                    training_data_hash=target.training_data_hash,
                    model_artifact_path=target.model_artifact_path,
                    tags=target.tags + ["rollback"]
                )
                
                session.add(rollback_version)
                session.commit()
                
                return rollback_version
            else:
                # Just update the production flag
                # First, clear production flag from all versions
                session.query(ModelVersion).filter_by(
                    model_id=model_id,
                    branch=branch,
                    is_production=True
                ).update({"is_production": False})
                
                # Set target version as production
                target.is_production = True
                target.status = VersionStatus.PRODUCTION.value
                target.updated_at = datetime.datetime.utcnow()
                
                session.merge(target)
                session.commit()
                
                return target
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def diff_versions(
        self,
        version_id1: str,
        version_id2: str
    ) -> Dict[str, Any]:
        """
        Generate diff between two model versions.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Dictionary with differences
        """
        session = self._get_session()
        try:
            version1 = session.query(ModelVersion).filter_by(id=version_id1).first()
            version2 = session.query(ModelVersion).filter_by(id=version_id2).first()
            
            if not version1 or not version2:
                raise ValueError("One or both versions not found")
            
            diff = {
                "metadata": {
                    "version1": version1.to_dict(),
                    "version2": version2.to_dict()
                },
                "hyperparameters": self._diff_dicts(
                    version1.hyperparameters,
                    version2.hyperparameters
                ),
                "metrics": self._diff_dicts(
                    version1.metrics,
                    version2.metrics
                ),
                "tags": {
                    "added": list(set(version2.tags) - set(version1.tags)),
                    "removed": list(set(version1.tags) - set(version2.tags)),
                    "common": list(set(version1.tags) & set(version2.tags))
                },
                "status_changed": version1.status != version2.status,
                "production_changed": version1.is_production != version2.is_production
            }
            
            return diff
        finally:
            session.close()
    
    def _diff_dicts(self, dict1: Dict, dict2: Dict) -> Dict[str, Any]:
        """Generate diff between two dictionaries"""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        diff = {}
        
        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            if val1 == val2:
                diff[key] = {"status": "unchanged", "value": val1}
            elif val1 is None:
                diff[key] = {"status": "added", "value": val2}
            elif val2 is None:
                diff[key] = {"status": "removed", "value": val1}
            else:
                diff[key] = {
                    "status": "changed",
                    "old_value": val1,
                    "new_value": val2,
                    "change": val2 - val1 if isinstance(val1, (int, float)) else None
                }
        
        return diff
    
    def get_version_history(
        self,
        model_id: str,
        branch: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a model.
        
        Args:
            model_id: Model identifier
            branch: Filter by branch
            limit: Maximum number of versions to return
            
        Returns:
            List of version dictionaries
        """
        session = self._get_session()
        try:
            query = session.query(ModelVersion).filter_by(model_id=model_id)
            
            if branch:
                query = query.filter_by(branch=branch)
            
            versions = query.order_by(ModelVersion.created_at.desc()).limit(limit).all()
            
            return [v.to_dict() for v in versions]
        finally:
            session.close()
    
    def get_experiment_history(
        self,
        model_id: Optional[str] = None,
        version_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get experiment history.
        
        Args:
            model_id: Filter by model ID
            version_id: Filter by version ID
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiment dictionaries
        """
        session = self._get_session()
        try:
            query = session.query(Experiment)
            
            if model_id:
                query = query.filter_by(model_id=model_id)
            if version_id:
                query = query.filter_by(version_id=version_id)
            
            experiments = query.order_by(Experiment.created_at.desc()).limit(limit).all()
            
            return [
                {
                    "id": e.id,
                    "name": e.name,
                    "model_id": e.model_id,
                    "version_id": e.version_id,
                    "status": e.status,
                    "hyperparameters": e.hyperparameters,
                    "metrics": e.metrics,
                    "created_at": e.created_at.isoformat(),
                    "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                    "duration_seconds": e.duration_seconds,
                    "tags": e.tags
                }
                for e in experiments
            ]
        finally:
            session.close()
    
    def generate_model_card(self, version_id: str) -> ModelCard:
        """
        Generate a model card for a specific version.
        
        Args:
            version_id: Model version ID
            
        Returns:
            ModelCard object
        """
        session = self._get_session()
        try:
            version = session.query(ModelVersion).filter_by(id=version_id).first()
            if not version:
                raise ValueError(f"Version {version_id} not found")
            
            # Get training data info
            training_data_info = {}
            if version.training_data_hash:
                # In a real implementation, you would look up training data info
                # from your data registry using the hash
                training_data_info = {
                    "hash": version.training_data_hash,
                    "description": "Training data information not available in this demo"
                }
            
            # Create model metrics object
            metrics = ModelMetrics(
                accuracy=version.metrics.get("accuracy"),
                loss=version.metrics.get("loss"),
                f1_score=version.metrics.get("f1_score"),
                precision=version.metrics.get("precision"),
                recall=version.metrics.get("recall"),
                custom_metrics={k: v for k, v in version.metrics.items() 
                               if k not in ["accuracy", "loss", "f1_score", "precision", "recall"]}
            )
            
            # Create model card
            model_card = ModelCard(
                model_id=version.model_id,
                version=version.version,
                description=version.description,
                author=version.author,
                created_at=version.created_at,
                updated_at=version.updated_at,
                hyperparameters=version.hyperparameters,
                metrics=metrics,
                training_data=training_data_info,
                intended_use=version.model_card.get("intended_use", "General purpose model"),
                limitations=version.model_card.get("limitations", ["No limitations specified"]),
                ethical_considerations=version.model_card.get("ethical_considerations", 
                                                          "No ethical considerations specified"),
                citations=version.model_card.get("citations", []),
                tags=version.tags,
                license=version.model_card.get("license", "Apache-2.0")
            )
            
            return model_card
        finally:
            session.close()
    
    def update_model_card(
        self,
        version_id: str,
        intended_use: Optional[str] = None,
        limitations: Optional[List[str]] = None,
        ethical_considerations: Optional[str] = None,
        citations: Optional[List[str]] = None,
        license: Optional[str] = None
    ) -> ModelVersion:
        """
        Update model card information.
        
        Args:
            version_id: Model version ID
            intended_use: Intended use description
            limitations: List of limitations
            ethical_considerations: Ethical considerations
            citations: List of citations
            license: License type
            
        Returns:
            Updated ModelVersion
        """
        session = self._get_session()
        try:
            version = session.query(ModelVersion).filter_by(id=version_id).first()
            if not version:
                raise ValueError(f"Version {version_id} not found")
            
            # Update model card
            model_card = version.model_card or {}
            
            if intended_use is not None:
                model_card["intended_use"] = intended_use
            if limitations is not None:
                model_card["limitations"] = limitations
            if ethical_considerations is not None:
                model_card["ethical_considerations"] = ethical_considerations
            if citations is not None:
                model_card["citations"] = citations
            if license is not None:
                model_card["license"] = license
            
            version.model_card = model_card
            version.updated_at = datetime.datetime.utcnow()
            
            session.merge(version)
            session.commit()
            
            return version
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def store_model_artifact(
        self,
        version_id: str,
        artifact_path: str,
        artifact_type: str = "model"
    ) -> str:
        """
        Store model artifact and update version record.
        
        Args:
            version_id: Model version ID
            artifact_path: Path to artifact file
            artifact_type: Type of artifact
            
        Returns:
            Stored artifact path
        """
        session = self._get_session()
        try:
            version = session.query(ModelVersion).filter_by(id=version_id).first()
            if not version:
                raise ValueError(f"Version {version_id} not found")
            
            # Create storage directory
            storage_dir = self.storage_path / version.model_id / version.version
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy artifact
            import shutil
            artifact_filename = Path(artifact_path).name
            stored_path = storage_dir / artifact_filename
            shutil.copy2(artifact_path, stored_path)
            
            # Update version record
            version.model_artifact_path = str(stored_path)
            version.updated_at = datetime.datetime.utcnow()
            
            session.merge(version)
            session.commit()
            
            # Log artifact to MLflow
            if self.mlflow_client and version.mlflow_run_id:
                try:
                    with mlflow.start_run(run_id=version.mlflow_run_id):
                        mlflow.log_artifact(str(stored_path), artifact_type)
                except Exception as e:
                    print(f"Warning: Failed to log artifact to MLflow: {e}")
            
            return str(stored_path)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_production_version(self, model_id: str, branch: str = "main") -> Optional[ModelVersion]:
        """
        Get the current production version of a model.
        
        Args:
            model_id: Model identifier
            branch: Branch name
            
        Returns:
            Production ModelVersion or None
        """
        session = self._get_session()
        try:
            return session.query(ModelVersion).filter_by(
                model_id=model_id,
                branch=branch,
                is_production=True,
                status=VersionStatus.PRODUCTION.value
            ).first()
        finally:
            session.close()
    
    def promote_to_production(self, version_id: str) -> ModelVersion:
        """
        Promote a version to production.
        
        Args:
            version_id: Model version ID
            
        Returns:
            Promoted ModelVersion
        """
        session = self._get_session()
        try:
            version = session.query(ModelVersion).filter_by(id=version_id).first()
            if not version:
                raise ValueError(f"Version {version_id} not found")
            
            # Clear production flag from all versions of this model on this branch
            session.query(ModelVersion).filter_by(
                model_id=version.model_id,
                branch=version.branch,
                is_production=True
            ).update({"is_production": False})
            
            # Set this version as production
            version.is_production = True
            version.status = VersionStatus.PRODUCTION.value
            version.updated_at = datetime.datetime.utcnow()
            
            session.merge(version)
            session.commit()
            
            return version
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def archive_version(self, version_id: str) -> ModelVersion:
        """
        Archive a model version.
        
        Args:
            version_id: Model version ID
            
        Returns:
            Archived ModelVersion
        """
        session = self._get_session()
        try:
            version = session.query(ModelVersion).filter_by(id=version_id).first()
            if not version:
                raise ValueError(f"Version {version_id} not found")
            
            version.status = VersionStatus.ARCHIVED.value
            version.updated_at = datetime.datetime.utcnow()
            
            session.merge(version)
            session.commit()
            
            return version
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[VersionStatus] = None,
        branch: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search for models based on various criteria.
        
        Args:
            query: Search query for model ID or description
            tags: Filter by tags
            status: Filter by status
            branch: Filter by branch
            limit: Maximum results
            
        Returns:
            List of matching model versions
        """
        session = self._get_session()
        try:
            query_obj = session.query(ModelVersion)
            
            if query:
                query_obj = query_obj.filter(
                    (ModelVersion.model_id.ilike(f"%{query}%")) |
                    (ModelVersion.description.ilike(f"%{query}%"))
                )
            
            if tags:
                for tag in tags:
                    query_obj = query_obj.filter(ModelVersion.tags.contains([tag]))
            
            if status:
                query_obj = query_obj.filter_by(status=status.value)
            
            if branch:
                query_obj = query_obj.filter_by(branch=branch)
            
            versions = query_obj.order_by(ModelVersion.updated_at.desc()).limit(limit).all()
            
            return [v.to_dict() for v in versions]
        finally:
            session.close()
    
    def export_version(self, version_id: str, export_path: str) -> str:
        """
        Export a model version with all metadata.
        
        Args:
            version_id: Model version ID
            export_path: Path to export directory
            
        Returns:
            Path to exported package
        """
        session = self._get_session()
        try:
            version = session.query(ModelVersion).filter_by(id=version_id).first()
            if not version:
                raise ValueError(f"Version {version_id} not found")
            
            export_dir = Path(export_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Export metadata
            metadata = version.to_dict()
            metadata_path = export_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Export model card
            model_card = self.generate_model_card(version_id)
            model_card_path = export_dir / "model_card.md"
            with open(model_card_path, "w") as f:
                f.write(model_card.to_markdown())
            
            # Copy model artifact if exists
            if version.model_artifact_path and os.path.exists(version.model_artifact_path):
                import shutil
                artifact_name = Path(version.model_artifact_path).name
                shutil.copy2(version.model_artifact_path, export_dir / artifact_name)
            
            # Create archive
            import zipfile
            archive_path = f"{export_dir}.zip"
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in export_dir.rglob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(export_dir))
            
            return archive_path
        finally:
            session.close()
    
    def import_version(self, archive_path: str, model_id: Optional[str] = None) -> ModelVersion:
        """
        Import a model version from an exported package.
        
        Args:
            archive_path: Path to exported archive
            model_id: Override model ID
            
        Returns:
            Imported ModelVersion
        """
        import tempfile
        import zipfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract archive
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            # Load metadata
            metadata_path = Path(temp_dir) / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Override model ID if provided
            if model_id:
                metadata["model_id"] = model_id
            
            # Create new version
            version = self.create_model_version(
                model_id=metadata["model_id"],
                version=metadata["version"],
                branch=metadata.get("branch", "main"),
                description=metadata.get("description", "Imported version"),
                hyperparameters=metadata.get("hyperparameters"),
                tags=metadata.get("tags")
            )
            
            # Copy model artifact if exists
            artifact_files = list(Path(temp_dir).glob("*.bin")) + list(Path(temp_dir).glob("*.pt"))
            if artifact_files:
                self.store_model_artifact(version.id, str(artifact_files[0]))
            
            return version

# Global instance for easy access
model_store = ModelStore()

# Convenience functions for common operations
def create_model_version(*args, **kwargs):
    """Convenience function for creating model versions"""
    return model_store.create_model_version(*args, **kwargs)

def log_experiment(*args, **kwargs):
    """Convenience function for logging experiments"""
    return model_store.log_experiment(*args, **kwargs)

def get_production_version(*args, **kwargs):
    """Convenience function for getting production version"""
    return model_store.get_production_version(*args, **kwargs)

def rollback_version(*args, **kwargs):
    """Convenience function for rolling back versions"""
    return model_store.rollback_version(*args, **kwargs)

def diff_versions(*args, **kwargs):
    """Convenience function for diffing versions"""
    return model_store.diff_versions(*args, **kwargs)