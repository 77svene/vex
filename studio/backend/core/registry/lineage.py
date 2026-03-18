"""
Model Registry & Version Control — Git-like versioning for models with lineage tracking, diff visualization between fine-tunes, and automatic model card generation. Enables reproducible ML workflows.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class ModelVersion(Base):
    """Database model for storing model versions with lineage tracking."""

    __tablename__ = "model_versions"

    id = Column(String(64), primary_key=True)  # SHA-256 hash of model weights + config
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    model_type = Column(String(100), nullable=False)  # e.g., "llama", "mistral", "custom"
    base_model = Column(String(255), nullable=True)  # Original model name before fine-tuning
    
    # Training metadata
    training_config = Column(JSONB, nullable=False)
    training_metrics = Column(JSONB, nullable=True)
    dataset_info = Column(JSONB, nullable=True)
    
    # Lineage tracking
    parent_id = Column(String(64), ForeignKey("model_versions.id"), nullable=True, index=True)
    parent = relationship("ModelVersion", remote_side=[id], backref="children")
    
    # Storage information
    storage_path = Column(String(512), nullable=False)  # Path to model weights in content-addressable storage
    config_path = Column(String(512), nullable=False)  # Path to model configuration
    tokenizer_path = Column(String(512), nullable=True)  # Path to tokenizer files
    
    # Model card
    model_card = Column(Text, nullable=True)  # Auto-generated model card in markdown
    
    # Timestamps and ownership
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(255), nullable=True)  # User identifier
    
    # Tags for organization
    tags = Column(JSONB, nullable=True)  # List of tags like ["fine-tuned", "production", "experimental"]
    
    # Visibility and access
    is_public = Column(Boolean, default=False, nullable=False)
    
    # Version metadata
    version_number = Column(String(50), nullable=True)  # Semantic version if provided
    commit_message = Column(Text, nullable=True)  # User-provided description of changes
    
    __table_args__ = (
        Index("idx_model_versions_name_created", "name", "created_at"),
        Index("idx_model_versions_model_type", "model_type"),
        Index("idx_model_versions_tags", "tags", postgresql_using="gin"),
    )


class ModelDiff(Base):
    """Stores computed diffs between model versions for quick retrieval."""

    __tablename__ = "model_diffs"

    id = Column(String(64), primary_key=True)  # SHA-256 of sorted version IDs
    version_a_id = Column(String(64), ForeignKey("model_versions.id"), nullable=False)
    version_b_id = Column(String(64), ForeignKey("model_versions.id"), nullable=False)
    
    # Diff results
    config_diff = Column(JSONB, nullable=True)  # Differences in training configuration
    metrics_diff = Column(JSONB, nullable=True)  # Differences in training metrics
    dataset_diff = Column(JSONB, nullable=True)  # Differences in dataset information
    summary = Column(Text, nullable=True)  # Human-readable summary of changes
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    version_a = relationship("ModelVersion", foreign_keys=[version_a_id])
    version_b = relationship("ModelVersion", foreign_keys=[version_b_id])
    
    __table_args__ = (
        UniqueConstraint("version_a_id", "version_b_id", name="uq_model_diffs_versions"),
        Index("idx_model_diffs_version_a", "version_a_id"),
        Index("idx_model_diffs_version_b", "version_b_id"),
    )


class ContentAddressableStorage:
    """Content-addressable storage for model weights using SHA-256 hashing."""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_storage_path(self, content_hash: str) -> Path:
        """Generate storage path using first 2 chars of hash for directory sharding."""
        return self.base_path / content_hash[:2] / content_hash
    
    def store(self, content: bytes) -> str:
        """Store content and return its SHA-256 hash."""
        content_hash = hashlib.sha256(content).hexdigest()
        storage_path = self._get_storage_path(content_hash)
        
        if not storage_path.exists():
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(storage_path, "wb") as f:
                f.write(content)
        
        return content_hash
    
    def retrieve(self, content_hash: str) -> Optional[bytes]:
        """Retrieve content by its hash."""
        storage_path = self._get_storage_path(content_hash)
        if storage_path.exists():
            with open(storage_path, "rb") as f:
                return f.read()
        return None
    
    def exists(self, content_hash: str) -> bool:
        """Check if content exists in storage."""
        return self._get_storage_path(content_hash).exists()


class ModelRegistry:
    """
    Git-like model registry with version control, lineage tracking, and diff capabilities.
    
    Features:
    - SHA-256 based versioning with parent pointers for lineage
    - Content-addressable storage for model weights
    - Automatic model card generation
    - Diff visualization between model versions
    - PostgreSQL backend for metadata storage
    """
    
    def __init__(self, database_url: str, storage_path: Union[str, Path]):
        """
        Initialize the model registry.
        
        Args:
            database_url: PostgreSQL connection URL
            storage_path: Path to content-addressable storage directory
        """
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.storage = ContentAddressableStorage(storage_path)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
    
    def _compute_model_hash(
        self,
        weights: bytes,
        config: Dict[str, Any],
        parent_id: Optional[str] = None
    ) -> str:
        """
        Compute deterministic hash for model version.
        
        Hash is computed from:
        1. SHA-256 of model weights
        2. JSON-serialized training config (sorted keys)
        3. Parent version ID (if exists)
        """
        weights_hash = hashlib.sha256(weights).hexdigest()
        config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        
        # Combine all components for final hash
        components = [weights_hash, config_hash]
        if parent_id:
            components.append(parent_id)
        
        combined = "".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def register_model(
        self,
        name: str,
        weights: bytes,
        config: Dict[str, Any],
        model_type: str,
        base_model: Optional[str] = None,
        parent_id: Optional[str] = None,
        description: Optional[str] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        version_number: Optional[str] = None,
        commit_message: Optional[str] = None,
        is_public: bool = False,
    ) -> str:
        """
        Register a new model version in the registry.
        
        Args:
            name: Model name
            weights: Model weights as bytes
            config: Training configuration dictionary
            model_type: Type of model (e.g., "llama", "mistral")
            base_model: Original model name before fine-tuning
            parent_id: ID of parent model version (for lineage tracking)
            description: Human-readable description
            training_metrics: Dictionary of training metrics
            dataset_info: Information about training dataset
            tags: List of tags for organization
            created_by: User identifier
            version_number: Semantic version (e.g., "1.0.0")
            commit_message: Description of changes from parent
            is_public: Whether model is publicly accessible
            
        Returns:
            Model version ID (SHA-256 hash)
        """
        # Compute model hash
        model_id = self._compute_model_hash(weights, config, parent_id)
        
        # Store weights in content-addressable storage
        weights_hash = self.storage.store(weights)
        
        # Generate storage paths
        storage_path = f"weights/{weights_hash[:2]}/{weights_hash}"
        config_path = f"configs/{model_id[:2]}/{model_id}.json"
        
        # Generate model card
        model_card = self._generate_model_card(
            model_id=model_id,
            name=name,
            model_type=model_type,
            base_model=base_model,
            config=config,
            metrics=training_metrics,
            dataset_info=dataset_info,
            description=description,
            parent_id=parent_id,
        )
        
        with self.SessionLocal() as session:
            # Check if model already exists
            existing = session.query(ModelVersion).filter_by(id=model_id).first()
            if existing:
                return model_id  # Model already registered
            
            # Verify parent exists if specified
            if parent_id:
                parent = session.query(ModelVersion).filter_by(id=parent_id).first()
                if not parent:
                    raise ValueError(f"Parent model {parent_id} not found")
            
            # Create model version record
            model_version = ModelVersion(
                id=model_id,
                name=name,
                description=description,
                model_type=model_type,
                base_model=base_model,
                training_config=config,
                training_metrics=training_metrics,
                dataset_info=dataset_info,
                parent_id=parent_id,
                storage_path=storage_path,
                config_path=config_path,
                model_card=model_card,
                tags=tags or [],
                created_by=created_by,
                version_number=version_number,
                commit_message=commit_message,
                is_public=is_public,
            )
            
            session.add(model_version)
            session.commit()
            
            return model_id
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve model metadata by ID."""
        with self.SessionLocal() as session:
            model = session.query(ModelVersion).filter_by(id=model_id).first()
            if not model:
                return None
            
            return {
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "model_type": model.model_type,
                "base_model": model.base_model,
                "training_config": model.training_config,
                "training_metrics": model.training_metrics,
                "dataset_info": model.dataset_info,
                "parent_id": model.parent_id,
                "storage_path": model.storage_path,
                "config_path": model.config_path,
                "model_card": model.model_card,
                "tags": model.tags,
                "created_by": model.created_by,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "updated_at": model.updated_at.isoformat() if model.updated_at else None,
                "version_number": model.version_number,
                "commit_message": model.commit_message,
                "is_public": model.is_public,
            }
    
    def get_lineage(self, model_id: str, max_depth: int = 100) -> List[Dict[str, Any]]:
        """
        Get full lineage tree for a model (ancestors and descendants).
        
        Args:
            model_id: Model version ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of model metadata dictionaries in lineage order
        """
        lineage = []
        
        with self.SessionLocal() as session:
            # Get ancestors (parents)
            current_id = model_id
            visited = set()
            
            while current_id and len(lineage) < max_depth:
                if current_id in visited:
                    break
                visited.add(current_id)
                
                model = session.query(ModelVersion).filter_by(id=current_id).first()
                if not model:
                    break
                
                lineage.append({
                    "id": model.id,
                    "name": model.name,
                    "model_type": model.model_type,
                    "parent_id": model.parent_id,
                    "created_at": model.created_at.isoformat() if model.created_at else None,
                    "version_number": model.version_number,
                    "commit_message": model.commit_message,
                    "direction": "ancestor",
                })
                
                current_id = model.parent_id
            
            # Reverse to get chronological order (oldest first)
            lineage.reverse()
            
            # Get descendants (children) - using recursive CTE for efficiency
            descendants = self._get_descendants(session, model_id, max_depth - len(lineage))
            lineage.extend(descendants)
        
        return lineage
    
    def _get_descendants(
        self, session: Session, model_id: str, max_depth: int
    ) -> List[Dict[str, Any]]:
        """Recursively get all descendants of a model."""
        descendants = []
        
        def _collect_descendants(parent_id: str, depth: int):
            if depth >= max_depth:
                return
            
            children = session.query(ModelVersion).filter_by(parent_id=parent_id).all()
            for child in children:
                descendants.append({
                    "id": child.id,
                    "name": child.name,
                    "model_type": child.model_type,
                    "parent_id": child.parent_id,
                    "created_at": child.created_at.isoformat() if child.created_at else None,
                    "version_number": child.version_number,
                    "commit_message": child.commit_message,
                    "direction": "descendant",
                })
                _collect_descendants(child.id, depth + 1)
        
        _collect_descendants(model_id, 0)
        return descendants
    
    def diff_models(
        self, model_a_id: str, model_b_id: str, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate diff between two model versions.
        
        Args:
            model_a_id: First model version ID
            model_b_id: Second model version ID
            use_cache: Whether to use cached diff if available
            
        Returns:
            Dictionary containing diff information
        """
        # Generate deterministic diff ID
        diff_components = sorted([model_a_id, model_b_id])
        diff_id = hashlib.sha256("".join(diff_components).encode()).hexdigest()
        
        with self.SessionLocal() as session:
            # Check cache first
            if use_cache:
                cached_diff = session.query(ModelDiff).filter_by(id=diff_id).first()
                if cached_diff:
                    return {
                        "id": cached_diff.id,
                        "model_a_id": cached_diff.version_a_id,
                        "model_b_id": cached_diff.version_b_id,
                        "config_diff": cached_diff.config_diff,
                        "metrics_diff": cached_diff.metrics_diff,
                        "dataset_diff": cached_diff.dataset_diff,
                        "summary": cached_diff.summary,
                        "cached": True,
                    }
            
            # Retrieve both models
            model_a = session.query(ModelVersion).filter_by(id=model_a_id).first()
            model_b = session.query(ModelVersion).filter_by(id=model_b_id).first()
            
            if not model_a or not model_b:
                raise ValueError("One or both models not found")
            
            # Compute diffs
            config_diff = self._diff_configs(model_a.training_config, model_b.training_config)
            metrics_diff = self._diff_metrics(model_a.training_metrics, model_b.training_metrics)
            dataset_diff = self._diff_datasets(model_a.dataset_info, model_b.dataset_info)
            
            # Generate summary
            summary = self._generate_diff_summary(
                model_a, model_b, config_diff, metrics_diff, dataset_diff
            )
            
            # Cache the diff
            model_diff = ModelDiff(
                id=diff_id,
                version_a_id=model_a_id,
                version_b_id=model_b_id,
                config_diff=config_diff,
                metrics_diff=metrics_diff,
                dataset_diff=dataset_diff,
                summary=summary,
            )
            
            session.merge(model_diff)  # Use merge to handle duplicates
            session.commit()
            
            return {
                "id": diff_id,
                "model_a_id": model_a_id,
                "model_b_id": model_b_id,
                "config_diff": config_diff,
                "metrics_diff": metrics_diff,
                "dataset_diff": dataset_diff,
                "summary": summary,
                "cached": False,
            }
    
    def _diff_configs(self, config_a: Dict, config_b: Dict) -> Dict[str, Any]:
        """Compare two training configurations."""
        diff = {}
        all_keys = set(config_a.keys()) | set(config_b.keys())
        
        for key in all_keys:
            val_a = config_a.get(key)
            val_b = config_b.get(key)
            
            if val_a != val_b:
                diff[key] = {
                    "model_a": val_a,
                    "model_b": val_b,
                    "change_type": self._get_change_type(val_a, val_b),
                }
        
        return diff
    
    def _diff_metrics(self, metrics_a: Optional[Dict], metrics_b: Optional[Dict]) -> Dict[str, Any]:
        """Compare two sets of training metrics."""
        if not metrics_a or not metrics_b:
            return {"error": "Metrics not available for one or both models"}
        
        diff = {}
        all_keys = set(metrics_a.keys()) | set(metrics_b.keys())
        
        for key in all_keys:
            val_a = metrics_a.get(key)
            val_b = metrics_b.get(key)
            
            if val_a != val_b:
                # Calculate percentage change if both are numbers
                change_pct = None
                if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)) and val_a != 0:
                    change_pct = ((val_b - val_a) / abs(val_a)) * 100
                
                diff[key] = {
                    "model_a": val_a,
                    "model_b": val_b,
                    "change_pct": change_pct,
                    "change_type": self._get_change_type(val_a, val_b),
                }
        
        return diff
    
    def _diff_datasets(self, dataset_a: Optional[Dict], dataset_b: Optional[Dict]) -> Dict[str, Any]:
        """Compare two dataset information dictionaries."""
        if not dataset_a or not dataset_b:
            return {"error": "Dataset info not available for one or both models"}
        
        diff = {}
        all_keys = set(dataset_a.keys()) | set(dataset_b.keys())
        
        for key in all_keys:
            val_a = dataset_a.get(key)
            val_b = dataset_b.get(key)
            
            if val_a != val_b:
                diff[key] = {
                    "model_a": val_a,
                    "model_b": val_b,
                    "change_type": self._get_change_type(val_a, val_b),
                }
        
        return diff
    
    def _get_change_type(self, val_a: Any, val_b: Any) -> str:
        """Determine the type of change between two values."""
        if val_a is None:
            return "added"
        elif val_b is None:
            return "removed"
        else:
            return "modified"
    
    def _generate_diff_summary(
        self,
        model_a: ModelVersion,
        model_b: ModelVersion,
        config_diff: Dict,
        metrics_diff: Dict,
        dataset_diff: Dict,
    ) -> str:
        """Generate a human-readable summary of model differences."""
        summary_parts = []
        
        # Model identification
        summary_parts.append(f"Comparing {model_a.name} (ID: {model_a.id[:8]}) → {model_b.name} (ID: {model_b.id[:8]})")
        
        # Configuration changes
        if config_diff:
            changed_keys = list(config_diff.keys())[:5]  # Show first 5 changes
            summary_parts.append(f"Configuration changes: {', '.join(changed_keys)}")
        
        # Metrics changes
        if metrics_diff and "error" not in metrics_diff:
            improved_metrics = []
            degraded_metrics = []
            
            for metric, change_info in metrics_diff.items():
                if change_info.get("change_pct") is not None:
                    if change_info["change_pct"] > 0:
                        improved_metrics.append(metric)
                    else:
                        degraded_metrics.append(metric)
            
            if improved_metrics:
                summary_parts.append(f"Improved metrics: {', '.join(improved_metrics[:3])}")
            if degraded_metrics:
                summary_parts.append(f"Degraded metrics: {', '.join(degraded_metrics[:3])}")
        
        # Relationship
        if model_b.parent_id == model_a.id:
            summary_parts.append("Relationship: Direct parent-child")
        elif model_a.parent_id == model_b.id:
            summary_parts.append("Relationship: Direct child-parent")
        else:
            # Check for common ancestor
            lineage_a = self.get_lineage(model_a.id, max_depth=10)
            lineage_b = self.get_lineage(model_b.id, max_depth=10)
            common_ancestor = next(
                (m for m in lineage_a if m["id"] in {m2["id"] for m2 in lineage_b}),
                None
            )
            if common_ancestor:
                summary_parts.append(f"Relationship: Share common ancestor {common_ancestor['id'][:8]}")
        
        return "\n".join(summary_parts)
    
    def _generate_model_card(
        self,
        model_id: str,
        name: str,
        model_type: str,
        base_model: Optional[str],
        config: Dict[str, Any],
        metrics: Optional[Dict[str, Any]],
        dataset_info: Optional[Dict[str, Any]],
        description: Optional[str],
        parent_id: Optional[str],
    ) -> str:
        """Generate automatic model card in markdown format."""
        lines = [
            f"# Model Card: {name}",
            "",
            "## Model Details",
            "",
            f"- **Model ID:** `{model_id}`",
            f"- **Model Type:** {model_type}",
            f"- **Base Model:** {base_model or 'N/A'}",
            f"- **Parent Version:** `{parent_id[:12] if parent_id else 'None'}`",
            "",
            "## Description",
            "",
            description or "No description provided.",
            "",
            "## Training Configuration",
            "",
            "```json",
            json.dumps(config, indent=2),
            "```",
            "",
        ]
        
        if metrics:
            lines.extend([
                "## Training Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ])
            for key, value in metrics.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")
        
        if dataset_info:
            lines.extend([
                "## Dataset Information",
                "",
                "```json",
                json.dumps(dataset_info, indent=2),
                "```",
                "",
            ])
        
        lines.extend([
            "## Reproducibility",
            "",
            "To reproduce this model:",
            "",
            f"1. Load base model: `{base_model or 'N/A'}`",
            "2. Apply training configuration above",
            "3. Use the same dataset or dataset configuration",
            "",
            "---",
            f"*Generated automatically by SOVEREIGN Model Registry on {datetime.utcnow().isoformat()}*",
        ])
        
        return "\n".join(lines)
    
    def search_models(
        self,
        name: Optional[str] = None,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        created_by: Optional[str] = None,
        is_public: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search for models with various filters.
        
        Args:
            name: Filter by model name (partial match)
            model_type: Filter by model type
            tags: Filter by tags (models containing any of the specified tags)
            created_by: Filter by creator
            is_public: Filter by visibility
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of model metadata dictionaries
        """
        with self.SessionLocal() as session:
            query = session.query(ModelVersion)
            
            if name:
                query = query.filter(ModelVersion.name.ilike(f"%{name}%"))
            
            if model_type:
                query = query.filter(ModelVersion.model_type == model_type)
            
            if tags:
                # PostgreSQL JSONB array contains any of the specified tags
                for tag in tags:
                    query = query.filter(ModelVersion.tags.contains([tag]))
            
            if created_by:
                query = query.filter(ModelVersion.created_by == created_by)
            
            if is_public is not None:
                query = query.filter(ModelVersion.is_public == is_public)
            
            query = query.order_by(ModelVersion.created_at.desc())
            query = query.limit(limit).offset(offset)
            
            models = query.all()
            
            return [
                {
                    "id": model.id,
                    "name": model.name,
                    "model_type": model.model_type,
                    "base_model": model.base_model,
                    "tags": model.tags,
                    "created_by": model.created_by,
                    "created_at": model.created_at.isoformat() if model.created_at else None,
                    "version_number": model.version_number,
                    "is_public": model.is_public,
                }
                for model in models
            ]
    
    def get_model_weights(self, model_id: str) -> Optional[bytes]:
        """Retrieve model weights by model ID."""
        with self.SessionLocal() as session:
            model = session.query(ModelVersion).filter_by(id=model_id).first()
            if not model:
                return None
            
            # Extract weights hash from storage path
            weights_hash = model.storage_path.split("/")[-1]
            return self.storage.retrieve(weights_hash)
    
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """
        Delete a model version from the registry.
        
        Args:
            model_id: Model version ID to delete
            force: If True, delete even if model has children
            
        Returns:
            True if deleted successfully
        """
        with self.SessionLocal() as session:
            model = session.query(ModelVersion).filter_by(id=model_id).first()
            if not model:
                return False
            
            # Check for children
            children = session.query(ModelVersion).filter_by(parent_id=model_id).first()
            if children and not force:
                raise ValueError(
                    f"Model {model_id} has child versions. Use force=True to delete anyway."
                )
            
            # Update children to point to grandparent
            if children and force:
                session.query(ModelVersion).filter_by(parent_id=model_id).update(
                    {"parent_id": model.parent_id}
                )
            
            # Delete associated diffs
            session.query(ModelDiff).filter(
                (ModelDiff.version_a_id == model_id) | 
                (ModelDiff.version_b_id == model_id)
            ).delete()
            
            # Delete model record
            session.delete(model)
            session.commit()
            
            # Note: We don't delete weights from storage as they might be shared
            # Garbage collection should be implemented separately
            
            return True


# Integration with existing SOVEREIGN components
def get_registry(database_url: Optional[str] = None, storage_path: Optional[str] = None) -> ModelRegistry:
    """
    Factory function to get a ModelRegistry instance.
    
    Uses environment variables or default paths if not provided.
    """
    if not database_url:
        database_url = os.getenv(
            "SOVEREIGN_DATABASE_URL",
            "postgresql://sovereign:sovereign@localhost:5432/sovereign"
        )
    
    if not storage_path:
        storage_path = os.getenv(
            "SOVEREIGN_MODEL_STORAGE",
            str(Path.home() / ".sovereign" / "model_storage")
        )
    
    return ModelRegistry(database_url, storage_path)


# CLI integration (to be called from studio/backend/cli.py)
def register_model_cli(args):
    """CLI handler for model registration."""
    registry = get_registry()
    
    # Load model weights and config from files
    with open(args.weights, "rb") as f:
        weights = f.read()
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    model_id = registry.register_model(
        name=args.name,
        weights=weights,
        config=config,
        model_type=args.model_type,
        base_model=args.base_model,
        parent_id=args.parent,
        description=args.description,
        tags=args.tags.split(",") if args.tags else None,
        created_by=args.user,
        version_number=args.version,
        commit_message=args.message,
        is_public=args.public,
    )
    
    print(f"Model registered successfully with ID: {model_id}")


# Example usage
if __name__ == "__main__":
    # Example: Register a model
    registry = get_registry()
    
    # This would typically be called from a training script
    example_config = {
        "learning_rate": 2e-5,
        "batch_size": 4,
        "epochs": 3,
        "optimizer": "adamw",
    }
    
    # Load model weights (example)
    # with open("model.bin", "rb") as f:
    #     weights = f.read()
    
    # model_id = registry.register_model(
    #     name="my-fine-tuned-model",
    #     weights=weights,
    #     config=example_config,
    #     model_type="llama",
    #     base_model="meta-llama/Llama-2-7b-hf",
    #     description="Fine-tuned on custom dataset",
    #     tags=["fine-tuned", "experimental"],
    # )
    
    # print(f"Registered model: {model_id}")