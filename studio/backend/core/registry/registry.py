# studio/backend/core/registry/registry.py

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine, Column, String, DateTime, ForeignKey, Text, Integer, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.pool import QueuePool

from studio.backend.auth.storage import get_database_url
from studio.backend.core.data_recipe.jobs.types import JobStatus

Base = declarative_base()


class ModelVersion(Base):
    """SQLAlchemy model for storing model version metadata."""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(255), nullable=False, index=True)
    version_hash = Column(String(64), unique=True, nullable=False, index=True)
    parent_hash = Column(String(64), ForeignKey('model_versions.version_hash'), nullable=True)
    description = Column(Text, nullable=True)
    metadata_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    is_latest = Column(Boolean, default=False)
    tags = Column(JSON, default=list)
    
    # Relationships
    parent = relationship("ModelVersion", remote_side=[version_hash])
    children = relationship("ModelVersion", back_populates="parent")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model version to dictionary representation."""
        return {
            'id': self.id,
            'model_name': self.model_name,
            'version_hash': self.version_hash,
            'parent_hash': self.parent_hash,
            'description': self.description,
            'metadata': self.metadata_json,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'created_by': self.created_by,
            'is_latest': self.is_latest,
            'tags': self.tags or []
        }


class ModelRegistry:
    """
    Git-like version control system for ML models.
    
    Features:
    - SHA-256 based versioning with parent pointers for lineage tracking
    - Content-addressable storage for model weights
    - Diff visualization between fine-tunes
    - Automatic model card generation
    - Reproducible ML workflows
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            storage_path: Path for content-addressable storage. 
                         Defaults to ~/.vex/model_registry
        """
        self.storage_path = Path(storage_path or os.path.expanduser("~/.vex/model_registry"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection
        self.engine = create_engine(
            get_database_url(),
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_recycle=3600
        )
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
    
    def _compute_hash(self, content: Union[str, bytes, Dict]) -> str:
        """Compute SHA-256 hash of content."""
        if isinstance(content, dict):
            content = json.dumps(content, sort_keys=True)
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
    
    def _store_weights(self, weights_path: str, content_hash: str) -> str:
        """
        Store model weights in content-addressable storage.
        
        Args:
            weights_path: Path to weights file or directory
            content_hash: SHA-256 hash for content-addressable storage
            
        Returns:
            Relative path to stored weights
        """
        target_dir = self.storage_path / "weights" / content_hash[:2]
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / content_hash
        
        if os.path.isdir(weights_path):
            # Store directory as tar archive
            import tarfile
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
                with tarfile.open(tmp.name, 'w:gz') as tar:
                    tar.add(weights_path, arcname=os.path.basename(weights_path))
                shutil.move(tmp.name, target_path)
        else:
            # Store single file
            shutil.copy2(weights_path, target_path)
        
        return str(target_path.relative_to(self.storage_path))
    
    def _load_weights(self, relative_path: str) -> Path:
        """Load weights from content-addressable storage."""
        return self.storage_path / relative_path
    
    def register_model(
        self,
        model_name: str,
        weights_path: str,
        metadata: Dict[str, Any],
        parent_hash: Optional[str] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Register a new model version in the registry.
        
        Args:
            model_name: Name of the model (e.g., "llama-2-7b")
            weights_path: Path to model weights file or directory
            metadata: Model metadata (hyperparameters, training config, etc.)
            parent_hash: Hash of parent model version for lineage tracking
            description: Optional description of this version
            created_by: User who created this version
            tags: Optional tags for this version
            
        Returns:
            Dictionary with model version information
        """
        # Validate parent hash exists if provided
        if parent_hash:
            with self.Session() as session:
                parent = session.query(ModelVersion).filter_by(version_hash=parent_hash).first()
                if not parent:
                    raise ValueError(f"Parent model with hash {parent_hash} not found")
        
        # Compute content hash from metadata and weights
        metadata_hash = self._compute_hash(metadata)
        
        # Compute weights hash
        if os.path.isdir(weights_path):
            # Hash directory contents
            import hashlib
            hasher = hashlib.sha256()
            for root, dirs, files in os.walk(weights_path):
                for file in sorted(files):
                    filepath = os.path.join(root, file)
                    with open(filepath, 'rb') as f:
                        hasher.update(f.read())
            weights_hash = hasher.hexdigest()
        else:
            with open(weights_path, 'rb') as f:
                weights_hash = self._compute_hash(f.read())
        
        # Combined hash for version
        combined_content = f"{model_name}:{metadata_hash}:{weights_hash}:{parent_hash or ''}"
        version_hash = self._compute_hash(combined_content)
        
        # Store weights in content-addressable storage
        weights_storage_path = self._store_weights(weights_path, version_hash)
        
        # Prepare metadata for storage
        full_metadata = {
            **metadata,
            'weights_hash': weights_hash,
            'weights_storage_path': weights_storage_path,
            'model_name': model_name
        }
        
        # Store in database
        with self.Session() as session:
            # Mark previous latest as not latest
            if parent_hash:
                session.query(ModelVersion).filter_by(
                    model_name=model_name,
                    is_latest=True
                ).update({'is_latest': False})
            
            model_version = ModelVersion(
                model_name=model_name,
                version_hash=version_hash,
                parent_hash=parent_hash,
                description=description,
                metadata_json=full_metadata,
                created_by=created_by,
                is_latest=True,
                tags=tags or []
            )
            
            session.add(model_version)
            session.commit()
            
            return model_version.to_dict()
    
    def get_model(self, version_hash: str) -> Optional[Dict[str, Any]]:
        """Get model version by hash."""
        with self.Session() as session:
            model = session.query(ModelVersion).filter_by(version_hash=version_hash).first()
            return model.to_dict() if model else None
    
    def get_latest_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get latest version of a model by name."""
        with self.Session() as session:
            model = session.query(ModelVersion).filter_by(
                model_name=model_name,
                is_latest=True
            ).first()
            return model.to_dict() if model else None
    
    def list_models(
        self,
        model_name: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List model versions with optional filtering.
        
        Args:
            model_name: Filter by model name
            tag: Filter by tag
            limit: Maximum number of results
            offset: Pagination offset
            
        Returns:
            List of model version dictionaries
        """
        with self.Session() as session:
            query = session.query(ModelVersion)
            
            if model_name:
                query = query.filter_by(model_name=model_name)
            
            if tag:
                query = query.filter(ModelVersion.tags.contains([tag]))
            
            query = query.order_by(ModelVersion.created_at.desc())
            query = query.limit(limit).offset(offset)
            
            return [model.to_dict() for model in query.all()]
    
    def get_lineage(self, version_hash: str) -> List[Dict[str, Any]]:
        """
        Get full lineage of a model version (ancestors and descendants).
        
        Args:
            version_hash: Hash of the model version
            
        Returns:
            List of model versions in lineage order (oldest to newest)
        """
        with self.Session() as session:
            # Get all ancestors
            ancestors = []
            current_hash = version_hash
            
            while current_hash:
                model = session.query(ModelVersion).filter_by(version_hash=current_hash).first()
                if not model:
                    break
                ancestors.append(model.to_dict())
                current_hash = model.parent_hash
            
            ancestors.reverse()  # Oldest first
            
            # Get all descendants
            descendants = []
            
            def get_children(hash: str):
                children = session.query(ModelVersion).filter_by(parent_hash=hash).all()
                for child in children:
                    descendants.append(child.to_dict())
                    get_children(child.version_hash)
            
            get_children(version_hash)
            
            return ancestors + descendants
    
    def diff_models(
        self,
        hash1: str,
        hash2: str,
        include_weights: bool = False
    ) -> Dict[str, Any]:
        """
        Generate diff between two model versions.
        
        Args:
            hash1: First model version hash
            hash2: Second model version hash
            include_weights: Whether to include weight differences
            
        Returns:
            Dictionary with diff information
        """
        model1 = self.get_model(hash1)
        model2 = self.get_model(hash2)
        
        if not model1 or not model2:
            raise ValueError("One or both model versions not found")
        
        diff = {
            'model1': model1,
            'model2': model2,
            'metadata_diff': self._diff_metadata(model1['metadata'], model2['metadata']),
            'lineage_relation': self._get_lineage_relation(hash1, hash2)
        }
        
        if include_weights:
            diff['weights_diff'] = self._diff_weights(
                model1['metadata']['weights_storage_path'],
                model2['metadata']['weights_storage_path']
            )
        
        return diff
    
    def _diff_metadata(self, meta1: Dict, meta2: Dict) -> Dict[str, Any]:
        """Compare metadata between two model versions."""
        diff = {
            'added': {},
            'removed': {},
            'changed': {},
            'unchanged': {}
        }
        
        all_keys = set(meta1.keys()) | set(meta2.keys())
        
        for key in all_keys:
            if key not in meta1:
                diff['added'][key] = meta2[key]
            elif key not in meta2:
                diff['removed'][key] = meta1[key]
            elif meta1[key] != meta2[key]:
                diff['changed'][key] = {
                    'old': meta1[key],
                    'new': meta2[key]
                }
            else:
                diff['unchanged'][key] = meta1[key]
        
        return diff
    
    def _diff_weights(self, path1: str, path2: str) -> Dict[str, Any]:
        """
        Compare weights between two model versions.
        
        Note: This is a simplified implementation. In production, you might want
        to use more sophisticated weight comparison techniques.
        """
        full_path1 = self._load_weights(path1)
        full_path2 = self._load_weights(path2)
        
        if not full_path1.exists() or not full_path2.exists():
            return {'error': 'Weights files not found'}
        
        # Simple size comparison
        size1 = full_path1.stat().st_size if full_path1.is_file() else sum(
            f.stat().st_size for f in full_path1.rglob('*') if f.is_file()
        )
        size2 = full_path2.stat().st_size if full_path2.is_file() else sum(
            f.stat().st_size for f in full_path2.rglob('*') if f.is_file()
        )
        
        return {
            'size_diff_bytes': size2 - size1,
            'size_diff_percent': ((size2 - size1) / size1 * 100) if size1 > 0 else 0,
            'identical': size1 == size2
        }
    
    def _get_lineage_relation(self, hash1: str, hash2: str) -> str:
        """Determine lineage relationship between two models."""
        lineage1 = {v['version_hash'] for v in self.get_lineage(hash1)}
        lineage2 = {v['version_hash'] for v in self.get_lineage(hash2)}
        
        if hash2 in lineage1:
            return 'ancestor'
        elif hash1 in lineage2:
            return 'descendant'
        elif lineage1 & lineage2:
            return 'cousin'
        else:
            return 'unrelated'
    
    def generate_model_card(self, version_hash: str) -> str:
        """
        Generate a model card in Markdown format.
        
        Args:
            version_hash: Hash of the model version
            
        Returns:
            Model card as Markdown string
        """
        model = self.get_model(version_hash)
        if not model:
            raise ValueError(f"Model with hash {version_hash} not found")
        
        metadata = model['metadata']
        
        # Generate model card
        card = f"""# Model Card: {model['model_name']}

## Version Information
- **Version Hash**: `{model['version_hash']}`
- **Parent Hash**: `{model['parent_hash'] or 'None (initial version)'}`
- **Created**: {model['created_at']}
- **Created By**: {model['created_by'] or 'Unknown'}
- **Description**: {model['description'] or 'No description provided'}

## Model Details
- **Model Name**: {model['model_name']}
- **Tags**: {', '.join(model['tags']) if model['tags'] else 'None'}

## Training Configuration
"""
        
        # Add training parameters
        training_params = {k: v for k, v in metadata.items() 
                         if k.startswith('training_') or k in ['learning_rate', 'epochs', 'batch_size']}
        
        if training_params:
            card += "\n### Training Parameters\n"
            for key, value in training_params.items():
                card += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        # Add model architecture
        if 'model_architecture' in metadata:
            card += f"\n### Model Architecture\n```\n{metadata['model_architecture']}\n```\n"
        
        # Add dataset information
        if 'dataset' in metadata:
            card += f"\n### Dataset\n- **Name**: {metadata['dataset']}\n"
            if 'dataset_size' in metadata:
                card += f"- **Size**: {metadata['dataset_size']} samples\n"
        
        # Add performance metrics
        metrics = {k: v for k, v in metadata.items() 
                  if k.startswith('metric_') or k in ['accuracy', 'loss', 'f1_score']}
        
        if metrics:
            card += "\n### Performance Metrics\n"
            for key, value in metrics.items():
                card += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        # Add usage example
        card += """
## Usage

```python
from vex import load_model

# Load this specific version
model = load_model("{model_name}", version="{version_hash}")

# Or load the latest version
model = load_model("{model_name}")
```

## Lineage

```mermaid
graph TD
    {lineage_mermaid}
```
""".format(
            model_name=model['model_name'],
            version_hash=model['version_hash'],
            lineage_mermaid=self._generate_lineage_mermaid(version_hash)
        )
        
        return card
    
    def _generate_lineage_mermaid(self, version_hash: str) -> str:
        """Generate Mermaid diagram for model lineage."""
        lineage = self.get_lineage(version_hash)
        
        if not lineage:
            return "A[No lineage information]"
        
        lines = []
        for i, model in enumerate(lineage):
            node_id = f"M{i}"
            label = f"{model['model_name']}\\n{model['version_hash'][:8]}"
            lines.append(f'    {node_id}["{label}"]')
            
            if i > 0:
                prev_id = f"M{i-1}"
                lines.append(f"    {prev_id} --> {node_id}")
        
        return "\n".join(lines)
    
    def delete_model(self, version_hash: str, force: bool = False) -> bool:
        """
        Delete a model version from the registry.
        
        Args:
            version_hash: Hash of the model version to delete
            force: If True, delete even if it has children
            
        Returns:
            True if successful, False otherwise
        """
        with self.Session() as session:
            model = session.query(ModelVersion).filter_by(version_hash=version_hash).first()
            if not model:
                return False
            
            # Check for children
            children = session.query(ModelVersion).filter_by(parent_hash=version_hash).all()
            if children and not force:
                raise ValueError(
                    f"Cannot delete model with {len(children)} children. "
                    "Use force=True to delete anyway."
                )
            
            # Update children to point to parent
            if children and force:
                for child in children:
                    child.parent_hash = model.parent_hash
            
            # Delete weights from storage
            weights_path = self._load_weights(model.metadata_json.get('weights_storage_path', ''))
            if weights_path.exists():
                if weights_path.is_file():
                    weights_path.unlink()
                else:
                    shutil.rmtree(weights_path)
            
            # Delete from database
            session.delete(model)
            session.commit()
            
            return True
    
    def export_model(self, version_hash: str, export_path: str) -> str:
        """
        Export a model version to a directory.
        
        Args:
            version_hash: Hash of the model version to export
            export_path: Path to export directory
            
        Returns:
            Path to exported model directory
        """
        model = self.get_model(version_hash)
        if not model:
            raise ValueError(f"Model with hash {version_hash} not found")
        
        export_dir = Path(export_path) / f"{model['model_name']}_{version_hash[:8]}"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export metadata
        metadata_path = export_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model, f, indent=2)
        
        # Export model card
        card_path = export_dir / "MODEL_CARD.md"
        with open(card_path, 'w') as f:
            f.write(self.generate_model_card(version_hash))
        
        # Copy weights
        weights_path = self._load_weights(model['metadata']['weights_storage_path'])
        if weights_path.exists():
            if weights_path.is_file():
                shutil.copy2(weights_path, export_dir / "weights")
            else:
                shutil.copytree(weights_path, export_dir / "weights")
        
        return str(export_dir)
    
    def import_model(
        self,
        import_path: str,
        model_name: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Import a model from an exported directory.
        
        Args:
            import_path: Path to exported model directory
            model_name: Override model name
            created_by: User importing the model
            
        Returns:
            Imported model version information
        """
        import_dir = Path(import_path)
        
        # Load metadata
        metadata_path = import_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError("Invalid export directory: metadata.json not found")
        
        with open(metadata_path, 'r') as f:
            exported_model = json.load(f)
        
        # Use provided model name or original
        final_model_name = model_name or exported_model['model_name']
        
        # Find weights
        weights_path = import_dir / "weights"
        if not weights_path.exists():
            raise ValueError("Invalid export directory: weights not found")
        
        # Register the imported model
        return self.register_model(
            model_name=final_model_name,
            weights_path=str(weights_path),
            metadata=exported_model['metadata'],
            parent_hash=exported_model.get('parent_hash'),
            description=f"Imported from {import_path}",
            created_by=created_by,
            tags=exported_model.get('tags', []) + ['imported']
        )


# Singleton instance for easy access
_registry_instance = None


def get_model_registry(storage_path: Optional[str] = None) -> ModelRegistry:
    """
    Get or create the model registry singleton instance.
    
    Args:
        storage_path: Optional custom storage path
        
    Returns:
        ModelRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(storage_path)
    return _registry_instance


# Convenience functions for common operations
def register_model(*args, **kwargs) -> Dict[str, Any]:
    """Convenience function for registering a model."""
    return get_model_registry().register_model(*args, **kwargs)


def get_model(version_hash: str) -> Optional[Dict[str, Any]]:
    """Convenience function for getting a model by hash."""
    return get_model_registry().get_model(version_hash)


def get_latest_model(model_name: str) -> Optional[Dict[str, Any]]:
    """Convenience function for getting the latest version of a model."""
    return get_model_registry().get_latest_model(model_name)


def list_models(**kwargs) -> List[Dict[str, Any]]:
    """Convenience function for listing models."""
    return get_model_registry().list_models(**kwargs)


def generate_model_card(version_hash: str) -> str:
    """Convenience function for generating a model card."""
    return get_model_registry().generate_model_card(version_hash)


def diff_models(hash1: str, hash2: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for diffing two model versions."""
    return get_model_registry().diff_models(hash1, hash2, **kwargs)