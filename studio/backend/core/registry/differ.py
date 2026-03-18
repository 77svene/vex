"""
Model Registry & Version Control — Git-like versioning for models with lineage tracking,
diff visualization between fine-tunes, and automatic model card generation.
Enables reproducible ML workflows.
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import psycopg2
from psycopg2.extras import Json, DictCursor
from psycopg2 import sql

from studio.backend.auth.storage import get_db_connection
from studio.backend.core.data_recipe.jobs.types import JobStatus


class ModelRegistryError(Exception):
    """Base exception for model registry operations."""
    pass


class ModelVersion:
    """Represents a specific version of a model in the registry."""
    
    def __init__(
        self,
        model_id: str,
        version_hash: str,
        parent_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_card: Optional[str] = None,
        created_at: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        status: JobStatus = JobStatus.COMPLETED
    ):
        self.model_id = model_id
        self.version_hash = version_hash
        self.parent_hash = parent_hash
        self.metadata = metadata or {}
        self.model_card = model_card or ""
        self.created_at = created_at or datetime.utcnow()
        self.tags = tags or []
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model version to dictionary representation."""
        return {
            "model_id": self.model_id,
            "version_hash": self.version_hash,
            "parent_hash": self.parent_hash,
            "metadata": self.metadata,
            "model_card": self.model_card,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "status": self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create ModelVersion from dictionary representation."""
        return cls(
            model_id=data["model_id"],
            version_hash=data["version_hash"],
            parent_hash=data.get("parent_hash"),
            metadata=data.get("metadata", {}),
            model_card=data.get("model_card", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            tags=data.get("tags", []),
            status=JobStatus(data.get("status", JobStatus.COMPLETED.value))
        )


class ContentAddressableStorage:
    """Content-addressable storage for model weights using SHA-256 hashing."""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_content_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of file contents."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def store(self, source_path: Union[str, Path], model_id: str) -> Tuple[str, Path]:
        """
        Store model weights in content-addressable storage.
        
        Returns:
            Tuple of (content_hash, storage_path)
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        content_hash = self._get_content_hash(source_path)
        storage_dir = self.base_path / model_id / content_hash[:2]
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / content_hash
        
        if not storage_path.exists():
            shutil.copy2(source_path, storage_path)
        
        return content_hash, storage_path
    
    def retrieve(self, model_id: str, content_hash: str) -> Path:
        """Retrieve model weights by content hash."""
        storage_dir = self.base_path / model_id / content_hash[:2]
        storage_path = storage_dir / content_hash
        
        if not storage_path.exists():
            raise FileNotFoundError(
                f"Model weights not found for hash {content_hash} in model {model_id}"
            )
        
        return storage_path
    
    def exists(self, model_id: str, content_hash: str) -> bool:
        """Check if model weights exist in storage."""
        storage_dir = self.base_path / model_id / content_hash[:2]
        storage_path = storage_dir / content_hash
        return storage_path.exists()


class ModelRegistry:
    """
    Git-like version control system for ML models with lineage tracking.
    
    Features:
    - SHA-256 based versioning with parent pointers
    - Content-addressable storage for model weights
    - Automatic model card generation
    - Diff visualization between fine-tunes
    - PostgreSQL backend for metadata
    """
    
    def __init__(self, storage_path: Union[str, Path] = "./model_storage"):
        """
        Initialize model registry.
        
        Args:
            storage_path: Base path for content-addressable storage
        """
        self.storage = ContentAddressableStorage(storage_path)
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database tables for model registry."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Create model_versions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_versions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        model_id VARCHAR(255) NOT NULL,
                        version_hash VARCHAR(64) NOT NULL,
                        parent_hash VARCHAR(64),
                        metadata JSONB DEFAULT '{}',
                        model_card TEXT DEFAULT '',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        tags TEXT[] DEFAULT '{}',
                        status VARCHAR(50) DEFAULT 'completed',
                        UNIQUE(model_id, version_hash)
                    )
                """)
                
                # Create indexes for efficient querying
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_versions_model_id 
                    ON model_versions(model_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_versions_parent_hash 
                    ON model_versions(parent_hash)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_versions_created_at 
                    ON model_versions(created_at)
                """)
                
                # Create model_diffs table for cached diffs
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS model_diffs (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        model_id VARCHAR(255) NOT NULL,
                        version_hash_1 VARCHAR(64) NOT NULL,
                        version_hash_2 VARCHAR(64) NOT NULL,
                        diff_data JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        UNIQUE(model_id, version_hash_1, version_hash_2)
                    )
                """)
                
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise ModelRegistryError(f"Failed to initialize database: {e}")
        finally:
            conn.close()
    
    def _calculate_version_hash(
        self,
        model_path: Union[str, Path],
        metadata: Dict[str, Any],
        parent_hash: Optional[str] = None
    ) -> str:
        """
        Calculate deterministic version hash for a model.
        
        Hash includes:
        - Model weights content hash
        - Metadata hash
        - Parent version hash (if exists)
        """
        # Get content hash of model weights
        weights_hash = self.storage._get_content_hash(model_path)
        
        # Create hash input combining weights, metadata, and lineage
        hash_input = {
            "weights_hash": weights_hash,
            "metadata": metadata,
            "parent_hash": parent_hash
        }
        
        # Sort keys for deterministic hashing
        hash_str = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()
    
    def register_model(
        self,
        model_id: str,
        model_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        model_card: Optional[str] = None,
        parent_hash: Optional[str] = None,
        tags: Optional[List[str]] = None,
        auto_generate_card: bool = True
    ) -> ModelVersion:
        """
        Register a new model version in the registry.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to model weights file/directory
            metadata: Model metadata (training params, dataset info, etc.)
            model_card: Model documentation (auto-generated if not provided)
            parent_hash: Hash of parent version (for lineage)
            tags: List of tags for the model version
            auto_generate_card: Whether to auto-generate model card if not provided
            
        Returns:
            ModelVersion object with registration details
        """
        metadata = metadata or {}
        tags = tags or []
        
        # Calculate version hash
        version_hash = self._calculate_version_hash(model_path, metadata, parent_hash)
        
        # Check if version already exists
        if self.exists(model_id, version_hash):
            raise ModelRegistryError(
                f"Model version {version_hash} already exists for model {model_id}"
            )
        
        # Store model weights in content-addressable storage
        content_hash, storage_path = self.storage.store(model_path, model_id)
        
        # Update metadata with storage info
        metadata["content_hash"] = content_hash
        metadata["storage_path"] = str(storage_path)
        metadata["file_size"] = os.path.getsize(storage_path)
        
        # Auto-generate model card if not provided
        if not model_card and auto_generate_card:
            model_card = self.generate_model_card(
                model_id=model_id,
                version_hash=version_hash,
                metadata=metadata,
                parent_hash=parent_hash
            )
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version_hash=version_hash,
            parent_hash=parent_hash,
            metadata=metadata,
            model_card=model_card or "",
            tags=tags,
            status=JobStatus.COMPLETED
        )
        
        # Store in database
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_versions 
                    (model_id, version_hash, parent_hash, metadata, model_card, tags, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        model_id,
                        version_hash,
                        parent_hash,
                        Json(metadata),
                        model_card,
                        tags,
                        JobStatus.COMPLETED.value
                    )
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise ModelRegistryError(f"Failed to register model: {e}")
        finally:
            conn.close()
        
        return model_version
    
    def get_model_version(
        self,
        model_id: str,
        version_hash: str
    ) -> Optional[ModelVersion]:
        """Retrieve a specific model version."""
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM model_versions 
                    WHERE model_id = %s AND version_hash = %s
                    """,
                    (model_id, version_hash)
                )
                row = cur.fetchone()
                
                if row:
                    return ModelVersion(
                        model_id=row["model_id"],
                        version_hash=row["version_hash"],
                        parent_hash=row["parent_hash"],
                        metadata=row["metadata"],
                        model_card=row["model_card"],
                        created_at=row["created_at"],
                        tags=row["tags"],
                        status=JobStatus(row["status"])
                    )
                return None
        finally:
            conn.close()
    
    def list_model_versions(
        self,
        model_id: str,
        limit: int = 100,
        offset: int = 0,
        tag_filter: Optional[str] = None
    ) -> List[ModelVersion]:
        """List all versions of a model, optionally filtered by tag."""
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                query = """
                    SELECT * FROM model_versions 
                    WHERE model_id = %s
                """
                params = [model_id]
                
                if tag_filter:
                    query += " AND %s = ANY(tags)"
                    params.append(tag_filter)
                
                query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                return [
                    ModelVersion(
                        model_id=row["model_id"],
                        version_hash=row["version_hash"],
                        parent_hash=row["parent_hash"],
                        metadata=row["metadata"],
                        model_card=row["model_card"],
                        created_at=row["created_at"],
                        tags=row["tags"],
                        status=JobStatus(row["status"])
                    )
                    for row in rows
                ]
        finally:
            conn.close()
    
    def get_lineage(
        self,
        model_id: str,
        version_hash: str,
        max_depth: int = 50
    ) -> List[ModelVersion]:
        """
        Get full lineage (ancestry) of a model version.
        
        Returns list from current version back to root.
        """
        lineage = []
        current_hash = version_hash
        depth = 0
        
        while current_hash and depth < max_depth:
            version = self.get_model_version(model_id, current_hash)
            if not version:
                break
            
            lineage.append(version)
            current_hash = version.parent_hash
            depth += 1
        
        return lineage
    
    def get_children(
        self,
        model_id: str,
        version_hash: str
    ) -> List[ModelVersion]:
        """Get all direct children (derivatives) of a model version."""
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM model_versions 
                    WHERE model_id = %s AND parent_hash = %s
                    ORDER BY created_at DESC
                    """,
                    (model_id, version_hash)
                )
                rows = cur.fetchall()
                
                return [
                    ModelVersion(
                        model_id=row["model_id"],
                        version_hash=row["version_hash"],
                        parent_hash=row["parent_hash"],
                        metadata=row["metadata"],
                        model_card=row["model_card"],
                        created_at=row["created_at"],
                        tags=row["tags"],
                        status=JobStatus(row["status"])
                    )
                    for row in rows
                ]
        finally:
            conn.close()
    
    def exists(self, model_id: str, version_hash: str) -> bool:
        """Check if a model version exists."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM model_versions 
                    WHERE model_id = %s AND version_hash = %s
                    """,
                    (model_id, version_hash)
                )
                return cur.fetchone() is not None
        finally:
            conn.close()
    
    def generate_model_card(
        self,
        model_id: str,
        version_hash: str,
        metadata: Dict[str, Any],
        parent_hash: Optional[str] = None
    ) -> str:
        """
        Auto-generate model card from metadata.
        
        Creates a structured markdown document with model information.
        """
        # Get parent model info if exists
        parent_info = ""
        if parent_hash:
            parent_version = self.get_model_version(model_id, parent_hash)
            if parent_version:
                parent_info = f"Fine-tuned from: `{parent_hash}`"
        
        # Extract metadata fields
        training_info = metadata.get("training", {})
        dataset_info = metadata.get("dataset", {})
        model_info = metadata.get("model", {})
        
        # Generate model card
        card = f"""# Model Card: {model_id}

## Version Information
- **Version Hash**: `{version_hash}`
- **Created**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
{parent_info}

## Model Details
- **Base Model**: {model_info.get('base_model', 'Unknown')}
- **Model Type**: {model_info.get('type', 'Unknown')}
- **Parameters**: {model_info.get('parameters', 'Unknown')}
- **Architecture**: {model_info.get('architecture', 'Unknown')}

## Training Information
- **Training Framework**: {training_info.get('framework', 'Unsloth')}
- **Training Date**: {training_info.get('date', 'Unknown')}
- **Training Duration**: {training_info.get('duration', 'Unknown')}
- **Hardware**: {training_info.get('hardware', 'Unknown')}
- **Batch Size**: {training_info.get('batch_size', 'Unknown')}
- **Learning Rate**: {training_info.get('learning_rate', 'Unknown')}
- **Epochs**: {training_info.get('epochs', 'Unknown')}

## Dataset Information
- **Dataset Name**: {dataset_info.get('name', 'Unknown')}
- **Dataset Size**: {dataset_info.get('size', 'Unknown')}
- **Dataset Split**: {dataset_info.get('split', 'Unknown')}
- **Preprocessing**: {dataset_info.get('preprocessing', 'Unknown')}

## Performance Metrics
{self._format_metrics(metadata.get('metrics', {}))}

## Usage
```python
# Load the model
from vex import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model_id}",
    revision="{version_hash}",
    load_in_4bit=True,
)

# Use the model
inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
```

## Limitations and Biases
{metadata.get('limitations', 'No limitations documented.')}

## Citation
{metadata.get('citation', 'No citation provided.')}
"""
        return card
    
    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics dictionary into markdown table."""
        if not metrics:
            return "No metrics recorded."
        
        lines = ["| Metric | Value |", "|--------|-------|"]
        for key, value in metrics.items():
            lines.append(f"| {key} | {value} |")
        
        return "\n".join(lines)
    
    def diff_models(
        self,
        model_id: str,
        version_hash_1: str,
        version_hash_2: str,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate diff between two model versions.
        
        Returns a structured diff including:
        - Metadata differences
        - Model card differences
        - Lineage differences
        - Performance metric changes
        """
        # Check cache first
        if use_cache:
            cached_diff = self._get_cached_diff(model_id, version_hash_1, version_hash_2)
            if cached_diff:
                return cached_diff
        
        # Get both versions
        version1 = self.get_model_version(model_id, version_hash_1)
        version2 = self.get_model_version(model_id, version_hash_2)
        
        if not version1 or not version2:
            raise ModelRegistryError("One or both model versions not found")
        
        # Calculate diff
        diff = {
            "model_id": model_id,
            "version_1": version_hash_1,
            "version_2": version_hash_2,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata_diff": self._diff_metadata(version1.metadata, version2.metadata),
            "model_card_diff": self._diff_text(version1.model_card, version2.model_card),
            "lineage_diff": self._diff_lineage(
                self.get_lineage(model_id, version_hash_1),
                self.get_lineage(model_id, version_hash_2)
            ),
            "performance_diff": self._diff_performance(
                version1.metadata.get("metrics", {}),
                version2.metadata.get("metrics", {})
            ),
            "summary": self._generate_diff_summary(version1, version2)
        }
        
        # Cache the diff
        if use_cache:
            self._cache_diff(model_id, version_hash_1, version_hash_2, diff)
        
        return diff
    
    def _diff_metadata(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare metadata between two versions."""
        diff = {
            "added": {},
            "removed": {},
            "changed": {}
        }
        
        all_keys = set(metadata1.keys()) | set(metadata2.keys())
        
        for key in all_keys:
            val1 = metadata1.get(key)
            val2 = metadata2.get(key)
            
            if val1 is None:
                diff["added"][key] = val2
            elif val2 is None:
                diff["removed"][key] = val1
            elif val1 != val2:
                diff["changed"][key] = {"from": val1, "to": val2}
        
        return diff
    
    def _diff_text(self, text1: str, text2: str) -> Dict[str, Any]:
        """Simple text diff between two strings."""
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        added = [line for line in lines2 if line not in lines1]
        removed = [line for line in lines1 if line not in lines2]
        
        return {
            "added_lines": len(added),
            "removed_lines": len(removed),
            "added": added[:10],  # Limit to first 10 for brevity
            "removed": removed[:10],
            "similarity_score": self._calculate_similarity(text1, text2)
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple similarity score between two texts."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 1.0
    
    def _diff_lineage(
        self,
        lineage1: List[ModelVersion],
        lineage2: List[ModelVersion]
    ) -> Dict[str, Any]:
        """Compare lineage between two versions."""
        hashes1 = [v.version_hash for v in lineage1]
        hashes2 = [v.version_hash for v in lineage2]
        
        common_ancestor = None
        for h1, h2 in zip(hashes1, hashes2):
            if h1 == h2:
                common_ancestor = h1
            else:
                break
        
        return {
            "common_ancestor": common_ancestor,
            "divergence_point": len(hashes1) - hashes1.index(common_ancestor) if common_ancestor in hashes1 else 0,
            "lineage_length_1": len(lineage1),
            "lineage_length_2": len(lineage2)
        }
    
    def _diff_performance(
        self,
        metrics1: Dict[str, Any],
        metrics2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare performance metrics between two versions."""
        diff = {
            "improved": {},
            "degraded": {},
            "unchanged": {}
        }
        
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            val1 = metrics1.get(metric)
            val2 = metrics2.get(metric)
            
            if val1 is None or val2 is None:
                continue
            
            try:
                # Try to compare as numbers
                num1 = float(val1)
                num2 = float(val2)
                
                if num2 > num1:
                    diff["improved"][metric] = {"from": num1, "to": num2, "change": num2 - num1}
                elif num2 < num1:
                    diff["degraded"][metric] = {"from": num1, "to": num2, "change": num2 - num1}
                else:
                    diff["unchanged"][metric] = num1
            except (ValueError, TypeError):
                # Not numbers, compare as strings
                if val1 != val2:
                    diff["changed"] = diff.get("changed", {})
                    diff["changed"][metric] = {"from": val1, "to": val2}
        
        return diff
    
    def _generate_diff_summary(
        self,
        version1: ModelVersion,
        version2: ModelVersion
    ) -> str:
        """Generate human-readable diff summary."""
        summary_parts = []
        
        # Check parent relationship
        if version2.parent_hash == version1.version_hash:
            summary_parts.append(f"Version {version2.version_hash[:8]} is a direct child of {version1.version_hash[:8]}")
        elif version1.parent_hash == version2.version_hash:
            summary_parts.append(f"Version {version1.version_hash[:8]} is a direct child of {version2.version_hash[:8]}")
        else:
            summary_parts.append(f"Versions {version1.version_hash[:8]} and {version2.version_hash[:8]} are siblings or cousins")
        
        # Compare creation times
        time_diff = version2.created_at - version1.created_at
        if time_diff.total_seconds() > 0:
            summary_parts.append(f"Version {version2.version_hash[:8]} is newer by {time_diff}")
        else:
            summary_parts.append(f"Version {version1.version_hash[:8]} is newer by {-time_diff}")
        
        # Compare tags
        tags1 = set(version1.tags)
        tags2 = set(version2.tags)
        if tags1 != tags2:
            added_tags = tags2 - tags1
            removed_tags = tags1 - tags2
            if added_tags:
                summary_parts.append(f"Added tags: {', '.join(added_tags)}")
            if removed_tags:
                summary_parts.append(f"Removed tags: {', '.join(removed_tags)}")
        
        return ". ".join(summary_parts) + "."
    
    def _get_cached_diff(
        self,
        model_id: str,
        version_hash_1: str,
        version_hash_2: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached diff if exists."""
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(
                    """
                    SELECT diff_data FROM model_diffs 
                    WHERE model_id = %s 
                    AND version_hash_1 = %s 
                    AND version_hash_2 = %s
                    """,
                    (model_id, version_hash_1, version_hash_2)
                )
                row = cur.fetchone()
                return row["diff_data"] if row else None
        finally:
            conn.close()
    
    def _cache_diff(
        self,
        model_id: str,
        version_hash_1: str,
        version_hash_2: str,
        diff_data: Dict[str, Any]
    ) -> None:
        """Cache diff for future retrieval."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_diffs 
                    (model_id, version_hash_1, version_hash_2, diff_data)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (model_id, version_hash_1, version_hash_2) 
                    DO UPDATE SET diff_data = EXCLUDED.diff_data
                    """,
                    (model_id, version_hash_1, version_hash_2, Json(diff_data))
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise ModelRegistryError(f"Failed to cache diff: {e}")
        finally:
            conn.close()
    
    def tag_version(
        self,
        model_id: str,
        version_hash: str,
        tags: List[str]
    ) -> None:
        """Add tags to a model version."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Get current tags
                cur.execute(
                    "SELECT tags FROM model_versions WHERE model_id = %s AND version_hash = %s",
                    (model_id, version_hash)
                )
                row = cur.fetchone()
                if not row:
                    raise ModelRegistryError(f"Model version not found: {model_id}/{version_hash}")
                
                current_tags = set(row[0] or [])
                new_tags = list(current_tags | set(tags))
                
                cur.execute(
                    """
                    UPDATE model_versions 
                    SET tags = %s 
                    WHERE model_id = %s AND version_hash = %s
                    """,
                    (new_tags, model_id, version_hash)
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise ModelRegistryError(f"Failed to tag version: {e}")
        finally:
            conn.close()
    
    def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ModelVersion]:
        """
        Search models by query, tags, or metadata filters.
        
        Args:
            query: Text search in model_id or model_card
            tags: Filter by tags (AND logic)
            metadata_filters: Filter by metadata key-value pairs
            limit: Maximum results to return
            offset: Pagination offset
        """
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=DictCursor) as cur:
                conditions = []
                params = []
                
                if query:
                    conditions.append("(model_id ILIKE %s OR model_card ILIKE %s)")
                    params.extend([f"%{query}%", f"%{query}%"])
                
                if tags:
                    for tag in tags:
                        conditions.append("%s = ANY(tags)")
                        params.append(tag)
                
                if metadata_filters:
                    for key, value in metadata_filters.items():
                        conditions.append("metadata->>%s = %s")
                        params.extend([key, str(value)])
                
                where_clause = " AND ".join(conditions) if conditions else "TRUE"
                
                query_sql = f"""
                    SELECT * FROM model_versions 
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """
                params.extend([limit, offset])
                
                cur.execute(query_sql, params)
                rows = cur.fetchall()
                
                return [
                    ModelVersion(
                        model_id=row["model_id"],
                        version_hash=row["version_hash"],
                        parent_hash=row["parent_hash"],
                        metadata=row["metadata"],
                        model_card=row["model_card"],
                        created_at=row["created_at"],
                        tags=row["tags"],
                        status=JobStatus(row["status"])
                    )
                    for row in rows
                ]
        finally:
            conn.close()
    
    def delete_version(
        self,
        model_id: str,
        version_hash: str,
        force: bool = False
    ) -> bool:
        """
        Delete a model version.
        
        Args:
            model_id: Model identifier
            version_hash: Version hash to delete
            force: If True, delete even if version has children
            
        Returns:
            True if deletion was successful
        """
        # Check if version has children
        if not force:
            children = self.get_children(model_id, version_hash)
            if children:
                raise ModelRegistryError(
                    f"Cannot delete version {version_hash}: it has {len(children)} children. "
                    "Use force=True to override."
                )
        
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Delete from database
                cur.execute(
                    """
                    DELETE FROM model_versions 
                    WHERE model_id = %s AND version_hash = %s
                    RETURNING metadata->>'content_hash' as content_hash
                    """,
                    (model_id, version_hash)
                )
                row = cur.fetchone()
                
                if not row:
                    return False
                
                content_hash = row[0]
                
                # Delete cached diffs
                cur.execute(
                    """
                    DELETE FROM model_diffs 
                    WHERE model_id = %s 
                    AND (version_hash_1 = %s OR version_hash_2 = %s)
                    """,
                    (model_id, version_hash, version_hash)
                )
                
                conn.commit()
                
                # Optionally delete from storage (commented out for safety)
                # if content_hash:
                #     storage_path = self.storage.retrieve(model_id, content_hash)
                #     if storage_path.exists():
                #         storage_path.unlink()
                
                return True
        except Exception as e:
            conn.rollback()
            raise ModelRegistryError(f"Failed to delete version: {e}")
        finally:
            conn.close()


# Singleton instance for easy import
_registry_instance = None


def get_model_registry(storage_path: Union[str, Path] = "./model_storage") -> ModelRegistry:
    """Get singleton instance of model registry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(storage_path)
    return _registry_instance


# CLI integration helpers
def format_model_version(version: ModelVersion) -> str:
    """Format model version for CLI display."""
    return (
        f"Model: {version.model_id}\n"
        f"Version: {version.version_hash[:16]}...\n"
        f"Parent: {version.parent_hash[:16] + '...' if version.parent_hash else 'None'}\n"
        f"Created: {version.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Tags: {', '.join(version.tags) if version.tags else 'None'}\n"
        f"Status: {version.status.value}"
    )


def format_diff(diff: Dict[str, Any]) -> str:
    """Format diff for CLI display."""
    output = []
    output.append(f"Diff between {diff['version_1'][:8]} and {diff['version_2'][:8]}")
    output.append("=" * 50)
    
    # Metadata changes
    metadata_diff = diff.get("metadata_diff", {})
    if metadata_diff.get("added"):
        output.append("\nAdded metadata:")
        for key, value in metadata_diff["added"].items():
            output.append(f"  + {key}: {value}")
    
    if metadata_diff.get("removed"):
        output.append("\nRemoved metadata:")
        for key, value in metadata_diff["removed"].items():
            output.append(f"  - {key}: {value}")
    
    if metadata_diff.get("changed"):
        output.append("\nChanged metadata:")
        for key, change in metadata_diff["changed"].items():
            output.append(f"  ~ {key}: {change['from']} -> {change['to']}")
    
    # Performance changes
    perf_diff = diff.get("performance_diff", {})
    if perf_diff.get("improved"):
        output.append("\nImproved metrics:")
        for metric, change in perf_diff["improved"].items():
            output.append(f"  ↑ {metric}: {change['from']} -> {change['to']} (+{change['change']:.4f})")
    
    if perf_diff.get("degraded"):
        output.append("\nDegraded metrics:")
        for metric, change in perf_diff["degraded"].items():
            output.append(f"  ↓ {metric}: {change['from']} -> {change['to']} ({change['change']:.4f})")
    
    # Summary
    output.append(f"\nSummary: {diff.get('summary', 'No summary available')}")
    
    return "\n".join(output)