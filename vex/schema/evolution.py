"""Automatic Schema Evolution & Data Validation for Scrapy

Built-in schema inference and evolution that automatically detects data structure changes,
validates against JSON Schema, and maintains backward compatibility.
"""

import json
import os
import re
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import defaultdict, OrderedDict
from copy import deepcopy
import warnings

logger = logging.getLogger(__name__)

# JSON Schema validation (optional dependency)
try:
    import jsonschema
    from jsonschema import validate, ValidationError, Draft7Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    warnings.warn(
        "jsonschema not installed. Schema validation will be limited. "
        "Install with: pip install jsonschema"
    )


class SchemaInference:
    """Infer JSON Schema from sample data structures."""
    
    def __init__(self, min_samples: int = 3, max_depth: int = 10):
        """
        Args:
            min_samples: Minimum number of samples needed for reliable inference
            max_depth: Maximum depth for nested schema inference
        """
        self.min_samples = min_samples
        self.max_depth = max_depth
        self._type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            type(None): "null",
            list: "array",
            dict: "object"
        }
    
    def infer_schema(self, samples: List[Dict]) -> Dict:
        """Infer JSON Schema from a list of sample items.
        
        Args:
            samples: List of dictionaries to infer schema from
            
        Returns:
            JSON Schema dictionary
        """
        if not samples:
            return {"type": "object", "properties": {}}
        
        if len(samples) < self.min_samples:
            logger.warning(
                f"Only {len(samples)} samples provided, "
                f"minimum recommended is {self.min_samples}"
            )
        
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }
        
        # Track field statistics across samples
        field_stats = defaultdict(lambda: {
            "types": set(),
            "required_count": 0,
            "examples": [],
            "nullable": False
        })
        
        # Analyze each sample
        for sample in samples:
            if not isinstance(sample, dict):
                logger.warning(f"Skipping non-dict sample: {type(sample)}")
                continue
            
            sample_fields = set(sample.keys())
            
            for field, value in sample.items():
                stats = field_stats[field]
                stats["types"].add(self._get_type(value))
                
                if value is None:
                    stats["nullable"] = True
                else:
                    stats["examples"].append(value)
                    if len(stats["examples"]) > 5:  # Keep only 5 examples
                        stats["examples"].pop(0)
            
            # Track required fields (present in all samples)
            for field in field_stats:
                if field in sample_fields:
                    field_stats[field]["required_count"] += 1
        
        # Build schema properties
        for field, stats in field_stats.items():
            field_schema = self._infer_field_schema(field, stats, len(samples))
            schema["properties"][field] = field_schema
            
            # Mark as required if present in all samples
            if stats["required_count"] == len(samples) and not stats["nullable"]:
                schema["required"].append(field)
        
        return schema
    
    def _get_type(self, value: Any) -> str:
        """Get JSON Schema type for a value."""
        if value is None:
            return "null"
        
        value_type = type(value)
        
        # Handle nested structures
        if value_type == dict:
            return "object"
        elif value_type == list:
            return "array"
        
        return self._type_mapping.get(value_type, "string")
    
    def _infer_field_schema(self, field: str, stats: Dict, total_samples: int) -> Dict:
        """Infer schema for a single field."""
        field_schema = {}
        types = stats["types"]
        
        # Handle nullable fields
        if "null" in types:
            types.discard("null")
            if len(types) == 1:
                field_schema["type"] = next(iter(types))
                field_schema["nullable"] = True
            else:
                field_schema["type"] = list(types)
                field_schema["nullable"] = True
        elif len(types) == 1:
            field_schema["type"] = next(iter(types))
        else:
            field_schema["type"] = list(types)
        
        # Add examples if available
        if stats["examples"]:
            field_schema["examples"] = stats["examples"][:3]
        
        # Infer additional constraints based on examples
        if "string" in types and stats["examples"]:
            field_schema.update(self._infer_string_constraints(stats["examples"]))
        
        return field_schema
    
    def _infer_string_constraints(self, examples: List[str]) -> Dict:
        """Infer constraints for string fields."""
        constraints = {}
        
        # Check for patterns
        if all(isinstance(ex, str) for ex in examples):
            # Check for email pattern
            if all(re.match(r'^[^@]+@[^@]+\.[^@]+$', ex) for ex in examples):
                constraints["format"] = "email"
            
            # Check for URL pattern
            elif all(re.match(r'^https?://', ex) for ex in examples):
                constraints["format"] = "uri"
            
            # Check for date pattern
            elif all(re.match(r'^\d{4}-\d{2}-\d{2}', ex) for ex in examples):
                constraints["format"] = "date"
        
        # Check for min/max length
        lengths = [len(ex) for ex in examples if isinstance(ex, str)]
        if lengths:
            constraints["minLength"] = min(lengths)
            constraints["maxLength"] = max(lengths)
        
        return constraints


class SchemaVersion:
    """Represents a versioned schema with metadata."""
    
    def __init__(
        self,
        schema: Dict,
        version: Optional[str] = None,
        description: str = "",
        created_at: Optional[datetime] = None
    ):
        self.schema = schema
        self.version = version or self._generate_version(schema)
        self.description = description
        self.created_at = created_at or datetime.now()
        self._hash = self._calculate_hash()
    
    def _generate_version(self, schema: Dict) -> str:
        """Generate version based on schema content."""
        schema_str = json.dumps(schema, sort_keys=True)
        hash_obj = hashlib.sha256(schema_str.encode())
        return f"v{hash_obj.hexdigest()[:8]}"
    
    def _calculate_hash(self) -> str:
        """Calculate hash of the schema for comparison."""
        schema_str = json.dumps(self.schema, sort_keys=True)
        return hashlib.md5(schema_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "schema": self.schema,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "hash": self._hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SchemaVersion':
        """Create from dictionary."""
        return cls(
            schema=data["schema"],
            version=data["version"],
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"])
        )
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return False
        return self._hash == other._hash
    
    def __hash__(self) -> int:
        return hash(self._hash)


class SchemaEvolution:
    """Manages schema evolution, versioning, and compatibility."""
    
    def __init__(self, storage_path: str = "schemas"):
        """
        Args:
            storage_path: Directory to store schema versions
        """
        self.storage_path = storage_path
        self._ensure_storage_dir()
        self.schemas: Dict[str, SchemaVersion] = {}
        self._load_schemas()
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path, exist_ok=True)
    
    def _load_schemas(self):
        """Load all schema versions from storage."""
        schema_file = os.path.join(self.storage_path, "schemas.json")
        if os.path.exists(schema_file):
            try:
                with open(schema_file, 'r') as f:
                    data = json.load(f)
                    for version_data in data.get("versions", []):
                        schema_version = SchemaVersion.from_dict(version_data)
                        self.schemas[schema_version.version] = schema_version
                logger.info(f"Loaded {len(self.schemas)} schema versions")
            except Exception as e:
                logger.error(f"Failed to load schemas: {e}")
    
    def _save_schemas(self):
        """Save all schema versions to storage."""
        schema_file = os.path.join(self.storage_path, "schemas.json")
        data = {
            "versions": [sv.to_dict() for sv in self.schemas.values()],
            "last_updated": datetime.now().isoformat()
        }
        
        with open(schema_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_schema(
        self,
        schema: Dict,
        description: str = "",
        force_new_version: bool = False
    ) -> SchemaVersion:
        """Register a new schema version.
        
        Args:
            schema: JSON Schema dictionary
            description: Description of this schema version
            force_new_version: Force creation of new version even if schema unchanged
            
        Returns:
            The registered SchemaVersion
        """
        new_version = SchemaVersion(schema, description=description)
        
        # Check if identical schema already exists
        if not force_new_version:
            for existing in self.schemas.values():
                if existing == new_version:
                    logger.info(f"Schema already exists as version {existing.version}")
                    return existing
        
        # Add new version
        self.schemas[new_version.version] = new_version
        self._save_schemas()
        
        logger.info(f"Registered new schema version: {new_version.version}")
        return new_version
    
    def get_latest_schema(self) -> Optional[SchemaVersion]:
        """Get the latest registered schema version."""
        if not self.schemas:
            return None
        
        # Sort by creation time
        versions = sorted(
            self.schemas.values(),
            key=lambda x: x.created_at,
            reverse=True
        )
        return versions[0]
    
    def get_schema(self, version: str) -> Optional[SchemaVersion]:
        """Get schema by version string."""
        return self.schemas.get(version)
    
    def compare_schemas(
        self,
        old_schema: Dict,
        new_schema: Dict
    ) -> Dict[str, List]:
        """Compare two schemas and identify changes.
        
        Returns:
            Dictionary with breaking and non-breaking changes
        """
        changes = {
            "breaking": [],
            "non_breaking": [],
            "additions": [],
            "removals": []
        }
        
        old_props = old_schema.get("properties", {})
        new_props = new_schema.get("properties", {})
        
        # Check for removed fields
        for field in old_props:
            if field not in new_props:
                changes["removals"].append(field)
                changes["breaking"].append(f"Removed field: {field}")
        
        # Check for added fields
        for field in new_props:
            if field not in old_props:
                changes["additions"].append(field)
                # Check if required in new schema
                if field in new_schema.get("required", []):
                    changes["breaking"].append(
                        f"Added required field: {field}"
                    )
                else:
                    changes["non_breaking"].append(
                        f"Added optional field: {field}"
                    )
        
        # Check for type changes
        for field in set(old_props.keys()) & set(new_props.keys()):
            old_type = old_props[field].get("type")
            new_type = new_props[field].get("type")
            
            if old_type != new_type:
                changes["breaking"].append(
                    f"Type changed for {field}: {old_type} -> {new_type}"
                )
        
        return changes
    
    def check_compatibility(
        self,
        old_schema: Dict,
        new_schema: Dict,
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """Check if new schema is backward compatible with old schema.
        
        Args:
            old_schema: Original schema
            new_schema: New schema to check
            strict: If True, any breaking change is considered incompatible
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        changes = self.compare_schemas(old_schema, new_schema)
        issues = []
        
        # Breaking changes always cause incompatibility
        if changes["breaking"]:
            issues.extend(changes["breaking"])
        
        # In strict mode, removals are also breaking
        if strict and changes["removals"]:
            issues.append(f"Strict mode: removals not allowed: {changes['removals']}")
        
        is_compatible = len(issues) == 0
        return is_compatible, issues
    
    def generate_migration_script(
        self,
        from_version: str,
        to_version: str
    ) -> Optional[Dict]:
        """Generate migration script between schema versions.
        
        Returns:
            Migration script dictionary or None if no migration needed
        """
        old_schema = self.get_schema(from_version)
        new_schema = self.get_schema(to_version)
        
        if not old_schema or not new_schema:
            logger.error("One or both schema versions not found")
            return None
        
        changes = self.compare_schemas(old_schema.schema, new_schema.schema)
        
        if not any(changes.values()):
            return None  # No changes
        
        migration = {
            "from_version": from_version,
            "to_version": to_version,
            "changes": changes,
            "transformations": []
        }
        
        # Generate transformations for each change
        for field in changes["additions"]:
            if field in new_schema.schema.get("required", []):
                migration["transformations"].append({
                    "type": "add_required_field",
                    "field": field,
                    "default": self._get_default_value(
                        new_schema.schema["properties"][field]
                    )
                })
            else:
                migration["transformations"].append({
                    "type": "add_optional_field",
                    "field": field
                })
        
        for field in changes["removals"]:
            migration["transformations"].append({
                "type": "remove_field",
                "field": field
            })
        
        return migration
    
    def _get_default_value(self, field_schema: Dict) -> Any:
        """Get default value for a field based on its schema."""
        field_type = field_schema.get("type", "string")
        
        defaults = {
            "string": "",
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "array": [],
            "object": {},
            "null": None
        }
        
        return defaults.get(field_type, None)


class SchemaValidator:
    """Validates data against JSON Schema with configurable strictness."""
    
    def __init__(
        self,
        schema: Dict,
        strictness: str = "warn",
        coerce_types: bool = True
    ):
        """
        Args:
            schema: JSON Schema to validate against
            strictness: 'strict', 'warn', or 'ignore'
            coerce_types: Attempt to coerce types when possible
        """
        if not JSONSCHEMA_AVAILABLE:
            raise ImportError(
                "jsonschema package required for validation. "
                "Install with: pip install jsonschema"
            )
        
        self.schema = schema
        self.strictness = strictness
        self.coerce_types = coerce_types
        self.validator = Draft7Validator(schema)
        self._coercion_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool
        }
    
    def validate(self, item: Dict) -> Tuple[bool, List[str]]:
        """Validate an item against the schema.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Try type coercion first if enabled
        if self.coerce_types:
            item = self._coerce_types(item)
        
        # Validate against schema
        validation_errors = list(self.validator.iter_errors(item))
        
        for error in validation_errors:
            error_msg = self._format_error(error)
            errors.append(error_msg)
            
            if self.strictness == "strict":
                break  # Stop at first error in strict mode
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _coerce_types(self, item: Dict) -> Dict:
        """Attempt to coerce item types to match schema."""
        coerced = deepcopy(item)
        properties = self.schema.get("properties", {})
        
        for field, field_schema in properties.items():
            if field not in coerced:
                continue
            
            value = coerced[field]
            target_type = field_schema.get("type")
            
            if target_type in self._coercion_map:
                try:
                    if target_type == "number" and isinstance(value, (int, float)):
                        continue  # Already numeric
                    elif target_type == "integer" and isinstance(value, int):
                        continue  # Already integer
                    else:
                        coerced[field] = self._coercion_map[target_type](value)
                except (ValueError, TypeError):
                    pass  # Keep original value if coercion fails
        
        return coerced
    
    def _format_error(self, error: jsonschema.ValidationError) -> str:
        """Format validation error into readable string."""
        path = ".".join(str(p) for p in error.absolute_path)
        if not path:
            path = "item"
        
        return f"{path}: {error.message}"


class ValidationPipeline:
    """Scrapy pipeline for schema validation and evolution."""
    
    def __init__(
        self,
        schema_evolution: SchemaEvolution,
        strictness: str = "warn",
        auto_evolve: bool = True,
        min_samples_for_inference: int = 10
    ):
        """
        Args:
            schema_evolution: SchemaEvolution instance
            strictness: Validation strictness level
            auto_evolve: Automatically evolve schema when changes detected
            min_samples_for_inference: Minimum samples before auto-inferring schema
        """
        self.schema_evolution = schema_evolution
        self.strictness = strictness
        self.auto_evolve = auto_evolve
        self.min_samples = min_samples_for_inference
        self.inference = SchemaInference()
        self.validator = None
        self.current_schema = None
        self._sample_buffer = []
        self._stats = {
            "validated": 0,
            "passed": 0,
            "failed": 0,
            "coerced": 0,
            "schema_updates": 0
        }
    
    def open_spider(self, spider):
        """Called when spider opens."""
        # Load or create initial schema
        self.current_schema = self.schema_evolution.get_latest_schema()
        
        if not self.current_schema:
            logger.info("No existing schema found. Will infer from samples.")
            return
        
        # Initialize validator
        self.validator = SchemaValidator(
            self.current_schema.schema,
            strictness=self.strictness
        )
        logger.info(f"Loaded schema version: {self.current_schema.version}")
    
    def process_item(self, item, spider):
        """Process and validate item."""
        self._stats["validated"] += 1
        
        # Collect samples for schema inference if no schema exists
        if not self.current_schema:
            self._sample_buffer.append(dict(item))
            
            if len(self._sample_buffer) >= self.min_samples:
                self._infer_and_register_schema()
            
            return item  # Pass through until we have schema
        
        # Validate against current schema
        is_valid, errors = self.validator.validate(dict(item))
        
        if is_valid:
            self._stats["passed"] += 1
        else:
            self._stats["failed"] += 1
            self._handle_validation_errors(item, errors, spider)
        
        # Check for schema evolution if enabled
        if self.auto_evolve:
            self._sample_buffer.append(dict(item))
            if len(self._sample_buffer) >= self.min_samples:
                self._check_for_schema_evolution()
        
        return item
    
    def close_spider(self, spider):
        """Called when spider closes."""
        # Process remaining samples
        if self._sample_buffer and self.auto_evolve:
            self._check_for_schema_evolution()
        
        # Log statistics
        logger.info(f"Validation statistics: {self._stats}")
    
    def _infer_and_register_schema(self):
        """Infer schema from samples and register it."""
        if not self._sample_buffer:
            return
        
        logger.info(f"Inferring schema from {len(self._sample_buffer)} samples")
        inferred_schema = self.inference.infer_schema(self._sample_buffer)
        
        # Register the new schema
        self.current_schema = self.schema_evolution.register_schema(
            inferred_schema,
            description=f"Auto-inferred from {len(self._sample_buffer)} samples"
        )
        
        # Initialize validator
        self.validator = SchemaValidator(
            self.current_schema.schema,
            strictness=self.strictness
        )
        
        # Clear buffer
        self._sample_buffer.clear()
        self._stats["schema_updates"] += 1
        
        logger.info(f"Registered new schema version: {self.current_schema.version}")
    
    def _check_for_schema_evolution(self):
        """Check if schema needs evolution based on recent samples."""
        if not self._sample_buffer or not self.current_schema:
            return
        
        # Infer schema from recent samples
        recent_schema = self.inference.infer_schema(self._sample_buffer)
        
        # Compare with current schema
        changes = self.schema_evolution.compare_schemas(
            self.current_schema.schema,
            recent_schema
        )
        
        # If significant changes detected, evolve schema
        if changes["additions"] or changes["breaking"]:
            logger.info(f"Schema changes detected: {changes}")
            
            # Check compatibility
            is_compatible, issues = self.schema_evolution.check_compatibility(
                self.current_schema.schema,
                recent_schema
            )
            
            if is_compatible or self.strictness != "strict":
                # Register evolved schema
                self.current_schema = self.schema_evolution.register_schema(
                    recent_schema,
                    description=f"Evolved from {self.current_schema.version}"
                )
                
                # Update validator
                self.validator = SchemaValidator(
                    self.current_schema.schema,
                    strictness=self.strictness
                )
                
                self._stats["schema_updates"] += 1
                logger.info(f"Schema evolved to version: {self.current_schema.version}")
            else:
                logger.warning(
                    f"Schema evolution blocked due to incompatibility: {issues}"
                )
        
        # Clear processed samples
        self._sample_buffer.clear()
    
    def _handle_validation_errors(self, item, errors, spider):
        """Handle validation errors based on strictness level."""
        error_msg = f"Validation failed for item: {errors}"
        
        if self.strictness == "strict":
            # Drop the item
            from vex.exceptions import DropItem
            raise DropItem(error_msg)
        
        elif self.strictness == "warn":
            # Log warning but keep item
            logger.warning(error_msg)
            
            # Add validation metadata to item
            if not hasattr(item, '_validation'):
                item['_validation'] = {}
            item['_validation']['errors'] = errors
            item['_validation']['schema_version'] = self.current_schema.version


class SchemaMigration:
    """Utilities for migrating data between schema versions."""
    
    def __init__(self, schema_evolution: SchemaEvolution):
        self.schema_evolution = schema_evolution
    
    def migrate_item(
        self,
        item: Dict,
        from_version: str,
        to_version: Optional[str] = None
    ) -> Dict:
        """Migrate an item from one schema version to another.
        
        Args:
            item: Item to migrate
            from_version: Source schema version
            to_version: Target schema version (defaults to latest)
            
        Returns:
            Migrated item
        """
        if to_version is None:
            latest = self.schema_evolution.get_latest_schema()
            if not latest:
                return item
            to_version = latest.version
        
        if from_version == to_version:
            return item
        
        migration = self.schema_evolution.generate_migration_script(
            from_version,
            to_version
        )
        
        if not migration:
            return item
        
        return self._apply_migration(item, migration)
    
    def _apply_migration(self, item: Dict, migration: Dict) -> Dict:
        """Apply migration transformations to an item."""
        migrated = deepcopy(item)
        
        for transformation in migration["transformations"]:
            t_type = transformation["type"]
            
            if t_type == "add_required_field":
                field = transformation["field"]
                if field not in migrated:
                    migrated[field] = transformation.get("default")
            
            elif t_type == "add_optional_field":
                # Optional fields don't need defaults
                pass
            
            elif t_type == "remove_field":
                field = transformation["field"]
                migrated.pop(field, None)
        
        return migrated
    
    def bulk_migrate(
        self,
        items: List[Dict],
        from_version: str,
        to_version: Optional[str] = None
    ) -> List[Dict]:
        """Migrate multiple items."""
        return [
            self.migrate_item(item, from_version, to_version)
            for item in items
        ]


# Integration with Scrapy settings
def get_schema_evolution_from_settings(settings) -> SchemaEvolution:
    """Create SchemaEvolution from Scrapy settings."""
    storage_path = settings.get(
        'SCHEMA_STORAGE_PATH',
        os.path.join(settings.get('PROJECT_DIR', '.'), 'schemas')
    )
    return SchemaEvolution(storage_path)


def get_validation_pipeline_from_settings(settings) -> ValidationPipeline:
    """Create ValidationPipeline from Scrapy settings."""
    schema_evolution = get_schema_evolution_from_settings(settings)
    
    return ValidationPipeline(
        schema_evolution=schema_evolution,
        strictness=settings.get('SCHEMA_VALIDATION_STRICTNESS', 'warn'),
        auto_evolve=settings.get('SCHEMA_AUTO_EVOLVE', True),
        min_samples_for_inference=settings.get('SCHEMA_MIN_SAMPLES', 10)
    )