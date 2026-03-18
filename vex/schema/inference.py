"""
Automatic Schema Evolution & Data Validation for Scrapy.

This module provides built-in schema inference, evolution, and validation
capabilities that automatically detect data structure changes, validate against
JSON Schema, and maintain backward compatibility.
"""

import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when data fails schema validation."""
    pass


class SchemaEvolutionError(Exception):
    """Raised when schema evolution fails."""
    pass


class StrictnessLevel(Enum):
    """Configurable strictness levels for validation."""
    STRICT = "strict"          # All fields must match exactly
    LENIENT = "lenient"        # Allow missing optional fields, type coercion
    PERMISSIVE = "permissive"  # Allow extra fields, minimal validation


class SchemaVersion:
    """Represents a versioned schema with metadata."""
    
    def __init__(self, 
                 schema: Dict[str, Any], 
                 version: int = 1,
                 created_at: Optional[datetime] = None,
                 description: str = "",
                 backward_compatible: bool = True):
        self.schema = schema
        self.version = version
        self.created_at = created_at or datetime.utcnow()
        self.description = description
        self.backward_compatible = backward_compatible
        self._hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute a hash of the schema for change detection."""
        schema_str = json.dumps(self.schema, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
    
    @property
    def hash(self) -> str:
        """Get the schema hash."""
        return self._hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "schema": self.schema,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "backward_compatible": self.backward_compatible,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaVersion":
        """Create from dictionary representation."""
        return cls(
            schema=data["schema"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            description=data.get("description", ""),
            backward_compatible=data.get("backward_compatible", True)
        )


class SchemaInference:
    """
    Infers JSON Schema from sample data with automatic type detection.
    
    Supports nested structures, arrays, optional fields, and type inference
    from multiple samples.
    """
    
    # Type priority for mixed type inference (higher priority wins)
    TYPE_PRIORITY = {
        "string": 1,
        "number": 2,
        "integer": 3,
        "boolean": 4,
        "null": 5,
        "array": 6,
        "object": 7
    }
    
    def __init__(self, 
                 min_samples: int = 3,
                 confidence_threshold: float = 0.8,
                 detect_optional: bool = True):
        """
        Initialize schema inference.
        
        Args:
            min_samples: Minimum samples needed for reliable inference
            confidence_threshold: Threshold for field presence to be considered required
            detect_optional: Whether to detect optional fields
        """
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.detect_optional = detect_optional
        self._field_stats = defaultdict(lambda: {"count": 0, "types": defaultdict(int)})
    
    def infer_from_samples(self, 
                          samples: List[Dict[str, Any]], 
                          root_name: str = "root") -> Dict[str, Any]:
        """
        Infer schema from multiple data samples.
        
        Args:
            samples: List of data samples to infer schema from
            root_name: Name for the root schema object
            
        Returns:
            JSON Schema dictionary
        """
        if not samples:
            raise ValueError("No samples provided for schema inference")
        
        # Reset statistics
        self._field_stats.clear()
        
        # Analyze all samples
        for sample in samples:
            self._analyze_value(sample, path="")
        
        # Build schema from statistics
        schema = self._build_schema(root_name)
        
        # Add metadata
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        schema["title"] = f"Auto-inferred schema from {len(samples)} samples"
        schema["description"] = f"Generated on {datetime.utcnow().isoformat()}"
        
        return schema
    
    def infer_from_single(self, 
                         sample: Dict[str, Any], 
                         root_name: str = "root") -> Dict[str, Any]:
        """
        Infer schema from a single sample (less reliable).
        
        Args:
            sample: Data sample to infer schema from
            root_name: Name for the root schema object
            
        Returns:
            JSON Schema dictionary
        """
        return self.infer_from_samples([sample], root_name)
    
    def _analyze_value(self, value: Any, path: str) -> None:
        """Recursively analyze a value and update statistics."""
        if value is None:
            self._record_field_type(path, "null")
        elif isinstance(value, bool):
            self._record_field_type(path, "boolean")
        elif isinstance(value, int):
            self._record_field_type(path, "integer")
        elif isinstance(value, float):
            self._record_field_type(path, "number")
        elif isinstance(value, str):
            self._record_field_type(path, "string")
        elif isinstance(value, list):
            self._record_field_type(path, "array")
            # Analyze array items
            for i, item in enumerate(value[:10]):  # Limit analysis for performance
                self._analyze_value(item, f"{path}[]")
        elif isinstance(value, dict):
            self._record_field_type(path, "object")
            # Analyze object properties
            for key, val in value.items():
                child_path = f"{path}.{key}" if path else key
                self._analyze_value(val, child_path)
        else:
            # Fallback for unknown types
            self._record_field_type(path, "string")
    
    def _record_field_type(self, path: str, type_name: str) -> None:
        """Record a field type observation."""
        self._field_stats[path]["count"] += 1
        self._field_stats[path]["types"][type_name] += 1
    
    def _build_schema(self, root_name: str) -> Dict[str, Any]:
        """Build JSON Schema from collected statistics."""
        schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Process each field
        for path, stats in self._field_stats.items():
            if not path:  # Root level
                continue
            
            # Determine field name and parent path
            parts = path.split(".")
            field_name = parts[-1]
            
            # Calculate type with highest confidence
            total_count = stats["count"]
            type_counts = stats["types"]
            
            # Find most common type
            most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
            confidence = type_counts[most_common_type] / total_count
            
            # Determine if field is required
            is_required = confidence >= self.confidence_threshold
            
            # Build field schema
            field_schema = self._build_field_schema(
                most_common_type, 
                path, 
                type_counts,
                total_count
            )
            
            # Add to parent schema
            self._add_field_to_schema(
                schema, 
                parts[:-1], 
                field_name, 
                field_schema, 
                is_required
            )
        
        return schema
    
    def _build_field_schema(self, 
                           primary_type: str, 
                           path: str,
                           type_counts: Dict[str, int],
                           total_count: int) -> Dict[str, Any]:
        """Build schema for a specific field."""
        if primary_type == "object":
            # Build nested object schema
            nested_schema = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            # Find all child fields
            child_prefix = f"{path}."
            for child_path in self._field_stats:
                if child_path.startswith(child_prefix):
                    child_parts = child_path[len(child_prefix):].split(".")
                    if len(child_parts) == 1:  # Direct child
                        child_name = child_parts[0]
                        child_stats = self._field_stats[child_path]
                        child_type = max(child_stats["types"].items(), 
                                        key=lambda x: x[1])[0]
                        child_schema = self._build_field_schema(
                            child_type,
                            child_path,
                            child_stats["types"],
                            child_stats["count"]
                        )
                        is_required = (child_stats["count"] / total_count >= 
                                     self.confidence_threshold)
                        
                        nested_schema["properties"][child_name] = child_schema
                        if is_required:
                            nested_schema["required"].append(child_name)
            
            return nested_schema
        
        elif primary_type == "array":
            # Try to infer array item type
            array_item_path = f"{path}[]"
            if array_item_path in self._field_stats:
                item_stats = self._field_stats[array_item_path]
                item_type = max(item_stats["types"].items(), 
                              key=lambda x: x[1])[0]
                item_schema = self._build_field_schema(
                    item_type,
                    array_item_path,
                    item_stats["types"],
                    item_stats["count"]
                )
                return {
                    "type": "array",
                    "items": item_schema
                }
            else:
                return {"type": "array"}
        
        else:
            # Primitive type
            return {"type": primary_type}
    
    def _add_field_to_schema(self, 
                            schema: Dict[str, Any], 
                            path_parts: List[str], 
                            field_name: str, 
                            field_schema: Dict[str, Any],
                            is_required: bool) -> None:
        """Add a field to the schema at the appropriate location."""
        current = schema
        
        # Navigate to parent object
        for part in path_parts:
            if part not in current["properties"]:
                current["properties"][part] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            current = current["properties"][part]
        
        # Add the field
        current["properties"][field_name] = field_schema
        
        # Mark as required if needed
        if is_required and field_name not in current["required"]:
            current["required"].append(field_name)


class SchemaRegistry:
    """
    Manages schema versions and evolution with backward compatibility.
    
    Tracks schema changes, validates compatibility, and provides migration paths.
    """
    
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 auto_version: bool = True):
        """
        Initialize schema registry.
        
        Args:
            storage_path: Path to store schema versions (JSON file)
            auto_version: Automatically increment version on schema changes
        """
        self.storage_path = storage_path
        self.auto_version = auto_version
        self._versions: Dict[int, SchemaVersion] = {}
        self._current_version: Optional[int] = None
        self._migration_handlers: Dict[Tuple[int, int], Callable] = {}
        
        # Load existing schemas if storage path provided
        if storage_path:
            self._load_schemas()
    
    @property
    def current_version(self) -> Optional[SchemaVersion]:
        """Get the current schema version."""
        if self._current_version is None:
            return None
        return self._versions.get(self._current_version)
    
    @property
    def latest_version(self) -> Optional[SchemaVersion]:
        """Get the latest schema version."""
        if not self._versions:
            return None
        latest_num = max(self._versions.keys())
        return self._versions[latest_num]
    
    def register_schema(self, 
                       schema: Dict[str, Any], 
                       description: str = "",
                       force_new_version: bool = False) -> SchemaVersion:
        """
        Register a new schema version.
        
        Args:
            schema: JSON Schema dictionary
            description: Description of schema changes
            force_new_version: Force creation of new version even if compatible
            
        Returns:
            The created SchemaVersion
        """
        if not self._versions:
            # First schema
            version = SchemaVersion(
                schema=schema,
                version=1,
                description=description or "Initial schema"
            )
            self._versions[1] = version
            self._current_version = 1
            self._save_schemas()
            return version
        
        current = self.current_version
        
        # Check if schema is identical
        if current and current.hash == SchemaVersion(schema).hash:
            logger.debug("Schema unchanged, returning current version")
            return current
        
        # Check backward compatibility
        is_compatible = self._check_compatibility(current.schema, schema)
        
        if is_compatible and not force_new_version:
            # Update current schema in place
            current.schema = schema
            current.description = description or current.description
            self._save_schemas()
            return current
        else:
            # Create new version
            new_version_num = max(self._versions.keys()) + 1
            new_version = SchemaVersion(
                schema=schema,
                version=new_version_num,
                description=description,
                backward_compatible=is_compatible
            )
            self._versions[new_version_num] = new_version
            self._current_version = new_version_num
            
            if not is_compatible:
                logger.warning(
                    f"Schema version {new_version_num} is not backward compatible "
                    f"with version {current.version if current else 'none'}"
                )
            
            self._save_schemas()
            return new_version
    
    def get_version(self, version: int) -> Optional[SchemaVersion]:
        """Get a specific schema version."""
        return self._versions.get(version)
    
    def validate_data(self, 
                     data: Dict[str, Any], 
                     version: Optional[int] = None,
                     strictness: StrictnessLevel = StrictnessLevel.STRICT) -> Tuple[bool, List[str]]:
        """
        Validate data against a schema version.
        
        Args:
            data: Data to validate
            version: Schema version to validate against (None for current)
            strictness: Validation strictness level
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        schema_version = self.get_version(version) if version else self.current_version
        if not schema_version:
            return False, ["No schema version available"]
        
        validator = DataValidator(schema_version.schema, strictness)
        return validator.validate(data)
    
    def evolve_schema(self, 
                     new_data: List[Dict[str, Any]],
                     inference_config: Optional[Dict[str, Any]] = None) -> SchemaVersion:
        """
        Evolve schema based on new data samples.
        
        Args:
            new_data: New data samples to analyze
            inference_config: Configuration for schema inference
            
        Returns:
            New or updated SchemaVersion
        """
        config = inference_config or {}
        inference = SchemaInference(**config)
        
        # Infer new schema
        new_schema = inference.infer_from_samples(new_data)
        
        # Register evolved schema
        return self.register_schema(
            new_schema,
            description=f"Schema evolved from {len(new_data)} new samples"
        )
    
    def get_migration_path(self, 
                          from_version: int, 
                          to_version: int) -> List[int]:
        """
        Get migration path between two versions.
        
        Returns:
            List of version numbers forming the migration path
        """
        if from_version == to_version:
            return [from_version]
        
        # Simple linear migration for now
        # Could be enhanced with graph-based path finding
        if from_version < to_version:
            return list(range(from_version, to_version + 1))
        else:
            return list(range(from_version, to_version - 1, -1))
    
    def register_migration_handler(self,
                                  from_version: int,
                                  to_version: int,
                                  handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """
        Register a custom migration handler between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            handler: Function that transforms data from source to target version
        """
        self._migration_handlers[(from_version, to_version)] = handler
    
    def migrate_data(self,
                    data: Dict[str, Any],
                    from_version: int,
                    to_version: int) -> Dict[str, Any]:
        """
        Migrate data between schema versions.
        
        Args:
            data: Data to migrate
            from_version: Source version
            to_version: Target version
            
        Returns:
            Migrated data
        """
        if from_version == to_version:
            return data
        
        path = self.get_migration_path(from_version, to_version)
        
        migrated_data = deepcopy(data)
        
        for i in range(len(path) - 1):
            current = path[i]
            next_ver = path[i + 1]
            
            # Check for custom handler
            handler = self._migration_handlers.get((current, next_ver))
            if handler:
                migrated_data = handler(migrated_data)
            else:
                # Default migration: add/remove fields based on schema differences
                migrated_data = self._default_migration(
                    migrated_data, 
                    self.get_version(current),
                    self.get_version(next_ver)
                )
        
        return migrated_data
    
    def _check_compatibility(self, 
                           old_schema: Dict[str, Any], 
                           new_schema: Dict[str, Any]) -> bool:
        """
        Check if new schema is backward compatible with old schema.
        
        Basic compatibility rules:
        - Can add new optional fields
        - Can remove fields (they become optional)
        - Cannot change field types in incompatible ways
        - Cannot make optional fields required
        """
        # For now, simple compatibility check
        # In production, would need more sophisticated analysis
        try:
            old_props = old_schema.get("properties", {})
            new_props = new_schema.get("properties", {})
            old_required = set(old_schema.get("required", []))
            new_required = set(new_schema.get("required", []))
            
            # Check if any required fields were removed
            removed_required = old_required - new_required
            if removed_required:
                return False
            
            # Check if any fields changed type incompatibly
            for field_name in set(old_props.keys()) & set(new_props.keys()):
                old_type = old_props[field_name].get("type")
                new_type = new_props[field_name].get("type")
                
                # Simple type compatibility check
                if old_type and new_type and old_type != new_type:
                    # Allow number -> integer, string -> number with coercion
                    compatible_pairs = {
                        ("number", "integer"),
                        ("string", "number"),
                        ("string", "integer")
                    }
                    if (old_type, new_type) not in compatible_pairs:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Compatibility check failed: {e}")
            return False
    
    def _default_migration(self,
                          data: Dict[str, Any],
                          from_version: Optional[SchemaVersion],
                          to_version: Optional[SchemaVersion]) -> Dict[str, Any]:
        """Perform default migration between versions."""
        if not from_version or not to_version:
            return data
        
        migrated = deepcopy(data)
        
        # Get field differences
        old_props = from_version.schema.get("properties", {})
        new_props = to_version.schema.get("properties", {})
        old_required = set(from_version.schema.get("required", []))
        new_required = set(to_version.schema.get("required", []))
        
        # Remove fields that no longer exist
        for field in set(migrated.keys()) - set(new_props.keys()):
            del migrated[field]
        
        # Add default values for new required fields
        for field in new_required - old_required:
            if field not in migrated:
                # Try to get default from schema
                field_schema = new_props.get(field, {})
                default = field_schema.get("default")
                if default is not None:
                    migrated[field] = default
                else:
                    # Use type-appropriate default
                    field_type = field_schema.get("type", "string")
                    migrated[field] = self._get_default_for_type(field_type)
        
        return migrated
    
    def _get_default_for_type(self, field_type: str) -> Any:
        """Get a default value for a given type."""
        defaults = {
            "string": "",
            "number": 0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {},
            "null": None
        }
        return defaults.get(field_type, None)
    
    def _save_schemas(self) -> None:
        """Save schemas to storage."""
        if not self.storage_path:
            return
        
        try:
            data = {
                "versions": {v: self._versions[v].to_dict() 
                           for v in self._versions},
                "current_version": self._current_version
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save schemas: {e}")
    
    def _load_schemas(self) -> None:
        """Load schemas from storage."""
        if not self.storage_path:
            return
        
        try:
            import os
            if not os.path.exists(self.storage_path):
                return
            
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self._versions = {
                int(v): SchemaVersion.from_dict(schema_data)
                for v, schema_data in data.get("versions", {}).items()
            }
            self._current_version = data.get("current_version")
            
        except Exception as e:
            logger.error(f"Failed to load schemas: {e}")


class DataValidator:
    """
    Validates data against JSON Schema with configurable strictness.
    
    Supports type coercion, optional fields, and custom validation rules.
    """
    
    def __init__(self, 
                 schema: Dict[str, Any],
                 strictness: StrictnessLevel = StrictnessLevel.STRICT):
        """
        Initialize validator.
        
        Args:
            schema: JSON Schema to validate against
            strictness: Validation strictness level
        """
        self.schema = schema
        self.strictness = strictness
        self._coercion_handlers = self._setup_coercion_handlers()
    
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """
        Validate data against schema.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        self._validate_value(data, self.schema, "", errors)
        return len(errors) == 0, errors
    
    def _validate_value(self, 
                       value: Any, 
                       schema: Dict[str, Any], 
                       path: str,
                       errors: List[str]) -> Any:
        """Recursively validate a value against schema."""
        # Handle $ref (not implemented for simplicity)
        if "$ref" in schema:
            errors.append(f"{path}: $ref not supported")
            return value
        
        # Get type from schema
        schema_type = schema.get("type")
        
        if schema_type:
            # Validate type
            value, type_error = self._validate_type(value, schema_type, path)
            if type_error:
                errors.append(type_error)
                return value
        
        # Type-specific validation
        if schema_type == "object":
            return self._validate_object(value, schema, path, errors)
        elif schema_type == "array":
            return self._validate_array(value, schema, path, errors)
        elif schema_type in ["string", "number", "integer", "boolean"]:
            return self._validate_primitive(value, schema, path, errors)
        
        return value
    
    def _validate_type(self, 
                      value: Any, 
                      expected_type: Union[str, List[str]], 
                      path: str) -> Tuple[Any, Optional[str]]:
        """Validate and optionally coerce type."""
        if isinstance(expected_type, list):
            # Multiple types allowed
            for type_name in expected_type:
                coerced, error = self._validate_type(value, type_name, path)
                if not error:
                    return coerced, None
            return value, f"{path}: Expected one of {expected_type}, got {type(value).__name__}"
        
        actual_type = self._get_json_type(value)
        
        if actual_type == expected_type:
            return value, None
        
        # Try coercion if allowed
        if self.strictness in [StrictnessLevel.LENIENT, StrictnessLevel.PERMISSIVE]:
            coerced = self._coerce_type(value, expected_type)
            if coerced is not None:
                return coerced, None
        
        return value, f"{path}: Expected {expected_type}, got {actual_type}"
    
    def _validate_object(self, 
                        value: Any, 
                        schema: Dict[str, Any], 
                        path: str,
                        errors: List[str]) -> Dict[str, Any]:
        """Validate object value."""
        if not isinstance(value, dict):
            if self.strictness == StrictnessLevel.PERMISSIVE:
                return {}
            errors.append(f"{path}: Expected object, got {type(value).__name__}")
            return value
        
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional_properties = schema.get("additionalProperties", True)
        
        result = {}
        
        # Check required fields
        for field in required:
            if field not in value:
                if self.strictness == StrictnessLevel.STRICT:
                    errors.append(f"{path}.{field}: Required field missing")
                else:
                    # Try to provide default
                    field_schema = properties.get(field, {})
                    default = field_schema.get("default")
                    if default is not None:
                        result[field] = default
                    else:
                        result[field] = self._get_default_for_type(
                            field_schema.get("type", "string")
                        )
        
        # Validate each field
        for field, field_value in value.items():
            field_path = f"{path}.{field}" if path else field
            
            if field in properties:
                # Validate against field schema
                field_schema = properties[field]
                validated_value = self._validate_value(
                    field_value, field_schema, field_path, errors
                )
                result[field] = validated_value
            elif additional_properties:
                # Allow additional properties
                result[field] = field_value
            elif self.strictness == StrictnessLevel.STRICT:
                errors.append(f"{field_path}: Unexpected field")
        
        return result
    
    def _validate_array(self, 
                       value: Any, 
                       schema: Dict[str, Any], 
                       path: str,
                       errors: List[str]) -> List[Any]:
        """Validate array value."""
        if not isinstance(value, list):
            if self.strictness == StrictnessLevel.PERMISSIVE:
                return []
            errors.append(f"{path}: Expected array, got {type(value).__name__}")
            return value
        
        items_schema = schema.get("items", {})
        result = []
        
        for i, item in enumerate(value):
            item_path = f"{path}[{i}]"
            validated_item = self._validate_value(
                item, items_schema, item_path, errors
            )
            result.append(validated_item)
        
        return result
    
    def _validate_primitive(self, 
                           value: Any, 
                           schema: Dict[str, Any], 
                           path: str,
                           errors: List[str]) -> Any:
        """Validate primitive value."""
        # Enum validation
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}: Value {value} not in enum {schema['enum']}")
            return value
        
        # Range validation for numbers
        if schema.get("type") in ["number", "integer"]:
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path}: Value {value} below minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path}: Value {value} above maximum {schema['maximum']}")
        
        # String length validation
        if schema.get("type") == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{path}: String too short (min {schema['minLength']})")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{path}: String too long (max {schema['maxLength']})")
        
        return value
    
    def _get_json_type(self, value: Any) -> str:
        """Get JSON Schema type for a value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"  # Fallback
    
    def _coerce_type(self, value: Any, target_type: str) -> Any:
        """Attempt to coerce value to target type."""
        handler = self._coercion_handlers.get(target_type)
        if handler:
            try:
                return handler(value)
            except (ValueError, TypeError):
                return None
        return None
    
    def _setup_coercion_handlers(self) -> Dict[str, Callable]:
        """Setup type coercion handlers."""
        return {
            "string": lambda x: str(x) if x is not None else "",
            "number": lambda x: float(x) if x is not None else 0.0,
            "integer": lambda x: int(float(x)) if x is not None else 0,
            "boolean": lambda x: bool(x) if x is not None else False,
            "array": lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, dict)) else [],
            "object": lambda x: dict(x) if isinstance(x, dict) else {},
            "null": lambda x: None
        }
    
    def _get_default_for_type(self, field_type: str) -> Any:
        """Get a default value for a given type."""
        defaults = {
            "string": "",
            "number": 0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {},
            "null": None
        }
        return defaults.get(field_type, None)


class SchemaEvolutionPipeline:
    """
    Pipeline for automatic schema evolution and validation in Scrapy spiders.
    
    Integrates with Scrapy's item pipeline to provide automatic schema
    inference, validation, and evolution.
    """
    
    def __init__(self, 
                 registry: SchemaRegistry,
                 inference_samples: int = 100,
                 validation_strictness: StrictnessLevel = StrictnessLevel.LENIENT,
                 auto_evolve: bool = True,
                 save_invalid: bool = False):
        """
        Initialize schema evolution pipeline.
        
        Args:
            registry: SchemaRegistry instance
            inference_samples: Number of samples to collect before inference
            validation_strictness: Strictness level for validation
            auto_evolve: Automatically evolve schema when new patterns detected
            save_invalid: Whether to save items that fail validation
        """
        self.registry = registry
        self.inference_samples = inference_samples
        self.validation_strictness = validation_strictness
        self.auto_evolve = auto_evolve
        self.save_invalid = save_invalid
        
        self._samples: List[Dict[str, Any]] = []
        self._stats = {
            "processed": 0,
            "valid": 0,
            "invalid": 0,
            "evolved": 0,
            "migrated": 0
        }
    
    def process_item(self, item: Dict[str, Any], spider) -> Dict[str, Any]:
        """
        Process item through schema validation and evolution.
        
        Args:
            item: Scrapy item to process
            spider: Spider instance
            
        Returns:
            Processed item
        """
        self._stats["processed"] += 1
        
        # Collect sample for inference
        if len(self._samples) < self.inference_samples:
            self._samples.append(deepcopy(item))
            
            # Evolve schema when we have enough samples
            if len(self._samples) == self.inference_samples and self.auto_evolve:
                self._evolve_schema()
        
        # Validate against current schema
        current_version = self.registry.current_version
        if current_version:
            is_valid, errors = self.registry.validate_data(
                item, 
                strictness=self.validation_strictness
            )
            
            if is_valid:
                self._stats["valid"] += 1
                return item
            else:
                self._stats["invalid"] += 1
                logger.warning(f"Item failed validation: {errors}")
                
                if self.save_invalid:
                    # Try to migrate or fix the item
                    migrated = self._attempt_migration(item, errors)
                    if migrated:
                        self._stats["migrated"] += 1
                        return migrated
                
                if self.validation_strictness == StrictnessLevel.STRICT:
                    from vex.exceptions import DropItem
                    raise DropItem(f"Item failed schema validation: {errors}")
        
        return item
    
    def _evolve_schema(self) -> None:
        """Evolve schema based on collected samples."""
        if not self._samples:
            return
        
        try:
            new_version = self.registry.evolve_schema(self._samples)
            self._stats["evolved"] += 1
            logger.info(f"Schema evolved to version {new_version.version}")
            
            # Clear samples after evolution
            self._samples.clear()
            
        except Exception as e:
            logger.error(f"Schema evolution failed: {e}")
    
    def _attempt_migration(self, 
                          item: Dict[str, Any], 
                          errors: List[str]) -> Optional[Dict[str, Any]]:
        """Attempt to migrate item to current schema version."""
        # Simple migration: try to add missing fields with defaults
        current = self.registry.current_version
        if not current:
            return None
        
        migrated = deepcopy(item)
        schema = current.schema
        properties = schema.get("properties", {})
        
        for error in errors:
            if "Required field missing" in error:
                # Extract field name from error
                parts = error.split(".")
                if len(parts) >= 2:
                    field_name = parts[-1].split(":")[0]
                    if field_name in properties:
                        field_schema = properties[field_name]
                        default = field_schema.get("default")
                        if default is not None:
                            migrated[field_name] = default
                        else:
                            field_type = field_schema.get("type", "string")
                            migrated[field_name] = self._get_default_for_type(field_type)
        
        # Validate again
        is_valid, _ = self.registry.validate_data(
            migrated, 
            strictness=self.validation_strictness
        )
        
        return migrated if is_valid else None
    
    def _get_default_for_type(self, field_type: str) -> Any:
        """Get a default value for a given type."""
        defaults = {
            "string": "",
            "number": 0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {},
            "null": None
        }
        return defaults.get(field_type, None)
    
    def close_spider(self, spider) -> None:
        """Called when spider closes."""
        # Final schema evolution if we have remaining samples
        if self._samples and self.auto_evolve:
            self._evolve_schema()
        
        # Log statistics
        logger.info(
            f"Schema evolution pipeline stats: "
            f"processed={self._stats['processed']}, "
            f"valid={self._stats['valid']}, "
            f"invalid={self._stats['invalid']}, "
            f"evolved={self._stats['evolved']}, "
            f"migrated={self._stats['migrated']}"
        )


# Convenience functions for quick usage
def infer_schema_from_items(items: List[Dict[str, Any]], 
                           **kwargs) -> Dict[str, Any]:
    """
    Quick function to infer schema from items.
    
    Args:
        items: List of items to infer schema from
        **kwargs: Additional arguments for SchemaInference
        
    Returns:
        JSON Schema dictionary
    """
    inference = SchemaInference(**kwargs)
    return inference.infer_from_samples(items)


def validate_item(item: Dict[str, Any], 
                 schema: Dict[str, Any],
                 strictness: StrictnessLevel = StrictnessLevel.STRICT) -> Tuple[bool, List[str]]:
    """
    Quick function to validate an item against a schema.
    
    Args:
        item: Item to validate
        schema: JSON Schema to validate against
        strictness: Validation strictness level
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    validator = DataValidator(schema, strictness)
    return validator.validate(item)


def create_schema_registry(storage_path: Optional[str] = None, 
                          **kwargs) -> SchemaRegistry:
    """
    Quick function to create a schema registry.
    
    Args:
        storage_path: Path to store schemas
        **kwargs: Additional arguments for SchemaRegistry
        
    Returns:
        SchemaRegistry instance
    """
    return SchemaRegistry(storage_path=storage_path, **kwargs)