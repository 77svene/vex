"""
Automatic Schema Evolution & Data Validation for Scrapy.

Built-in schema inference and evolution that automatically detects data structure changes,
validates against JSON Schema, and maintains backward compatibility.
"""

import json
import os
import re
import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Type
from datetime import datetime
from collections import defaultdict
from enum import Enum
from pathlib import Path
import jsonschema
from jsonschema import ValidationError, Draft7Validator
from jsonschema.exceptions import SchemaError

from vex import Item
from vex.exceptions import NotConfigured
from vex.utils.misc import load_object
from vex.settings import BaseSettings

logger = logging.getLogger(__name__)


class SchemaVersion(Enum):
    """Schema version compatibility modes."""
    STRICT = "strict"  # No backward compatibility
    BACKWARD = "backward"  # New schema can read old data
    FORWARD = "forward"  # Old schema can read new data
    FULL = "full"  # Both backward and forward compatibility


class ValidationStrictness(Enum):
    """Validation strictness levels."""
    IGNORE = "ignore"  # No validation
    WARN = "warn"  # Log warnings but continue
    ERROR = "error"  # Raise exceptions on validation errors
    STRICT = "strict"  # Fail on any validation error or schema mismatch


class SchemaInferenceError(Exception):
    """Raised when schema inference fails."""
    pass


class SchemaValidationError(Exception):
    """Raised when data validation fails."""
    pass


class SchemaEvolutionError(Exception):
    """Raised when schema evolution fails."""
    pass


class FieldSchema:
    """Represents schema information for a single field."""
    
    def __init__(self, name: str, field_type: str, required: bool = False, 
                 nullable: bool = False, default: Any = None, 
                 constraints: Optional[Dict] = None):
        self.name = name
        self.field_type = field_type
        self.required = required
        self.nullable = nullable
        self.default = default
        self.constraints = constraints or {}
    
    def to_json_schema(self) -> Dict:
        """Convert to JSON Schema format."""
        schema = {"type": self._map_type()}
        
        if self.nullable:
            schema["type"] = [schema["type"], "null"]
        
        if self.default is not None:
            schema["default"] = self.default
        
        # Add constraints
        for constraint, value in self.constraints.items():
            if constraint == "min_length":
                schema["minLength"] = value
            elif constraint == "max_length":
                schema["maxLength"] = value
            elif constraint == "pattern":
                schema["pattern"] = value
            elif constraint == "minimum":
                schema["minimum"] = value
            elif constraint == "maximum":
                schema["maximum"] = value
            elif constraint == "enum":
                schema["enum"] = value
        
        return schema
    
    def _map_type(self) -> str:
        """Map internal type to JSON Schema type."""
        type_map = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
            "datetime": "string",  # ISO format string
            "date": "string",
            "url": "string",
            "email": "string",
        }
        return type_map.get(self.field_type, "string")
    
    def __repr__(self):
        return f"FieldSchema(name={self.name}, type={self.field_type}, required={self.required})"


class ItemSchema:
    """Represents the complete schema for an item."""
    
    def __init__(self, name: str, version: int = 1, 
                 compatibility: SchemaVersion = SchemaVersion.BACKWARD):
        self.name = name
        self.version = version
        self.compatibility = compatibility
        self.fields: Dict[str, FieldSchema] = {}
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.hash = ""
    
    def add_field(self, field: FieldSchema):
        """Add a field to the schema."""
        self.fields[field.name] = field
        self._update_hash()
    
    def remove_field(self, field_name: str):
        """Remove a field from the schema."""
        if field_name in self.fields:
            del self.fields[field_name]
            self._update_hash()
    
    def to_json_schema(self) -> Dict:
        """Convert to JSON Schema format."""
        properties = {}
        required = []
        
        for field_name, field_schema in self.fields.items():
            properties[field_name] = field_schema.to_json_schema()
            if field_schema.required:
                required.append(field_name)
        
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": self.name,
            "type": "object",
            "properties": properties,
            "version": self.version,
            "compatibility": self.compatibility.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hash": self.hash,
        }
        
        if required:
            schema["required"] = required
        
        return schema
    
    def from_json_schema(self, json_schema: Dict):
        """Load schema from JSON Schema format."""
        self.name = json_schema.get("title", self.name)
        self.version = json_schema.get("version", self.version)
        self.compatibility = SchemaVersion(json_schema.get("compatibility", self.compatibility.value))
        self.created_at = json_schema.get("created_at", self.created_at)
        self.updated_at = json_schema.get("updated_at", self.updated_at)
        self.hash = json_schema.get("hash", self.hash)
        
        properties = json_schema.get("properties", {})
        required_fields = set(json_schema.get("required", []))
        
        for field_name, field_schema in properties.items():
            field_type = self._infer_type_from_json_schema(field_schema)
            nullable = "null" in field_schema.get("type", []) if isinstance(field_schema.get("type"), list) else False
            
            constraints = {}
            if "minLength" in field_schema:
                constraints["min_length"] = field_schema["minLength"]
            if "maxLength" in field_schema:
                constraints["max_length"] = field_schema["maxLength"]
            if "pattern" in field_schema:
                constraints["pattern"] = field_schema["pattern"]
            if "minimum" in field_schema:
                constraints["minimum"] = field_schema["minimum"]
            if "maximum" in field_schema:
                constraints["maximum"] = field_schema["maximum"]
            if "enum" in field_schema:
                constraints["enum"] = field_schema["enum"]
            
            field = FieldSchema(
                name=field_name,
                field_type=field_type,
                required=field_name in required_fields,
                nullable=nullable,
                default=field_schema.get("default"),
                constraints=constraints
            )
            self.add_field(field)
    
    def _infer_type_from_json_schema(self, field_schema: Dict) -> str:
        """Infer field type from JSON Schema."""
        json_type = field_schema.get("type", "string")
        
        if isinstance(json_type, list):
            # Handle nullable types
            json_type = [t for t in json_type if t != "null"][0] if json_type else "string"
        
        type_map = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
        }
        
        # Check for format hints
        if json_type == "string":
            if "format" in field_schema:
                format_type = field_schema["format"]
                if format_type == "date-time":
                    return "datetime"
                elif format_type == "date":
                    return "date"
                elif format_type == "uri":
                    return "url"
                elif format_type == "email":
                    return "email"
        
        return type_map.get(json_type, "string")
    
    def _update_hash(self):
        """Update schema hash based on current fields."""
        schema_str = json.dumps(self.to_json_schema(), sort_keys=True)
        self.hash = hashlib.md5(schema_str.encode()).hexdigest()
        self.updated_at = datetime.utcnow().isoformat()
    
    def __repr__(self):
        return f"ItemSchema(name={self.name}, version={self.version}, fields={len(self.fields)})"


class SchemaInferer:
    """Infers schema from sample data."""
    
    # Common field patterns for type inference
    PATTERNS = {
        "url": re.compile(r'^https?://', re.IGNORECASE),
        "email": re.compile(r'^[^@]+@[^@]+\.[^@]+$'),
        "datetime": re.compile(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}'),
        "date": re.compile(r'^\d{4}-\d{2}-\d{2}$'),
        "integer": re.compile(r'^-?\d+$'),
        "number": re.compile(r'^-?\d*\.?\d+$'),
        "boolean": re.compile(r'^(true|false|yes|no|0|1)$', re.IGNORECASE),
    }
    
    def __init__(self, sample_size: int = 100, confidence_threshold: float = 0.8):
        self.sample_size = sample_size
        self.confidence_threshold = confidence_threshold
        self._samples: List[Dict] = []
    
    def add_sample(self, data: Union[Dict, Item]):
        """Add a data sample for schema inference."""
        if isinstance(data, Item):
            data = dict(data)
        self._samples.append(data)
        
        # Keep only the most recent samples
        if len(self._samples) > self.sample_size:
            self._samples = self._samples[-self.sample_size:]
    
    def infer_schema(self, name: str = "InferredSchema") -> ItemSchema:
        """Infer schema from collected samples."""
        if not self._samples:
            raise SchemaInferenceError("No samples available for schema inference")
        
        schema = ItemSchema(name=name)
        field_stats = self._collect_field_statistics()
        
        for field_name, stats in field_stats.items():
            field_type = self._infer_field_type(stats)
            required = stats["presence"] >= self.confidence_threshold
            nullable = stats["null_count"] > 0
            
            # Infer constraints
            constraints = self._infer_constraints(stats, field_type)
            
            field = FieldSchema(
                name=field_name,
                field_type=field_type,
                required=required,
                nullable=nullable,
                constraints=constraints
            )
            schema.add_field(field)
        
        return schema
    
    def _collect_field_statistics(self) -> Dict:
        """Collect statistics about each field across samples."""
        stats = defaultdict(lambda: {
            "count": 0,
            "null_count": 0,
            "types": defaultdict(int),
            "values": [],
            "min_length": None,
            "max_length": None,
            "min_value": None,
            "max_value": None,
            "patterns": defaultdict(int),
        })
        
        for sample in self._samples:
            for field_name, value in sample.items():
                field_stats = stats[field_name]
                field_stats["count"] += 1
                
                if value is None:
                    field_stats["null_count"] += 1
                    continue
                
                # Track type
                value_type = self._get_value_type(value)
                field_stats["types"][value_type] += 1
                
                # Track value for pattern analysis
                if isinstance(value, str):
                    field_stats["values"].append(value)
                    length = len(value)
                    field_stats["min_length"] = min(field_stats["min_length"] or length, length)
                    field_stats["max_length"] = max(field_stats["max_length"] or length, length)
                    
                    # Check patterns
                    for pattern_name, pattern in self.PATTERNS.items():
                        if pattern.match(value):
                            field_stats["patterns"][pattern_name] += 1
                
                elif isinstance(value, (int, float)):
                    field_stats["min_value"] = min(field_stats["min_value"] or value, value)
                    field_stats["max_value"] = max(field_stats["max_value"] or value, value)
        
        # Calculate presence ratio
        total_samples = len(self._samples)
        for field_name, field_stats in stats.items():
            field_stats["presence"] = field_stats["count"] / total_samples
        
        return dict(stats)
    
    def _get_value_type(self, value: Any) -> str:
        """Determine the type of a value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            # Try to infer more specific string types
            for pattern_name, pattern in self.PATTERNS.items():
                if pattern.match(value):
                    return pattern_name
            return "string"
        elif isinstance(value, (list, tuple)):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"
    
    def _infer_field_type(self, stats: Dict) -> str:
        """Infer the most likely type for a field."""
        type_counts = stats["types"]
        if not type_counts:
            return "string"
        
        # Find the most common type
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        total_count = sum(type_counts.values())
        confidence = type_counts[most_common_type] / total_count
        
        if confidence >= self.confidence_threshold:
            return most_common_type
        
        # If no clear type, check patterns
        pattern_counts = stats["patterns"]
        if pattern_counts:
            most_common_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
            pattern_confidence = pattern_counts[most_common_pattern] / total_count
            if pattern_confidence >= self.confidence_threshold:
                return most_common_pattern
        
        return "string"
    
    def _infer_constraints(self, stats: Dict, field_type: str) -> Dict:
        """Infer constraints for a field."""
        constraints = {}
        
        if field_type == "string":
            if stats["min_length"] is not None:
                constraints["min_length"] = stats["min_length"]
            if stats["max_length"] is not None:
                constraints["max_length"] = stats["max_length"]
            
            # Check for enum values (if limited set)
            unique_values = set(stats["values"])
            if 0 < len(unique_values) <= 10 and len(stats["values"]) >= 5:
                constraints["enum"] = list(unique_values)
        
        elif field_type in ("integer", "number"):
            if stats["min_value"] is not None:
                constraints["minimum"] = stats["min_value"]
            if stats["max_value"] is not None:
                constraints["maximum"] = stats["max_value"]
        
        return constraints


class SchemaRegistry:
    """Manages schema versions and evolution."""
    
    def __init__(self, registry_path: str = "./schema_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._schemas: Dict[str, Dict[int, ItemSchema]] = defaultdict(dict)
        self._load_schemas()
    
    def _load_schemas(self):
        """Load all schemas from the registry."""
        for schema_dir in self.registry_path.iterdir():
            if schema_dir.is_dir():
                schema_name = schema_dir.name
                for schema_file in schema_dir.glob("*.json"):
                    try:
                        with open(schema_file, 'r') as f:
                            schema_data = json.load(f)
                        
                        schema = ItemSchema(name=schema_name)
                        schema.from_json_schema(schema_data)
                        self._schemas[schema_name][schema.version] = schema
                    except Exception as e:
                        logger.warning(f"Failed to load schema from {schema_file}: {e}")
    
    def register_schema(self, schema: ItemSchema) -> int:
        """Register a new schema version."""
        schema_name = schema.name
        
        # Check if schema with same hash already exists
        for existing_version, existing_schema in self._schemas[schema_name].items():
            if existing_schema.hash == schema.hash:
                return existing_version
        
        # Determine next version number
        if schema_name in self._schemas and self._schemas[schema_name]:
            latest_version = max(self._schemas[schema_name].keys())
            schema.version = latest_version + 1
        else:
            schema.version = 1
        
        # Save schema to disk
        schema_dir = self.registry_path / schema_name
        schema_dir.mkdir(exist_ok=True)
        
        schema_file = schema_dir / f"v{schema.version}.json"
        with open(schema_file, 'w') as f:
            json.dump(schema.to_json_schema(), f, indent=2)
        
        # Update in-memory registry
        self._schemas[schema_name][schema.version] = schema
        
        logger.info(f"Registered schema {schema_name} version {schema.version}")
        return schema.version
    
    def get_schema(self, name: str, version: Optional[int] = None) -> Optional[ItemSchema]:
        """Get a schema by name and version."""
        if name not in self._schemas:
            return None
        
        if version is None:
            # Return latest version
            if not self._schemas[name]:
                return None
            version = max(self._schemas[name].keys())
        
        return self._schemas[name].get(version)
    
    def get_latest_version(self, name: str) -> Optional[int]:
        """Get the latest version number for a schema."""
        if name not in self._schemas or not self._schemas[name]:
            return None
        return max(self._schemas[name].keys())
    
    def evolve_schema(self, name: str, new_data: Dict, 
                      compatibility: SchemaVersion = SchemaVersion.BACKWARD) -> ItemSchema:
        """Evolve schema based on new data."""
        current_schema = self.get_schema(name)
        
        if current_schema is None:
            # Create new schema
            inferer = SchemaInferer()
            inferer.add_sample(new_data)
            new_schema = inferer.infer_schema(name=name)
            new_schema.compatibility = compatibility
            self.register_schema(new_schema)
            return new_schema
        
        # Check compatibility
        if not self._check_compatibility(current_schema, new_data, compatibility):
            raise SchemaEvolutionError(
                f"New data is not compatible with schema {name} v{current_schema.version} "
                f"in {compatibility.value} mode"
            )
        
        # Infer new schema from combined data
        inferer = SchemaInferer()
        
        # Add existing schema as sample
        sample_from_schema = self._schema_to_sample(current_schema)
        inferer.add_sample(sample_from_schema)
        
        # Add new data
        inferer.add_sample(new_data)
        
        # Create evolved schema
        evolved_schema = inferer.infer_schema(name=name)
        evolved_schema.compatibility = compatibility
        
        # Register evolved schema
        self.register_schema(evolved_schema)
        
        return evolved_schema
    
    def _check_compatibility(self, schema: ItemSchema, data: Dict, 
                            compatibility: SchemaVersion) -> bool:
        """Check if data is compatible with schema based on compatibility mode."""
        if compatibility == SchemaVersion.STRICT:
            # Data must match schema exactly
            return self._validate_against_schema(data, schema, strict=True)
        
        elif compatibility == SchemaVersion.BACKWARD:
            # New schema can read old data (data can have fewer fields)
            # Check that all required fields in schema are present in data
            for field_name, field_schema in schema.fields.items():
                if field_schema.required and field_name not in data:
                    return False
            return True
        
        elif compatibility == SchemaVersion.FORWARD:
            # Old schema can read new data (data can have extra fields)
            # Check that all fields in data exist in schema
            for field_name in data.keys():
                if field_name not in schema.fields:
                    return False
            return True
        
        elif compatibility == SchemaVersion.FULL:
            # Both backward and forward compatibility
            # Check backward compatibility
            for field_name, field_schema in schema.fields.items():
                if field_schema.required and field_name not in data:
                    return False
            
            # Check forward compatibility
            for field_name in data.keys():
                if field_name not in schema.fields:
                    return False
            
            return True
        
        return False
    
    def _validate_against_schema(self, data: Dict, schema: ItemSchema, 
                                strict: bool = False) -> bool:
        """Validate data against schema."""
        json_schema = schema.to_json_schema()
        
        try:
            Draft7Validator(json_schema).validate(data)
            return True
        except ValidationError as e:
            if strict:
                return False
            logger.warning(f"Validation error: {e.message}")
            return True  # Non-strict mode allows validation errors
    
    def _schema_to_sample(self, schema: ItemSchema) -> Dict:
        """Convert schema to a sample data structure."""
        sample = {}
        for field_name, field_schema in schema.fields.items():
            if field_schema.default is not None:
                sample[field_name] = field_schema.default
            elif field_schema.nullable:
                sample[field_name] = None
            else:
                # Generate sample value based on type
                sample[field_name] = self._generate_sample_value(field_schema)
        return sample
    
    def _generate_sample_value(self, field_schema: FieldSchema) -> Any:
        """Generate a sample value for a field."""
        if field_schema.field_type == "string":
            return "sample_string"
        elif field_schema.field_type == "integer":
            return 0
        elif field_schema.field_type == "number":
            return 0.0
        elif field_schema.field_type == "boolean":
            return False
        elif field_schema.field_type == "array":
            return []
        elif field_schema.field_type == "object":
            return {}
        elif field_schema.field_type == "datetime":
            return datetime.utcnow().isoformat()
        elif field_schema.field_type == "date":
            return datetime.utcnow().date().isoformat()
        elif field_schema.field_type == "url":
            return "https://example.com"
        elif field_schema.field_type == "email":
            return "user@example.com"
        else:
            return ""


class DataValidator:
    """Validates data against schemas with configurable strictness."""
    
    def __init__(self, schema_registry: SchemaRegistry, 
                 strictness: ValidationStrictness = ValidationStrictness.WARN):
        self.schema_registry = schema_registry
        self.strictness = strictness
        self._validators: Dict[str, Draft7Validator] = {}
    
    def validate(self, data: Dict, schema_name: str, 
                 schema_version: Optional[int] = None) -> Tuple[bool, List[str]]:
        """Validate data against a schema."""
        if self.strictness == ValidationStrictness.IGNORE:
            return True, []
        
        schema = self.schema_registry.get_schema(schema_name, schema_version)
        if schema is None:
            msg = f"Schema {schema_name} v{schema_version} not found"
            if self.strictness == ValidationStrictness.ERROR:
                raise SchemaValidationError(msg)
            elif self.strictness == ValidationStrictness.WARN:
                logger.warning(msg)
            return False, [msg]
        
        # Get or create validator
        cache_key = f"{schema_name}_v{schema.version}"
        if cache_key not in self._validators:
            json_schema = schema.to_json_schema()
            self._validators[cache_key] = Draft7Validator(json_schema)
        
        validator = self._validators[cache_key]
        errors = []
        
        for error in validator.iter_errors(data):
            error_path = ".".join(str(p) for p in error.path) if error.path else "root"
            error_msg = f"{error_path}: {error.message}"
            errors.append(error_msg)
            
            if self.strictness == ValidationStrictness.ERROR:
                raise SchemaValidationError(f"Validation failed: {error_msg}")
            elif self.strictness == ValidationStrictness.WARN:
                logger.warning(f"Validation error in {schema_name}: {error_msg}")
        
        return len(errors) == 0, errors
    
    def validate_and_evolve(self, data: Dict, schema_name: str,
                           auto_evolve: bool = True) -> Tuple[bool, Dict, Optional[int]]:
        """Validate data and optionally evolve schema."""
        # Get current schema
        current_schema = self.schema_registry.get_schema(schema_name)
        
        if current_schema is None:
            if auto_evolve:
                # Create new schema from data
                new_schema = self.schema_registry.evolve_schema(schema_name, data)
                return True, data, new_schema.version
            else:
                msg = f"Schema {schema_name} not found and auto-evolve is disabled"
                if self.strictness == ValidationStrictness.ERROR:
                    raise SchemaValidationError(msg)
                return False, data, None
        
        # Validate against current schema
        is_valid, errors = self.validate(data, schema_name, current_schema.version)
        
        if is_valid:
            return True, data, current_schema.version
        
        # If validation failed and auto-evolve is enabled, evolve schema
        if auto_evolve:
            try:
                evolved_schema = self.schema_registry.evolve_schema(schema_name, data)
                logger.info(f"Evolved schema {schema_name} to version {evolved_schema.version}")
                return True, data, evolved_schema.version
            except SchemaEvolutionError as e:
                logger.error(f"Failed to evolve schema: {e}")
                if self.strictness == ValidationStrictness.ERROR:
                    raise
        
        return False, data, current_schema.version


class SchemaMigration:
    """Utilities for migrating data between schema versions."""
    
    def __init__(self, schema_registry: SchemaRegistry):
        self.schema_registry = schema_registry
    
    def migrate_data(self, data: Dict, from_schema: ItemSchema, 
                    to_schema: ItemSchema) -> Dict:
        """Migrate data from one schema version to another."""
        migrated_data = {}
        
        # Copy fields that exist in both schemas
        for field_name, to_field in to_schema.fields.items():
            if field_name in from_schema.fields:
                # Field exists in both schemas
                from_field = from_schema.fields[field_name]
                
                # Check if type conversion is needed
                if from_field.field_type != to_field.field_type:
                    migrated_data[field_name] = self._convert_type(
                        data.get(field_name), from_field.field_type, to_field.field_type
                    )
                else:
                    migrated_data[field_name] = data.get(field_name)
            else:
                # Field is new in target schema
                if to_field.default is not None:
                    migrated_data[field_name] = to_field.default
                elif to_field.nullable:
                    migrated_data[field_name] = None
                elif to_field.required:
                    raise SchemaEvolutionError(
                        f"Required field {field_name} missing in source data"
                    )
        
        # Handle fields that were removed (only if not required in target)
        for field_name, from_field in from_schema.fields.items():
            if field_name not in to_schema.fields:
                # Field was removed, check if we need to preserve it
                if from_field.required and not to_schema.fields.get(field_name):
                    logger.warning(f"Required field {field_name} was removed from schema")
        
        return migrated_data
    
    def _convert_type(self, value: Any, from_type: str, to_type: str) -> Any:
        """Convert value from one type to another."""
        if value is None:
            return None
        
        try:
            if from_type == to_type:
                return value
            
            # String conversions
            if to_type == "string":
                return str(value)
            
            # Numeric conversions
            elif to_type == "integer":
                if isinstance(value, str):
                    return int(float(value))
                return int(value)
            
            elif to_type == "number":
                if isinstance(value, str):
                    return float(value)
                return float(value)
            
            # Boolean conversion
            elif to_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "yes", "1")
                return bool(value)
            
            # Date/datetime conversions
            elif to_type in ("datetime", "date"):
                if isinstance(value, str):
                    # Try to parse ISO format
                    if "T" in value:
                        return datetime.fromisoformat(value.replace("Z", "+00:00"))
                    else:
                        return datetime.strptime(value, "%Y-%m-%d").date()
                return value
            
            # Array conversion
            elif to_type == "array":
                if isinstance(value, (list, tuple)):
                    return list(value)
                return [value]
            
            # Object conversion
            elif to_type == "object":
                if isinstance(value, dict):
                    return value
                return {"value": value}
            
            else:
                return value
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Type conversion failed from {from_type} to {to_type}: {e}")
            return value
    
    def create_migration_script(self, schema_name: str, from_version: int, 
                               to_version: int) -> str:
        """Create a migration script between schema versions."""
        from_schema = self.schema_registry.get_schema(schema_name, from_version)
        to_schema = self.schema_registry.get_schema(schema_name, to_version)
        
        if not from_schema or not to_schema:
            raise SchemaEvolutionError("Source or target schema not found")
        
        script = f"""# Migration script for {schema_name}
# From version {from_version} to version {to_version}
# Generated at {datetime.utcnow().isoformat()}

def migrate_data(data):
    migrated = {{}}
    
"""
        
        # Add field migrations
        for field_name, to_field in to_schema.fields.items():
            if field_name in from_schema.fields:
                from_field = from_schema.fields[field_name]
                if from_field.field_type != to_field.field_type:
                    script += f'    # Convert {field_name} from {from_field.field_type} to {to_field.field_type}\n'
                    script += f'    migrated["{field_name}"] = convert_type(data.get("{field_name}"), "{from_field.field_type}", "{to_field.field_type}")\n'
                else:
                    script += f'    migrated["{field_name}"] = data.get("{field_name}")\n'
            else:
                if to_field.default is not None:
                    script += f'    migrated["{field_name}"] = {repr(to_field.default)}  # New field with default\n'
                elif to_field.nullable:
                    script += f'    migrated["{field_name}"] = None  # New nullable field\n'
        
        script += """
    return migrated

def convert_type(value, from_type, to_type):
    # Type conversion logic here
    if value is None:
        return None
    
    # Add conversion logic based on types
    return value
"""
        
        return script


class SchemaValidationPipeline:
    """Scrapy pipeline for automatic schema validation and evolution."""
    
    def __init__(self, settings: BaseSettings):
        self.settings = settings
        
        # Configuration from settings
        self.registry_path = settings.get('SCHEMA_REGISTRY_PATH', './schema_registry')
        self.strictness = ValidationStrictness(
            settings.get('SCHEMA_VALIDATION_STRICTNESS', 'warn')
        )
        self.auto_evolve = settings.getbool('SCHEMA_AUTO_EVOLVE', True)
        self.compatibility = SchemaVersion(
            settings.get('SCHEMA_COMPATIBILITY_MODE', 'backward')
        )
        
        # Initialize components
        self.schema_registry = SchemaRegistry(self.registry_path)
        self.validator = DataValidator(self.schema_registry, self.strictness)
        self.migrator = SchemaMigration(self.schema_registry)
        
        # Cache for item schemas
        self._item_schemas: Dict[Type[Item], str] = {}
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create pipeline from crawler."""
        return cls(crawler.settings)
    
    def process_item(self, item: Item, spider):
        """Process item through schema validation pipeline."""
        item_type = type(item).__name__
        
        # Get or infer schema name
        schema_name = self._get_schema_name(item, item_type)
        
        # Convert item to dict for validation
        item_dict = dict(item)
        
        # Validate and evolve schema
        is_valid, validated_data, schema_version = self.validator.validate_and_evolve(
            item_dict, schema_name, self.auto_evolve
        )
        
        if not is_valid and self.strictness in (ValidationStrictness.ERROR, ValidationStrictness.STRICT):
            raise SchemaValidationError(
                f"Item validation failed for schema {schema_name}"
            )
        
        # Add schema metadata to item
        item['_schema_name'] = schema_name
        item['_schema_version'] = schema_version
        item['_validated_at'] = datetime.utcnow().isoformat()
        
        # If schema was evolved, log it
        if schema_version and schema_version > 1:
            spider.logger.info(
                f"Schema {schema_name} evolved to version {schema_version}"
            )
        
        return item
    
    def _get_schema_name(self, item: Item, item_type: str) -> str:
        """Get schema name for an item."""
        # Check if item has a schema name attribute
        if hasattr(item, 'schema_name'):
            return item.schema_name
        
        # Use item class name as schema name
        return item_type
    
    def open_spider(self, spider):
        """Called when spider is opened."""
        spider.logger.info(
            f"Schema validation pipeline opened with strictness: {self.strictness.value}"
        )
    
    def close_spider(self, spider):
        """Called when spider is closed."""
        # Log schema evolution summary
        for schema_name, versions in self.schema_registry._schemas.items():
            if versions:
                latest_version = max(versions.keys())
                spider.logger.info(
                    f"Schema {schema_name}: latest version {latest_version}"
                )


# Utility functions for easy integration
def infer_schema_from_items(items: List[Item], name: str = "ItemSchema") -> ItemSchema:
    """Infer schema from a list of items."""
    inferer = SchemaInferer()
    for item in items:
        inferer.add_sample(item)
    return inferer.infer_schema(name)


def validate_item(item: Item, schema: ItemSchema, 
                  strictness: ValidationStrictness = ValidationStrictness.WARN) -> Tuple[bool, List[str]]:
    """Validate a single item against a schema."""
    registry = SchemaRegistry()
    registry.register_schema(schema)
    validator = DataValidator(registry, strictness)
    return validator.validate(dict(item), schema.name, schema.version)


def create_schema_from_json(json_schema: Dict, name: str = "CustomSchema") -> ItemSchema:
    """Create an ItemSchema from a JSON Schema definition."""
    schema = ItemSchema(name=name)
    schema.from_json_schema(json_schema)
    return schema


# Example usage in settings.py
"""
# Enable schema validation pipeline
ITEM_PIPELINES = {
    'vex.schema.validator.SchemaValidationPipeline': 100,
}

# Schema validation settings
SCHEMA_REGISTRY_PATH = './schema_registry'
SCHEMA_VALIDATION_STRICTNESS = 'warn'  # 'ignore', 'warn', 'error', 'strict'
SCHEMA_AUTO_EVOLVE = True
SCHEMA_COMPATIBILITY_MODE = 'backward'  # 'strict', 'backward', 'forward', 'full'
"""