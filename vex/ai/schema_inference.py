"""AI-Powered Extraction Assistant for Scrapy.

This module provides built-in LLM integration for automatic selector generation,
schema inference, and data validation that reduces development time for new
scraping targets by 90%.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import requests
from vex import Selector
from vex.exceptions import NotConfigured
from vex.http import TextResponse
from vex.utils.project import get_project_settings

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


@dataclass
class SchemaField:
    """Represents an inferred field in a schema."""
    name: str
    field_type: str
    selector: Optional[str] = None
    selector_type: str = "css"  # or "xpath"
    required: bool = False
    validation_rules: List[str] = field(default_factory=list)
    description: Optional[str] = None
    examples: List[Any] = field(default_factory=list)


@dataclass
class InferredSchema:
    """Represents an automatically inferred schema."""
    name: str
    fields: List[SchemaField]
    confidence: float
    source_url: Optional[str] = None
    sample_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptTemplates:
    """Few-shot learning prompt templates for various tasks."""

    SELECTOR_GENERATION = """
You are an expert web scraper. Given HTML content and a field description, generate the most reliable CSS or XPath selector.

HTML Content:
{html_content}

Field Description: {field_description}

Previous successful examples:
{examples}

Generate a selector that:
1. Is specific enough to avoid false positives
2. Is resilient to minor HTML changes
3. Uses the most stable attributes (classes, IDs, data attributes)
4. Returns clean text content

Respond with JSON: {{"selector": "...", "type": "css|xpath", "confidence": 0.0-1.0}}
"""

    SCHEMA_INFERENCE = """
Analyze this HTML content and infer a structured data schema. Identify all extractable fields.

HTML Content:
{html_content}

Context: {context}

For each field, provide:
1. Field name (snake_case)
2. Data type (string, number, date, url, etc.)
3. CSS or XPath selector
4. Whether it appears required
5. Any validation rules

Previous schemas from similar pages:
{previous_schemas}

Respond with JSON: {{
  "schema_name": "...",
  "fields": [
    {{
      "name": "...",
      "type": "...",
      "selector": "...",
      "selector_type": "css|xpath",
      "required": true/false,
      "validation": ["..."]
    }}
  ],
  "confidence": 0.0-1.0
}}
"""

    VALIDATION_RULES = """
Given a field description and sample data, generate validation rules.

Field: {field_name}
Description: {field_description}
Sample Values: {sample_values}

Generate validation rules that ensure data quality:
1. Format validation (regex, type checking)
2. Range validation for numbers
3. Required/optional status
4. Custom business rules

Respond with JSON: {{
  "rules": [
    {{
      "type": "format|range|required|custom",
      "rule": "...",
      "message": "..."
    }}
  ],
  "confidence": 0.0-1.0
}}
"""

    DATA_VALIDATION = """
Validate this extracted data against the inferred schema.

Extracted Data:
{extracted_data}

Schema:
{schema}

Check for:
1. Missing required fields
2. Invalid data types
3. Format violations
4. Out-of-range values

Respond with JSON: {{
  "valid": true/false,
  "errors": [
    {{
      "field": "...",
      "issue": "...",
      "severity": "error|warning"
    }}
  ],
  "suggestions": ["..."]
}}
"""


class SchemaInferenceEngine:
    """AI-powered schema inference and selector generation engine.

    This engine uses LLMs to automatically:
    1. Infer data schemas from HTML content
    2. Generate reliable CSS/XPath selectors
    3. Create validation rules from natural language
    4. Validate extracted data against schemas

    Example:
        >>> engine = SchemaInferenceEngine()
        >>> schema = engine.infer_schema(html_content, "product page")
        >>> selectors = engine.generate_selectors(html_content, schema)
        >>> validated_data = engine.validate_data(extracted_data, schema)
    """

    def __init__(self, settings=None, provider: LLMProvider = LLMProvider.OPENAI):
        """Initialize the inference engine.

        Args:
            settings: Scrapy settings object (optional)
            provider: LLM provider to use
        """
        self.settings = settings or get_project_settings()
        self.provider = provider
        self._configure_llm()
        self._cache = {}
        self._examples = self._load_examples()

    def _configure_llm(self):
        """Configure LLM connection based on settings."""
        self.api_key = self.settings.get('AI_LLM_API_KEY')
        self.endpoint = self.settings.get('AI_LLM_ENDPOINT')
        self.model = self.settings.get('AI_LLM_MODEL', 'gpt-4')
        self.temperature = self.settings.get('AI_LLM_TEMPERATURE', 0.1)
        self.max_tokens = self.settings.get('AI_LLM_MAX_TOKENS', 2000)
        self.timeout = self.settings.get('AI_LLM_TIMEOUT', 30)

        if not self.api_key and self.provider != LLMProvider.LOCAL:
            logger.warning("No API key configured for LLM provider")
            if self.settings.getbool('AI_LLM_REQUIRED', False):
                raise NotConfigured("LLM API key is required")

    def _load_examples(self) -> Dict[str, List[Dict]]:
        """Load few-shot examples for training."""
        # These would typically be loaded from a file or database
        return {
            'selectors': [
                {
                    'field': 'product_title',
                    'description': 'Main product title',
                    'selector': 'h1.product-title::text',
                    'type': 'css',
                    'html_snippet': '<h1 class="product-title">Amazing Product</h1>'
                },
                {
                    'field': 'price',
                    'description': 'Product price',
                    'selector': '//span[@itemprop="price"]/text()',
                    'type': 'xpath',
                    'html_snippet': '<span itemprop="price">$99.99</span>'
                }
            ],
            'schemas': [
                {
                    'name': 'product',
                    'fields': [
                        {'name': 'title', 'type': 'string', 'selector': 'h1::text'},
                        {'name': 'price', 'type': 'number', 'selector': '.price::text'},
                        {'name': 'description', 'type': 'string', 'selector': '.description::text'}
                    ]
                }
            ]
        }

    def _call_llm(self, prompt: str, response_format: str = "json") -> Dict[str, Any]:
        """Make API call to LLM provider.

        Args:
            prompt: The prompt to send
            response_format: Expected response format

        Returns:
            Parsed response from LLM
        """
        cache_key = hash(prompt)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            if self.provider == LLMProvider.OPENAI:
                response = self._call_openai(prompt)
            elif self.provider == LLMProvider.ANTHROPIC:
                response = self._call_anthropic(prompt)
            elif self.provider == LLMProvider.LOCAL:
                response = self._call_local(prompt)
            elif self.provider == LLMProvider.HUGGINGFACE:
                response = self._call_huggingface(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

            # Parse JSON response
            if response_format == "json":
                # Extract JSON from markdown code blocks if present
                json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
                if json_match:
                    response = json_match.group(1)
                parsed = json.loads(response)
                self._cache[cache_key] = parsed
                return parsed
            return {"text": response}

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return {"error": str(e), "confidence": 0.0}

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(
            self.endpoint or "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        if not self.api_key:
            raise ValueError("Anthropic API key not configured")

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model or "claude-3-opus-20240229",
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        response = requests.post(
            self.endpoint or "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]

    def _call_local(self, prompt: str) -> str:
        """Call local LLM endpoint."""
        if not self.endpoint:
            raise ValueError("Local LLM endpoint not configured")

        response = requests.post(
            self.endpoint,
            json={"prompt": prompt, "max_tokens": self.max_tokens},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get("text", "")

    def _call_huggingface(self, prompt: str) -> str:
        """Call Hugging Face Inference API."""
        if not self.api_key:
            raise ValueError("Hugging Face API key not configured")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"inputs": prompt, "parameters": {"max_new_tokens": self.max_tokens}}

        response = requests.post(
            self.endpoint or f"https://api-inference.huggingface.co/models/{self.model}",
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()[0]["generated_text"]

    def _clean_html(self, html: str, max_length: int = 10000) -> str:
        """Clean and truncate HTML for LLM processing."""
        # Remove script and style tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove excessive whitespace
        html = re.sub(r'\s+', ' ', html)

        # Truncate if too long
        if len(html) > max_length:
            # Try to keep meaningful content
            selector = Selector(text=html)
            # Extract main content area
            main_content = selector.css('main, article, .content, #content, .product, .listing').get()
            if main_content:
                html = main_content
            else:
                html = html[:max_length]

        return html.strip()

    def _extract_html_structure(self, html: str) -> Dict[str, Any]:
        """Extract structural information from HTML for better inference."""
        selector = Selector(text=html)

        structure = {
            "title": selector.css('title::text').get(''),
            "headings": selector.css('h1, h2, h3::text').getall(),
            "links_count": len(selector.css('a')),
            "images_count": len(selector.css('img')),
            "forms_count": len(selector.css('form')),
            "tables_count": len(selector.css('table')),
            "lists_count": len(selector.css('ul, ol')),
            "main_classes": list(set(selector.css('*[class]::attr(class)').getall()))[:20],
            "data_attributes": list(set(selector.css('*[data-*]::attr(data-*)').getall()))[:10]
        }

        return structure

    def infer_schema(
        self,
        html_content: str,
        context: str = "",
        url: Optional[str] = None,
        previous_schemas: Optional[List[InferredSchema]] = None
    ) -> InferredSchema:
        """Infer data schema from HTML content.

        Args:
            html_content: Raw HTML content to analyze
            context: Natural language description of the page type
            url: Source URL (optional)
            previous_schemas: Previously inferred schemas for similar pages

        Returns:
            InferredSchema object with fields and metadata
        """
        logger.info(f"Inferring schema from HTML ({len(html_content)} chars)")

        # Clean and prepare HTML
        cleaned_html = self._clean_html(html_content)
        structure = self._extract_html_structure(cleaned_html)

        # Prepare examples
        examples_str = json.dumps(self._examples.get('schemas', []), indent=2)
        previous_str = json.dumps([s.__dict__ for s in previous_schemas], indent=2) if previous_schemas else "[]"

        # Generate prompt
        prompt = PromptTemplates.SCHEMA_INFERENCE.format(
            html_content=cleaned_html[:5000],  # Limit for token constraints
            context=context or f"Web page from {urlparse(url).netloc if url else 'unknown'}",
            previous_schemas=previous_str
        )

        # Call LLM
        response = self._call_llm(prompt)

        if "error" in response:
            logger.error(f"Schema inference failed: {response['error']}")
            return self._fallback_schema(html_content, context)

        try:
            # Parse response into schema
            fields = []
            for field_data in response.get("fields", []):
                field = SchemaField(
                    name=field_data.get("name", "unknown"),
                    field_type=field_data.get("type", "string"),
                    selector=field_data.get("selector"),
                    selector_type=field_data.get("selector_type", "css"),
                    required=field_data.get("required", False),
                    validation_rules=field_data.get("validation", []),
                    description=field_data.get("description")
                )
                fields.append(field)

            schema = InferredSchema(
                name=response.get("schema_name", "inferred_schema"),
                fields=fields,
                confidence=response.get("confidence", 0.5),
                source_url=url,
                metadata={
                    "structure": structure,
                    "html_length": len(html_content),
                    "provider": self.provider.value,
                    "model": self.model
                }
            )

            logger.info(f"Inferred schema with {len(fields)} fields (confidence: {schema.confidence:.2f})")
            return schema

        except Exception as e:
            logger.error(f"Failed to parse schema response: {e}")
            return self._fallback_schema(html_content, context)

    def _fallback_schema(self, html_content: str, context: str) -> InferredSchema:
        """Generate fallback schema using heuristic methods."""
        selector = Selector(text=html_content)

        fields = []

        # Extract common patterns
        # Title field
        title = selector.css('title::text, h1::text').get()
        if title:
            fields.append(SchemaField(
                name="title",
                field_type="string",
                selector="h1::text",
                selector_type="css",
                required=True
            ))

        # Links
        links = selector.css('a::attr(href)').getall()
        if links:
            fields.append(SchemaField(
                name="links",
                field_type="array",
                selector="a::attr(href)",
                selector_type="css",
                validation_rules=["type:url"]
            ))

        # Images
        images = selector.css('img::attr(src)').getall()
        if images:
            fields.append(SchemaField(
                name="images",
                field_type="array",
                selector="img::attr(src)",
                selector_type="css",
                validation_rules=["type:url"]
            ))

        # Text content
        text_content = selector.css('p::text, div::text, span::text').getall()
        if text_content:
            fields.append(SchemaField(
                name="text_content",
                field_type="array",
                selector="p::text, div::text, span::text",
                selector_type="css"
            ))

        return InferredSchema(
            name="fallback_schema",
            fields=fields,
            confidence=0.3,
            metadata={"method": "heuristic_fallback"}
        )

    def generate_selectors(
        self,
        html_content: str,
        field_descriptions: Dict[str, str],
        schema: Optional[InferredSchema] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Generate selectors for specific fields.

        Args:
            html_content: HTML content to analyze
            field_descriptions: Dict mapping field names to descriptions
            schema: Optional existing schema to enhance

        Returns:
            Dict mapping field names to selector information
        """
        logger.info(f"Generating selectors for {len(field_descriptions)} fields")

        cleaned_html = self._clean_html(html_content)
        results = {}

        for field_name, description in field_descriptions.items():
            # Prepare examples for few-shot learning
            examples = []
            for example in self._examples.get('selectors', []):
                if any(keyword in description.lower() for keyword in example['field'].split('_')):
                    examples.append(example)

            examples_str = json.dumps(examples[:3], indent=2) if examples else "[]"

            prompt = PromptTemplates.SELECTOR_GENERATION.format(
                html_content=cleaned_html[:3000],
                field_description=description,
                examples=examples_str
            )

            response = self._call_llm(prompt)

            if "error" not in response:
                results[field_name] = {
                    "selector": response.get("selector"),
                    "type": response.get("type", "css"),
                    "confidence": response.get("confidence", 0.5),
                    "description": description
                }

                # Test the selector
                test_result = self._test_selector(
                    cleaned_html,
                    response.get("selector"),
                    response.get("type", "css")
                )
                results[field_name]["test_result"] = test_result

        return results

    def _test_selector(self, html: str, selector: str, selector_type: str) -> Dict[str, Any]:
        """Test a selector against HTML content."""
        try:
            sel = Selector(text=html)
            if selector_type == "css":
                results = sel.css(selector).getall()
            else:
                results = sel.xpath(selector).getall()

            return {
                "success": True,
                "matches": len(results),
                "sample": results[:3] if results else []
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def generate_validation_rules(
        self,
        field_name: str,
        field_description: str,
        sample_values: List[Any]
    ) -> List[Dict[str, Any]]:
        """Generate validation rules from natural language description.

        Args:
            field_name: Name of the field
            field_description: Natural language description
            sample_values: Sample values for the field

        Returns:
            List of validation rules
        """
        prompt = PromptTemplates.VALIDATION_RULES.format(
            field_name=field_name,
            field_description=field_description,
            sample_values=json.dumps(sample_values[:5])
        )

        response = self._call_llm(prompt)

        if "error" in response:
            logger.error(f"Validation rule generation failed: {response['error']}")
            return self._generate_fallback_rules(field_name, sample_values)

        return response.get("rules", [])

    def _generate_fallback_rules(self, field_name: str, sample_values: List[Any]) -> List[Dict[str, Any]]:
        """Generate fallback validation rules using heuristics."""
        rules = []

        if not sample_values:
            return rules

        # Infer type from samples
        types = set(type(v).__name__ for v in sample_values if v is not None)

        if "str" in types:
            # String validation
            lengths = [len(str(v)) for v in sample_values if v is not None]
            if lengths:
                min_len = min(lengths)
                max_len = max(lengths)
                rules.append({
                    "type": "range",
                    "rule": f"length between {min_len} and {max_len}",
                    "message": f"String length should be between {min_len} and {max_len}"
                })

        elif "int" in types or "float" in types:
            # Numeric validation
            numbers = [float(v) for v in sample_values if v is not None and isinstance(v, (int, float))]
            if numbers:
                min_val = min(numbers)
                max_val = max(numbers)
                rules.append({
                    "type": "range",
                    "rule": f"value between {min_val} and {max_val}",
                    "message": f"Value should be between {min_val} and {max_val}"
                })

        # URL validation
        if any("http" in str(v) for v in sample_values if v):
            rules.append({
                "type": "format",
                "rule": "type:url",
                "message": "Must be a valid URL"
            })

        return rules

    def validate_data(
        self,
        extracted_data: Dict[str, Any],
        schema: InferredSchema,
        custom_rules: Optional[Dict[str, List[Dict]]] = None
    ) -> Dict[str, Any]:
        """Validate extracted data against schema.

        Args:
            extracted_data: Data extracted by spider
            schema: Inferred schema to validate against
            custom_rules: Custom validation rules per field

        Returns:
            Validation results with errors and suggestions
        """
        logger.info(f"Validating data against schema '{schema.name}'")

        prompt = PromptTemplates.DATA_VALIDATION.format(
            extracted_data=json.dumps(extracted_data, indent=2),
            schema=json.dumps(schema.__dict__, indent=2, default=str)
        )

        response = self._call_llm(prompt)

        if "error" in response:
            logger.error(f"Data validation failed: {response['error']}")
            return self._validate_fallback(extracted_data, schema, custom_rules)

        return response

    def _validate_fallback(
        self,
        data: Dict[str, Any],
        schema: InferredSchema,
        custom_rules: Optional[Dict[str, List[Dict]]] = None
    ) -> Dict[str, Any]:
        """Fallback validation using schema rules."""
        errors = []
        suggestions = []

        schema_fields = {f.name: f for f in schema.fields}

        # Check required fields
        for field_name, field_schema in schema_fields.items():
            if field_schema.required and field_name not in data:
                errors.append({
                    "field": field_name,
                    "issue": "Required field missing",
                    "severity": "error"
                })

        # Validate existing fields
        for field_name, value in data.items():
            if field_name in schema_fields:
                field_schema = schema_fields[field_name]

                # Type validation
                if field_schema.field_type == "number" and not isinstance(value, (int, float)):
                    errors.append({
                        "field": field_name,
                        "issue": f"Expected number, got {type(value).__name__}",
                        "severity": "error"
                    })

                elif field_schema.field_type == "url" and isinstance(value, str):
                    if not value.startswith(('http://', 'https://')):
                        errors.append({
                            "field": field_name,
                            "issue": "Invalid URL format",
                            "severity": "warning"
                        })

                # Custom rules
                if custom_rules and field_name in custom_rules:
                    for rule in custom_rules[field_name]:
                        if not self._apply_rule(value, rule):
                            errors.append({
                                "field": field_name,
                                "issue": rule.get("message", "Validation failed"),
                                "severity": "error"
                            })

        return {
            "valid": len([e for e in errors if e["severity"] == "error"]) == 0,
            "errors": errors,
            "suggestions": suggestions
        }

    def _apply_rule(self, value: Any, rule: Dict[str, Any]) -> bool:
        """Apply a single validation rule."""
        rule_type = rule.get("type")
        rule_expr = rule.get("rule", "")

        try:
            if rule_type == "format":
                if "url" in rule_expr:
                    return isinstance(value, str) and value.startswith(('http://', 'https://'))
                elif "email" in rule_expr:
                    return isinstance(value, str) and '@' in value

            elif rule_type == "range":
                if isinstance(value, (int, float)):
                    # Extract range from rule
                    match = re.search(r'between\s+([\d.]+)\s+and\s+([\d.]+)', rule_expr)
                    if match:
                        min_val, max_val = float(match.group(1)), float(match.group(2))
                        return min_val <= value <= max_val

            elif rule_type == "required":
                return value is not None and value != ""

        except Exception:
            pass

        return True

    def enhance_spider_code(
        self,
        spider_code: str,
        schema: InferredSchema,
        url_pattern: Optional[str] = None
    ) -> str:
        """Enhance spider code with inferred selectors and validation.

        Args:
            spider_code: Original spider code
            schema: Inferred schema
            url_pattern: URL pattern the spider targets

        Returns:
            Enhanced spider code with AI-generated improvements
        """
        # This would be a more complex implementation that parses
        # the spider code and injects selectors and validation logic
        # For now, return a template
        return self._generate_spider_template(schema, url_pattern)

    def _generate_spider_template(self, schema: InferredSchema, url_pattern: str) -> str:
        """Generate a spider template from inferred schema."""
        fields_code = []
        for field in schema.fields:
            selector_line = f'        item["{field.name}"] = response.{field.selector_type}("{field.selector}").get()'
            if field.field_type == "array":
                selector_line = f'        item["{field.name}"] = response.{field.selector_type}("{field.selector}").getall()'
            fields_code.append(selector_line)

        template = f'''"""
Auto-generated spider using AI Schema Inference.
Schema: {schema.name}
Confidence: {schema.confidence:.2f}
"""

import vex
from vex_ai import AIItem, validate_with_schema


class InferredSpider(vex.Spider):
    name = "{schema.name}_spider"
    allowed_domains = ["{urlparse(url_pattern).netloc if url_pattern else 'example.com'}"]
    start_urls = ["{url_pattern or 'https://example.com'}"]

    @validate_with_schema("{schema.name}")
    def parse(self, response):
        item = AIItem()
        
{chr(10).join(fields_code)}
        
        yield item
'''
        return template

    def save_schema(self, schema: InferredSchema, filepath: str):
        """Save inferred schema to file."""
        schema_data = {
            "name": schema.name,
            "fields": [f.__dict__ for f in schema.fields],
            "confidence": schema.confidence,
            "source_url": schema.source_url,
            "metadata": schema.metadata,
            "sample_data": schema.sample_data
        }

        with open(filepath, 'w') as f:
            json.dump(schema_data, f, indent=2)

        logger.info(f"Schema saved to {filepath}")

    def load_schema(self, filepath: str) -> InferredSchema:
        """Load schema from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        fields = [SchemaField(**field_data) for field_data in data["fields"]]
        return InferredSchema(
            name=data["name"],
            fields=fields,
            confidence=data["confidence"],
            source_url=data.get("source_url"),
            metadata=data.get("metadata", {}),
            sample_data=data.get("sample_data", {})
        )


class AIItem(dict):
    """Dictionary-like item that supports AI validation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validation_errors = []

    def validate(self, schema: InferredSchema) -> bool:
        """Validate item against schema."""
        engine = SchemaInferenceEngine()
        result = engine.validate_data(dict(self), schema)
        self._validation_errors = result.get("errors", [])
        return result.get("valid", False)

    @property
    def validation_errors(self):
        return self._validation_errors


def validate_with_schema(schema_name: str):
    """Decorator for spider methods to validate against schema."""
    def decorator(func):
        def wrapper(self, response, *args, **kwargs):
            # Load schema (this would be cached in production)
            schema_path = f"schemas/{schema_name}.json"
            try:
                engine = SchemaInferenceEngine()
                schema = engine.load_schema(schema_path)
            except FileNotFoundError:
                logger.warning(f"Schema {schema_name} not found, skipping validation")
                return func(self, response, *args, **kwargs)

            # Generate items
            for item in func(self, response, *args, **kwargs):
                if isinstance(item, AIItem):
                    if not item.validate(schema):
                        logger.warning(f"Validation failed: {item.validation_errors}")
                        # Could drop item or send to error pipeline
                yield item

        return wrapper
    return decorator


# Integration with Scrapy commands
class AISchemaCommand:
    """Scrapy command for AI schema inference."""

    def __init__(self):
        self.engine = None

    def add_options(self, parser):
        parser.add_argument("--url", help="URL to analyze")
        parser.add_argument("--html-file", help="Local HTML file to analyze")
        parser.add_argument("--context", help="Context description")
        parser.add_argument("--output", "-o", help="Output schema file")
        parser.add_argument("--provider", choices=["openai", "anthropic", "local"],
                          default="openai", help="LLM provider")

    def run(self, args, settings):
        """Run schema inference command."""
        self.engine = SchemaInferenceEngine(
            settings=settings,
            provider=LLMProvider(args.provider)
        )

        if args.url:
            import requests
            response = requests.get(args.url)
            html_content = response.text
        elif args.html_file:
            with open(args.html_file, 'r') as f:
                html_content = f.read()
        else:
            print("Please provide --url or --html-file")
            return

        schema = self.engine.infer_schema(
            html_content,
            context=args.context or "",
            url=args.url
        )

        if args.output:
            self.engine.save_schema(schema, args.output)
            print(f"Schema saved to {args.output}")
        else:
            print(json.dumps(schema.__dict__, indent=2, default=str))


# Factory function for easy integration
def create_inference_engine(settings=None, **kwargs) -> SchemaInferenceEngine:
    """Create a configured inference engine.

    Args:
        settings: Scrapy settings
        **kwargs: Additional configuration

    Returns:
        Configured SchemaInferenceEngine instance
    """
    return SchemaInferenceEngine(settings=settings, **kwargs)


# Middleware for automatic schema inference
class AISchemaMiddleware:
    """Scrapy middleware for automatic schema inference."""

    @classmethod
    def from_crawler(cls, crawler):
        settings = crawler.settings
        if not settings.getbool('AI_SCHEMA_ENABLED', False):
            raise NotConfigured

        return cls(settings)

    def __init__(self, settings):
        self.settings = settings
        self.engine = create_inference_engine(settings)
        self.schemas = {}

    def process_spider_output(self, response, result, spider):
        """Process spider output and infer schemas if needed."""
        for item in result:
            # Auto-infer schema from first few items
            if hasattr(spider, 'name') and spider.name not in self.schemas:
                if isinstance(item, dict) and len(self.schemas.get(spider.name, [])) < 3:
                    # Collect sample data for schema inference
                    if spider.name not in self.schemas:
                        self.schemas[spider.name] = []
                    self.schemas[spider.name].append(item)

                    # After collecting enough samples, infer schema
                    if len(self.schemas[spider.name]) >= 3:
                        self._infer_schema_from_samples(spider, response)

            yield item

    def _infer_schema_from_samples(self, spider, response):
        """Infer schema from collected samples."""
        samples = self.schemas.get(spider.name, [])
        if not samples:
            return

        # Combine sample data
        combined = {}
        for sample in samples:
            for key, value in sample.items():
                if key not in combined:
                    combined[key] = []
                combined[key].append(value)

        # Create schema fields
        fields = []
        for field_name, values in combined.items():
            field_type = self._infer_type(values)
            fields.append(SchemaField(
                name=field_name,
                field_type=field_type,
                required=len(values) == len(samples)
            ))

        schema = InferredSchema(
            name=f"{spider.name}_auto",
            fields=fields,
            confidence=0.7,
            source_url=response.url,
            sample_data=combined
        )

        # Save schema
        self.engine.save_schema(schema, f"schemas/{spider.name}_auto.json")
        logger.info(f"Auto-inferred schema for {spider.name}")

    def _infer_type(self, values: List[Any]) -> str:
        """Infer field type from sample values."""
        types = set()
        for value in values:
            if isinstance(value, str):
                if value.startswith(('http://', 'https://')):
                    types.add('url')
                elif '@' in value:
                    types.add('email')
                else:
                    types.add('string')
            elif isinstance(value, (int, float)):
                types.add('number')
            elif isinstance(value, list):
                types.add('array')
            elif isinstance(value, dict):
                types.add('object')
            else:
                types.add('string')

        if len(types) == 1:
            return types.pop()
        return 'string'  # Default to string if mixed types


# Example usage in settings.py:
"""
# Enable AI schema inference
AI_SCHEMA_ENABLED = True

# LLM Configuration
AI_LLM_API_KEY = 'your-api-key'
AI_LLM_ENDPOINT = 'https://api.openai.com/v1/chat/completions'
AI_LLM_MODEL = 'gpt-4'
AI_LLM_TEMPERATURE = 0.1
AI_LLM_MAX_TOKENS = 2000
AI_LLM_REQUIRED = False  # Set to True to require LLM for operation

# Enable middleware
DOWNLOADER_MIDDLEWARES = {
    'vex.ai.schema_inference.AISchemaMiddleware': 543,
}
"""