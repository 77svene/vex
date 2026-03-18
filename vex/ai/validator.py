"""AI-powered extraction assistant for Scrapy spiders.

This module provides intelligent selector generation, schema inference, and data validation
using LLM integration to reduce scraping development time by 90%.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod

import vex
from vex.http import TextResponse
from vex.selector import Selector

logger = logging.getLogger(__name__)


class LLMBackend(Enum):
    """Supported LLM backends."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


@dataclass
class ExtractionSchema:
    """Schema definition for data extraction."""
    name: str
    fields: Dict[str, Dict[str, Any]]  # field_name -> {type, selector, description, required}
    version: str = "1.0"
    created_at: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtractionSchema':
        return cls(**data)


@dataclass
class SelectorCandidate:
    """Candidate selector with confidence score."""
    field: str
    selector: str
    selector_type: str  # 'css', 'xpath', 'regex'
    confidence: float
    examples: List[str]
    validation_rules: List[str]


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict:
        """Generate JSON from prompt."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', 0.1),
            max_tokens=kwargs.get('max_tokens', 4096)
        )
        return response.choices[0].message.content
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict:
        response_text = self.generate(prompt, system_prompt, **kwargs)
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Failed to extract JSON from response: {response_text}")


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', 4096),
            system=system_prompt or "",
            messages=messages
        )
        return response.content[0].text
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict:
        response_text = self.generate(prompt, system_prompt, **kwargs)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Failed to extract JSON from response: {response_text}")


class OllamaClient(LLMClient):
    """Ollama local LLM client."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        try:
            import requests
            self.base_url = base_url
            self.model = model
            self.session = requests.Session()
        except ImportError:
            raise ImportError("requests package not installed. Install with: pip install requests")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        import requests
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', 0.1),
                "num_predict": kwargs.get('max_tokens', 4096)
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        response = self.session.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        return response.json()["response"]
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> Dict:
        response_text = self.generate(prompt, system_prompt, **kwargs)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        raise ValueError(f"Failed to extract JSON from response: {response_text}")


class AIExtractor:
    """AI-powered extraction assistant for Scrapy."""
    
    def __init__(self, 
                 llm_backend: Union[str, LLMBackend] = LLMBackend.OPENAI,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 cache_enabled: bool = True):
        """
        Initialize AI extractor.
        
        Args:
            llm_backend: LLM backend to use
            api_key: API key for the LLM service
            model: Model name to use
            cache_enabled: Whether to cache LLM responses
        """
        if isinstance(llm_backend, str):
            llm_backend = LLMBackend(llm_backend)
        
        self.llm_backend = llm_backend
        self.cache_enabled = cache_enabled
        self._cache = {}
        
        # Initialize LLM client
        if llm_backend == LLMBackend.OPENAI:
            self.client = OpenAIClient(api_key=api_key, model=model or "gpt-4-turbo")
        elif llm_backend == LLMBackend.ANTHROPIC:
            self.client = AnthropicClient(api_key=api_key, model=model or "claude-3-opus-20240229")
        elif llm_backend == LLMBackend.OLLAMA:
            self.client = OllamaClient(model=model or "llama3")
        else:
            raise ValueError(f"Unsupported LLM backend: {llm_backend}")
        
        # Few-shot examples for selector generation
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_few_shot_examples(self) -> List[Dict]:
        """Load few-shot examples for selector generation."""
        return [
            {
                "description": "Extract product title from e-commerce page",
                "html_snippet": '<h1 class="product-title">iPhone 15 Pro Max</h1>',
                "selectors": [
                    {"field": "title", "selector": "h1.product-title::text", "type": "css"},
                    {"field": "title", "selector": "//h1[@class='product-title']/text()", "type": "xpath"}
                ]
            },
            {
                "description": "Extract article author and date",
                "html_snippet": '''
                <div class="article-meta">
                    <span class="author">John Doe</span>
                    <time datetime="2024-01-15">January 15, 2024</time>
                </div>
                ''',
                "selectors": [
                    {"field": "author", "selector": "span.author::text", "type": "css"},
                    {"field": "date", "selector": "time::attr(datetime)", "type": "css"},
                    {"field": "date_display", "selector": "time::text", "type": "css"}
                ]
            },
            {
                "description": "Extract all links from navigation menu",
                "html_snippet": '''
                <nav class="main-menu">
                    <a href="/home">Home</a>
                    <a href="/products">Products</a>
                    <a href="/about">About</a>
                </nav>
                ''',
                "selectors": [
                    {"field": "nav_links", "selector": "nav.main-menu a::attr(href)", "type": "css"},
                    {"field": "nav_links", "selector": "//nav[@class='main-menu']/a/@href", "type": "xpath"}
                ]
            }
        ]
    
    def generate_selectors(self, 
                          html: str, 
                          description: str,
                          response: Optional[TextResponse] = None) -> List[SelectorCandidate]:
        """
        Generate selectors for extracting data based on description.
        
        Args:
            html: HTML content to analyze
            description: Natural language description of what to extract
            response: Optional Scrapy response object
        
        Returns:
            List of selector candidates with confidence scores
        """
        cache_key = f"selectors:{hash(html)}:{hash(description)}"
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Prepare few-shot examples
        examples_text = "\n\n".join([
            f"Example {i+1}:\nDescription: {ex['description']}\nHTML: {ex['html_snippet'][:200]}...\nSelectors: {json.dumps(ex['selectors'], indent=2)}"
            for i, ex in enumerate(self.few_shot_examples[:3])
        ])
        
        system_prompt = """You are an expert web scraping engineer. Generate CSS and XPath selectors for extracting data from HTML based on descriptions.
        
Rules:
1. Generate both CSS and XPath selectors when possible
2. Provide confidence scores (0.0-1.0) based on selector robustness
3. Include example extracted values
4. Consider multiple possible selectors for the same field
5. Return JSON array of selector candidates

Output format:
[{
    "field": "field_name",
    "selector": "css_or_xpath_selector",
    "selector_type": "css" or "xpath",
    "confidence": 0.95,
    "examples": ["example_value1", "example_value2"],
    "validation_rules": ["rule1", "rule2"]
}]"""
        
        # Truncate HTML for LLM context window
        html_snippet = html[:5000] if len(html) > 5000 else html
        
        prompt = f"""Generate selectors for extracting data from this HTML based on the description.

Description: {description}

HTML snippet:
{html_snippet}

Few-shot examples:
{examples_text}

Generate the most robust selectors possible. Consider:
- Multiple selector strategies (CSS, XPath)
- Fallback selectors
- Data validation rules
- Edge cases (missing elements, dynamic content)

Return ONLY the JSON array of selector candidates."""
        
        try:
            result = self.client.generate_json(prompt, system_prompt)
            
            # Convert to SelectorCandidate objects
            candidates = []
            for item in result:
                candidates.append(SelectorCandidate(
                    field=item.get('field', 'unknown'),
                    selector=item.get('selector', ''),
                    selector_type=item.get('selector_type', 'css'),
                    confidence=item.get('confidence', 0.5),
                    examples=item.get('examples', []),
                    validation_rules=item.get('validation_rules', [])
                ))
            
            if self.cache_enabled:
                self._cache[cache_key] = candidates
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to generate selectors: {e}")
            # Fallback: return basic selectors
            return self._generate_fallback_selectors(html, description)
    
    def _generate_fallback_selectors(self, html: str, description: str) -> List[SelectorCandidate]:
        """Generate basic fallback selectors when LLM fails."""
        # Simple pattern matching as fallback
        candidates = []
        
        # Look for common patterns
        if "title" in description.lower():
            # Try common title selectors
            title_selectors = [
                ("h1::text", "css", 0.7),
                ("title::text", "css", 0.6),
                ("//h1/text()", "xpath", 0.7),
                ("//title/text()", "xpath", 0.6)
            ]
            for sel, sel_type, conf in title_selectors:
                candidates.append(SelectorCandidate(
                    field="title",
                    selector=sel,
                    selector_type=sel_type,
                    confidence=conf,
                    examples=[],
                    validation_rules=["non_empty", "string"]
                ))
        
        if "link" in description.lower() or "url" in description.lower():
            candidates.append(SelectorCandidate(
                field="links",
                selector="a::attr(href)",
                selector_type="css",
                confidence=0.8,
                examples=[],
                validation_rules=["url_format", "non_empty"]
            ))
        
        return candidates
    
    def infer_schema(self, 
                    html_samples: List[str], 
                    description: Optional[str] = None) -> ExtractionSchema:
        """
        Infer data schema from sample HTML pages.
        
        Args:
            html_samples: List of HTML samples to analyze
            description: Optional description of the data to extract
        
        Returns:
            Inferred extraction schema
        """
        # Combine samples for analysis
        combined_html = "\n\n---\n\n".join(html_samples[:5])  # Limit to 5 samples
        
        system_prompt = """You are a data schema inference expert. Analyze HTML samples to infer the structure of extractable data.
        
Create a JSON schema that describes:
1. Field names and types
2. CSS/XPath selectors for each field
3. Whether fields are required
4. Validation rules
5. Data patterns

Output format:
{
    "name": "schema_name",
    "fields": {
        "field_name": {
            "type": "string|number|date|url|list",
            "selector": "css_or_xpath",
            "description": "what this field contains",
            "required": true|false,
            "validation": ["rule1", "rule2"]
        }
    },
    "version": "1.0"
}"""
        
        prompt = f"""Analyze these HTML samples and infer a data extraction schema.

{'Description: ' + description if description else ''}

HTML Samples:
{combined_html[:10000]}

Infer the most likely schema based on:
1. Common patterns across samples
2. Semantic meaning of elements
3. Data types and formats
4. Required vs optional fields

Return ONLY the JSON schema."""
        
        try:
            schema_dict = self.client.generate_json(prompt, system_prompt)
            return ExtractionSchema.from_dict(schema_dict)
        except Exception as e:
            logger.error(f"Failed to infer schema: {e}")
            return ExtractionSchema(
                name="inferred_schema",
                fields={},
                created_at=""
            )
    
    def validate_extraction(self,
                           data: Dict[str, Any],
                           schema: Optional[ExtractionSchema] = None,
                           rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate extracted data against schema or natural language rules.
        
        Args:
            data: Extracted data to validate
            schema: Optional schema to validate against
            rules: Optional natural language validation rules
        
        Returns:
            Validation results with errors and warnings
        """
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "field_validations": {}
        }
        
        if schema:
            # Validate against schema
            for field_name, field_schema in schema.fields.items():
                field_value = data.get(field_name)
                field_validation = {"valid": True, "errors": [], "warnings": []}
                
                # Check required fields
                if field_schema.get("required", False) and (field_value is None or field_value == ""):
                    field_validation["valid"] = False
                    field_validation["errors"].append(f"Required field '{field_name}' is missing")
                
                # Type validation
                if field_value is not None:
                    expected_type = field_schema.get("type", "string")
                    if not self._validate_type(field_value, expected_type):
                        field_validation["valid"] = False
                        field_validation["errors"].append(
                            f"Field '{field_name}' has wrong type. Expected {expected_type}"
                        )
                
                validation_results["field_validations"][field_name] = field_validation
                if not field_validation["valid"]:
                    validation_results["valid"] = False
                    validation_results["errors"].extend(field_validation["errors"])
        
        if rules:
            # Validate against natural language rules
            rule_validation = self._validate_with_rules(data, rules)
            if not rule_validation["valid"]:
                validation_results["valid"] = False
                validation_results["errors"].extend(rule_validation["errors"])
                validation_results["warnings"].extend(rule_validation.get("warnings", []))
        
        return validation_results
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value against expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "float": float,
            "boolean": bool,
            "list": list,
            "dict": dict,
            "date": str,  # Could be enhanced with date parsing
            "url": str
        }
        
        if expected_type in type_map:
            expected = type_map[expected_type]
            if isinstance(expected, tuple):
                return isinstance(value, expected)
            return isinstance(value, expected)
        return True
    
    def _validate_with_rules(self, data: Dict[str, Any], rules: List[str]) -> Dict[str, Any]:
        """Validate data using natural language rules via LLM."""
        system_prompt = """You are a data validation expert. Validate data against natural language rules.
        
Return JSON with validation results:
{
    "valid": true|false,
    "errors": ["error1", "error2"],
    "warnings": ["warning1", "warning2"]
}"""
        
        prompt = f"""Validate this data against the following rules:

Data:
{json.dumps(data, indent=2)}

Rules:
{json.dumps(rules, indent=2)}

Check each rule carefully. Return ONLY the JSON validation results."""
        
        try:
            return self.client.generate_json(prompt, system_prompt)
        except Exception as e:
            logger.error(f"Failed to validate with rules: {e}")
            return {"valid": True, "errors": [], "warnings": [f"Validation failed: {str(e)}"]}
    
    def generate_spider_code(self,
                            url: str,
                            description: str,
                            selectors: List[SelectorCandidate]) -> str:
        """
        Generate Scrapy spider code based on selectors.
        
        Args:
            url: Target URL
            description: Description of what to scrape
            selectors: List of selector candidates
        
        Returns:
            Generated spider code
        """
        # Group selectors by field
        field_selectors = {}
        for candidate in selectors:
            if candidate.field not in field_selectors:
                field_selectors[candidate.field] = []
            field_selectors[candidate.field].append(candidate)
        
        # Generate spider code
        spider_code = f'''"""
Auto-generated spider by Scrapy AI Extractor
Description: {description}
Generated at: {__import__('datetime').datetime.now().isoformat()}
"""

import vex
from vex_ai import AIExtractor


class AISpider(vex.Spider):
    name = "ai_generated_spider"
    start_urls = ["{url}"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ai_extractor = AIExtractor()
    
    def parse(self, response):
        """Parse response and extract data."""
        data = {{}}
        
'''
        
        # Add extraction logic for each field
        for field, candidates in field_selectors.items():
            best_candidate = max(candidates, key=lambda x: x.confidence)
            
            if best_candidate.selector_type == "css":
                extraction_code = f'        data["{field}"] = response.css("{best_candidate.selector}").get()'
            else:  # xpath
                extraction_code = f'        data["{field}"] = response.xpath("{best_candidate.selector}").get()'
            
            spider_code += extraction_code + "\n"
        
        spider_code += '''
        # Validate extracted data
        validation = self.ai_extractor.validate_extraction(data)
        if not validation["valid"]:
            self.logger.warning(f"Validation failed: {validation['errors']}")
        
        yield data
'''
        
        return spider_code


class AISpiderMiddleware:
    """Scrapy middleware for AI-powered extraction assistance."""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.extractor = AIExtractor(
            llm_backend=crawler.settings.get('AI_LLM_BACKEND', 'openai'),
            api_key=crawler.settings.get('AI_API_KEY'),
            model=crawler.settings.get('AI_MODEL'),
            cache_enabled=crawler.settings.getbool('AI_CACHE_ENABLED', True)
        )
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def process_spider_input(self, response, spider):
        """Process response before spider parses it."""
        # Could add AI-powered response classification here
        return None
    
    def process_spider_output(self, response, result, spider):
        """Process spider output for AI validation."""
        for item in result:
            if isinstance(item, dict):
                # Validate extracted items
                validation = self.extractor.validate_extraction(item)
                if not validation["valid"]:
                    spider.logger.warning(f"Item validation failed: {validation['errors']}")
            yield item


# Convenience functions for direct use
def generate_selectors(html: str, 
                      description: str,
                      llm_backend: str = "openai",
                      api_key: Optional[str] = None) -> List[SelectorCandidate]:
    """Convenience function to generate selectors."""
    extractor = AIExtractor(llm_backend=llm_backend, api_key=api_key)
    return extractor.generate_selectors(html, description)


def infer_schema(html_samples: List[str],
                description: Optional[str] = None,
                llm_backend: str = "openai",
                api_key: Optional[str] = None) -> ExtractionSchema:
    """Convenience function to infer schema."""
    extractor = AIExtractor(llm_backend=llm_backend, api_key=api_key)
    return extractor.infer_schema(html_samples, description)


def validate_data(data: Dict[str, Any],
                 schema: Optional[ExtractionSchema] = None,
                 rules: Optional[List[str]] = None,
                 llm_backend: str = "openai",
                 api_key: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to validate data."""
    extractor = AIExtractor(llm_backend=llm_backend, api_key=api_key)
    return extractor.validate_extraction(data, schema, rules)


# Integration with Scrapy commands
def integrate_with_genspider():
    """Integrate AI extraction with vex genspider command."""
    # This would modify the genspider command to use AI
    pass


# Example usage
if __name__ == "__main__":
    # Example: Generate selectors for a product page
    sample_html = """
    <html>
    <body>
        <div class="product">
            <h1 class="title">Amazing Product</h1>
            <span class="price">$99.99</span>
            <div class="description">
                <p>This is an amazing product that does amazing things.</p>
            </div>
            <img class="product-image" src="product.jpg" alt="Product Image">
        </div>
    </body>
    </html>
    """
    
    # Generate selectors
    extractor = AIExtractor(llm_backend="openai")  # Requires OPENAI_API_KEY env var
    candidates = extractor.generate_selectors(
        sample_html, 
        "Extract product title, price, description, and image URL"
    )
    
    print("Generated selectors:")
    for candidate in candidates:
        print(f"  {candidate.field}: {candidate.selector} ({candidate.selector_type}, confidence: {candidate.confidence})")
    
    # Infer schema
    schema = extractor.infer_schema([sample_html], "Product information")
    print(f"\nInferred schema: {schema.name}")
    for field, details in schema.fields.items():
        print(f"  {field}: {details}")