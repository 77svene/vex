"""AI-Powered Extraction Assistant for Scrapy

Provides LLM integration for automatic selector generation, schema inference,
and data validation to reduce development time for new scraping targets.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import vex
from vex import Selector
from vex.http import Response
from vex.utils.project import get_project_settings

logger = logging.getLogger(__name__)


class SelectorGenerator:
    """AI-powered selector generator using LLM integration.
    
    Integrates with local/remote LLM APIs to automatically generate
    CSS/XPath selectors, infer schemas from sample pages, and validate
    extracted data against natural language descriptions.
    """

    DEFAULT_SYSTEM_PROMPT = """You are an expert web scraping assistant. Your task is to analyze HTML content and generate optimal CSS or XPath selectors for extracting structured data. 

Key capabilities:
1. Generate precise selectors for specified fields
2. Infer data schemas from sample HTML
3. Validate extracted data against expected patterns
4. Suggest alternative selectors when primary ones fail

Always respond with valid JSON containing the requested selectors or schema information."""

    def __init__(self, 
                 llm_provider: str = "openai",
                 model_name: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        """Initialize the SelectorGenerator.
        
        Args:
            llm_provider: LLM provider ("openai", "local", "anthropic")
            model_name: Name of the LLM model to use
            api_key: API key for the LLM provider
            api_base: Base URL for API requests
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens in response
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        settings = get_project_settings()
        
        # Get API configuration from settings if not provided
        if api_key is None:
            api_key = settings.get(f'{llm_provider.upper()}_API_KEY')
        if api_base is None:
            api_base = settings.get(f'{llm_provider.upper()}_API_BASE')
        
        self.api_key = api_key
        self.api_base = api_base
        
        # Initialize the LLM client based on provider
        self.client = self._initialize_llm_client()
        
        # Cache for few-shot examples
        self._few_shot_examples = []
        
        # Schema cache for reuse
        self._schema_cache = {}

    def _initialize_llm_client(self):
        """Initialize the appropriate LLM client based on provider."""
        if self.llm_provider == "openai":
            try:
                import openai
                client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base
                )
                return client
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        elif self.llm_provider == "anthropic":
            try:
                import anthropic
                client = anthropic.Anthropic(
                    api_key=self.api_key,
                    base_url=self.api_base
                )
                return client
            except ImportError:
                raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
        
        elif self.llm_provider == "local":
            # For local LLMs via API (e.g., Ollama, vLLM)
            import requests
            return requests.Session()
        
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _call_llm(self, 
                  messages: List[Dict[str, str]], 
                  response_format: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a call to the configured LLM.
        
        Args:
            messages: List of message dictionaries for the conversation
            response_format: Optional format specification for response
            
        Returns:
            Parsed response from the LLM
        """
        try:
            if self.llm_provider == "openai":
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                return json.loads(content) if content else {}
            
            elif self.llm_provider == "anthropic":
                # Convert OpenAI-style messages to Anthropic format
                system_msg = ""
                user_msgs = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_msg = msg["content"]
                    else:
                        user_msgs.append(msg)
                
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_msg,
                    messages=user_msgs
                )
                content = response.content[0].text
                return json.loads(content) if content else {}
            
            elif self.llm_provider == "local":
                # Generic API call for local LLMs
                import requests
                
                url = f"{self.api_base}/v1/chat/completions"
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
                
                response = self.client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return json.loads(content) if content else {}
        
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def add_few_shot_example(self, 
                            html_snippet: str, 
                            field_descriptions: Dict[str, str],
                            generated_selectors: Dict[str, str],
                            selector_type: str = "css"):
        """Add a few-shot example for selector generation.
        
        Args:
            html_snippet: Sample HTML content
            field_descriptions: Natural language descriptions of fields
            generated_selectors: Generated selectors for the fields
            selector_type: Type of selectors ("css" or "xpath")
        """
        example = {
            "html_snippet": html_snippet[:5000],  # Truncate for token limits
            "field_descriptions": field_descriptions,
            "generated_selectors": generated_selectors,
            "selector_type": selector_type
        }
        self._few_shot_examples.append(example)

    def generate_selectors(self,
                          html_content: str,
                          field_descriptions: Dict[str, str],
                          selector_type: str = "css",
                          context: Optional[str] = None) -> Dict[str, str]:
        """Generate selectors for specified fields from HTML content.
        
        Args:
            html_content: HTML content to analyze
            field_descriptions: Dictionary mapping field names to descriptions
            selector_type: Type of selectors to generate ("css" or "xpath")
            context: Additional context about the page structure
            
        Returns:
            Dictionary mapping field names to generated selectors
        """
        # Prepare the prompt with few-shot examples
        messages = [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]
        
        # Add few-shot examples
        for example in self._few_shot_examples[-3:]:  # Use last 3 examples
            messages.append({
                "role": "user",
                "content": f"""Given this HTML snippet:
{example['html_snippet']}

Generate {example['selector_type']} selectors for these fields:
{json.dumps(example['field_descriptions'], indent=2)}"""
            })
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "selectors": example['generated_selectors'],
                    "selector_type": example['selector_type']
                })
            })
        
        # Add the current request
        user_message = f"""Analyze this HTML and generate {selector_type} selectors for the following fields:

Fields to extract:
{json.dumps(field_descriptions, indent=2)}

HTML content:
{html_content[:8000]}  # Truncate for token limits

{f"Additional context: {context}" if context else ""}

Return a JSON object with:
1. "selectors": dictionary mapping field names to {selector_type} selectors
2. "confidence": confidence score (0-1) for each selector
3. "alternative_selectors": alternative selectors for fields with low confidence
4. "notes": any important observations about the page structure"""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self._call_llm(messages)
            return response.get("selectors", {})
        except Exception as e:
            logger.error(f"Failed to generate selectors: {e}")
            return {}

    def infer_schema(self,
                    html_content: str,
                    sample_selectors: Optional[Dict[str, str]] = None,
                    schema_type: str = "json_schema") -> Dict[str, Any]:
        """Infer data schema from HTML content.
        
        Args:
            html_content: HTML content to analyze
            sample_selectors: Optional sample selectors to guide inference
            schema_type: Type of schema to generate ("json_schema", "pydantic", "sql")
            
        Returns:
            Inferred schema definition
        """
        messages = [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]
        
        user_message = f"""Analyze this HTML content and infer the data schema:

HTML content:
{html_content[:10000]}

{f"Sample selectors that might be useful: {json.dumps(sample_selectors, indent=2)}" if sample_selectors else ""}

Infer the complete schema of the data that can be extracted from this page. Consider:
1. All extractable fields and their data types
2. Relationships between fields
3. Nested structures (lists, objects)
4. Validation rules (required fields, patterns, ranges)

Return a {schema_type} schema definition in JSON format."""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self._call_llm(messages)
            schema_key = f"schema_{hash(html_content[:1000])}"
            self._schema_cache[schema_key] = response
            return response
        except Exception as e:
            logger.error(f"Failed to infer schema: {e}")
            return {}

    def validate_extraction(self,
                          extracted_data: Dict[str, Any],
                          validation_rules: Dict[str, str],
                          html_content: Optional[str] = None) -> Dict[str, Any]:
        """Validate extracted data against natural language rules.
        
        Args:
            extracted_data: Data extracted by selectors
            validation_rules: Natural language validation rules
            html_content: Original HTML for context
            
        Returns:
            Validation results with issues and suggestions
        """
        messages = [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]
        
        user_message = f"""Validate this extracted data against the provided rules:

Extracted data:
{json.dumps(extracted_data, indent=2)}

Validation rules:
{json.dumps(validation_rules, indent=2)}

{f"Original HTML context available: {len(html_content)} chars" if html_content else ""}

Check for:
1. Missing required fields
2. Data type mismatches
3. Pattern violations
4. Logical inconsistencies
5. Potential extraction errors

Return a JSON object with:
1. "valid": boolean indicating if all validations pass
2. "issues": list of validation issues found
3. "suggestions": suggestions for fixing issues
4. "confidence": overall confidence in the extraction quality"""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self._call_llm(messages)
            return response
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"valid": False, "issues": [str(e)], "suggestions": []}

    def generate_spider(self,
                       url: str,
                       sample_html: str,
                       fields: Dict[str, str],
                       spider_name: str = "generated_spider") -> str:
        """Generate a complete Scrapy spider from sample HTML.
        
        Args:
            url: Target URL pattern
            sample_html: Sample HTML for selector generation
            fields: Dictionary of field names and descriptions
            spider_name: Name for the generated spider
            
        Returns:
            Generated spider code as string
        """
        # Generate selectors
        selectors = self.generate_selectors(sample_html, fields)
        
        # Infer schema
        schema = self.infer_schema(sample_html, selectors)
        
        # Generate spider code
        spider_code = f'''"""
Auto-generated spider using AI Selector Generator
Generated for: {url}
"""

import vex
from vex import Selector


class {spider_name.title().replace('_', '')}Spider(vex.Spider):
    name = "{spider_name}"
    allowed_domains = ["{url.split("//")[-1].split("/")[0]}"]
    start_urls = ["{url}"]
    
    custom_settings = {{
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS': 1,
    }}
    
    def parse(self, response):
        """Parse the response and extract data."""
        selector = Selector(response)
        
        # Generated selectors
        selectors = {json.dumps(selectors, indent=8)}
        
        # Extract data using generated selectors
        item = {{}}
        for field, selector_str in selectors.items():
            try:
                item[field] = selector.css(selector_str).get()
            except Exception as e:
                self.logger.warning(f"Failed to extract {{field}}: {{e}}")
                item[field] = None
        
        # Validate extraction
        if self.validate_item(item):
            yield item
        
        # Follow pagination if present
        next_page = response.css('a.next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)
    
    def validate_item(self, item):
        """Validate extracted item."""
        # Basic validation - can be enhanced with AI validation
        required_fields = {list(fields.keys())[:3]}  # First 3 fields as required
        for field in required_fields:
            if not item.get(field):
                self.logger.warning(f"Missing required field: {{field}}")
                return False
        return True
'''
        
        return spider_code

    def analyze_page_structure(self, html_content: str) -> Dict[str, Any]:
        """Analyze page structure to identify patterns and components.
        
        Args:
            html_content: HTML content to analyze
            
        Returns:
            Analysis of page structure including lists, forms, tables, etc.
        """
        messages = [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]
        
        user_message = f"""Analyze the structure of this HTML page:

HTML content:
{html_content[:12000]}

Identify:
1. Main content areas and their purposes
2. Repeating patterns (product listings, article cards, etc.)
3. Navigation elements
4. Forms and input fields
5. Data tables
6. Pagination elements
7. Potential anti-bot protections

Return a JSON object with:
1. "page_type": type of page (e.g., "product_listing", "article", "search_results")
2. "components": identified UI components
3. "patterns": repeating patterns with sample selectors
4. "challenges": potential scraping challenges
5. "recommendations": scraping strategy recommendations"""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self._call_llm(messages)
            return response
        except Exception as e:
            logger.error(f"Page analysis failed: {e}")
            return {}


class AIScrapyMiddleware:
    """Scrapy middleware for AI-powered extraction assistance."""
    
    def __init__(self, selector_generator: SelectorGenerator):
        self.selector_generator = selector_generator
        self.stats = {
            "ai_selectors_generated": 0,
            "ai_validations_performed": 0,
            "ai_schema_inferences": 0
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware from crawler."""
        settings = crawler.settings
        
        # Initialize selector generator from settings
        selector_generator = SelectorGenerator(
            llm_provider=settings.get('AI_LLM_PROVIDER', 'openai'),
            model_name=settings.get('AI_MODEL_NAME', 'gpt-4'),
            api_key=settings.get('AI_API_KEY'),
            api_base=settings.get('AI_API_BASE'),
            temperature=settings.get('AI_TEMPERATURE', 0.1),
            max_tokens=settings.get('AI_MAX_TOKENS', 2000)
        )
        
        return cls(selector_generator)
    
    def process_spider_input(self, response, spider):
        """Process response before spider callback."""
        if hasattr(spider, 'ai_enabled') and spider.ai_enabled:
            # Store response for potential AI analysis
            response.meta['ai_html'] = response.text
        return None
    
    def process_spider_output(self, response, result, spider):
        """Process spider output for AI validation."""
        if hasattr(spider, 'ai_enabled') and spider.ai_enabled:
            for item in result:
                if isinstance(item, dict) and 'ai_validation_rules' in spider.__dict__:
                    # Perform AI validation
                    validation = self.selector_generator.validate_extraction(
                        item,
                        spider.ai_validation_rules,
                        response.meta.get('ai_html')
                    )
                    
                    if not validation.get('valid', True):
                        spider.logger.warning(f"AI validation failed: {validation.get('issues', [])}")
                    
                    self.stats['ai_validations_performed'] += 1
                
                yield item
        else:
            yield from result


# Utility functions for easy integration
def generate_selectors_for_url(url: str, 
                              fields: Dict[str, str],
                              selector_type: str = "css") -> Dict[str, str]:
    """Convenience function to generate selectors for a URL.
    
    Args:
        url: URL to fetch and analyze
        fields: Dictionary of field names and descriptions
        selector_type: Type of selectors to generate
        
    Returns:
        Dictionary of field names to selectors
    """
    import requests
    
    # Fetch the page
    response = requests.get(url)
    response.raise_for_status()
    
    # Generate selectors
    generator = SelectorGenerator()
    return generator.generate_selectors(response.text, fields, selector_type)


def infer_schema_from_url(url: str) -> Dict[str, Any]:
    """Convenience function to infer schema from a URL.
    
    Args:
        url: URL to fetch and analyze
        
    Returns:
        Inferred schema
    """
    import requests
    
    # Fetch the page
    response = requests.get(url)
    response.raise_for_status()
    
    # Infer schema
    generator = SelectorGenerator()
    return generator.infer_schema(response.text)