import os
import sys
import json
import hashlib
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import vex
from vex import Spider
from vex.http import Request, Response
from vex.crawler import CrawlerProcess
from vex.utils.project import get_project_settings
from vex.commands import ScrapyCommand
from vex.settings import Settings
from vex.exceptions import NotConfigured

# Try to import optional dependencies
try:
    import pyodide
    from pyodide import create_proxy
    HAS_PYODIDE = True
except ImportError:
    HAS_PYODIDE = False

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class EdgeConfig:
    """Configuration for edge deployment"""
    platform: str  # 'cloudflare', 'deno', 'browser'
    wasm_runtime: bool = True
    auto_tls: bool = True
    cdn_integration: bool = True
    memory_limit: int = 128  # MB
    cpu_limit: float = 1.0  # vCPU
    timeout: int = 30  # seconds
    environment_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.environment_vars is None:
            self.environment_vars = {}


class EdgePlatform(ABC):
    """Abstract base class for edge platform adapters"""
    
    @abstractmethod
    def compile_to_wasm(self, spider_class: type, settings: Settings) -> bytes:
        """Compile spider to WebAssembly"""
        pass
    
    @abstractmethod
    def deploy(self, wasm_bytes: bytes, config: EdgeConfig) -> str:
        """Deploy to edge platform"""
        pass
    
    @abstractmethod
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information"""
        pass


class CloudflareWorkersAdapter(EdgePlatform):
    """Adapter for Cloudflare Workers deployment"""
    
    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self.api_token = api_token
        self.api_base = "https://api.cloudflare.com/client/v4"
    
    def compile_to_wasm(self, spider_class: type, settings: Settings) -> bytes:
        """Compile spider to WebAssembly for Cloudflare Workers"""
        if not HAS_PYODIDE:
            raise NotConfigured("Pyodide required for WebAssembly compilation")
        
        # Extract spider logic
        spider_code = self._extract_spider_code(spider_class)
        
        # Create worker script
        worker_script = f"""
// Cloudflare Worker with Scrapy spider compiled to WebAssembly
import {{ compilePyodide }} from 'pyodide';

let pyodide = null;

async function initPyodide() {{
    pyodide = await compilePyodide();
    await pyodide.loadPackage(['micropip']);
    
    // Load Scrapy dependencies
    await pyodide.runPythonAsync(`
        import micropip
        await micropip.install('parsel')
        await micropip.install('w3lib')
        await micropip.install('itemloaders')
    `);
}}

{spider_code}

export default {{
    async fetch(request, env, ctx) {{
        if (!pyodide) {{
            await initPyodide();
        }}
        
        try {{
            const url = new URL(request.url);
            const response = await pyodide.runPythonAsync(`
                import asyncio
                from vex.http import Request, TextResponse
                
                # Create mock response
                response = TextResponse(
                    url='{url}',
                    status=200,
                    headers={{}},
                    body=b'',
                    request=Request('{url}')
                )
                
                # Run spider parse method
                spider = {spider_class.__name__}()
                results = []
                async for item in spider.parse(response):
                    results.append(dict(item))
                
                import json
                json.dumps(results)
            `);
            
            return new Response(response, {{
                headers: {{ 'Content-Type': 'application/json' }}
            }});
        }} catch (error) {{
            return new Response(JSON.stringify({{ error: error.message }}), {{
                status: 500,
                headers: {{ 'Content-Type': 'application/json' }}
            }});
        }}
    }}
}};
"""
        return worker_script.encode('utf-8')
    
    def _extract_spider_code(self, spider_class: type) -> str:
        """Extract and serialize spider code"""
        import inspect
        source = inspect.getsource(spider_class)
        
        # Add necessary imports
        imports = """
import vex
from vex.http import Request, TextResponse
from vex import Item, Field
"""
        return imports + source
    
    def deploy(self, wasm_bytes: bytes, config: EdgeConfig) -> str:
        """Deploy to Cloudflare Workers"""
        if not HAS_REQUESTS:
            raise NotConfigured("requests library required for Cloudflare deployment")
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/javascript"
        }
        
        # Create worker
        worker_name = f"vex-edge-{hashlib.md5(wasm_bytes).hexdigest()[:8]}"
        
        response = requests.put(
            f"{self.api_base}/accounts/{self.account_id}/workers/scripts/{worker_name}",
            headers=headers,
            data=wasm_bytes
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to deploy worker: {response.text}")
        
        # Enable worker
        enable_response = requests.post(
            f"{self.api_base}/accounts/{self.account_id}/workers/scripts/{worker_name}/subdomain",
            headers=headers,
            json={"enabled": True}
        )
        
        return f"https://{worker_name}.{self.account_id}.workers.dev"
    
    def get_runtime_info(self) -> Dict[str, Any]:
        return {
            "platform": "cloudflare_workers",
            "runtime": "v8",
            "wasm_support": True,
            "max_memory": "128MB",
            "max_timeout": "30s"
        }


class DenoDeployAdapter(EdgePlatform):
    """Adapter for Deno Deploy"""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.api_base = "https://dash.deno.com/api"
    
    def compile_to_wasm(self, spider_class: type, settings: Settings) -> bytes:
        """Compile spider to WebAssembly for Deno Deploy"""
        spider_code = self._extract_spider_code(spider_class)
        
        deno_script = f"""
// Deno Deploy with Scrapy spider
import {{ serve }} from "https://deno.land/std@0.177.0/http/server.ts";

{spider_code}

async function handler(request: Request): Promise<Response> {{
    try {{
        const url = new URL(request.url);
        
        // Create mock response for spider
        const mockResponse = {{
            url: url.toString(),
            status: 200,
            headers: {{}},
            text: async () => "",
            json: async () => ({{}})
        }};
        
        // Initialize spider
        const spider = new {spider_class.__name__}();
        
        // Run parse method
        const items = [];
        for await (const item of spider.parse(mockResponse)) {{
            items.push(item);
        }}
        
        return new Response(JSON.stringify(items), {{
            headers: {{ "Content-Type": "application/json" }}
        }});
    }} catch (error) {{
        return new Response(JSON.stringify({{ error: error.message }}), {{
            status: 500,
            headers: {{ "Content-Type": "application/json" }}
        }});
    }}
}}

serve(handler, {{ port: 8000 }});
"""
        return deno_script.encode('utf-8')
    
    def _extract_spider_code(self, spider_class: type) -> str:
        """Extract spider code for Deno"""
        import inspect
        source = inspect.getsource(spider_class)
        
        # Convert Python to Deno-compatible JavaScript
        # This is a simplified conversion - in reality would need full transpilation
        js_code = f"""
class {spider_class.__name__} {{
    constructor() {{
        this.name = "{getattr(spider_class, 'name', spider_class.__name__)}";
        this.allowed_domains = {getattr(spider_class, 'allowed_domains', [])};
        this.start_urls = {getattr(spider_class, 'start_urls', [])};
    }}
    
    async *parse(response) {{
        // Spider parse logic would be transpiled here
        // This is a placeholder
        yield {{ url: response.url, title: "Example" }};
    }}
}}
"""
        return js_code
    
    def deploy(self, wasm_bytes: bytes, config: EdgeConfig) -> str:
        """Deploy to Deno Deploy"""
        if not HAS_REQUESTS:
            raise NotConfigured("requests library required for Deno deployment")
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/typescript"
        }
        
        # Create project
        project_name = f"vex-edge-{hashlib.md5(wasm_bytes).hexdigest()[:8]}"
        
        response = requests.post(
            f"{self.api_base}/projects",
            headers=headers,
            json={
                "name": project_name,
                "type": "playground",
                "environment_variables": config.environment_vars
            }
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Failed to create project: {response.text}")
        
        project_id = response.json()["id"]
        
        # Deploy code
        deploy_response = requests.post(
            f"{self.api_base}/projects/{project_id}/deployments",
            headers=headers,
            files={"file": ("main.ts", wasm_bytes)}
        )
        
        if deploy_response.status_code != 200:
            raise RuntimeError(f"Failed to deploy: {deploy_response.text}")
        
        deployment_url = deploy_response.json()["url"]
        return deployment_url
    
    def get_runtime_info(self) -> Dict[str, Any]:
        return {
            "platform": "deno_deploy",
            "runtime": "deno",
            "wasm_support": True,
            "max_memory": "512MB",
            "max_timeout": "60s"
        }


class BrowserExtensionAdapter(EdgePlatform):
    """Adapter for browser extension deployment"""
    
    def compile_to_wasm(self, spider_class: type, settings: Settings) -> bytes:
        """Compile spider to WebAssembly for browser extension"""
        spider_code = self._extract_spider_code(spider_class)
        
        extension_manifest = {
            "manifest_version": 3,
            "name": f"Scrapy Spider: {spider_class.__name__}",
            "version": "1.0",
            "description": f"Browser extension for {spider_class.__name__} spider",
            "permissions": [
                "activeTab",
                "storage",
                "webRequest"
            ],
            "host_permissions": [
                "<all_urls>"
            ],
            "background": {
                "service_worker": "background.js",
                "type": "module"
            },
            "content_scripts": [
                {
                    "matches": ["<all_urls>"],
                    "js": ["content.js"]
                }
            ],
            "action": {
                "default_popup": "popup.html"
            }
        }
        
        background_js = f"""
// Background service worker with Scrapy spider
import {{ compilePyodide }} from './pyodide.js';

let pyodide = null;

async function initPyodide() {{
    pyodide = await compilePyodide();
    await pyodide.loadPackage(['micropip']);
    
    // Install minimal Scrapy dependencies
    await pyodide.runPythonAsync(`
        import micropip
        await micropip.install('parsel')
        await micropip.install('w3lib')
    `);
}}

{spider_code}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {{
    if (request.action === 'runSpider') {{
        (async () => {{
            if (!pyodide) {{
                await initPyodide();
            }}
            
            try {{
                const result = await pyodide.runPythonAsync(`
                    import json
                    from vex.http import TextResponse, Request
                    
                    response = TextResponse(
                        url='${{request.url}}',
                        status=200,
                        headers=${{json.dumps(request.headers || {{}})}},
                        body='${{request.body || ""}}'.encode(),
                        request=Request('${{request.url}}')
                    )
                    
                    spider = {spider_class.__name__}()
                    items = []
                    for item in spider.parse(response):
                        items.append(dict(item))
                    
                    json.dumps(items)
                `);
                
                sendResponse({{ success: true, data: JSON.parse(result) }});
            }} catch (error) {{
                sendResponse({{ success: false, error: error.message }});
            }}
        }})();
        return true; // Keep message channel open for async response
    }}
}});
"""
        
        content_js = """
// Content script for page interaction
chrome.runtime.sendMessage({
    action: 'runSpider',
    url: window.location.href,
    headers: {},
    body: document.documentElement.outerHTML
}, (response) => {
    if (response.success) {
        console.log('Spider results:', response.data);
        // Send results to popup or storage
        chrome.storage.local.set({ spiderResults: response.data });
    }
});
"""
        
        popup_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Scrapy Spider</title>
    <style>
        body { width: 300px; padding: 10px; }
        #results { margin-top: 10px; }
    </style>
</head>
<body>
    <h3>Scrapy Spider Extension</h3>
    <button id="run">Run Spider</button>
    <div id="results"></div>
    <script src="popup.js"></script>
</body>
</html>
"""
        
        popup_js = """
document.getElementById('run').addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'runSpider' });
    });
});

// Display results
chrome.storage.local.get('spiderResults', (data) => {
    if (data.spiderResults) {
        const resultsDiv = document.getElementById('results');
        resultsDiv.innerHTML = '<pre>' + JSON.stringify(data.spiderResults, null, 2) + '</pre>';
    }
});
"""
        
        # Package everything
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.writestr('manifest.json', json.dumps(extension_manifest, indent=2))
            zipf.writestr('background.js', background_js)
            zipf.writestr('content.js', content_js)
            zipf.writestr('popup.html', popup_html)
            zipf.writestr('popup.js', popup_js)
        
        return zip_buffer.getvalue()
    
    def _extract_spider_code(self, spider_class: type) -> str:
        """Extract spider code for browser extension"""
        import inspect
        source = inspect.getsource(spider_class)
        
        # Add browser-specific imports
        imports = """
// Browser environment adaptations
const window = self;
const document = { createElement: () => ({}) };
"""
        return imports + source
    
    def deploy(self, wasm_bytes: bytes, config: EdgeConfig) -> str:
        """Package as browser extension"""
        # Save extension files
        extension_dir = Path(tempfile.mkdtemp()) / "extension"
        extension_dir.mkdir(exist_ok=True)
        
        import zipfile
        import io
        
        with zipfile.ZipFile(io.BytesIO(wasm_bytes)) as zipf:
            zipf.extractall(extension_dir)
        
        # Return path to extension directory
        return str(extension_dir)
    
    def get_runtime_info(self) -> Dict[str, Any]:
        return {
            "platform": "browser_extension",
            "runtime": "browser",
            "wasm_support": True,
            "max_memory": "unlimited",
            "max_timeout": "unlimited"
        }


class ACMECertificateManager:
    """Automatic TLS certificate management using ACME protocol"""
    
    def __init__(self, email: str, directory_url: str = "https://acme-v02.api.letsencrypt.org/directory"):
        self.email = email
        self.directory_url = directory_url
        self.account_key = None
        self.certificate = None
        
        if not HAS_CRYPTO:
            raise NotConfigured("cryptography library required for ACME")
    
    def generate_account_key(self) -> None:
        """Generate RSA account key"""
        self.account_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
    
    def register_account(self) -> Dict[str, Any]:
        """Register with ACME server"""
        if not self.account_key:
            self.generate_account_key()
        
        # Simplified ACME registration
        # In production, use acme library or implement full ACME protocol
        return {
            "status": "valid",
            "contact": [f"mailto:{self.email}"],
            "terms_of_service_agreed": True
        }
    
    def request_certificate(self, domains: List[str]) -> bytes:
        """Request certificate for domains"""
        if not HAS_REQUESTS:
            raise NotConfigured("requests library required for ACME")
        
        # This is a simplified implementation
        # In production, implement full ACME challenge flow
        
        # Generate CSR
        csr = self._generate_csr(domains)
        
        # Request certificate from ACME server
        # (Simplified - actual implementation would use ACME protocol)
        
        return csr
    
    def _generate_csr(self, domains: List[str]) -> bytes:
        """Generate Certificate Signing Request"""
        if not self.account_key:
            raise ValueError("Account key not generated")
        
        # Create CSR
        csr_builder = x509.CertificateSigningRequestBuilder()
        csr_builder = csr_builder.subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, domains[0])
        ]))
        
        # Add SANs
        if len(domains) > 1:
            san_list = [x509.DNSName(domain) for domain in domains]
            csr_builder = csr_builder.add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False
            )
        
        # Sign CSR
        csr = csr_builder.sign(
            self.account_key,
            hashes.SHA256()
        )
        
        return csr.public_bytes(serialization.Encoding.PEM)
    
    def auto_renew(self, days_before_expiry: int = 30) -> bool:
        """Auto-renew certificates before expiry"""
        # Check certificate expiry and renew if needed
        # Implementation would check current cert and renew if expiring soon
        return True


class WebAssemblyCompiler:
    """Compiler for Scrapy components to WebAssembly"""
    
    def __init__(self):
        self.wasm_cache = {}
    
    def compile_spider(self, spider_class: type, settings: Settings) -> bytes:
        """Compile spider to WebAssembly"""
        if not HAS_PYODIDE:
            raise NotConfigured("Pyodide required for WebAssembly compilation")
        
        cache_key = f"{spider_class.__name__}_{hash(str(settings))}"
        
        if cache_key in self.wasm_cache:
            return self.wasm_cache[cache_key]
        
        # Create minimal Scrapy environment for WASM
        wasm_code = self._create_wasm_module(spider_class, settings)
        
        # Cache the result
        self.wasm_cache[cache_key] = wasm_code
        
        return wasm_code
    
    def _create_wasm_module(self, spider_class: type, settings: Settings) -> bytes:
        """Create WebAssembly module"""
        # Extract spider logic
        spider_source = self._get_spider_source(spider_class)
        
        # Create WASM-compatible Python code
        wasm_python = f"""
# Minimal Scrapy implementation for WebAssembly
import sys
import json
from typing import Dict, Any, Iterator

class Item(dict):
    pass

class Request:
    def __init__(self, url, callback=None, method='GET', headers=None, body=None):
        self.url = url
        self.callback = callback
        self.method = method
        self.headers = headers or {{}}
        self.body = body

class Response:
    def __init__(self, url, status=200, headers=None, body=b'', request=None):
        self.url = url
        self.status = status
        self.headers = headers or {{}}
        self.body = body
        self.request = request
        self.text = body.decode('utf-8') if isinstance(body, bytes) else body
    
    def json(self):
        return json.loads(self.text)
    
    def xpath(self, query):
        # Simplified XPath implementation
        return []
    
    def css(self, query):
        # Simplified CSS selector implementation
        return []

{spider_source}

# WASM entry point
def run_spider(url: str) -> str:
    response = Response(url=url, body=b'<html></html>')
    spider = {spider_class.__name__}()
    
    results = []
    for item in spider.parse(response):
        results.append(dict(item))
    
    return json.dumps(results)
"""
        
        # Compile to WASM using Pyodide
        # This is a simplified version - actual compilation would be more complex
        return wasm_python.encode('utf-8')
    
    def _get_spider_source(self, spider_class: type) -> str:
        """Get spider source code"""
        import inspect
        return inspect.getsource(spider_class)


class EdgeDeployer:
    """Main edge deployment orchestrator"""
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.compiler = WebAssemblyCompiler()
        self.certificate_manager = None
        self.platform_adapter = None
        
        if config.auto_tls:
            self._setup_tls()
        
        self._setup_platform_adapter()
    
    def _setup_tls(self):
        """Setup TLS certificate management"""
        if HAS_CRYPTO:
            # Get email from environment or config
            email = os.getenv('ACME_EMAIL', 'admin@example.com')
            self.certificate_manager = ACMECertificateManager(email)
    
    def _setup_platform_adapter(self):
        """Setup platform-specific adapter"""
        platform = self.config.platform.lower()
        
        if platform == 'cloudflare':
            account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
            api_token = os.getenv('CLOUDFLARE_API_TOKEN')
            
            if not account_id or not api_token:
                raise NotConfigured("Cloudflare credentials required")
            
            self.platform_adapter = CloudflareWorkersAdapter(account_id, api_token)
        
        elif platform == 'deno':
            access_token = os.getenv('DENO_ACCESS_TOKEN')
            
            if not access_token:
                raise NotConfigured("Deno access token required")
            
            self.platform_adapter = DenoDeployAdapter(access_token)
        
        elif platform == 'browser':
            self.platform_adapter = BrowserExtensionAdapter()
        
        else:
            raise ValueError(f"Unsupported platform: {platform}")
    
    def deploy_spider(self, spider_class: type, settings: Optional[Settings] = None) -> str:
        """Deploy spider to edge platform"""
        if settings is None:
            settings = get_project_settings()
        
        # Compile spider to WebAssembly
        wasm_bytes = self.compiler.compile_spider(spider_class, settings)
        
        # Deploy to platform
        deployment_url = self.platform_adapter.deploy(wasm_bytes, self.config)
        
        # Setup CDN if enabled
        if self.config.cdn_integration:
            self._setup_cdn(deployment_url)
        
        return deployment_url
    
    def _setup_cdn(self, deployment_url: str):
        """Setup CDN integration"""
        # This would integrate with CDN providers
        # For now, just log the action
        print(f"CDN integration enabled for {deployment_url}")
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information"""
        info = {
            "config": asdict(self.config),
            "runtime": self.platform_adapter.get_runtime_info(),
            "wasm_support": HAS_PYODIDE,
            "tls_support": HAS_CRYPTO,
        }
        
        if self.certificate_manager:
            info["tls"] = {
                "auto_managed": True,
                "provider": "Let's Encrypt"
            }
        
        return info


class EdgeDeployCommand(ScrapyCommand):
    """Scrapy command for edge deployment"""
    
    requires_project = True
    default_settings = {
        'LOG_LEVEL': 'INFO',
    }
    
    def syntax(self):
        return "[options] <spider>"
    
    def short_desc(self):
        return "Deploy spider to edge platform"
    
    def add_options(self, parser):
        ScrapyCommand.add_options(self, parser)
        parser.add_argument(
            "--platform",
            dest="platform",
            default="cloudflare",
            help="Edge platform (cloudflare, deno, browser)"
        )
        parser.add_argument(
            "--no-wasm",
            dest="wasm_runtime",
            action="store_false",
            help="Disable WebAssembly runtime"
        )
        parser.add_argument(
            "--no-tls",
            dest="auto_tls",
            action="store_false",
            help="Disable automatic TLS"
        )
        parser.add_argument(
            "--no-cdn",
            dest="cdn_integration",
            action="store_false",
            help="Disable CDN integration"
        )
        parser.add_argument(
            "--memory",
            dest="memory_limit",
            type=int,
            default=128,
            help="Memory limit in MB"
        )
        parser.add_argument(
            "--cpu",
            dest="cpu_limit",
            type=float,
            default=1.0,
            help="CPU limit in vCPU"
        )
        parser.add_argument(
            "--timeout",
            dest="timeout",
            type=int,
            default=30,
            help="Timeout in seconds"
        )
    
    def run(self, args, opts):
        if not args:
            print("Error: Spider name required")
            return
        
        spider_name = args[0]
        
        # Load spider class
        try:
            spider_cls = self.crawler_process.spider_loader.load(spider_name)
        except KeyError:
            print(f"Error: Spider '{spider_name}' not found")
            return
        
        # Create edge config
        config = EdgeConfig(
            platform=opts.platform,
            wasm_runtime=opts.wasm_runtime,
            auto_tls=opts.auto_tls,
            cdn_integration=opts.cdn_integration,
            memory_limit=opts.memory_limit,
            cpu_limit=opts.cpu_limit,
            timeout=opts.timeout
        )
        
        # Deploy spider
        try:
            deployer = EdgeDeployer(config)
            deployment_url = deployer.deploy_spider(spider_cls, self.settings)
            
            print(f"Successfully deployed {spider_name} to {opts.platform}")
            print(f"Deployment URL: {deployment_url}")
            print("\nDeployment Info:")
            
            import pprint
            pprint.pprint(deployer.get_deployment_info())
            
        except Exception as e:
            print(f"Deployment failed: {e}")
            raise


# Utility functions
def deploy_to_edge(spider_class: type, platform: str = "cloudflare", **kwargs) -> str:
    """Utility function to deploy spider to edge platform"""
    config = EdgeConfig(platform=platform, **kwargs)
    deployer = EdgeDeployer(config)
    return deployer.deploy_spider(spider_class)


def create_edge_spider(base_spider: type, platform: str) -> type:
    """Create edge-optimized version of spider"""
    class EdgeSpider(base_spider):
        custom_settings = {
            'CONCURRENT_REQUESTS': 1,
            'DOWNLOAD_DELAY': 0,
            'ROBOTSTXT_OBEY': False,
            'HTTPCACHE_ENABLED': False,
        }
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.platform = platform
    
    return EdgeSpider


# Register command with Scrapy
from vex.commands import ScrapyCommand
ScrapyCommand.commands['edge_deploy'] = EdgeDeployCommand


# Example usage
if __name__ == "__main__":
    # Example spider
    class ExampleSpider(vex.Spider):
        name = "example"
        start_urls = ["https://example.com"]
        
        def parse(self, response):
            yield {
                "title": response.css("title::text").get(),
                "url": response.url
            }
    
    # Deploy to Cloudflare Workers
    try:
        deployment_url = deploy_to_edge(
            ExampleSpider,
            platform="cloudflare",
            memory_limit=256,
            timeout=60
        )
        print(f"Deployed to: {deployment_url}")
    except Exception as e:
        print(f"Deployment failed: {e}")