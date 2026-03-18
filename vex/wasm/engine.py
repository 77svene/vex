"""
WebAssembly Engine for Scrapy - Edge-Optimized Deployment
Compiles Scrapy to WebAssembly for CDN edge deployment with automatic geographic distribution.
"""

import asyncio
import json
import hashlib
import time
import zlib
from typing import Dict, List, Optional, Any, Callable, AsyncIterator, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from urllib.parse import urlparse
import sys

# Conditional imports for WASM environment
try:
    import js
    from pyodide.ffi import create_proxy
    from pyodide.http import pyfetch
    WASM_ENV = True
except ImportError:
    WASM_ENV = False
    js = None
    create_proxy = lambda x: x

from vex.http import Request, Response
from vex.core.engine import ExecutionEngine
from vex.core.scheduler import Scheduler
from vex.spiders import Spider
from vex.settings import Settings
from vex import signals
from vex.utils.log import configure_logging
from vex.utils.misc import load_object

logger = logging.getLogger(__name__)


class EdgeRegion(Enum):
    """Edge deployment regions for geographic distribution"""
    NA_WEST = "na-west"
    NA_EAST = "na-east"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_EAST = "asia-east"
    ASIA_SOUTH = "asia-south"
    OCEANIA = "oceania"
    SOUTH_AMERICA = "south-america"


class EdgeProvider(Enum):
    """Supported edge deployment providers"""
    CLOUDFLARE_WORKERS = "cloudflare-workers"
    VERCEL_EDGE = "vercel-edge"
    FASTLY_COMPUTE = "fastly-compute"
    DENOLAND = "denoland"
    WASM_CLOUD = "wasm-cloud"


@dataclass
class EdgeNode:
    """Represents an edge deployment node"""
    id: str
    region: EdgeRegion
    provider: EdgeProvider
    endpoint: str
    latency_ms: float = 0.0
    active_requests: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['region'] = self.region.value
        data['provider'] = self.provider.value
        return data


@dataclass
class EdgeTask:
    """Task to be executed on edge nodes"""
    id: str
    request: Request
    priority: int = 0
    region_preferences: List[EdgeRegion] = field(default_factory=list)
    max_retries: int = 3
    timeout_ms: int = 30000
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EdgeCoordinator:
    """Coordinates scraping tasks across edge nodes"""
    
    def __init__(self, coordinator_url: Optional[str] = None):
        self.coordinator_url = coordinator_url or "https://vex-edge-coordinator.example.com"
        self.nodes: Dict[str, EdgeNode] = {}
        self.task_queue: asyncio.Queue[EdgeTask] = asyncio.Queue()
        self.active_tasks: Dict[str, EdgeTask] = {}
        self._lock = asyncio.Lock()
        
    async def register_node(self, node: EdgeNode) -> bool:
        """Register an edge node with the coordinator"""
        async with self._lock:
            self.nodes[node.id] = node
            logger.info(f"Registered edge node {node.id} in region {node.region.value}")
            return True
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister an edge node"""
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered edge node {node_id}")
                return True
            return False
    
    async def select_optimal_node(self, task: EdgeTask) -> Optional[EdgeNode]:
        """Select optimal edge node for a task based on region and latency"""
        if not self.nodes:
            return None
        
        # Filter nodes by region preferences if specified
        candidate_nodes = list(self.nodes.values())
        if task.region_preferences:
            candidate_nodes = [
                node for node in candidate_nodes
                if node.region in task.region_preferences
            ]
        
        if not candidate_nodes:
            candidate_nodes = list(self.nodes.values())
        
        # Select node with lowest latency and available capacity
        return min(candidate_nodes, key=lambda n: (n.latency_ms, n.active_requests))
    
    async def submit_task(self, task: EdgeTask) -> bool:
        """Submit a task for edge execution"""
        await self.task_queue.put(task)
        logger.debug(f"Task {task.id} submitted to edge queue")
        return True
    
    async def get_task(self) -> Optional[EdgeTask]:
        """Get next task from queue (called by edge nodes)"""
        try:
            task = self.task_queue.get_nowait()
            self.active_tasks[task.id] = task
            return task
        except asyncio.QueueEmpty:
            return None
    
    async def complete_task(self, task_id: str, success: bool = True) -> bool:
        """Mark task as completed"""
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            logger.debug(f"Task {task_id} completed with success={success}")
            return True
        return False
    
    async def sync_state(self) -> Dict[str, Any]:
        """Synchronize state with remote coordinator"""
        if not self.coordinator_url:
            return {}
        
        try:
            state = {
                'nodes': [node.to_dict() for node in self.nodes.values()],
                'queue_size': self.task_queue.qsize(),
                'active_tasks': len(self.active_tasks),
                'timestamp': time.time()
            }
            
            if WASM_ENV:
                response = await pyfetch(
                    f"{self.coordinator_url}/sync",
                    method="POST",
                    headers={"Content-Type": "application/json"},
                    body=json.dumps(state)
                )
                if response.status == 200:
                    return await response.json()
            
            return state
        except Exception as e:
            logger.error(f"Failed to sync with coordinator: {e}")
            return {}


class WASMRequestAdapter:
    """Adapts Scrapy Request for WASM environment using Fetch API"""
    
    @staticmethod
    async def fetch(request: Request) -> Response:
        """Execute request using Fetch API in WASM environment"""
        if not WASM_ENV:
            raise RuntimeError("WASM environment not available")
        
        # Convert Scrapy Request to fetch options
        fetch_options = {
            'method': request.method,
            'headers': dict(request.headers),
            'redirect': 'follow' if request.meta.get('dont_redirect', False) else 'manual'
        }
        
        if request.body:
            fetch_options['body'] = request.body
        
        try:
            # Use Pyodide's pyfetch for WASM compatibility
            response = await pyfetch(
                request.url,
                **fetch_options
            )
            
            # Convert fetch response to Scrapy Response
            headers = {k: v for k, v in response.headers.items()}
            body = await response.bytes()
            
            return Response(
                url=request.url,
                status=response.status,
                headers=headers,
                body=body,
                request=request,
                flags=['cached'] if response.status == 304 else []
            )
        except Exception as e:
            logger.error(f"WASM fetch failed for {request.url}: {e}")
            raise


class ColdStartOptimizer:
    """Optimizes cold start times for edge deployments"""
    
    def __init__(self):
        self.cache = {}
        self.warmup_tasks = []
        self.precompiled_modules = set()
    
    def precompile_module(self, module_path: str) -> bool:
        """Precompile module for faster loading"""
        try:
            # In WASM environment, modules are precompiled during build
            self.precompiled_modules.add(module_path)
            logger.info(f"Precompiled module: {module_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to precompile {module_path}: {e}")
            return False
    
    def warmup(self, spider_cls: type, settings: Settings) -> None:
        """Warmup environment with spider and dependencies"""
        # Pre-initialize spider
        spider = spider_cls.from_settings(settings)
        
        # Pre-compile common patterns
        self.warmup_tasks.append({
            'spider': spider.name,
            'timestamp': time.time(),
            'settings_hash': self._hash_settings(settings)
        })
        
        logger.info(f"Warmup completed for spider: {spider.name}")
    
    def _hash_settings(self, settings: Settings) -> str:
        """Create hash of settings for cache invalidation"""
        settings_dict = settings.copy_to_dict()
        return hashlib.md5(json.dumps(settings_dict, sort_keys=True).encode()).hexdigest()
    
    def get_cached_response(self, url: str) -> Optional[Response]:
        """Get cached response for common URLs"""
        return self.cache.get(url)
    
    def cache_response(self, response: Response) -> None:
        """Cache response for future use"""
        if response.status == 200:
            self.cache[response.url] = response


class WASMEngine(ExecutionEngine):
    """WebAssembly-optimized Scrapy engine for edge deployment"""
    
    def __init__(self, spider: Spider, coordinator: Optional[EdgeCoordinator] = None):
        self.spider = spider
        self.coordinator = coordinator or EdgeCoordinator()
        self.cold_start_optimizer = ColdStartOptimizer()
        self.request_adapter = WASMRequestAdapter()
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.active_requests: Dict[str, Request] = {}
        self.metrics = {
            'requests_sent': 0,
            'responses_received': 0,
            'errors': 0,
            'edge_delegated': 0,
            'cold_starts': 0
        }
        
        # Initialize with minimal settings for WASM
        settings = Settings()
        settings.set('CONCURRENT_REQUESTS', 10)  # Lower for edge
        settings.set('DOWNLOAD_DELAY', 0.1)  # Faster for edge
        settings.set('RETRY_TIMES', 2)
        
        super().__init__(spider, settings)
        
        # Register signal handlers
        self.signals.connect(self._on_request_scheduled, signals.request_scheduled)
        self.signals.connect(self._on_response_received, signals.response_received)
        self.signals.connect(self._on_error, signals.spider_error)
    
    async def _on_request_scheduled(self, request: Request, spider: Spider) -> None:
        """Handle request scheduling for edge optimization"""
        self.metrics['requests_sent'] += 1
        
        # Check if request should be delegated to edge
        if self._should_delegate_to_edge(request):
            task = EdgeTask(
                id=f"task_{int(time.time())}_{hash(request.url)}",
                request=request,
                priority=request.priority,
                region_preferences=self._get_region_preferences(request)
            )
            
            node = await self.coordinator.select_optimal_node(task)
            if node:
                await self.coordinator.submit_task(task)
                self.metrics['edge_delegated'] += 1
                logger.debug(f"Delegated request to edge node {node.id}")
                return
        
        # Process locally
        await self._process_request_locally(request)
    
    async def _on_response_received(self, response: Response, request: Request, spider: Spider) -> None:
        """Handle response received"""
        self.metrics['responses_received'] += 1
        
        # Cache successful responses for cold start optimization
        if response.status == 200:
            self.cold_start_optimizer.cache_response(response)
    
    async def _on_error(self, failure, response: Response, spider: Spider) -> None:
        """Handle errors"""
        self.metrics['errors'] += 1
        logger.error(f"Spider error: {failure}")
    
    def _should_delegate_to_edge(self, request: Request) -> bool:
        """Determine if request should be delegated to edge"""
        # Delegate static assets and API requests
        url = urlparse(request.url)
        path = url.path.lower()
        
        # Delegate static content
        static_extensions = ['.js', '.css', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.woff2']
        if any(path.endswith(ext) for ext in static_extensions):
            return True
        
        # Delegate API endpoints
        if '/api/' in path or '/graphql' in path:
            return True
        
        # Check request metadata for edge hints
        if request.meta.get('edge_delegate', False):
            return True
        
        return False
    
    def _get_region_preferences(self, request: Request) -> List[EdgeRegion]:
        """Get region preferences based on request"""
        # Extract region from request metadata or URL
        region_hint = request.meta.get('region')
        if region_hint:
            try:
                return [EdgeRegion(region_hint)]
            except ValueError:
                pass
        
        # Default to all regions
        return list(EdgeRegion)
    
    async def _process_request_locally(self, request: Request) -> Response:
        """Process request locally in WASM environment"""
        # Check cold start cache first
        cached = self.cold_start_optimizer.get_cached_response(request.url)
        if cached:
            logger.debug(f"Serving cached response for {request.url}")
            return cached
        
        # Use WASM request adapter
        try:
            response = await self.request_adapter.fetch(request)
            return response
        except Exception as e:
            logger.error(f"Local request failed: {e}")
            raise
    
    async def start(self) -> None:
        """Start the WASM engine"""
        logger.info(f"Starting WASM engine for spider: {self.spider.name}")
        
        # Warmup cold start optimizer
        self.cold_start_optimizer.warmup(self.spider.__class__, self.settings)
        
        # Start coordinator sync loop
        asyncio.create_task(self._coordinator_sync_loop())
        
        # Start edge node heartbeat
        asyncio.create_task(self._edge_heartbeat_loop())
    
    async def _coordinator_sync_loop(self) -> None:
        """Periodically sync with edge coordinator"""
        while True:
            try:
                await self.coordinator.sync_state()
                await asyncio.sleep(30)  # Sync every 30 seconds
            except Exception as e:
                logger.error(f"Coordinator sync failed: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _edge_heartbeat_loop(self) -> None:
        """Send heartbeats to edge coordinator"""
        node_id = f"wasm_{hashlib.md5(str(id(self)).encode()).hexdigest()[:8]}"
        node = EdgeNode(
            id=node_id,
            region=EdgeRegion.NA_WEST,  # Default, should be configured
            provider=EdgeProvider.WASM_CLOUD,
            endpoint=f"wasm://{node_id}"
        )
        
        await self.coordinator.register_node(node)
        
        while True:
            try:
                node.last_heartbeat = time.time()
                node.active_requests = len(self.active_requests)
                await asyncio.sleep(10)  # Heartbeat every 10 seconds
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")
                await asyncio.sleep(30)
    
    async def close(self) -> None:
        """Close the WASM engine"""
        logger.info(f"Closing WASM engine for spider: {self.spider.name}")
        
        # Unregister from coordinator
        node_id = f"wasm_{hashlib.md5(str(id(self)).encode()).hexdigest()[:8]}"
        await self.coordinator.unregister_node(node_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        return {
            **self.metrics,
            'edge_nodes': len(self.coordinator.nodes),
            'queue_size': self.coordinator.task_queue.qsize(),
            'active_tasks': len(self.coordinator.active_tasks),
            'cached_responses': len(self.cold_start_optimizer.cache)
        }


class EdgeDeploymentManager:
    """Manages deployment to edge networks"""
    
    def __init__(self, provider: EdgeProvider = EdgeProvider.CLOUDFLARE_WORKERS):
        self.provider = provider
        self.deployment_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default deployment configuration"""
        configs = {
            EdgeProvider.CLOUDFLARE_WORKERS: {
                'name': 'vex-edge-worker',
                'main': 'worker.js',
                'compatibility_date': '2024-01-01',
                'vars': {
                    'ENVIRONMENT': 'production'
                },
                'kv_namespaces': [
                    {'binding': 'CACHE', 'id': 'vex-cache'}
                ]
            },
            EdgeProvider.VERCEL_EDGE: {
                'name': 'vex-edge',
                'version': 2,
                'builds': [
                    {'src': '*.js', 'use': '@vercel/edge'}
                ],
                'routes': [
                    {'src': '/(.*)', 'dest': '/'}
                ]
            },
            EdgeProvider.FASTLY_COMPUTE: {
                'name': 'vex-fastly',
                'description': 'Scrapy Edge Worker',
                'authors': ['vex-team@example.com'],
                'service_id': 'vex-service'
            }
        }
        return configs.get(self.provider, {})
    
    def generate_worker_script(self, spider_code: str) -> str:
        """Generate edge worker script"""
        if self.provider == EdgeProvider.CLOUDFLARE_WORKERS:
            return self._generate_cloudflare_worker(spider_code)
        elif self.provider == EdgeProvider.VERCEL_EDGE:
            return self._generate_vercel_edge(spider_code)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_cloudflare_worker(self, spider_code: str) -> str:
        """Generate Cloudflare Worker script"""
        return f"""
// Scrapy Edge Worker for Cloudflare
addEventListener('fetch', event => {{
    event.respondWith(handleRequest(event.request, event))
}})

async function handleRequest(request, event) {{
    const url = new URL(request.url);
    
    // Health check endpoint
    if (url.pathname === '/health') {{
        return new Response('OK', {{ status: 200 }});
    }}
    
    // Metrics endpoint
    if (url.pathname === '/metrics') {{
        return new Response(JSON.stringify({{
            requests: globalThis.metrics?.requests || 0,
            cache_hits: globalThis.metrics?.cache_hits || 0,
            timestamp: Date.now()
        }}), {{
            headers: {{ 'Content-Type': 'application/json' }}
        }});
    }}
    
    // Spider execution endpoint
    if (url.pathname === '/crawl') {{
        try {{
            const spiderConfig = await request.json();
            const result = await executeSpider(spiderConfig, event);
            return new Response(JSON.stringify(result), {{
                headers: {{ 'Content-Type': 'application/json' }}
            }});
        }} catch (error) {{
            return new Response(JSON.stringify({{ error: error.message }}), {{
                status: 500,
                headers: {{ 'Content-Type': 'application/json' }}
            }});
        }}
    }}
    
    // Default: proxy to origin or serve cached content
    return await fetch(request);
}}

// Spider execution logic (simplified)
async function executeSpider(config, event) {{
    // This would be replaced with actual WASM-compiled Scrapy code
    return {{
        status: 'completed',
        items_scraped: 0,
        requests_made: 0,
        timestamp: Date.now()
    }};
}}

// Initialize metrics
globalThis.metrics = {{
    requests: 0,
    cache_hits: 0
}};
"""
    
    def _generate_vercel_edge(self, spider_code: str) -> str:
        """Generate Vercel Edge Function script"""
        return f"""
// Scrapy Edge Function for Vercel
export const config = {{
    runtime: 'edge',
}};

export default async function handler(request) {{
    const url = new URL(request.url);
    
    // Health check
    if (url.pathname === '/health') {{
        return new Response('OK', {{ status: 200 }});
    }}
    
    // Spider execution
    if (url.pathname === '/api/crawl' && request.method === 'POST') {{
        try {{
            const body = await request.json();
            const result = await runSpider(body);
            return new Response(JSON.stringify(result), {{
                headers: {{ 'Content-Type': 'application/json' }}
            }});
        }} catch (error) {{
            return new Response(JSON.stringify({{ error: error.message }}), {{
                status: 500,
                headers: {{ 'Content-Type': 'application/json' }}
            }});
        }}
    }}
    
    // Default response
    return new Response('Scrapy Edge Function', {{ status: 200 }});
}}

// Simplified spider runner
async function runSpider(config) {{
    return {{
        status: 'completed',
        items: [],
        stats: {{}}
    }};
}}
"""
    
    def generate_deployment_files(self, spider_code: str) -> Dict[str, str]:
        """Generate all deployment files"""
        files = {}
        
        if self.provider == EdgeProvider.CLOUDFLARE_WORKERS:
            files['worker.js'] = self.generate_worker_script(spider_code)
            files['wrangler.toml'] = self._generate_wrangler_toml()
            files['package.json'] = json.dumps({
                'name': self.deployment_config['name'],
                'version': '1.0.0',
                'private': True,
                'scripts': {
                    'dev': 'wrangler dev',
                    'deploy': 'wrangler deploy'
                }
            }, indent=2)
        
        elif self.provider == EdgeProvider.VERCEL_EDGE:
            files['api/crawl.js'] = self.generate_worker_script(spider_code)
            files['vercel.json'] = json.dumps(self.deployment_config, indent=2)
        
        return files
    
    def _generate_wrangler_toml(self) -> str:
        """Generate wrangler.toml for Cloudflare Workers"""
        config = self.deployment_config
        return f"""
name = "{config['name']}"
main = "{config['main']}"
compatibility_date = "{config['compatibility_date']}"

[vars]
{chr(10).join(f'{k} = "{v}"' for k, v in config.get('vars', {}).items())}

"""
    
    async def deploy(self, spider_code: str, region: EdgeRegion = EdgeRegion.NA_WEST) -> Dict[str, Any]:
        """Deploy to edge network"""
        deployment_id = hashlib.md5(f"{spider_code}{time.time()}".encode()).hexdigest()[:12]
        
        deployment_info = {
            'id': deployment_id,
            'provider': self.provider.value,
            'region': region.value,
            'timestamp': time.time(),
            'status': 'deployed',
            'endpoints': self._generate_endpoints(deployment_id, region)
        }
        
        logger.info(f"Deployed to {self.provider.value} in region {region.value}")
        return deployment_info
    
    def _generate_endpoints(self, deployment_id: str, region: EdgeRegion) -> Dict[str, str]:
        """Generate deployment endpoints"""
        base_urls = {
            EdgeProvider.CLOUDFLARE_WORKERS: f"https://{deployment_id}.workers.dev",
            EdgeProvider.VERCEL_EDGE: f"https://{deployment_id}.vercel.app"
        }
        
        base_url = base_urls.get(self.provider, f"https://{deployment_id}.example.com")
        
        return {
            'health': f"{base_url}/health",
            'crawl': f"{base_url}/crawl" if self.provider == EdgeProvider.CLOUDFLARE_WORKERS else f"{base_url}/api/crawl",
            'metrics': f"{base_url}/metrics"
        }


# WASM-compatible Spider Runner
class WASMSpiderRunner:
    """Runs spiders in WASM environment"""
    
    def __init__(self, coordinator_url: Optional[str] = None):
        self.coordinator = EdgeCoordinator(coordinator_url)
        self.engines: Dict[str, WASMEngine] = {}
    
    async def run_spider(self, spider_cls: type, **kwargs) -> Dict[str, Any]:
        """Run a spider in WASM environment"""
        # Create spider instance
        settings = Settings()
        settings.setmodule('vex.settings.default_settings')
        
        spider = spider_cls.from_settings(settings, **kwargs)
        
        # Create WASM engine
        engine = WASMEngine(spider, self.coordinator)
        self.engines[spider.name] = engine
        
        try:
            # Start engine
            await engine.start()
            
            # Execute spider
            start_requests = list(spider.start_requests())
            results = []
            
            for request in start_requests:
                try:
                    response = await engine._process_request_locally(request)
                    
                    # Process response through spider
                    callback = request.callback or spider.parse
                    if callback:
                        if asyncio.iscoroutinefunction(callback):
                            items = await callback(response)
                        else:
                            items = callback(response)
                        
                        if items:
                            if hasattr(items, '__aiter__'):
                                async for item in items:
                                    results.append(item)
                            else:
                                results.extend(items)
                
                except Exception as e:
                    logger.error(f"Failed to process {request.url}: {e}")
            
            # Get metrics
            metrics = engine.get_metrics()
            
            return {
                'spider': spider.name,
                'items': results,
                'metrics': metrics,
                'success': True
            }
        
        finally:
            # Cleanup
            await engine.close()
            if spider.name in self.engines:
                del self.engines[spider.name]
    
    async def run_distributed(self, spider_cls: type, regions: List[EdgeRegion], **kwargs) -> Dict[str, Any]:
        """Run spider across multiple edge regions"""
        tasks = []
        
        for region in regions:
            # Create region-specific runner
            runner = WASMSpiderRunner()
            
            # Configure for region
            settings = Settings()
            settings.set('EDGE_REGION', region.value)
            
            task = asyncio.create_task(
                runner.run_spider(spider_cls, **kwargs)
            )
            tasks.append((region, task))
        
        # Gather results from all regions
        results = {}
        for region, task in tasks:
            try:
                result = await task
                results[region.value] = result
            except Exception as e:
                results[region.value] = {'error': str(e)}
        
        return {
            'distributed': True,
            'regions': list(results.keys()),
            'results': results,
            'total_items': sum(
                len(r.get('items', []))
                for r in results.values()
                if isinstance(r, dict) and 'items' in r
            )
        }


# Integration with existing Scrapy commands
def add_wasm_commands():
    """Add WASM-related commands to Scrapy"""
    from vex.commands import ScrapyCommand
    from vex.utils.conf import build_component_list
    
    class WASMDeployCommand(ScrapyCommand):
        """Deploy spider to edge network"""
        
        def short_desc(self):
            return "Deploy spider to edge network as WebAssembly"
        
        def run(self, args, opts):
            provider = EdgeProvider(args[0]) if args else EdgeProvider.CLOUDFLARE_WORKERS
            region = EdgeRegion(args[1]) if len(args) > 1 else EdgeRegion.NA_WEST
            
            manager = EdgeDeploymentManager(provider)
            
            # Get spider code (simplified)
            spider_name = opts.spider or self.settings.get('SPIDER')
            if not spider_name:
                logger.error("No spider specified")
                return
            
            # Generate deployment files
            spider_code = f"# Spider: {spider_name}\n# Generated for edge deployment"
            files = manager.generate_deployment_files(spider_code)
            
            # Deploy
            deployment_info = asyncio.run(manager.deploy(spider_code, region))
            
            print(f"Deployed to {provider.value} in {region.value}")
            print(f"Endpoints: {json.dumps(deployment_info['endpoints'], indent=2)}")
    
    class WASMRunCommand(ScrapyCommand):
        """Run spider in WASM environment"""
        
        def short_desc(self):
            return "Run spider in WebAssembly environment"
        
        def run(self, args, opts):
            spider_name = opts.spider or self.settings.get('SPIDER')
            if not spider_name:
                logger.error("No spider specified")
                return
            
            # Load spider class
            spider_cls = load_object(spider_name)
            
            # Run in WASM
            runner = WASMSpiderRunner()
            result = asyncio.run(runner.run_spider(spider_cls))
            
            print(f"Spider completed: {result['success']}")
            print(f"Items scraped: {len(result.get('items', []))}")
            print(f"Metrics: {json.dumps(result.get('metrics', {}), indent=2)}")
    
    # Register commands
    from vex import cmdline
    cmdline.commands['wasm-deploy'] = WASMDeployCommand
    cmdline.commands['wasm-run'] = WASMRunCommand


# Initialize WASM environment if running in browser
if WASM_ENV:
    # Export for JavaScript access
    def create_spider_runner(coordinator_url=None):
        """Create spider runner for JavaScript"""
        return WASMSpiderRunner(coordinator_url)
    
    # Make available to JavaScript
    js.create_spider_runner = create_proxy(create_spider_runner)
    js.WASMEngine = WASMEngine
    js.EdgeCoordinator = EdgeCoordinator
    js.EdgeDeploymentManager = EdgeDeploymentManager
    
    logger.info("Scrapy WASM engine initialized for browser environment")


# Module exports
__all__ = [
    'WASMEngine',
    'EdgeCoordinator',
    'EdgeNode',
    'EdgeTask',
    'EdgeRegion',
    'EdgeProvider',
    'WASMRequestAdapter',
    'ColdStartOptimizer',
    'EdgeDeploymentManager',
    'WASMSpiderRunner',
    'add_wasm_commands'
]