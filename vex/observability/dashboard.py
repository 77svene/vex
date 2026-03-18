"""Real-time Observability Stack for Scrapy.

This module provides comprehensive observability capabilities including distributed tracing,
metrics collection, live debugging dashboard, request replay, and automatic anomaly detection.
"""

import asyncio
import json
import time
import uuid
import pickle
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import traceback
import sys
import os

# Optional imports with fallbacks
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

try:
    from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import websockets
    from websockets.server import serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.misc import load_object
from vex.utils.project import get_project_settings
from vex.utils.log import configure_logging


class SpanType(Enum):
    """Types of spans for distributed tracing."""
    REQUEST = "request"
    RESPONSE = "response"
    ITEM = "item"
    ERROR = "error"
    MIDDLEWARE = "middleware"
    PIPELINE = "pipeline"
    SCHEDULER = "scheduler"
    DOWNLOADER = "downloader"


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class RequestTrace:
    """Tracing information for a request."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    request_id: str = ""
    url: str = ""
    method: str = "GET"
    start_time: float = 0.0
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Anomaly:
    """Detected anomaly in scraping patterns."""
    id: str
    timestamp: float
    anomaly_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_requests: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class StoredRequest:
    """Stored request for replay capabilities."""
    id: str
    request: Dict[str, Any]
    response: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0
    spider_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObservabilityDashboard:
    """Main observability dashboard with tracing, metrics, and debugging capabilities.
    
    This class integrates with Scrapy to provide real-time monitoring, distributed tracing,
    metrics collection, request replay, and anomaly detection.
    """
    
    def __init__(self, crawler):
        """Initialize the observability dashboard.
        
        Args:
            crawler: The Scrapy crawler instance
        """
        self.crawler = crawler
        self.settings = crawler.settings
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.enabled = self.settings.getbool('OBSERVABILITY_ENABLED', True)
        self.dashboard_port = self.settings.getint('OBSERVABILITY_DASHBOARD_PORT', 6800)
        self.metrics_port = self.settings.getint('OBSERVABILITY_METRICS_PORT', 9090)
        self.trace_sample_rate = self.settings.getfloat('OBSERVABILITY_TRACE_SAMPLE_RATE', 1.0)
        self.max_stored_requests = self.settings.getint('OBSERVABILITY_MAX_STORED_REQUESTS', 10000)
        self.anomaly_detection_enabled = self.settings.getbool('OBSERVABILITY_ANOMALY_DETECTION', True)
        self.anomaly_detection_interval = self.settings.getint('OBSERVABILITY_ANOMALY_INTERVAL', 60)
        
        # State
        self._active_traces: Dict[str, RequestTrace] = {}
        self._completed_traces: deque = deque(maxlen=10000)
        self._metrics: Dict[str, Any] = {}
        self._anomalies: deque = deque(maxlen=1000)
        self._stored_requests: Dict[str, StoredRequest] = {}
        self._websocket_clients: Set = set()
        self._running = False
        self._lock = threading.RLock()
        
        # Initialize components
        self._setup_tracing()
        self._setup_metrics()
        self._setup_anomaly_detection()
        
        # Connect to Scrapy signals
        self._connect_signals()
        
        self.logger.info(f"Observability dashboard initialized (enabled: {self.enabled})")
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create dashboard from crawler instance."""
        if not crawler.settings.getbool('OBSERVABILITY_ENABLED', True):
            raise NotConfigured("Observability dashboard disabled")
        
        dashboard = cls(crawler)
        return dashboard
    
    def _setup_tracing(self):
        """Set up distributed tracing with OpenTelemetry."""
        if not OPENTELEMETRY_AVAILABLE:
            self.logger.warning("OpenTelemetry not available. Tracing disabled.")
            self.tracer = None
            return
        
        try:
            resource = Resource.create({
                "service.name": "vex",
                "service.version": "2.0.0",
            })
            
            # Set up tracer
            trace.set_tracer_provider(TracerProvider(resource=resource))
            self.tracer = trace.get_tracer(__name__)
            
            # Set up OTLP exporter if configured
            otlp_endpoint = self.settings.get('OBSERVABILITY_OTLP_ENDPOINT')
            if otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.logger.info("Distributed tracing initialized")
        except Exception as e:
            self.logger.error(f"Failed to setup tracing: {e}")
            self.tracer = None
    
    def _setup_metrics(self):
        """Set up Prometheus metrics collection."""
        if not PROMETHEUS_AVAILABLE:
            self.logger.warning("Prometheus client not available. Metrics collection disabled.")
            self.metrics_enabled = False
            return
        
        try:
            # Start Prometheus HTTP server
            start_http_server(self.metrics_port)
            
            # Define metrics
            self.metrics = {
                'requests_total': Counter('vex_requests_total', 'Total requests', ['spider', 'status']),
                'requests_duration': Histogram('vex_requests_duration_seconds', 'Request duration', 
                                             ['spider', 'status'], buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]),
                'items_scraped': Counter('vex_items_scraped_total', 'Total items scraped', ['spider']),
                'errors_total': Counter('vex_errors_total', 'Total errors', ['spider', 'error_type']),
                'active_requests': Gauge('vex_active_requests', 'Active requests', ['spider']),
                'queue_size': Gauge('vex_queue_size', 'Queue size', ['spider']),
                'response_size': Summary('vex_response_size_bytes', 'Response size', ['spider']),
                'anomalies_detected': Counter('vex_anomalies_detected_total', 'Anomalies detected', 
                                            ['spider', 'severity']),
            }
            
            self.metrics_enabled = True
            self.logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            self.logger.error(f"Failed to setup metrics: {e}")
            self.metrics_enabled = False
    
    def _setup_anomaly_detection(self):
        """Set up ML-based anomaly detection."""
        if not self.anomaly_detection_enabled:
            self.anomaly_detector = None
            return
        
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn not available. Using rule-based anomaly detection.")
            self.anomaly_detector = RuleBasedAnomalyDetector()
        else:
            self.anomaly_detector = MLAnomalyDetector()
        
        # Start anomaly detection thread
        self._anomaly_thread = threading.Thread(target=self._anomaly_detection_loop, daemon=True)
        self._anomaly_thread.start()
    
    def _connect_signals(self):
        """Connect to Scrapy signals for observability."""
        self.crawler.signals.connect(self.request_scheduled, signal=signals.request_scheduled)
        self.crawler.signals.connect(self.request_downloaded, signal=signals.request_downloaded)
        self.crawler.signals.connect(self.response_received, signal=signals.response_received)
        self.crawler.signals.connect(self.item_scraped, signal=signals.item_scraped)
        self.crawler.signals.connect(self.item_error, signal=signals.item_error)
        self.crawler.signals.connect(self.spider_error, signal=signals.spider_error)
        self.crawler.signals.connect(self.spider_opened, signal=signals.spider_opened)
        self.crawler.signals.connect(self.spider_closed, signal=signals.spider_closed)
        
        # Start dashboard server
        if self.enabled:
            self._start_dashboard_server()
    
    def _start_dashboard_server(self):
        """Start the WebSocket-based live dashboard server."""
        if not WEBSOCKETS_AVAILABLE:
            self.logger.warning("websockets library not available. Live dashboard disabled.")
            return
        
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def handler(websocket, path):
                self._websocket_clients.add(websocket)
                try:
                    # Send initial state
                    await self._send_initial_state(websocket)
                    
                    # Keep connection alive
                    async for message in websocket:
                        await self._handle_websocket_message(websocket, message)
                finally:
                    self._websocket_clients.discard(websocket)
            
            start_server = serve(handler, "0.0.0.0", self.dashboard_port)
            self.logger.info(f"Dashboard WebSocket server started on port {self.dashboard_port}")
            
            loop.run_until_complete(start_server)
            loop.run_forever()
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
    
    async def _send_initial_state(self, websocket):
        """Send initial dashboard state to a new client."""
        initial_state = {
            'type': 'initial_state',
            'timestamp': time.time(),
            'traces': list(self._completed_traces)[-100:],  # Last 100 traces
            'metrics': self._get_current_metrics(),
            'anomalies': list(self._anomalies)[-50:],  # Last 50 anomalies
            'stored_requests': list(self._stored_requests.values())[-100:],  # Last 100 requests
        }
        await websocket.send(json.dumps(initial_state, default=str))
    
    async def _handle_websocket_message(self, websocket, message):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'replay_request':
                request_id = data.get('request_id')
                await self._replay_request(websocket, request_id)
            elif msg_type == 'get_trace':
                trace_id = data.get('trace_id')
                await self._send_trace_details(websocket, trace_id)
            elif msg_type == 'acknowledge_anomaly':
                anomaly_id = data.get('anomaly_id')
                self._acknowledge_anomaly(anomaly_id)
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
    
    async def _replay_request(self, websocket, request_id):
        """Replay a stored request."""
        if request_id not in self._stored_requests:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Request {request_id} not found'
            }))
            return
        
        stored_request = self._stored_requests[request_id]
        
        try:
            # Create a new request from stored data
            request_data = stored_request.request
            request = Request(
                url=request_data['url'],
                method=request_data.get('method', 'GET'),
                headers=request_data.get('headers', {}),
                body=request_data.get('body'),
                meta=request_data.get('meta', {}),
                dont_filter=True  # Allow replaying same request
            )
            
            # Schedule the request
            self.crawler.engine.crawl(request)
            
            await websocket.send(json.dumps({
                'type': 'replay_started',
                'request_id': request_id,
                'timestamp': time.time()
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                'type': 'replay_error',
                'request_id': request_id,
                'error': str(e)
            }))
    
    async def _send_trace_details(self, websocket, trace_id):
        """Send detailed trace information."""
        trace = None
        for t in self._completed_traces:
            if t.trace_id == trace_id:
                trace = t
                break
        
        if trace:
            await websocket.send(json.dumps({
                'type': 'trace_details',
                'trace': asdict(trace)
            }, default=str))
    
    def _acknowledge_anomaly(self, anomaly_id):
        """Mark an anomaly as acknowledged."""
        for anomaly in self._anomalies:
            if anomaly.id == anomaly_id:
                anomaly.resolved = True
                anomaly.resolution_time = time.time()
                break
    
    def _anomaly_detection_loop(self):
        """Background thread for anomaly detection."""
        while self._running:
            try:
                time.sleep(self.anomaly_detection_interval)
                self._detect_anomalies()
            except Exception as e:
                self.logger.error(f"Error in anomaly detection loop: {e}")
    
    def _detect_anomalies(self):
        """Detect anomalies in scraping patterns."""
        if not self.anomaly_detector:
            return
        
        try:
            # Collect metrics for analysis
            metrics_data = self._collect_metrics_for_analysis()
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(metrics_data)
            
            # Store and notify about new anomalies
            for anomaly_data in anomalies:
                anomaly = Anomaly(
                    id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    **anomaly_data
                )
                self._anomalies.append(anomaly)
                
                # Update metrics
                if self.metrics_enabled:
                    self.metrics['anomalies_detected'].labels(
                        spider=self.crawler.spider.name if self.crawler.spider else 'unknown',
                        severity=anomaly.severity
                    ).inc()
                
                # Notify via WebSocket
                asyncio.run(self._notify_anomaly(anomaly))
                
                self.logger.warning(f"Anomaly detected: {anomaly.description}")
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
    
    def _collect_metrics_for_analysis(self) -> Dict[str, Any]:
        """Collect metrics for anomaly detection analysis."""
        with self._lock:
            # Calculate error rate
            total_requests = len(self._completed_traces)
            error_requests = sum(1 for t in self._completed_traces if t.error)
            error_rate = error_requests / total_requests if total_requests > 0 else 0
            
            # Calculate average response time
            response_times = [t.end_time - t.start_time for t in self._completed_traces 
                            if t.end_time and t.start_time]
            avg_response_time = statistics.mean(response_times) if response_times else 0
            
            # Calculate response time variance
            response_time_variance = statistics.variance(response_times) if len(response_times) > 1 else 0
            
            # Get recent anomalies
            recent_anomalies = [a for a in self._anomalies 
                              if time.time() - a.timestamp < 300]  # Last 5 minutes
            
            return {
                'error_rate': error_rate,
                'avg_response_time': avg_response_time,
                'response_time_variance': response_time_variance,
                'total_requests': total_requests,
                'error_requests': error_requests,
                'recent_anomalies_count': len(recent_anomalies),
                'timestamp': time.time(),
            }
    
    async def _notify_anomaly(self, anomaly: Anomaly):
        """Notify WebSocket clients about a new anomaly."""
        if not self._websocket_clients:
            return
        
        message = json.dumps({
            'type': 'anomaly_detected',
            'anomaly': asdict(anomaly)
        }, default=str)
        
        # Send to all connected clients
        for client in self._websocket_clients.copy():
            try:
                await client.send(message)
            except:
                self._websocket_clients.discard(client)
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return {
                'active_traces': len(self._active_traces),
                'completed_traces': len(self._completed_traces),
                'stored_requests': len(self._stored_requests),
                'anomalies': len(self._anomalies),
                'websocket_clients': len(self._websocket_clients),
            }
    
    def request_scheduled(self, request, spider):
        """Handle request scheduled signal."""
        if not self.enabled:
            return
        
        # Generate trace context
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        # Create trace
        trace = RequestTrace(
            trace_id=trace_id,
            span_id=span_id,
            request_id=request.meta.get('request_id', str(uuid.uuid4())),
            url=request.url,
            method=request.method,
            start_time=time.time(),
            metadata={
                'spider': spider.name,
                'meta': dict(request.meta),
            },
            tags={
                'component': 'scheduler',
                'spider.name': spider.name,
            }
        )
        
        with self._lock:
            self._active_traces[trace_id] = trace
        
        # Add trace context to request
        request.meta['_observability'] = {
            'trace_id': trace_id,
            'span_id': span_id,
        }
        
        # Update metrics
        if self.metrics_enabled:
            self.metrics['active_requests'].labels(spider=spider.name).inc()
            self.metrics['queue_size'].labels(spider=spider.name).set(
                len(self.crawler.engine.slot.scheduler) if hasattr(self.crawler.engine, 'slot') else 0
            )
    
    def request_downloaded(self, request, response, spider):
        """Handle request downloaded signal."""
        if not self.enabled:
            return
        
        trace_id = request.meta.get('_observability', {}).get('trace_id')
        if not trace_id:
            return
        
        with self._lock:
            trace = self._active_traces.get(trace_id)
            if trace:
                trace.status_code = response.status if response else None
                trace.end_time = time.time()
                
                # Store request for replay
                self._store_request(request, response, spider)
    
    def response_received(self, request, response, spider):
        """Handle response received signal."""
        if not self.enabled:
            return
        
        trace_id = request.meta.get('_observability', {}).get('trace_id')
        if not trace_id:
            return
        
        with self._lock:
            trace = self._active_traces.pop(trace_id, None)
            if trace:
                trace.end_time = time.time()
                trace.status_code = response.status
                self._completed_traces.append(trace)
        
        # Update metrics
        if self.metrics_enabled:
            duration = trace.end_time - trace.start_time if trace else 0
            self.metrics['requests_total'].labels(
                spider=spider.name, 
                status=response.status if response else 'error'
            ).inc()
            self.metrics['requests_duration'].labels(
                spider=spider.name,
                status=response.status if response else 'error'
            ).observe(duration)
            self.metrics['active_requests'].labels(spider=spider.name).dec()
            self.metrics['response_size'].labels(spider=spider.name).observe(
                len(response.body) if response else 0
            )
    
    def item_scraped(self, item, response, spider):
        """Handle item scraped signal."""
        if not self.enabled:
            return
        
        if self.metrics_enabled:
            self.metrics['items_scraped'].labels(spider=spider.name).inc()
    
    def item_error(self, item, response, spider, failure):
        """Handle item error signal."""
        if not self.enabled:
            return
        
        if self.metrics_enabled:
            self.metrics['errors_total'].labels(
                spider=spider.name,
                error_type='item_error'
            ).inc()
    
    def spider_error(self, failure, response, spider):
        """Handle spider error signal."""
        if not self.enabled:
            return
        
        trace_id = response.meta.get('_observability', {}).get('trace_id') if response else None
        
        with self._lock:
            if trace_id:
                trace = self._active_traces.get(trace_id)
                if trace:
                    trace.error = str(failure)
                    trace.end_time = time.time()
        
        if self.metrics_enabled:
            self.metrics['errors_total'].labels(
                spider=spider.name,
                error_type='spider_error'
            ).inc()
    
    def spider_opened(self, spider):
        """Handle spider opened signal."""
        self.logger.info(f"Spider opened: {spider.name}")
        
        # Start OpenTelemetry span for spider
        if self.tracer:
            self._spider_span = self.tracer.start_span(f"spider_{spider.name}")
            self._spider_span.set_attribute("spider.name", spider.name)
    
    def spider_closed(self, spider):
        """Handle spider closed signal."""
        self.logger.info(f"Spider closed: {spider.name}")
        
        # End OpenTelemetry span
        if hasattr(self, '_spider_span'):
            self._spider_span.end()
        
        # Clean up
        self._running = False
    
    def _store_request(self, request, response, spider):
        """Store request for replay capabilities."""
        request_id = hashlib.md5(f"{request.url}{request.method}{time.time()}".encode()).hexdigest()
        
        request_data = {
            'url': request.url,
            'method': request.method,
            'headers': dict(request.headers),
            'body': request.body.decode('utf-8', errors='ignore') if request.body else None,
            'meta': dict(request.meta),
        }
        
        response_data = None
        if response:
            response_data = {
                'url': response.url,
                'status': response.status,
                'headers': dict(response.headers),
                'body': response.body.decode('utf-8', errors='ignore')[:10000],  # Limit body size
                'flags': list(response.flags),
            }
        
        stored_request = StoredRequest(
            id=request_id,
            request=request_data,
            response=response_data,
            timestamp=time.time(),
            spider_name=spider.name,
            metadata={
                'trace_id': request.meta.get('_observability', {}).get('trace_id'),
            }
        )
        
        with self._lock:
            self._stored_requests[request_id] = stored_request
            
            # Limit stored requests
            if len(self._stored_requests) > self.max_stored_requests:
                # Remove oldest requests
                oldest_id = min(self._stored_requests.keys(), 
                              key=lambda k: self._stored_requests[k].timestamp)
                del self._stored_requests[oldest_id]
    
    def get_trace(self, trace_id: str) -> Optional[RequestTrace]:
        """Get a specific trace by ID."""
        with self._lock:
            # Check active traces
            if trace_id in self._active_traces:
                return self._active_traces[trace_id]
            
            # Check completed traces
            for trace in self._completed_traces:
                if trace.trace_id == trace_id:
                    return trace
        
        return None
    
    def get_recent_traces(self, limit: int = 100) -> List[RequestTrace]:
        """Get recent traces."""
        with self._lock:
            return list(self._completed_traces)[-limit:]
    
    def get_stored_requests(self, limit: int = 100) -> List[StoredRequest]:
        """Get stored requests for replay."""
        with self._lock:
            return sorted(self._stored_requests.values(), 
                         key=lambda r: r.timestamp, reverse=True)[:limit]
    
    def get_anomalies(self, limit: int = 100) -> List[Anomaly]:
        """Get detected anomalies."""
        with self._lock:
            return list(self._anomalies)[-limit:]
    
    def replay_request(self, request_id: str) -> bool:
        """Replay a stored request."""
        if request_id not in self._stored_requests:
            return False
        
        stored_request = self._stored_requests[request_id]
        
        try:
            request_data = stored_request.request
            request = Request(
                url=request_data['url'],
                method=request_data.get('method', 'GET'),
                headers=request_data.get('headers', {}),
                body=request_data.get('body'),
                meta=request_data.get('meta', {}),
                dont_filter=True
            )
            
            self.crawler.engine.crawl(request)
            return True
        except Exception as e:
            self.logger.error(f"Failed to replay request {request_id}: {e}")
            return False


class RuleBasedAnomalyDetector:
    """Rule-based anomaly detector for when ML is not available."""
    
    def __init__(self):
        self.rules = [
            self._check_error_rate,
            self._check_response_time,
            self._check_request_volume,
        ]
    
    def detect_anomalies(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies using rules."""
        anomalies = []
        
        for rule in self.rules:
            anomaly = rule(metrics_data)
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    def _check_error_rate(self, metrics_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for high error rate."""
        error_rate = metrics_data.get('error_rate', 0)
        
        if error_rate > 0.5:  # More than 50% errors
            return {
                'anomaly_type': 'high_error_rate',
                'severity': 'high',
                'description': f'High error rate detected: {error_rate:.1%}',
                'metrics': {'error_rate': error_rate},
            }
        elif error_rate > 0.2:  # More than 20% errors
            return {
                'anomaly_type': 'elevated_error_rate',
                'severity': 'medium',
                'description': f'Elevated error rate detected: {error_rate:.1%}',
                'metrics': {'error_rate': error_rate},
            }
        
        return None
    
    def _check_response_time(self, metrics_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for slow response times."""
        avg_response_time = metrics_data.get('avg_response_time', 0)
        
        if avg_response_time > 10.0:  # More than 10 seconds average
            return {
                'anomaly_type': 'slow_response_time',
                'severity': 'high',
                'description': f'Slow response time detected: {avg_response_time:.1f}s average',
                'metrics': {'avg_response_time': avg_response_time},
            }
        
        return None
    
    def _check_request_volume(self, metrics_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for abnormal request volume."""
        total_requests = metrics_data.get('total_requests', 0)
        
        # This would need historical data for proper detection
        # For now, just check if we have a sudden drop
        if total_requests == 0 and time.time() - metrics_data.get('timestamp', 0) > 300:
            return {
                'anomaly_type': 'no_requests',
                'severity': 'critical',
                'description': 'No requests processed in the last 5 minutes',
                'metrics': {'total_requests': total_requests},
            }
        
        return None


class MLAnomalyDetector:
    """ML-based anomaly detector using scikit-learn."""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.training_data = []
        self.is_trained = False
    
    def detect_anomalies(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies using ML model."""
        # Extract features
        features = self._extract_features(metrics_data)
        
        if not self.is_trained:
            # Collect training data
            self.training_data.append(features)
            
            # Train after collecting enough data
            if len(self.training_data) >= 100:
                self._train_model()
            
            return []
        
        # Predict anomaly
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)
        
        if prediction[0] == -1:  # Anomaly detected
            # Calculate anomaly score
            anomaly_score = self.model.score_samples(features_scaled)[0]
            
            return [{
                'anomaly_type': 'ml_detected_anomaly',
                'severity': self._score_to_severity(anomaly_score),
                'description': f'ML anomaly detected (score: {anomaly_score:.2f})',
                'metrics': metrics_data,
            }]
        
        return []
    
    def _extract_features(self, metrics_data: Dict[str, Any]) -> List[float]:
        """Extract features from metrics data."""
        return [
            metrics_data.get('error_rate', 0),
            metrics_data.get('avg_response_time', 0),
            metrics_data.get('response_time_variance', 0),
            metrics_data.get('total_requests', 0),
            metrics_data.get('error_requests', 0),
            metrics_data.get('recent_anomalies_count', 0),
        ]
    
    def _train_model(self):
        """Train the anomaly detection model."""
        try:
            X = self.scaler.fit_transform(self.training_data)
            self.model.fit(X)
            self.is_trained = True
            self.logger.info("ML anomaly detection model trained")
        except Exception as e:
            self.logger.error(f"Failed to train ML model: {e}")
    
    def _score_to_severity(self, score: float) -> str:
        """Convert anomaly score to severity level."""
        if score < -0.5:
            return 'critical'
        elif score < -0.3:
            return 'high'
        elif score < -0.1:
            return 'medium'
        else:
            return 'low'


# Extension integration
class ObservabilityExtension:
    """Scrapy extension for observability dashboard."""
    
    def __init__(self, crawler):
        self.dashboard = ObservabilityDashboard.from_crawler(crawler)
    
    @classmethod
    def from_crawler(cls, crawler):
        extension = cls(crawler)
        crawler.signals.connect(extension.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(extension.spider_closed, signal=signals.spider_closed)
        return extension
    
    def spider_opened(self, spider):
        self.dashboard.spider_opened(spider)
    
    def spider_closed(self, spider):
        self.dashboard.spider_closed(spider)


# Middleware integration
class ObservabilityMiddleware:
    """Scrapy middleware for observability."""
    
    def __init__(self, crawler):
        self.dashboard = ObservabilityDashboard.from_crawler(crawler)
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler)
    
    def process_request(self, request, spider):
        self.dashboard.request_scheduled(request, spider)
    
    def process_response(self, request, response, spider):
        self.dashboard.response_received(request, response, spider)
        return response
    
    def process_exception(self, request, exception, spider):
        # Handle exception
        pass


# Settings defaults
DEFAULT_SETTINGS = {
    'OBSERVABILITY_ENABLED': True,
    'OBSERVABILITY_DASHBOARD_PORT': 6800,
    'OBSERVABILITY_METRICS_PORT': 9090,
    'OBSERVABILITY_TRACE_SAMPLE_RATE': 1.0,
    'OBSERVABILITY_MAX_STORED_REQUESTS': 10000,
    'OBSERVABILITY_ANOMALY_DETECTION': True,
    'OBSERVABILITY_ANOMALY_INTERVAL': 60,
    'EXTENSIONS': {
        'vex.observability.dashboard.ObservabilityExtension': 500,
    },
    'DOWNLOADER_MIDDLEWARES': {
        'vex.observability.dashboard.ObservabilityMiddleware': 500,
    },
}


def configure_observability(settings):
    """Configure observability settings."""
    for key, value in DEFAULT_SETTINGS.items():
        if key not in settings:
            settings.set(key, value)
    return settings


# CLI command integration
class ObservabilityCommand:
    """Scrapy command for observability dashboard."""
    
    def short_desc(self):
        return "Start observability dashboard"
    
    def add_options(self, parser):
        parser.add_argument("--port", dest="port", type=int, default=6800,
                          help="Dashboard port (default: 6800)")
        parser.add_argument("--metrics-port", dest="metrics_port", type=int, default=9090,
                          help="Metrics port (default: 9090)")
    
    def run(self, args, opts):
        # This would start a standalone dashboard server
        # For integration, the dashboard is started with the spider
        pass


# Utility functions
def start_dashboard(crawler):
    """Start the observability dashboard for a crawler."""
    return ObservabilityDashboard.from_crawler(crawler)


def get_dashboard():
    """Get the current dashboard instance (if available)."""
    # This would need to be implemented with a global registry
    return None


# Example usage in settings.py:
"""
# Enable observability
OBSERVABILITY_ENABLED = True
OBSERVABILITY_DASHBOARD_PORT = 6800
OBSERVABILITY_METRICS_PORT = 9090
OBSERVABILITY_TRACE_SAMPLE_RATE = 1.0
OBSERVABILITY_MAX_STORED_REQUESTS = 10000
OBSERVABILITY_ANOMALY_DETECTION = True
OBSERVABILITY_ANOMALY_INTERVAL = 60

# OpenTelemetry configuration (optional)
OBSERVABILITY_OTLP_ENDPOINT = "http://localhost:4317"

# Extensions and middleware
EXTENSIONS = {
    'vex.observability.dashboard.ObservabilityExtension': 500,
}

DOWNLOADER_MIDDLEWARES = {
    'vex.observability.dashboard.ObservabilityMiddleware': 500,
}
"""