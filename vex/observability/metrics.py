# vex/observability/metrics.py
"""
Real-time Observability Stack for Scrapy
Integrated distributed tracing, metrics collection, live debugging dashboard,
automatic anomaly detection, and request replay capabilities.
"""

import asyncio
import json
import time
import uuid
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Awaitable
from urllib.parse import urlparse
import pickle
import hashlib

# Optional imports with graceful fallback
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    from prometheus_client.core import CollectorRegistry, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    import websockets
    from websockets.server import serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

from twisted.internet import reactor, defer
from twisted.internet.task import LoopingCall
from twisted.web import server, resource
from twisted.web.http import Request
from twisted.python import log

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request as ScrapyRequest, Response
from vex.utils.misc import load_object
from vex.utils.defer import maybe_deferred_to_future
from vex.utils.log import failure_to_exc_info


@dataclass
class TraceContext:
    """Distributed tracing context for request correlation."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None
    sampled: bool = True
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to W3C Trace Context headers."""
        return {
            'traceparent': f'00-{self.trace_id}-{self.span_id}-01',
            'tracestate': ''
        }
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> 'TraceContext':
        """Extract trace context from request headers."""
        traceparent = headers.get('traceparent', '')
        if traceparent and len(traceparent.split('-')) == 4:
            _, trace_id, span_id, _ = traceparent.split('-')
            return cls(trace_id=trace_id, span_id=span_id)
        return cls()


@dataclass
class RequestMetrics:
    """Comprehensive metrics for a single request."""
    request_id: str
    url: str
    domain: str
    method: str
    start_time: float
    end_time: Optional[float] = None
    status_code: Optional[int] = None
    response_size: Optional[int] = None
    retries: int = 0
    error: Optional[str] = None
    error_type: Optional[str] = None
    trace_context: Optional[TraceContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_success(self) -> bool:
        return self.status_code and 200 <= self.status_code < 300
    
    @property
    def is_failure(self) -> bool:
        return self.error is not None or (self.status_code and self.status_code >= 400)


@dataclass
class AnomalyPattern:
    """Pattern for anomaly detection."""
    pattern_id: str
    description: str
    condition: Callable[[RequestMetrics], bool]
    severity: str = 'warning'  # info, warning, critical
    cooldown_seconds: int = 300
    last_triggered: Optional[float] = None


class MetricsCollector:
    """Central metrics collection and aggregation."""
    
    def __init__(self, settings):
        self.settings = settings
        self.requests: Dict[str, RequestMetrics] = {}
        self.completed_requests: deque = deque(maxlen=10000)
        self.domain_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'failures': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'last_error': None,
            'error_types': defaultdict(int)
        })
        
        # Time-series data for anomaly detection
        self.time_series: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_patterns: List[AnomalyPattern] = []
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        # OpenTelemetry tracer (if available)
        self.tracer = None
        if OPENTELEMETRY_AVAILABLE and settings.getbool('OTEL_ENABLED', False):
            self._setup_opentelemetry()
        
        # WebSocket connections for live dashboard
        self.websocket_clients: Set[Any] = set()
        
        # ML model for anomaly detection
        self.ml_model = None
        self.scaler = None
        if ML_AVAILABLE and settings.getbool('ANOMALY_DETECTION_ML', True):
            self._setup_ml_model()
    
    def _setup_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        self.request_counter = Counter(
            'vex_requests_total',
            'Total requests made',
            ['domain', 'method', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'vex_request_duration_seconds',
            'Request duration in seconds',
            ['domain', 'method'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.response_size = Histogram(
            'vex_response_size_bytes',
            'Response size in bytes',
            ['domain'],
            buckets=[100, 1000, 10000, 100000, 1000000],
            registry=self.registry
        )
        
        self.active_requests = Gauge(
            'vex_active_requests',
            'Number of active requests',
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'vex_errors_total',
            'Total errors by type',
            ['domain', 'error_type'],
            registry=self.registry
        )
    
    def _setup_opentelemetry(self):
        """Initialize OpenTelemetry tracing."""
        try:
            provider = TracerProvider()
            processor = BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=self.settings.get('OTEL_EXPORTER_OTLP_ENDPOINT', 'localhost:4317')
                )
            )
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer("vex.observability")
        except Exception as e:
            log.msg(f"Failed to setup OpenTelemetry: {e}", level=logging.WARNING)
    
    def _setup_ml_model(self):
        """Initialize ML model for anomaly detection."""
        try:
            self.ml_model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
        except Exception as e:
            log.msg(f"Failed to setup ML model: {e}", level=logging.WARNING)
            self.ml_model = None
    
    def start_request(self, request: ScrapyRequest, trace_context: Optional[TraceContext] = None) -> str:
        """Start tracking a request."""
        request_id = str(uuid.uuid4())
        parsed_url = urlparse(request.url)
        
        metrics = RequestMetrics(
            request_id=request_id,
            url=request.url,
            domain=parsed_url.netloc,
            method=request.method,
            start_time=time.time(),
            trace_context=trace_context or TraceContext()
        )
        
        self.requests[request_id] = metrics
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.active_requests.inc()
        
        # Broadcast to WebSocket clients
        self._broadcast_event({
            'type': 'request_start',
            'request_id': request_id,
            'url': request.url,
            'domain': parsed_url.netloc,
            'method': request.method,
            'timestamp': metrics.start_time,
            'trace_id': metrics.trace_context.trace_id if metrics.trace_context else None
        })
        
        return request_id
    
    def end_request(self, request_id: str, response: Optional[Response] = None, 
                   error: Optional[Exception] = None, error_type: Optional[str] = None):
        """End tracking a request."""
        if request_id not in self.requests:
            return
        
        metrics = self.requests[request_id]
        metrics.end_time = time.time()
        
        if response:
            metrics.status_code = response.status
            metrics.response_size = len(response.body)
        
        if error:
            metrics.error = str(error)
            metrics.error_type = error_type or type(error).__name__
        
        # Update domain statistics
        domain_stats = self.domain_stats[metrics.domain]
        domain_stats['total'] += 1
        
        if metrics.is_success:
            domain_stats['success'] += 1
        elif metrics.is_failure:
            domain_stats['failures'] += 1
            if metrics.error_type:
                domain_stats['error_types'][metrics.error_type] += 1
            domain_stats['last_error'] = metrics.error
        
        if metrics.duration:
            domain_stats['total_time'] += metrics.duration
            domain_stats['avg_time'] = domain_stats['total_time'] / domain_stats['total']
        
        # Update time series for anomaly detection
        self._update_time_series(metrics)
        
        # Check for anomalies
        self._check_anomalies(metrics)
        
        # Update Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self.active_requests.dec()
            status_label = 'success' if metrics.is_success else 'failure'
            self.request_counter.labels(
                domain=metrics.domain,
                method=metrics.method,
                status=status_label
            ).inc()
            
            if metrics.duration:
                self.request_duration.labels(
                    domain=metrics.domain,
                    method=metrics.method
                ).observe(metrics.duration)
            
            if metrics.response_size:
                self.response_size.labels(domain=metrics.domain).observe(metrics.response_size)
            
            if metrics.error_type:
                self.error_counter.labels(
                    domain=metrics.domain,
                    error_type=metrics.error_type
                ).inc()
        
        # Create OpenTelemetry span
        if self.tracer and metrics.trace_context:
            with self.tracer.start_as_current_span(
                name=f"{metrics.method} {metrics.url}",
                attributes={
                    "http.url": metrics.url,
                    "http.method": metrics.method,
                    "http.status_code": metrics.status_code,
                    "vex.domain": metrics.domain,
                    "vex.request_id": request_id
                }
            ) as span:
                if error:
                    span.record_exception(error)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
        
        # Broadcast to WebSocket clients
        self._broadcast_event({
            'type': 'request_end',
            'request_id': request_id,
            'url': metrics.url,
            'domain': metrics.domain,
            'status_code': metrics.status_code,
            'duration': metrics.duration,
            'success': metrics.is_success,
            'error': metrics.error,
            'error_type': metrics.error_type,
            'timestamp': metrics.end_time,
            'trace_id': metrics.trace_context.trace_id if metrics.trace_context else None
        })
        
        # Move to completed requests
        self.completed_requests.append(metrics)
        del self.requests[request_id]
    
    def _update_time_series(self, metrics: RequestMetrics):
        """Update time series data for anomaly detection."""
        timestamp = metrics.end_time or time.time()
        
        # Add to time series
        self.time_series['requests'].append((timestamp, 1))
        self.time_series['success'].append((timestamp, 1 if metrics.is_success else 0))
        self.time_series['failures'].append((timestamp, 1 if metrics.is_failure else 0))
        
        if metrics.duration:
            self.time_series['duration'].append((timestamp, metrics.duration))
        
        # Clean old data (keep last hour)
        cutoff = time.time() - 3600
        for key in self.time_series:
            while self.time_series[key] and self.time_series[key][0][0] < cutoff:
                self.time_series[key].popleft()
    
    def _check_anomalies(self, metrics: RequestMetrics):
        """Check for anomalies in request metrics."""
        current_time = time.time()
        
        # Rule-based anomaly detection
        for pattern in self.anomaly_patterns:
            if pattern.condition(metrics):
                if (pattern.last_triggered is None or 
                    current_time - pattern.last_triggered > pattern.cooldown_seconds):
                    pattern.last_triggered = current_time
                    self._trigger_anomaly(pattern, metrics)
        
        # ML-based anomaly detection
        if self.ml_model and len(self.completed_requests) > 100:
            self._check_ml_anomalies(metrics)
    
    def _check_ml_anomalies(self, metrics: RequestMetrics):
        """Use ML model to detect anomalies."""
        try:
            # Extract features from recent requests
            recent_requests = list(self.completed_requests)[-100:]
            features = []
            
            for req in recent_requests:
                if req.duration and req.status_code:
                    features.append([
                        req.duration,
                        1 if req.is_success else 0,
                        req.response_size or 0,
                        hash(req.domain) % 1000,  # Domain hash as feature
                        req.retries
                    ])
            
            if len(features) < 50:
                return
            
            # Scale features
            features_array = np.array(features)
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Predict anomalies
            predictions = self.ml_model.fit_predict(features_scaled)
            
            # Check if current request is an anomaly
            current_features = np.array([[
                metrics.duration or 0,
                1 if metrics.is_success else 0,
                metrics.response_size or 0,
                hash(metrics.domain) % 1000,
                metrics.retries
            ]])
            
            current_scaled = self.scaler.transform(current_features)
            prediction = self.ml_model.predict(current_scaled)
            
            if prediction[0] == -1:  # Anomaly detected
                self._trigger_ml_anomaly(metrics)
                
        except Exception as e:
            log.msg(f"ML anomaly detection failed: {e}", level=logging.DEBUG)
    
    def _trigger_anomaly(self, pattern: AnomalyPattern, metrics: RequestMetrics):
        """Trigger an anomaly alert."""
        anomaly_data = {
            'type': 'anomaly',
            'pattern_id': pattern.pattern_id,
            'description': pattern.description,
            'severity': pattern.severity,
            'request_id': metrics.request_id,
            'url': metrics.url,
            'domain': metrics.domain,
            'timestamp': time.time(),
            'metrics': {
                'duration': metrics.duration,
                'status_code': metrics.status_code,
                'error': metrics.error
            }
        }
        
        self._broadcast_event(anomaly_data)
        log.msg(f"Anomaly detected: {pattern.description} for {metrics.url}", 
                level=logging.WARNING)
    
    def _trigger_ml_anomaly(self, metrics: RequestMetrics):
        """Trigger ML-based anomaly alert."""
        anomaly_data = {
            'type': 'ml_anomaly',
            'description': 'ML model detected anomalous request pattern',
            'severity': 'warning',
            'request_id': metrics.request_id,
            'url': metrics.url,
            'domain': metrics.domain,
            'timestamp': time.time(),
            'confidence': 0.8,  # Could be calculated from model
            'features': {
                'duration': metrics.duration,
                'success': metrics.is_success,
                'response_size': metrics.response_size
            }
        }
        
        self._broadcast_event(anomaly_data)
    
    def _broadcast_event(self, event_data: Dict[str, Any]):
        """Broadcast event to all WebSocket clients."""
        if not self.websocket_clients:
            return
        
        message = json.dumps(event_data)
        disconnected = set()
        
        for client in self.websocket_clients:
            try:
                if hasattr(client, 'sendMessage'):
                    client.sendMessage(message.encode('utf-8'))
                elif hasattr(client, 'send'):
                    asyncio.create_task(client.send(message))
            except Exception:
                disconnected.add(client)
        
        # Clean up disconnected clients
        self.websocket_clients -= disconnected
    
    def register_websocket(self, client):
        """Register a WebSocket client for live updates."""
        self.websocket_clients.add(client)
    
    def unregister_websocket(self, client):
        """Unregister a WebSocket client."""
        self.websocket_clients.discard(client)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        total_requests = sum(s['total'] for s in self.domain_stats.values())
        total_success = sum(s['success'] for s in self.domain_stats.values())
        total_failures = sum(s['failures'] for s in self.domain_stats.values())
        
        success_rate = total_success / total_requests if total_requests > 0 else 0
        
        # Calculate percentiles for response times
        durations = [r.duration for r in self.completed_requests if r.duration]
        durations.sort()
        
        percentiles = {}
        if durations:
            percentiles = {
                'p50': durations[len(durations) // 2],
                'p90': durations[int(len(durations) * 0.9)],
                'p95': durations[int(len(durations) * 0.95)],
                'p99': durations[int(len(durations) * 0.99)]
            }
        
        return {
            'total_requests': total_requests,
            'success_rate': success_rate,
            'total_success': total_success,
            'total_failures': total_failures,
            'active_requests': len(self.requests),
            'completed_requests': len(self.completed_requests),
            'domain_stats': dict(self.domain_stats),
            'response_time_percentiles': percentiles,
            'timestamp': time.time()
        }
    
    def replay_request(self, request_id: str) -> Optional[ScrapyRequest]:
        """Replay a request from completed requests."""
        for metrics in self.completed_requests:
            if metrics.request_id == request_id:
                # Create a new request with the same parameters
                request = ScrapyRequest(
                    url=metrics.url,
                    method=metrics.method,
                    callback=None,  # Will need to be set by caller
                    errback=None,
                    dont_filter=True,  # Allow replaying even if duplicate
                    meta={
                        'replay_from': request_id,
                        'original_trace_id': metrics.trace_context.trace_id if metrics.trace_context else None
                    }
                )
                return request
        return None


class PrometheusResource(resource.Resource):
    """Twisted resource for Prometheus metrics endpoint."""
    
    isLeaf = True
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def render_GET(self, request: Request) -> bytes:
        """Handle GET request for metrics."""
        if not PROMETHEUS_AVAILABLE:
            request.setResponseCode(503)
            return b"Prometheus client not available"
        
        output = generate_latest(self.metrics_collector.registry)
        request.setHeader(b'content-type', CONTENT_TYPE_LATEST.encode('utf-8'))
        return output


class DashboardResource(resource.Resource):
    """Twisted resource for the live debugging dashboard."""
    
    isLeaf = False
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.putChild(b'', self._render_dashboard())
        self.putChild(b'api', self._render_api())
        self.putChild(b'replay', self._render_replay())
    
    def _render_dashboard(self) -> resource.Resource:
        """Render the main dashboard HTML."""
        class DashboardHTML(resource.Resource):
            isLeaf = True
            
            def render_GET(self, request: Request) -> bytes:
                html = self._get_dashboard_html()
                request.setHeader(b'content-type', b'text/html')
                return html.encode('utf-8')
            
            def _get_dashboard_html(self) -> str:
                return """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Scrapy Observability Dashboard</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                        .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; }
                        .metric { font-size: 24px; font-weight: bold; }
                        .success { color: green; }
                        .failure { color: red; }
                        .warning { color: orange; }
                        #requests { height: 400px; overflow-y: scroll; }
                        .request { padding: 5px; border-bottom: 1px solid #eee; }
                        .request:hover { background: #f5f5f5; }
                    </style>
                </head>
                <body>
                    <h1>Scrapy Observability Dashboard</h1>
                    <div class="container">
                        <div class="card">
                            <h3>Metrics</h3>
                            <div id="metrics">Loading...</div>
                        </div>
                        <div class="card">
                            <h3>Anomalies</h3>
                            <div id="anomalies">No anomalies detected</div>
                        </div>
                        <div class="card" style="grid-column: span 2;">
                            <h3>Live Requests</h3>
                            <div id="requests"></div>
                        </div>
                    </div>
                    <script>
                        const ws = new WebSocket('ws://' + window.location.host + '/observability/ws');
                        ws.onmessage = function(event) {
                            const data = JSON.parse(event.data);
                            if (data.type === 'request_start' || data.type === 'request_end') {
                                updateRequests(data);
                            } else if (data.type === 'anomaly' || data.type === 'ml_anomaly') {
                                updateAnomalies(data);
                            }
                        };
                        
                        function updateRequests(data) {
                            const requests = document.getElementById('requests');
                            const div = document.createElement('div');
                            div.className = 'request';
                            div.innerHTML = `
                                <strong>${data.method}</strong> ${data.url}<br>
                                <small>Status: ${data.status_code || 'pending'} | 
                                Duration: ${data.duration ? data.duration.toFixed(2) + 's' : 'pending'}</small>
                            `;
                            requests.insertBefore(div, requests.firstChild);
                        }
                        
                        function updateAnomalies(data) {
                            const anomalies = document.getElementById('anomalies');
                            const div = document.createElement('div');
                            div.className = data.severity === 'critical' ? 'failure' : 'warning';
                            div.innerHTML = `
                                <strong>${data.description}</strong><br>
                                <small>${data.url} at ${new Date(data.timestamp * 1000).toLocaleTimeString()}</small>
                            `;
                            anomalies.insertBefore(div, anomalies.firstChild);
                        }
                        
                        // Fetch initial metrics
                        fetch('/observability/api/metrics')
                            .then(r => r.json())
                            .then(data => {
                                document.getElementById('metrics').innerHTML = `
                                    <div class="metric success">${(data.success_rate * 100).toFixed(1)}%</div>
                                    <div>Success Rate</div>
                                    <div>${data.total_requests} total requests</div>
                                `;
                            });
                    </script>
                </body>
                </html>
                """
        
        return DashboardHTML()
    
    def _render_api(self) -> resource.Resource:
        """Render API endpoints."""
        class APIResource(resource.Resource):
            isLeaf = False
            
            def __init__(self, metrics_collector):
                self.metrics_collector = metrics_collector
                self.putChild(b'metrics', self._render_metrics())
                self.putChild(b'requests', self._render_requests())
                self.putChild(b'anomalies', self._render_anomalies())
            
            def _render_metrics(self) -> resource.Resource:
                class MetricsResource(resource.Resource):
                    isLeaf = True
                    
                    def render_GET(self, request: Request) -> bytes:
                        summary = self.metrics_collector.get_metrics_summary()
                        request.setHeader(b'content-type', b'application/json')
                        return json.dumps(summary).encode('utf-8')
                
                return MetricsResource(self.metrics_collector)
            
            def _render_requests(self) -> resource.Resource:
                class RequestsResource(resource.Resource):
                    isLeaf = True
                    
                    def render_GET(self, request: Request) -> bytes:
                        limit = int(request.args.get(b'limit', [100])[0])
                        requests_list = list(self.metrics_collector.completed_requests)[-limit:]
                        
                        requests_data = []
                        for req in requests_list:
                            requests_data.append({
                                'request_id': req.request_id,
                                'url': req.url,
                                'domain': req.domain,
                                'method': req.method,
                                'status_code': req.status_code,
                                'duration': req.duration,
                                'success': req.is_success,
                                'error': req.error,
                                'timestamp': req.end_time or req.start_time
                            })
                        
                        request.setHeader(b'content-type', b'application/json')
                        return json.dumps(requests_data).encode('utf-8')
                
                return RequestsResource(self.metrics_collector)
            
            def _render_anomalies(self) -> resource.Resource:
                class AnomaliesResource(resource.Resource):
                    isLeaf = True
                    
                    def render_GET(self, request: Request) -> bytes:
                        # Return recent anomalies (would need to store them)
                        anomalies = []
                        request.setHeader(b'content-type', b'application/json')
                        return json.dumps(anomalies).encode('utf-8')
                
                return AnomaliesResource(self.metrics_collector)
        
        return APIResource(self.metrics_collector)
    
    def _render_replay(self) -> resource.Resource:
        """Render request replay endpoint."""
        class ReplayResource(resource.Resource):
            isLeaf = True
            
            def render_POST(self, request: Request) -> bytes:
                try:
                    data = json.loads(request.content.read())
                    request_id = data.get('request_id')
                    
                    if not request_id:
                        request.setResponseCode(400)
                        return b"Missing request_id"
                    
                    # This would need integration with Scrapy's crawler
                    request.setHeader(b'content-type', b'application/json')
                    return json.dumps({
                        'status': 'replay_initiated',
                        'request_id': request_id,
                        'message': 'Request replay functionality requires crawler integration'
                    }).encode('utf-8')
                    
                except Exception as e:
                    request.setResponseCode(500)
                    return str(e).encode('utf-8')
        
        return ReplayResource(self.metrics_collector)


class WebSocketServer:
    """WebSocket server for real-time updates."""
    
    def __init__(self, metrics_collector: MetricsCollector, port: int = 8765):
        self.metrics_collector = metrics_collector
        self.port = port
        self.server = None
    
    async def handler(self, websocket, path):
        """Handle WebSocket connections."""
        self.metrics_collector.register_websocket(websocket)
        try:
            async for message in websocket:
                # Handle incoming messages if needed
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.metrics_collector.unregister_websocket(websocket)
    
    def start(self):
        """Start the WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            log.msg("WebSockets library not available, WebSocket server disabled", 
                    level=logging.WARNING)
            return
        
        async def start_server():
            self.server = await serve(self.handler, 'localhost', self.port)
            log.msg(f"WebSocket server started on ws://localhost:{self.port}")
        
        # Run in a separate thread
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_server())
            loop.run_forever()
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
    
    def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()


class ObservabilityExtension:
    """
    Scrapy extension for real-time observability.
    
    Provides distributed tracing, metrics collection, live debugging dashboard,
    anomaly detection, and request replay capabilities.
    """
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.metrics_collector = MetricsCollector(self.settings)
        self.websocket_server = None
        self.http_server = None
        
        # Configuration
        self.enabled = self.settings.getbool('OBSERVABILITY_ENABLED', True)
        self.dashboard_port = self.settings.getint('OBSERVABILITY_DASHBOARD_PORT', 6800)
        self.websocket_port = self.settings.getint('OBSERVABILITY_WEBSOCKET_PORT', 8765)
        self.metrics_path = self.settings.get('OBSERVABILITY_METRICS_PATH', '/metrics')
        self.dashboard_path = self.settings.get('OBSERVABILITY_DASHBOARD_PATH', '/observability')
        
        # Request tracking
        self.request_mapping: Dict[int, str] = {}  # Scrapy request fingerprint -> our request_id
        
        # Setup anomaly patterns
        self._setup_anomaly_patterns()
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create extension from crawler."""
        if not crawler.settings.getbool('OBSERVABILITY_ENABLED', True):
            raise NotConfigured
        
        extension = cls(crawler)
        crawler.signals.connect(extension.engine_started, signal=signals.engine_started)
        crawler.signals.connect(extension.engine_stopped, signal=signals.engine_stopped)
        crawler.signals.connect(extension.request_scheduled, signal=signals.request_scheduled)
        crawler.signals.connect(extension.response_received, signal=signals.response_received)
        crawler.signals.connect(extension.response_downloaded, signal=signals.response_downloaded)
        crawler.signals.connect(extension.spider_error, signal=signals.spider_error)
        crawler.signals.connect(extension.request_dropped, signal=signals.request_dropped)
        
        return extension
    
    def _setup_anomaly_patterns(self):
        """Setup default anomaly detection patterns."""
        patterns = [
            AnomalyPattern(
                pattern_id='high_failure_rate',
                description='High failure rate detected',
                condition=lambda m: m.is_failure and self._calculate_failure_rate(m.domain) > 0.3,
                severity='critical',
                cooldown_seconds=60
            ),
            AnomalyPattern(
                pattern_id='slow_response',
                description='Slow response detected',
                condition=lambda m: m.duration and m.duration > 30.0,
                severity='warning',
                cooldown_seconds=300
            ),
            AnomalyPattern(
                pattern_id='large_response',
                description='Unusually large response',
                condition=lambda m: m.response_size and m.response_size > 10 * 1024 * 1024,  # 10MB
                severity='info',
                cooldown_seconds=600
            ),
            AnomalyPattern(
                pattern_id='error_spike',
                description='Error spike detected',
                condition=lambda m: m.is_failure and self._check_error_spike(m.domain),
                severity='critical',
                cooldown_seconds=120
            )
        ]
        
        self.metrics_collector.anomaly_patterns = patterns
    
    def _calculate_failure_rate(self, domain: str) -> float:
        """Calculate failure rate for a domain."""
        stats = self.metrics_collector.domain_stats.get(domain, {})
        total = stats.get('total', 0)
        failures = stats.get('failures', 0)
        return failures / total if total > 0 else 0
    
    def _check_error_spike(self, domain: str) -> bool:
        """Check if there's an error spike for a domain."""
        # Get recent errors for this domain
        recent_errors = [
            r for r in self.metrics_collector.completed_requests
            if r.domain == domain and r.is_failure and 
            r.end_time and r.end_time > time.time() - 300  # Last 5 minutes
        ]
        
        # Check if error rate in last minute is significantly higher
        last_minute_errors = [
            r for r in recent_errors
            if r.end_time > time.time() - 60
        ]
        
        return len(last_minute_errors) > 5  # More than 5 errors in last minute
    
    def engine_started(self):
        """Called when the engine starts."""
        if not self.enabled:
            return
        
        log.msg("Starting Observability Stack", level=logging.INFO)
        
        # Start WebSocket server
        self.websocket_server = WebSocketServer(
            self.metrics_collector,
            self.websocket_port
        )
        self.websocket_server.start()
        
        # Setup HTTP endpoints
        self._setup_http_endpoints()
        
        log.msg(f"Observability dashboard available at http://localhost:{self.dashboard_port}{self.dashboard_path}",
                level=logging.INFO)
        log.msg(f"Prometheus metrics available at http://localhost:{self.dashboard_port}{self.metrics_path}",
                level=logging.INFO)
        log.msg(f"WebSocket server available at ws://localhost:{self.websocket_port}",
                level=logging.INFO)
    
    def engine_stopped(self):
        """Called when the engine stops."""
        if self.websocket_server:
            self.websocket_server.stop()
        
        if self.http_server:
            self.http_server.stopListening()
        
        log.msg("Observability Stack stopped", level=logging.INFO)
    
    def _setup_http_endpoints(self):
        """Setup HTTP endpoints for metrics and dashboard."""
        from twisted.web import server as twisted_server
        from twisted.internet import reactor
        
        root = resource.Resource()
        
        # Prometheus metrics endpoint
        if PROMETHEUS_AVAILABLE:
            root.putChild(
                self.metrics_path.lstrip('/').encode('utf-8'),
                PrometheusResource(self.metrics_collector)
            )
        
        # Dashboard endpoints
        dashboard_root = DashboardResource(self.metrics_collector)
        root.putChild(
            self.dashboard_path.lstrip('/').encode('utf-8'),
            dashboard_root
        )
        
        # Start HTTP server
        site = twisted_server.Site(root)
        self.http_server = reactor.listenTCP(self.dashboard_port, site)
    
    def request_scheduled(self, request: ScrapyRequest, spider):
        """Called when a request is scheduled."""
        if not self.enabled:
            return
        
        # Extract or create trace context
        trace_context = None
        if 'traceparent' in request.headers:
            trace_context = TraceContext.from_headers(
                dict(request.headers)
            )
        else:
            trace_context = TraceContext()
            # Add trace headers to request
            for key, value in trace_context.to_headers().items():
                request.headers[key] = value
        
        # Start tracking the request
        request_id = self.metrics_collector.start_request(request, trace_context)
        
        # Store mapping
        request_fingerprint = request fingerprint
        self.request_mapping[request_fingerprint] = request_id
        
        # Store in request meta for later retrieval
        request.meta['observability_request_id'] = request_id
        request.meta['observability_trace_context'] = trace_context
    
    def response_received(self, response: Response, request: ScrapyRequest, spider):
        """Called when a response is received."""
        if not self.enabled:
            return
        
        request_id = request.meta.get('observability_request_id')
        if request_id:
            self.metrics_collector.end_request(request_id, response=response)
    
    def response_downloaded(self, response: Response, request: ScrapyRequest, spider):
        """Called when a response is downloaded."""
        # This is called before response_received, so we might not need it
        pass
    
    def spider_error(self, failure, response, spider):
        """Called when a spider error occurs."""
        if not self.enabled:
            return
        
        request = response.request
        request_id = request.meta.get('observability_request_id')
        
        if request_id:
            error_type = failure.type.__name__ if hasattr(failure, 'type') else 'Unknown'
            self.metrics_collector.end_request(
                request_id,
                error=failure.value,
                error_type=error_type
            )
    
    def request_dropped(self, request: ScrapyRequest, spider):
        """Called when a request is dropped."""
        if not self.enabled:
            return
        
        request_id = request.meta.get('observability_request_id')
        if request_id:
            self.metrics_collector.end_request(
                request_id,
                error="Request dropped",
                error_type="RequestDropped"
            )
    
    def get_request_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get trace information for a request."""
        # Look in active requests
        if request_id in self.metrics_collector.requests:
            metrics = self.metrics_collector.requests[request_id]
            return {
                'request_id': metrics.request_id,
                'url': metrics.url,
                'trace_id': metrics.trace_context.trace_id if metrics.trace_context else None,
                'span_id': metrics.trace_context.span_id if metrics.trace_context else None,
                'status': 'active'
            }
        
        # Look in completed requests
        for metrics in self.metrics_collector.completed_requests:
            if metrics.request_id == request_id:
                return {
                    'request_id': metrics.request_id,
                    'url': metrics.url,
                    'trace_id': metrics.trace_context.trace_id if metrics.trace_context else None,
                    'span_id': metrics.trace_context.span_id if metrics.trace_context else None,
                    'status': 'completed',
                    'success': metrics.is_success,
                    'duration': metrics.duration
                }
        
        return None
    
    def replay_request(self, request_id: str) -> Optional[ScrapyRequest]:
        """Replay a completed request."""
        return self.metrics_collector.replay_request(request_id)
    
    def get_domain_health(self, domain: str) -> Dict[str, Any]:
        """Get health metrics for a specific domain."""
        stats = self.metrics_collector.domain_stats.get(domain, {})
        
        if not stats:
            return {'status': 'unknown', 'message': 'No data for domain'}
        
        total = stats.get('total', 0)
        failures = stats.get('failures', 0)
        failure_rate = failures / total if total > 0 else 0
        
        if failure_rate == 0:
            status = 'healthy'
        elif failure_rate < 0.1:
            status = 'warning'
        else:
            status = 'critical'
        
        return {
            'domain': domain,
            'status': status,
            'total_requests': total,
            'success_rate': 1 - failure_rate,
            'failure_rate': failure_rate,
            'avg_response_time': stats.get('avg_time', 0),
            'last_error': stats.get('last_error'),
            'error_breakdown': dict(stats.get('error_types', {}))
        }


class RequestReplayMiddleware:
    """Middleware for replaying requests from the observability stack."""
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.enabled = self.settings.getbool('REQUEST_REPLAY_ENABLED', True)
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('REQUEST_REPLAY_ENABLED', True):
            raise NotConfigured
        
        return cls(crawler)
    
    def process_request(self, request: ScrapyRequest, spider):
        """Process request for replay functionality."""
        if not self.enabled:
            return None
        
        # Check if this is a replay request
        if 'replay_from' in request.meta:
            original_id = request.meta['replay_from']
            
            # Add replay headers
            request.headers['X-Replay-From'] = original_id
            request.headers['X-Replay-Timestamp'] = str(time.time())
            
            log.msg(f"Replaying request from {original_id}: {request.url}", 
                    level=logging.INFO)
        
        return None


# Utility functions for integration with existing Scrapy code

def setup_observability(settings):
    """Setup observability stack with given settings."""
    if not settings.getbool('OBSERVABILITY_ENABLED', True):
        return
    
    # Add extension to settings
    extensions = settings.getdict('EXTENSIONS', {})
    extensions['vex.observability.metrics.ObservabilityExtension'] = 500
    settings.set('EXTENSIONS', extensions)
    
    # Add middleware if replay is enabled
    if settings.getbool('REQUEST_REPLAY_ENABLED', True):
        middlewares = settings.getdict('DOWNLOADER_MIDDLEWARES', {})
        middlewares['vex.observability.metrics.RequestReplayMiddleware'] = 100
        settings.set('DOWNLOADER_MIDDLEWARES', middlewares)
    
    # Set default settings
    settings.setdefault('OBSERVABILITY_ENABLED', True)
    settings.setdefault('OBSERVABILITY_DASHBOARD_PORT', 6800)
    settings.setdefault('OBSERVABILITY_WEBSOCKET_PORT', 8765)
    settings.setdefault('OBSERVABILITY_METRICS_PATH', '/metrics')
    settings.setdefault('OBSERVABILITY_DASHBOARD_PATH', '/observability')
    settings.setdefault('OTEL_ENABLED', False)
    settings.setdefault('ANOMALY_DETECTION_ML', True)
    settings.setdefault('REQUEST_REPLAY_ENABLED', True)


def get_metrics_collector(crawler) -> Optional[MetricsCollector]:
    """Get metrics collector from crawler extension."""
    for extension in crawler.extensions.middlewares:
        if isinstance(extension, ObservabilityExtension):
            return extension.metrics_collector
    return None


# Example anomaly detection patterns that can be added
CUSTOM_ANOMALY_PATTERNS = [
    AnomalyPattern(
        pattern_id='captcha_detected',
        description='CAPTCHA challenge detected',
        condition=lambda m: m.error and 'captcha' in str(m.error).lower(),
        severity='critical',
        cooldown_seconds=300
    ),
    AnomalyPattern(
        pattern_id='rate_limited',
        description='Rate limiting detected',
        condition=lambda m: m.status_code in [429, 503],
        severity='warning',
        cooldown_seconds=60
    ),
    AnomalyPattern(
        pattern_id='ip_blocked',
        description='IP address blocked',
        condition=lambda m: m.status_code in [403, 407] and 
                          any(term in str(m.error).lower() 
                              for term in ['blocked', 'banned', 'forbidden']),
        severity='critical',
        cooldown_seconds=600
    )
]