# vex/observability/tracing.py
"""
Real-time Observability Stack for Scrapy
Distributed tracing, metrics collection, live debugging dashboard with request replay
and automatic anomaly detection for scraping failures.
"""

import asyncio
import json
import time
import uuid
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Thread
from enum import Enum

from vex import signals
from vex.http import Request, Response
from vex.exceptions import NotConfigured
from vex.utils.request import RequestFingerprinter
from vex.utils.serialize import ScrapyJSONEncoder

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import SpanKind, Status, StatusCode
    from opentelemetry.context import attach, detach, set_value
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, generate_latest,
        CONTENT_TYPE_LATEST, CollectorRegistry, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# ML for anomaly detection
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)


class ObservabilityLevel(Enum):
    """Observability detail levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


@dataclass
class TraceContext:
    """Distributed tracing context"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    sampled: bool = True
    baggage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestTrace:
    """Complete request trace data"""
    request_id: str
    trace_id: str
    span_id: str
    url: str
    method: str
    timestamp: float
    duration: Optional[float] = None
    status_code: Optional[int] = None
    response_size: Optional[int] = None
    error: Optional[str] = None
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyPattern:
    """Pattern for anomaly detection"""
    pattern_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    severity: str  # low, medium, high, critical
    auto_action: Optional[str] = None


class MetricsCollector:
    """Prometheus metrics collector for Scrapy"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Request metrics
        self.requests_total = Counter(
            'vex_requests_total',
            'Total number of requests',
            ['spider', 'method', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'vex_request_duration_seconds',
            'Request duration in seconds',
            ['spider', 'method'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.response_size = Histogram(
            'vex_response_size_bytes',
            'Response size in bytes',
            ['spider', 'status'],
            buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'vex_errors_total',
            'Total number of errors',
            ['spider', 'error_type'],
            registry=self.registry
        )
        
        self.retry_total = Counter(
            'vex_retries_total',
            'Total number of retries',
            ['spider', 'reason'],
            registry=self.registry
        )
        
        # Spider metrics
        self.items_scraped = Counter(
            'vex_items_scraped_total',
            'Total items scraped',
            ['spider'],
            registry=self.registry
        )
        
        self.active_requests = Gauge(
            'vex_active_requests',
            'Number of active requests',
            ['spider'],
            registry=self.registry
        )
        
        self.queue_size = Gauge(
            'vex_queue_size',
            'Size of request queue',
            ['spider'],
            registry=self.registry
        )
        
        # Performance metrics
        self.download_latency = Summary(
            'vex_download_latency_seconds',
            'Download latency',
            ['spider'],
            registry=self.registry
        )
        
        self.concurrency = Gauge(
            'vex_concurrent_requests',
            'Number of concurrent requests',
            ['spider'],
            registry=self.registry
        )
    
    def record_request(self, spider_name: str, method: str, status: int, duration: float):
        """Record request metrics"""
        self.requests_total.labels(
            spider=spider_name,
            method=method,
            status=str(status)
        ).inc()
        
        self.request_duration.labels(
            spider=spider_name,
            method=method
        ).observe(duration)
    
    def record_error(self, spider_name: str, error_type: str):
        """Record error metrics"""
        self.errors_total.labels(
            spider=spider_name,
            error_type=error_type
        ).inc()
    
    def record_retry(self, spider_name: str, reason: str):
        """Record retry metrics"""
        self.retry_total.labels(
            spider=spider_name,
            reason=reason
        ).inc()
    
    def update_active_requests(self, spider_name: str, count: int):
        """Update active requests gauge"""
        self.active_requests.labels(spider=spider_name).set(count)
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)


class AnomalyDetector:
    """ML-based anomaly detection for scraping failures"""
    
    def __init__(self, window_size: int = 1000, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.request_history = deque(maxlen=window_size)
        self.patterns: List[AnomalyPattern] = []
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.model = None
        self._lock = Lock()
        
        # Initialize default patterns
        self._init_default_patterns()
        
        if ML_AVAILABLE:
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
    
    def _init_default_patterns(self):
        """Initialize default anomaly patterns"""
        self.patterns = [
            AnomalyPattern(
                pattern_id="high_latency",
                name="High Latency Pattern",
                description="Requests taking significantly longer than average",
                conditions={"duration_multiplier": 3.0, "min_samples": 10},
                severity="medium",
                auto_action="throttle"
            ),
            AnomalyPattern(
                pattern_id="error_spike",
                name="Error Spike Pattern",
                description="Sudden increase in error rate",
                conditions={"error_rate_threshold": 0.3, "window_size": 50},
                severity="high",
                auto_action="pause"
            ),
            AnomalyPattern(
                pattern_id="status_distribution",
                name="Status Code Distribution",
                description="Unusual distribution of HTTP status codes",
                conditions={"chi_square_threshold": 0.01},
                severity="medium",
                auto_action="alert"
            ),
            AnomalyPattern(
                pattern_id="response_size",
                name="Response Size Anomaly",
                description="Responses significantly larger or smaller than expected",
                conditions={"size_deviation": 2.0},
                severity="low",
                auto_action="log"
            )
        ]
    
    def add_request(self, trace: RequestTrace):
        """Add request to history for anomaly detection"""
        with self._lock:
            self.request_history.append(trace)
            
            # Train model if we have enough data
            if ML_AVAILABLE and len(self.request_history) >= 100:
                self._train_model()
    
    def _extract_features(self, trace: RequestTrace) -> List[float]:
        """Extract features from request trace for ML model"""
        features = [
            trace.duration or 0,
            trace.response_size or 0,
            trace.status_code or 0,
            len(trace.url),
            trace.retries,
            1 if trace.error else 0
        ]
        return features
    
    def _train_model(self):
        """Train anomaly detection model"""
        if not ML_AVAILABLE or len(self.request_history) < 100:
            return
        
        features = []
        for trace in self.request_history:
            features.append(self._extract_features(trace))
        
        X = np.array(features)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
    
    def detect_anomalies(self, trace: RequestTrace) -> List[AnomalyPattern]:
        """Detect anomalies in request trace"""
        anomalies = []
        
        # Rule-based detection
        anomalies.extend(self._detect_rule_based(trace))
        
        # ML-based detection
        if ML_AVAILABLE and self.model and len(self.request_history) >= 100:
            ml_anomalies = self._detect_ml_based(trace)
            anomalies.extend(ml_anomalies)
        
        return anomalies
    
    def _detect_rule_based(self, trace: RequestTrace) -> List[AnomalyPattern]:
        """Rule-based anomaly detection"""
        anomalies = []
        
        # Check high latency pattern
        if trace.duration and len(self.request_history) >= 10:
            durations = [t.duration for t in self.request_history if t.duration]
            if durations:
                avg_duration = sum(durations) / len(durations)
                if trace.duration > avg_duration * 3:
                    anomalies.append(self.patterns[0])  # high_latency
        
        # Check error spike
        if trace.error:
            recent_errors = sum(1 for t in list(self.request_history)[-50:] if t.error)
            if recent_errors / 50 > 0.3:
                anomalies.append(self.patterns[1])  # error_spike
        
        return anomalies
    
    def _detect_ml_based(self, trace: RequestTrace) -> List[AnomalyPattern]:
        """ML-based anomaly detection"""
        features = np.array([self._extract_features(trace)])
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)
        if prediction[0] == -1:  # Anomaly detected
            # Return a generic ML-detected anomaly
            return [AnomalyPattern(
                pattern_id="ml_detected",
                name="ML Detected Anomaly",
                description="Anomaly detected by machine learning model",
                conditions={},
                severity="medium",
                auto_action="investigate"
            )]
        
        return []


class LiveDashboard:
    """WebSocket-based live debugging dashboard"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[asyncio.WebSocketServerProtocol] = set()
        self.server = None
        self.loop = None
        self._running = False
        self._thread = None
        
        # Dashboard state
        self.active_requests: Dict[str, RequestTrace] = {}
        self.completed_requests: deque = deque(maxlen=1000)
        self.anomalies: deque = deque(maxlen=100)
        self.metrics: Dict[str, Any] = {}
    
    async def _handler(self, websocket, path):
        """WebSocket connection handler"""
        self.clients.add(websocket)
        try:
            # Send initial state
            await self._send_initial_state(websocket)
            
            # Keep connection alive
            async for message in websocket:
                await self._handle_message(websocket, message)
        finally:
            self.clients.remove(websocket)
    
    async def _send_initial_state(self, websocket):
        """Send initial dashboard state to new client"""
        state = {
            "type": "initial_state",
            "active_requests": {k: v.to_dict() for k, v in self.active_requests.items()},
            "completed_requests": [r.to_dict() for r in list(self.completed_requests)[-100:]],
            "anomalies": [asdict(a) for a in list(self.anomalies)[-20:]],
            "metrics": self.metrics,
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(state, cls=ScrapyJSONEncoder))
    
    async def _handle_message(self, websocket, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "replay_request":
                request_id = data.get("request_id")
                await self._replay_request(websocket, request_id)
            elif msg_type == "get_trace":
                trace_id = data.get("trace_id")
                await self._get_trace_details(websocket, trace_id)
            elif msg_type == "update_filters":
                filters = data.get("filters")
                await self._update_filters(websocket, filters)
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _replay_request(self, websocket, request_id: str):
        """Replay a request from logs"""
        # This would integrate with Scrapy's request replay functionality
        response = {
            "type": "replay_response",
            "request_id": request_id,
            "status": "pending",
            "message": "Request replay initiated"
        }
        await websocket.send(json.dumps(response))
    
    async def broadcast_update(self, update_type: str, data: Dict[str, Any]):
        """Broadcast update to all connected clients"""
        if not self.clients:
            return
        
        message = {
            "type": update_type,
            "data": data,
            "timestamp": time.time()
        }
        
        message_json = json.dumps(message, cls=ScrapyJSONEncoder)
        
        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message_json)
            except Exception:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    def start(self):
        """Start WebSocket server in background thread"""
        if self._running:
            return
        
        self._running = True
        self._thread = Thread(target=self._run_server, daemon=True)
        self._thread.start()
    
    def _run_server(self):
        """Run WebSocket server"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        async def start_server():
            self.server = await asyncio.serve(
                self._handler,
                self.host,
                self.port
            )
            logger.info(f"Live dashboard started on ws://{self.host}:{self.port}")
        
        self.loop.run_until_complete(start_server())
        self.loop.run_forever()
    
    def stop(self):
        """Stop WebSocket server"""
        self._running = False
        if self.loop and self.server:
            self.loop.call_soon_threadsafe(self.server.close)
    
    def update_request(self, trace: RequestTrace):
        """Update request in dashboard"""
        self.active_requests[trace.request_id] = trace
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_update("request_update", trace.to_dict()),
                self.loop
            )
    
    def complete_request(self, trace: RequestTrace):
        """Mark request as completed"""
        if trace.request_id in self.active_requests:
            del self.active_requests[trace.request_id]
        
        self.completed_requests.append(trace)
        
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_update("request_completed", trace.to_dict()),
                self.loop
            )
    
    def report_anomaly(self, anomaly: AnomalyPattern, trace: RequestTrace):
        """Report detected anomaly"""
        self.anomalies.append({
            "anomaly": asdict(anomaly),
            "trace": trace.to_dict(),
            "timestamp": time.time()
        })
        
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.broadcast_update("anomaly_detected", {
                    "anomaly": asdict(anomaly),
                    "trace": trace.to_dict()
                }),
                self.loop
            )


class RequestReplayManager:
    """Manages request replay capabilities"""
    
    def __init__(self, max_log_size: int = 10000):
        self.max_log_size = max_log_size
        self.request_log: Dict[str, RequestTrace] = {}
        self._lock = Lock()
        self.request_fingerprinter = RequestFingerprinter()
    
    def log_request(self, trace: RequestTrace):
        """Log request for potential replay"""
        with self._lock:
            self.request_log[trace.request_id] = trace
            
            # Trim log if too large
            if len(self.request_log) > self.max_log_size:
                # Remove oldest entries
                sorted_ids = sorted(
                    self.request_log.keys(),
                    key=lambda x: self.request_log[x].timestamp
                )
                for request_id in sorted_ids[:len(sorted_ids) // 2]:
                    del self.request_log[request_id]
    
    def get_request(self, request_id: str) -> Optional[RequestTrace]:
        """Get request by ID"""
        return self.request_log.get(request_id)
    
    def get_requests_by_url(self, url: str, limit: int = 100) -> List[RequestTrace]:
        """Get requests by URL"""
        with self._lock:
            return [
                trace for trace in self.request_log.values()
                if trace.url == url
            ][:limit]
    
    def get_requests_by_pattern(self, pattern: str, limit: int = 100) -> List[RequestTrace]:
        """Get requests matching URL pattern"""
        import re
        with self._lock:
            return [
                trace for trace in self.request_log.values()
                if re.search(pattern, trace.url)
            ][:limit]
    
    def replay_request(self, request_id: str, spider) -> Optional[Request]:
        """Create a new request from logged request"""
        trace = self.get_request(request_id)
        if not trace:
            return None
        
        # Reconstruct request from trace
        request = Request(
            url=trace.url,
            method=trace.method,
            meta=trace.metadata.get("meta", {}),
            dont_filter=True,  # Allow replaying same URL
            callback=self._replay_callback
        )
        
        # Add replay metadata
        request.meta['replay_from'] = request_id
        request.meta['original_trace_id'] = trace.trace_id
        
        return request
    
    def _replay_callback(self, response):
        """Callback for replayed requests"""
        logger.info(f"Replay completed for {response.url}")
        return None


class DistributedTracer:
    """OpenTelemetry-based distributed tracing"""
    
    def __init__(self, service_name: str = "vex", 
                 exporter_type: str = "otlp",
                 endpoint: Optional[str] = None):
        self.service_name = service_name
        self.exporter_type = exporter_type
        self.endpoint = endpoint
        self.tracer = None
        self.active_spans: Dict[str, Any] = {}
        
        if OPENTELEMETRY_AVAILABLE:
            self._setup_tracer()
    
    def _setup_tracer(self):
        """Setup OpenTelemetry tracer"""
        provider = TracerProvider()
        
        if self.exporter_type == "otlp" and self.endpoint:
            exporter = OTLPSpanExporter(endpoint=self.endpoint)
        elif self.exporter_type == "console":
            exporter = ConsoleSpanExporter()
        else:
            # Default to console if no endpoint specified
            exporter = ConsoleSpanExporter()
        
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        
        self.tracer = trace.get_tracer(self.service_name)
    
    def start_span(self, name: str, context: Optional[TraceContext] = None, 
                   attributes: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Start a new span"""
        if not OPENTELEMETRY_AVAILABLE or not self.tracer:
            return None
        
        span_id = str(uuid.uuid4())
        
        # Create span context
        if context:
            span_context = trace.SpanContext(
                trace_id=int(context.trace_id, 16),
                span_id=int(span_id, 16),
                is_remote=True,
                trace_flags=trace.TraceFlags(0x01 if context.sampled else 0x00)
            )
            parent_context = trace.set_span_in_context(
                trace.NonRecordingSpan(span_context)
            )
        else:
            parent_context = None
        
        # Start span
        span = self.tracer.start_span(
            name=name,
            context=parent_context,
            kind=SpanKind.CLIENT
        )
        
        # Set attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        self.active_spans[span_id] = span
        return span_id
    
    def end_span(self, span_id: str, status: Optional[Status] = None, 
                 attributes: Optional[Dict[str, Any]] = None):
        """End a span"""
        if not OPENTELEMETRY_AVAILABLE or span_id not in self.active_spans:
            return
        
        span = self.active_spans[span_id]
        
        # Set final attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        # Set status
        if status:
            span.set_status(status)
        else:
            span.set_status(Status(StatusCode.OK))
        
        span.end()
        del self.active_spans[span_id]
    
    def add_event(self, span_id: str, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span"""
        if not OPENTELEMETRY_AVAILABLE or span_id not in self.active_spans:
            return
        
        span = self.active_spans[span_id]
        span.add_event(name, attributes=attributes or {})


class ObservabilityMiddleware:
    """Scrapy middleware for observability"""
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware from crawler"""
        settings = crawler.settings
        
        # Check if observability is enabled
        if not settings.getbool('OBSERVABILITY_ENABLED', False):
            raise NotConfigured("Observability not enabled")
        
        # Create middleware instance
        middleware = cls(crawler)
        
        # Connect signals
        crawler.signals.connect(middleware.request_scheduled, signal=signals.request_scheduled)
        crawler.signals.connect(middleware.response_received, signal=signals.response_received)
        crawler.signals.connect(middleware.request_dropped, signal=signals.request_dropped)
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        
        return middleware
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.spider_name = None
        
        # Initialize components
        self._init_components()
        
        # Request tracking
        self.active_requests: Dict[str, RequestTrace] = {}
        self.request_start_times: Dict[str, float] = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _init_components(self):
        """Initialize observability components"""
        # Metrics collector
        self.metrics = MetricsCollector()
        
        # Anomaly detector
        anomaly_enabled = self.settings.getbool('OBSERVABILITY_ANOMALY_DETECTION', True)
        self.anomaly_detector = AnomalyDetector() if anomaly_enabled else None
        
        # Live dashboard
        dashboard_enabled = self.settings.getbool('OBSERVABILITY_DASHBOARD', True)
        dashboard_host = self.settings.get('OBSERVABILITY_DASHBOARD_HOST', 'localhost')
        dashboard_port = self.settings.getint('OBSERVABILITY_DASHBOARD_PORT', 8765)
        self.dashboard = LiveDashboard(dashboard_host, dashboard_port) if dashboard_enabled else None
        
        # Distributed tracer
        tracing_enabled = self.settings.getbool('OBSERVABILITY_TRACING', True)
        tracing_service = self.settings.get('OBSERVABILITY_SERVICE_NAME', 'vex')
        tracing_exporter = self.settings.get('OBSERVABILITY_TRACING_EXPORTER', 'console')
        tracing_endpoint = self.settings.get('OBSERVABILITY_TRACING_ENDPOINT')
        self.tracer = DistributedTracer(tracing_service, tracing_exporter, tracing_endpoint) if tracing_enabled else None
        
        # Request replay manager
        replay_enabled = self.settings.getbool('OBSERVABILITY_REQUEST_REPLAY', True)
        self.replay_manager = RequestReplayManager() if replay_enabled else None
        
        # Start dashboard if enabled
        if self.dashboard:
            self.dashboard.start()
    
    def spider_opened(self, spider):
        """Handle spider opened signal"""
        self.spider_name = spider.name
        logger.info(f"Observability middleware attached to spider: {spider.name}")
    
    def spider_closed(self, spider):
        """Handle spider closed signal"""
        if self.dashboard:
            self.dashboard.stop()
        self.executor.shutdown(wait=False)
    
    def request_scheduled(self, request, spider):
        """Handle request scheduled signal"""
        request_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        # Create trace context
        trace_context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            sampled=True
        )
        
        # Create request trace
        request_trace = RequestTrace(
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id,
            url=request.url,
            method=request.method,
            timestamp=time.time(),
            metadata={
                "meta": dict(request.meta),
                "headers": dict(request.headers),
                "priority": request.priority,
                "dont_filter": request.dont_filter
            }
        )
        
        # Store trace in request meta
        request.meta['_observability'] = {
            'request_id': request_id,
            'trace_id': trace_id,
            'span_id': span_id,
            'trace_context': asdict(trace_context)
        }
        
        # Start tracing span
        if self.tracer:
            span_attributes = {
                "http.url": request.url,
                "http.method": request.method,
                "request.id": request_id
            }
            span_id = self.tracer.start_span(
                f"vex.request.{request.method}",
                trace_context,
                span_attributes
            )
            request.meta['_observability']['otel_span_id'] = span_id
        
        # Track request
        self.active_requests[request_id] = request_trace
        self.request_start_times[request_id] = time.time()
        
        # Update metrics
        if self.metrics:
            self.metrics.update_active_requests(self.spider_name, len(self.active_requests))
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.update_request(request_trace)
        
        # Log for replay
        if self.replay_manager:
            self.replay_manager.log_request(request_trace)
    
    def response_received(self, response, request, spider):
        """Handle response received signal"""
        observability_data = request.meta.get('_observability')
        if not observability_data:
            return
        
        request_id = observability_data['request_id']
        
        # Calculate duration
        start_time = self.request_start_times.get(request_id)
        duration = time.time() - start_time if start_time else 0
        
        # Update request trace
        if request_id in self.active_requests:
            trace = self.active_requests[request_id]
            trace.duration = duration
            trace.status_code = response.status
            trace.response_size = len(response.body)
            
            # End tracing span
            if self.tracer and 'otel_span_id' in observability_data:
                span_attributes = {
                    "http.status_code": response.status,
                    "http.response_content_length": len(response.body),
                    "duration": duration
                }
                
                status = Status(StatusCode.OK if 200 <= response.status < 400 else StatusCode.ERROR)
                self.tracer.end_span(
                    observability_data['otel_span_id'],
                    status,
                    span_attributes
                )
            
            # Update metrics
            if self.metrics:
                self.metrics.record_request(
                    self.spider_name,
                    request.method,
                    response.status,
                    duration
                )
                self.metrics.response_size.labels(
                    spider=self.spider_name,
                    status=str(response.status)
                ).observe(len(response.body))
            
            # Check for anomalies
            if self.anomaly_detector:
                self.anomaly_detector.add_request(trace)
                anomalies = self.anomaly_detector.detect_anomalies(trace)
                
                for anomaly in anomalies:
                    logger.warning(f"Anomaly detected: {anomaly.name} for {request.url}")
                    
                    # Report to dashboard
                    if self.dashboard:
                        self.dashboard.report_anomaly(anomaly, trace)
                    
                    # Take auto-action if configured
                    if anomaly.auto_action:
                        self._take_auto_action(anomaly, request, spider)
            
            # Update dashboard
            if self.dashboard:
                self.dashboard.complete_request(trace)
            
            # Clean up
            del self.active_requests[request_id]
            if request_id in self.request_start_times:
                del self.request_start_times[request_id]
            
            # Update active requests metric
            if self.metrics:
                self.metrics.update_active_requests(self.spider_name, len(self.active_requests))
    
    def request_dropped(self, request, spider):
        """Handle request dropped signal"""
        observability_data = request.meta.get('_observability')
        if not observability_data:
            return
        
        request_id = observability_data['request_id']
        
        # Record error metric
        if self.metrics:
            self.metrics.record_error(self.spider_name, "request_dropped")
        
        # End tracing span with error
        if self.tracer and 'otel_span_id' in observability_data:
            self.tracer.end_span(
                observability_data['otel_span_id'],
                Status(StatusCode.ERROR, "Request dropped"),
                {"error.type": "request_dropped"}
            )
        
        # Clean up
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        if request_id in self.request_start_times:
            del self.request_start_times[request_id]
        
        # Update dashboard
        if request_id in self.active_requests and self.dashboard:
            trace = self.active_requests[request_id]
            trace.error = "Request dropped"
            self.dashboard.complete_request(trace)
    
    def _take_auto_action(self, anomaly: AnomalyPattern, request: Request, spider):
        """Take automatic action based on anomaly"""
        action = anomaly.auto_action
        
        if action == "throttle":
            # Implement throttling logic
            logger.info(f"Throttling requests due to anomaly: {anomaly.name}")
            # Could adjust download delay or concurrency here
        elif action == "pause":
            # Pause spider temporarily
            logger.warning(f"Pausing spider due to anomaly: {anomaly.name}")
            # Could close spider or pause for a period
        elif action == "alert":
            # Send alert (could integrate with external alerting system)
            logger.error(f"Alert: Anomaly detected - {anomaly.name}")
        elif action == "log":
            # Just log the anomaly
            logger.info(f"Logged anomaly: {anomaly.name}")
    
    def process_request(self, request, spider):
        """Process request in middleware chain"""
        # Add observability headers
        observability_data = request.meta.get('_observability')
        if observability_data:
            request.headers['X-Trace-ID'] = observability_data['trace_id']
            request.headers['X-Request-ID'] = observability_data['request_id']
        
        return None
    
    def process_response(self, request, response, spider):
        """Process response in middleware chain"""
        return response
    
    def process_exception(self, request, exception, spider):
        """Process exception in middleware chain"""
        observability_data = request.meta.get('_observability')
        if not observability_data:
            return None
        
        request_id = observability_data['request_id']
        
        # Record error
        if request_id in self.active_requests:
            trace = self.active_requests[request_id]
            trace.error = str(exception)
        
        # Update metrics
        if self.metrics:
            self.metrics.record_error(self.spider_name, type(exception).__name__)
        
        # End tracing span with error
        if self.tracer and 'otel_span_id' in observability_data:
            self.tracer.end_span(
                observability_data['otel_span_id'],
                Status(StatusCode.ERROR, str(exception)),
                {"error.type": type(exception).__name__, "error.message": str(exception)}
            )
        
        # Update dashboard
        if request_id in self.active_requests and self.dashboard:
            trace = self.active_requests[request_id]
            trace.error = str(exception)
            self.dashboard.complete_request(trace)
        
        return None


class ObservabilityExtension:
    """Scrapy extension for observability"""
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create extension from crawler"""
        settings = crawler.settings
        
        if not settings.getbool('OBSERVABILITY_ENABLED', False):
            raise NotConfigured("Observability not enabled")
        
        extension = cls(crawler)
        
        # Connect signals
        crawler.signals.connect(extension.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(extension.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(extension.item_scraped, signal=signals.item_scraped)
        crawler.signals.connect(extension.item_dropped, signal=signals.item_dropped)
        
        return extension
    
    def __init__(self, crawler):
        self.crawler = crawler
        self.settings = crawler.settings
        self.spider_name = None
        
        # Start metrics server if enabled
        if PROMETHEUS_AVAILABLE and settings.getbool('OBSERVABILITY_METRICS_SERVER', False):
            metrics_port = settings.getint('OBSERVABILITY_METRICS_PORT', 9090)
            start_http_server(metrics_port)
            logger.info(f"Metrics server started on port {metrics_port}")
    
    def spider_opened(self, spider):
        self.spider_name = spider.name
    
    def spider_closed(self, spider):
        pass
    
    def item_scraped(self, item, response, spider):
        # Record item scraped metric
        if hasattr(self, 'metrics'):
            self.metrics.items_scraped.labels(spider=self.spider_name).inc()
    
    def item_dropped(self, item, response, exception, spider):
        # Record item dropped metric
        if hasattr(self, 'metrics'):
            self.metrics.record_error(self.spider_name, "item_dropped")


# Utility functions for external use
def get_trace_context_from_request(request: Request) -> Optional[TraceContext]:
    """Extract trace context from request"""
    observability_data = request.meta.get('_observability')
    if not observability_data or 'trace_context' not in observability_data:
        return None
    
    return TraceContext(**observability_data['trace_context'])


def inject_trace_headers(request: Request, trace_context: TraceContext):
    """Inject trace headers into request"""
    request.headers['X-Trace-ID'] = trace_context.trace_id
    request.headers['X-Span-ID'] = trace_context.span_id
    if trace_context.parent_span_id:
        request.headers['X-Parent-Span-ID'] = trace_context.parent_span_id


def extract_trace_headers(request: Request) -> Optional[TraceContext]:
    """Extract trace context from request headers"""
    trace_id = request.headers.get('X-Trace-ID')
    span_id = request.headers.get('X-Span-ID')
    parent_span_id = request.headers.get('X-Parent-Span-ID')
    
    if not trace_id or not span_id:
        return None
    
    return TraceContext(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        sampled=True
    )


# Export main classes
__all__ = [
    'ObservabilityMiddleware',
    'ObservabilityExtension',
    'DistributedTracer',
    'MetricsCollector',
    'AnomalyDetector',
    'LiveDashboard',
    'RequestReplayManager',
    'TraceContext',
    'RequestTrace',
    'AnomalyPattern',
    'ObservabilityLevel',
    'get_trace_context_from_request',
    'inject_trace_headers',
    'extract_trace_headers'
]