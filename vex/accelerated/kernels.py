"""
GPU-Accelerated Parsing & Extraction for Scrapy
Offloads parsing and extraction to GPU using CUDA/OpenCL for massive parallel processing
of large document collections with automatic CPU fallback.
"""

import os
import time
import logging
import threading
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib

# Optional GPU imports - gracefully handle if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import cupy as cp
    HAS_CUPY = True
    CUPY_AVAILABLE = True
except ImportError:
    HAS_CUPY = False
    CUPY_AVAILABLE = False
    cp = None

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False
    cuda = None

try:
    import pyopencl as cl
    HAS_PYOPENCL = True
except ImportError:
    HAS_PYOPENCL = False
    cl = None

from vex.http import TextResponse
from vex.selector import Selector
from vex.utils.python import to_unicode
from vex.exceptions import NotConfigured

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_BATCH_SIZE = 1024
DEFAULT_GPU_MEMORY_LIMIT = 0.8  # 80% of available GPU memory
MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB max per document


@dataclass
class GPUMetrics:
    """Metrics for GPU acceleration performance"""
    total_documents: int = 0
    gpu_processed: int = 0
    cpu_processed: int = 0
    gpu_time: float = 0.0
    cpu_time: float = 0.0
    memory_transfers: int = 0
    batch_count: int = 0
    speedup_factor: float = 0.0


class GPUBackend:
    """Base class for GPU backend implementations"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.available = False
        self.device_name = "Unknown"
        self.total_memory = 0
        self.free_memory = 0
        
    def initialize(self) -> bool:
        """Initialize GPU backend"""
        raise NotImplementedError
        
    def allocate_memory(self, size: int) -> Any:
        """Allocate GPU memory"""
        raise NotImplementedError
        
    def free_memory(self, ptr: Any) -> None:
        """Free GPU memory"""
        raise NotImplementedError
        
    def transfer_to_gpu(self, data: Any) -> Any:
        """Transfer data to GPU"""
        raise NotImplementedError
        
    def transfer_from_gpu(self, data: Any) -> Any:
        """Transfer data from GPU"""
        raise NotImplementedError
        
    def execute_kernel(self, kernel: str, inputs: List[Any], 
                      outputs: List[Any], **kwargs) -> None:
        """Execute GPU kernel"""
        raise NotImplementedError
        
    def synchronize(self) -> None:
        """Synchronize GPU operations"""
        raise NotImplementedError


class CUDABackend(GPUBackend):
    """CUDA backend implementation"""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.context = None
        self.stream = None
        
    def initialize(self) -> bool:
        if not HAS_PYCUDA:
            return False
            
        try:
            cuda.init()
            self.device = cuda.Device(self.device_id)
            self.context = self.device.make_context()
            self.stream = cuda.Stream()
            
            # Get device info
            attrs = self.device.get_attributes()
            self.device_name = self.device.name()
            self.total_memory = self.device.total_memory()
            self.free_memory = self.total_memory * 0.9  # Conservative estimate
            
            self.available = True
            logger.info(f"CUDA initialized: {self.device_name}")
            return True
        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}")
            return False
            
    def allocate_memory(self, size: int) -> Any:
        return cuda.mem_alloc(size)
        
    def free_memory(self, ptr: Any) -> None:
        if ptr:
            ptr.free()
            
    def transfer_to_gpu(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            gpu_data = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod_async(gpu_data, data, self.stream)
            return gpu_data
        return data
        
    def transfer_from_gpu(self, data: Any) -> Any:
        if hasattr(data, 'get'):  # It's a GPU buffer
            result = np.empty(data.size, dtype=np.uint8)
            cuda.memcpy_dtoh_async(result, data, self.stream)
            return result
        return data
        
    def synchronize(self) -> None:
        if self.stream:
            self.stream.synchronize()
            
    def __del__(self):
        if self.context:
            self.context.pop()


class OpenCLBackend(GPUBackend):
    """OpenCL backend implementation"""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.context = None
        self.queue = None
        self.platform = None
        
    def initialize(self) -> bool:
        if not HAS_PYOPENCL:
            return False
            
        try:
            platforms = cl.get_platforms()
            if not platforms:
                return False
                
            self.platform = platforms[0]
            devices = self.platform.get_devices(cl.device_type.GPU)
            if not devices:
                devices = self.platform.get_devices(cl.device_type.ALL)
                
            if not devices or self.device_id >= len(devices):
                return False
                
            self.device = devices[self.device_id]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context)
            
            self.device_name = self.device.name
            self.total_memory = self.device.global_mem_size
            self.free_memory = self.total_memory * 0.9
            
            self.available = True
            logger.info(f"OpenCL initialized: {self.device_name}")
            return True
        except Exception as e:
            logger.warning(f"OpenCL initialization failed: {e}")
            return False
            
    def allocate_memory(self, size: int) -> Any:
        return cl.Buffer(self.context, cl.mem_flags.READ_WRITE, size)
        
    def free_memory(self, ptr: Any) -> None:
        if ptr:
            ptr.release()
            
    def transfer_to_gpu(self, data: Any) -> Any:
        if isinstance(data, np.ndarray):
            return cl.Buffer(self.context, cl.mem_flags.READ_ONLY | 
                           cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        return data
        
    def transfer_from_gpu(self, data: Any) -> Any:
        if isinstance(data, cl.Buffer):
            result = np.empty(data.size, dtype=np.uint8)
            cl.enqueue_copy(self.queue, result, data)
            return result
        return data
        
    def synchronize(self) -> None:
        if self.queue:
            self.queue.finish()


class CPUEmulationBackend(GPUBackend):
    """CPU emulation backend for fallback"""
    
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.device_name = "CPU Emulation"
        self.available = True
        
    def initialize(self) -> bool:
        logger.info("Using CPU emulation backend")
        return True
        
    def allocate_memory(self, size: int) -> Any:
        return bytearray(size)
        
    def free_memory(self, ptr: Any) -> None:
        pass
        
    def transfer_to_gpu(self, data: Any) -> Any:
        return data
        
    def transfer_from_gpu(self, data: Any) -> Any:
        return data
        
    def synchronize(self) -> None:
        pass


class XPathGPUEvaluator:
    """GPU-accelerated XPath evaluator"""
    
    # CUDA kernel for basic XPath operations
    CUDA_KERNEL = """
    __global__ void evaluate_xpath_batch(
        const unsigned char* documents, 
        const int* doc_offsets,
        const int* doc_lengths,
        const unsigned char* xpath_patterns,
        const int* pattern_offsets,
        const int* pattern_lengths,
        unsigned char* results,
        int* result_offsets,
        int num_docs,
        int num_patterns
    ) {
        int doc_idx = blockIdx.x * blockDim.x + threadIdx.x;
        int pattern_idx = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (doc_idx >= num_docs || pattern_idx >= num_patterns) return;
        
        // Get document boundaries
        int doc_start = doc_offsets[doc_idx];
        int doc_length = doc_lengths[doc_idx];
        
        // Get pattern boundaries
        int pattern_start = pattern_offsets[pattern_idx];
        int pattern_length = pattern_lengths[pattern_idx];
        
        // Simple tag matching (simplified for demonstration)
        // In production, this would be a full XPath parser
        int result_idx = doc_idx * num_patterns + pattern_idx;
        int matches = 0;
        
        // Basic tag matching implementation
        for (int i = doc_start; i < doc_start + doc_length - pattern_length; i++) {
            bool match = true;
            for (int j = 0; j < pattern_length; j++) {
                if (documents[i + j] != xpath_patterns[pattern_start + j]) {
                    match = false;
                    break;
                }
            }
            if (match) matches++;
        }
        
        // Store result
        results[result_idx] = matches > 0 ? 1 : 0;
        result_offsets[result_idx] = matches;
    }
    """
    
    # OpenCL kernel for basic XPath operations
    OPENCL_KERNEL = """
    __kernel void evaluate_xpath_batch(
        __global const unsigned char* documents,
        __global const int* doc_offsets,
        __global const int* doc_lengths,
        __global const unsigned char* xpath_patterns,
        __global const int* pattern_offsets,
        __global const int* pattern_lengths,
        __global unsigned char* results,
        __global int* result_offsets,
        const int num_docs,
        const int num_patterns
    ) {
        int doc_idx = get_global_id(0);
        int pattern_idx = get_global_id(1);
        
        if (doc_idx >= num_docs || pattern_idx >= num_patterns) return;
        
        // Get document boundaries
        int doc_start = doc_offsets[doc_idx];
        int doc_length = doc_lengths[doc_idx];
        
        // Get pattern boundaries
        int pattern_start = pattern_offsets[pattern_idx];
        int pattern_length = pattern_lengths[pattern_idx];
        
        // Simple tag matching
        int result_idx = doc_idx * num_patterns + pattern_idx;
        int matches = 0;
        
        for (int i = doc_start; i < doc_start + doc_length - pattern_length; i++) {
            bool match = true;
            for (int j = 0; j < pattern_length; j++) {
                if (documents[i + j] != xpath_patterns[pattern_start + j]) {
                    match = false;
                    break;
                }
            }
            if (match) matches++;
        }
        
        // Store result
        results[result_idx] = matches > 0 ? 1 : 0;
        result_offsets[result_idx] = matches;
    }
    """
    
    def __init__(self, backend: GPUBackend):
        self.backend = backend
        self.compiled_kernels = {}
        
    def compile_kernels(self) -> bool:
        """Compile GPU kernels"""
        try:
            if isinstance(self.backend, CUDABackend) and HAS_PYCUDA:
                mod = SourceModule(self.CUDA_KERNEL)
                self.compiled_kernels['cuda'] = mod.get_function("evaluate_xpath_batch")
                return True
            elif isinstance(self.backend, OpenCLBackend) and HAS_PYOPENCL:
                program = cl.Program(self.backend.context, self.OPENCL_KERNEL).build()
                self.compiled_kernels['opencl'] = program.evaluate_xpath_batch
                return True
            return False
        except Exception as e:
            logger.error(f"Kernel compilation failed: {e}")
            return False
            
    def evaluate_batch(self, documents: List[str], xpaths: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate XPath expressions on a batch of documents using GPU
        
        Args:
            documents: List of HTML document strings
            xpaths: List of XPath expressions to evaluate
            
        Returns:
            List of dictionaries with extraction results
        """
        if not documents or not xpaths:
            return []
            
        # Convert documents to bytes
        doc_bytes = [doc.encode('utf-8') for doc in documents]
        doc_lengths = [len(d) for d in doc_bytes]
        doc_offsets = [0]
        for length in doc_lengths[:-1]:
            doc_offsets.append(doc_offsets[-1] + length)
            
        # Convert XPath patterns to bytes (simplified - in reality would parse XPath)
        xpath_bytes = []
        for xpath in xpaths:
            # Extract tag names from XPath (simplified)
            # In production, this would be a proper XPath parser
            if '//' in xpath:
                tag = xpath.split('//')[1].split('/')[0].split('[')[0]
                xpath_bytes.append(f"<{tag}".encode('utf-8'))
            else:
                xpath_bytes.append(xpath.encode('utf-8'))
                
        pattern_lengths = [len(p) for p in xpath_bytes]
        pattern_offsets = [0]
        for length in pattern_lengths[:-1]:
            pattern_offsets.append(pattern_offsets[-1] + length)
            
        # Flatten data
        all_docs = b''.join(doc_bytes)
        all_patterns = b''.join(xpath_bytes)
        
        # Prepare results
        num_docs = len(documents)
        num_patterns = len(xpaths)
        results = []
        
        try:
            if isinstance(self.backend, CUDABackend) and 'cuda' in self.compiled_kernels:
                results = self._evaluate_cuda(
                    all_docs, doc_offsets, doc_lengths,
                    all_patterns, pattern_offsets, pattern_lengths,
                    num_docs, num_patterns
                )
            elif isinstance(self.backend, OpenCLBackend) and 'opencl' in self.compiled_kernels:
                results = self._evaluate_opencl(
                    all_docs, doc_offsets, doc_lengths,
                    all_patterns, pattern_offsets, pattern_lengths,
                    num_docs, num_patterns
                )
            else:
                # Fallback to CPU
                results = self._evaluate_cpu(documents, xpaths)
        except Exception as e:
            logger.error(f"GPU evaluation failed: {e}, falling back to CPU")
            results = self._evaluate_cpu(documents, xpaths)
            
        return results
        
    def _evaluate_cuda(self, all_docs, doc_offsets, doc_lengths,
                      all_patterns, pattern_offsets, pattern_lengths,
                      num_docs, num_patterns) -> List[Dict[str, Any]]:
        """Evaluate using CUDA backend"""
        import pycuda.gpuarray as gpuarray
        
        # Transfer data to GPU
        docs_gpu = gpuarray.to_gpu(np.frombuffer(all_docs, dtype=np.uint8))
        doc_offsets_gpu = gpuarray.to_gpu(np.array(doc_offsets, dtype=np.int32))
        doc_lengths_gpu = gpuarray.to_gpu(np.array(doc_lengths, dtype=np.int32))
        
        patterns_gpu = gpuarray.to_gpu(np.frombuffer(all_patterns, dtype=np.uint8))
        pattern_offsets_gpu = gpuarray.to_gpu(np.array(pattern_offsets, dtype=np.int32))
        pattern_lengths_gpu = gpuarray.to_gpu(np.array(pattern_lengths, dtype=np.int32))
        
        # Prepare output arrays
        results_gpu = gpuarray.zeros((num_docs * num_patterns,), dtype=np.uint8)
        result_offsets_gpu = gpuarray.zeros((num_docs * num_patterns,), dtype=np.int32)
        
        # Configure grid and block dimensions
        block_size = (16, 16, 1)
        grid_size = (
            (num_docs + block_size[0] - 1) // block_size[0],
            (num_patterns + block_size[1] - 1) // block_size[1],
            1
        )
        
        # Execute kernel
        kernel = self.compiled_kernels['cuda']
        kernel(
            docs_gpu, doc_offsets_gpu, doc_lengths_gpu,
            patterns_gpu, pattern_offsets_gpu, pattern_lengths_gpu,
            results_gpu, result_offsets_gpu,
            np.int32(num_docs), np.int32(num_patterns),
            block=block_size, grid=grid_size
        )
        
        # Get results
        results = results_gpu.get().reshape((num_docs, num_patterns))
        counts = result_offsets_gpu.get().reshape((num_docs, num_patterns))
        
        # Format results
        output = []
        for i in range(num_docs):
            doc_results = {}
            for j, xpath in enumerate(xpaths):
                doc_results[xpath] = {
                    'found': bool(results[i, j]),
                    'count': int(counts[i, j])
                }
            output.append(doc_results)
            
        return output
        
    def _evaluate_opencl(self, all_docs, doc_offsets, doc_lengths,
                        all_patterns, pattern_offsets, pattern_lengths,
                        num_docs, num_patterns) -> List[Dict[str, Any]]:
        """Evaluate using OpenCL backend"""
        # Transfer data to GPU
        docs_buf = self.backend.transfer_to_gpu(np.frombuffer(all_docs, dtype=np.uint8))
        doc_offsets_buf = self.backend.transfer_to_gpu(np.array(doc_offsets, dtype=np.int32))
        doc_lengths_buf = self.backend.transfer_to_gpu(np.array(doc_lengths, dtype=np.int32))
        
        patterns_buf = self.backend.transfer_to_gpu(np.frombuffer(all_patterns, dtype=np.uint8))
        pattern_offsets_buf = self.backend.transfer_to_gpu(np.array(pattern_offsets, dtype=np.int32))
        pattern_lengths_buf = self.backend.transfer_to_gpu(np.array(pattern_lengths, dtype=np.int32))
        
        # Prepare output buffers
        results_buf = self.backend.allocate_memory(num_docs * num_patterns)
        result_offsets_buf = self.backend.allocate_memory(num_docs * num_patterns * 4)
        
        # Execute kernel
        kernel = self.compiled_kernels['opencl']
        kernel(
            self.backend.queue,
            (num_docs, num_patterns),
            None,
            docs_buf, doc_offsets_buf, doc_lengths_buf,
            patterns_buf, pattern_offsets_buf, pattern_lengths_buf,
            results_buf, result_offsets_buf,
            np.int32(num_docs), np.int32(num_patterns)
        )
        
        # Get results
        results = np.empty((num_docs * num_patterns,), dtype=np.uint8)
        counts = np.empty((num_docs * num_patterns,), dtype=np.int32)
        
        cl.enqueue_copy(self.backend.queue, results, results_buf)
        cl.enqueue_copy(self.backend.queue, counts, result_offsets_buf)
        self.backend.synchronize()
        
        results = results.reshape((num_docs, num_patterns))
        counts = counts.reshape((num_docs, num_patterns))
        
        # Format results
        output = []
        for i in range(num_docs):
            doc_results = {}
            for j, xpath in enumerate(xpaths):
                doc_results[xpath] = {
                    'found': bool(results[i, j]),
                    'count': int(counts[i, j])
                }
            output.append(doc_results)
            
        return output
        
    def _evaluate_cpu(self, documents: List[str], xpaths: List[str]) -> List[Dict[str, Any]]:
        """CPU fallback implementation"""
        results = []
        for doc in documents:
            selector = Selector(text=doc)
            doc_results = {}
            for xpath in xpaths:
                try:
                    matches = selector.xpath(xpath)
                    doc_results[xpath] = {
                        'found': len(matches) > 0,
                        'count': len(matches)
                    }
                except Exception as e:
                    doc_results[xpath] = {
                        'found': False,
                        'count': 0,
                        'error': str(e)
                    }
            results.append(doc_results)
        return results


class CSSGPUEvaluator:
    """GPU-accelerated CSS selector evaluator"""
    
    def __init__(self, backend: GPUBackend):
        self.backend = backend
        self.xpath_evaluator = XPathGPUEvaluator(backend)
        
    def evaluate_batch(self, documents: List[str], css_selectors: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate CSS selectors on a batch of documents using GPU
        
        Args:
            documents: List of HTML document strings
            css_selectors: List of CSS selectors to evaluate
            
        Returns:
            List of dictionaries with extraction results
        """
        # Convert CSS selectors to XPath (simplified)
        xpaths = []
        for css in css_selectors:
            # Very basic CSS to XPath conversion
            # In production, use a proper CSS to XPath converter
            if css.startswith('.'):
                xpath = f"//*[contains(@class, '{css[1:]}')]"
            elif css.startswith('#'):
                xpath = f"//*[@id='{css[1:]}']"
            elif '[' in css:
                # Handle attribute selectors
                xpath = f"//{css.split('[')[0]}[{css.split('[')[1]}"
            else:
                xpath = f"//{css}"
            xpaths.append(xpath)
            
        return self.xpath_evaluator.evaluate_batch(documents, xpaths)


class GPUAcceleratedParser:
    """
    Main class for GPU-accelerated parsing and extraction
    
    Provides automatic GPU detection, memory management, and fallback to CPU
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GPU-accelerated parser
        
        Args:
            config: Configuration dictionary with keys:
                - use_gpu: bool = True
                - backend: str = 'auto'  # 'cuda', 'opencl', 'cpu', 'auto'
                - device_id: int = 0
                - batch_size: int = 1024
                - memory_limit: float = 0.8
                - enable_profiling: bool = False
        """
        self.config = config or {}
        self.use_gpu = self.config.get('use_gpu', True)
        self.backend_type = self.config.get('backend', 'auto')
        self.device_id = self.config.get('device_id', 0)
        self.batch_size = self.config.get('batch_size', DEFAULT_BATCH_SIZE)
        self.memory_limit = self.config.get('memory_limit', DEFAULT_GPU_MEMORY_LIMIT)
        self.enable_profiling = self.config.get('enable_profiling', False)
        
        self.backend = None
        self.xpath_evaluator = None
        self.css_evaluator = None
        self.metrics = GPUMetrics()
        self._lock = threading.RLock()
        self._initialized = False
        
        # Performance tracking
        self._timings = {
            'gpu_processing': [],
            'cpu_processing': [],
            'memory_transfer': []
        }
        
    def initialize(self) -> bool:
        """Initialize GPU backend and evaluators"""
        if self._initialized:
            return True
            
        with self._lock:
            if self._initialized:
                return True
                
            # Try to initialize GPU backend
            if self.use_gpu:
                self.backend = self._initialize_backend()
                
            # Fall back to CPU if GPU not available
            if not self.backend or not self.backend.available:
                logger.warning("GPU not available, using CPU emulation")
                self.backend = CPUEmulationBackend(self.device_id)
                self.use_gpu = False
                
            # Initialize evaluators
            self.xpath_evaluator = XPathGPUEvaluator(self.backend)
            self.css_evaluator = CSSGPUEvaluator(self.backend)
            
            # Compile kernels if using GPU
            if self.use_gpu and not self.xpath_evaluator.compile_kernels():
                logger.warning("Failed to compile GPU kernels, using CPU")
                self.backend = CPUEmulationBackend(self.device_id)
                self.use_gpu = False
                self.xpath_evaluator = XPathGPUEvaluator(self.backend)
                self.css_evaluator = CSSGPUEvaluator(self.backend)
                
            self._initialized = True
            logger.info(f"GPUAcceleratedParser initialized with backend: {self.backend.device_name}")
            return True
            
    def _initialize_backend(self) -> Optional[GPUBackend]:
        """Initialize appropriate GPU backend"""
        if self.backend_type == 'auto':
            # Try CUDA first, then OpenCL
            backends = [
                ('cuda', CUDABackend),
                ('opencl', OpenCLBackend)
            ]
        elif self.backend_type == 'cuda':
            backends = [('cuda', CUDABackend)]
        elif self.backend_type == 'opencl':
            backends = [('opencl', OpenCLBackend)]
        else:
            return None
            
        for backend_name, backend_class in backends:
            try:
                backend = backend_class(self.device_id)
                if backend.initialize():
                    logger.info(f"Successfully initialized {backend_name} backend")
                    return backend
            except Exception as e:
                logger.debug(f"Failed to initialize {backend_name}: {e}")
                continue
                
        return None
        
    def parse_batch(self, 
                   documents: List[Union[str, bytes, TextResponse]],
                   selectors: List[str],
                   selector_type: str = 'xpath',
                   extractors: Optional[List[Callable]] = None) -> List[Dict[str, Any]]:
        """
        Parse a batch of documents using GPU acceleration
        
        Args:
            documents: List of documents (HTML strings, bytes, or TextResponse objects)
            selectors: List of XPath or CSS selectors
            selector_type: 'xpath' or 'css'
            extractors: Optional list of extractor functions for custom processing
            
        Returns:
            List of extraction results for each document
        """
        if not self._initialized:
            self.initialize()
            
        # Convert documents to strings
        html_docs = []
        for doc in documents:
            if isinstance(doc, TextResponse):
                html_docs.append(doc.text)
            elif isinstance(doc, bytes):
                html_docs.append(to_unicode(doc))
            elif isinstance(doc, str):
                html_docs.append(doc)
            else:
                raise ValueError(f"Unsupported document type: {type(doc)}")
                
        # Validate document sizes
        for i, doc in enumerate(html_docs):
            if len(doc) > MAX_DOCUMENT_SIZE:
                logger.warning(f"Document {i} exceeds maximum size, truncating")
                html_docs[i] = doc[:MAX_DOCUMENT_SIZE]
                
        # Process in batches
        results = []
        total_docs = len(html_docs)
        
        for batch_start in range(0, total_docs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_docs)
            batch_docs = html_docs[batch_start:batch_end]
            
            start_time = time.time()
            
            if self.use_gpu and len(batch_docs) >= 8:  # GPU only beneficial for larger batches
                try:
                    if selector_type == 'xpath':
                        batch_results = self.xpath_evaluator.evaluate_batch(batch_docs, selectors)
                    else:
                        batch_results = self.css_evaluator.evaluate_batch(batch_docs, selectors)
                        
                    self.metrics.gpu_processed += len(batch_docs)
                    self.metrics.gpu_time += time.time() - start_time
                except Exception as e:
                    logger.error(f"GPU processing failed: {e}, falling back to CPU")
                    batch_results = self._process_cpu_batch(batch_docs, selectors, selector_type)
                    self.metrics.cpu_processed += len(batch_docs)
                    self.metrics.cpu_time += time.time() - start_time
            else:
                batch_results = self._process_cpu_batch(batch_docs, selectors, selector_type)
                self.metrics.cpu_processed += len(batch_docs)
                self.metrics.cpu_time += time.time() - start_time
                
            # Apply custom extractors if provided
            if extractors:
                for i, doc_results in enumerate(batch_results):
                    doc_index = batch_start + i
                    for j, extractor in enumerate(extractors):
                        if j < len(selectors):
                            try:
                                selector = Selector(text=batch_docs[i])
                                extracted = extractor(selector, selectors[j])
                                doc_results[f'extractor_{j}'] = extracted
                            except Exception as e:
                                doc_results[f'extractor_{j}_error'] = str(e)
                                
            results.extend(batch_results)
            self.metrics.batch_count += 1
            
        self.metrics.total_documents = total_docs
        self._calculate_speedup()
        
        return results
        
    def _process_cpu_batch(self, 
                          documents: List[str], 
                          selectors: List[str],
                          selector_type: str) -> List[Dict[str, Any]]:
        """Process batch using CPU"""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(8, len(documents))) as executor:
            futures = []
            for doc in documents:
                future = executor.submit(self._process_single_cpu, doc, selectors, selector_type)
                futures.append(future)
                
            for future in as_completed(futures):
                results.append(future.result())
                
        return results
        
    def _process_single_cpu(self, 
                           document: str, 
                           selectors: List[str],
                           selector_type: str) -> Dict[str, Any]:
        """Process single document using CPU"""
        selector = Selector(text=document)
        results = {}
        
        for sel in selectors:
            try:
                if selector_type == 'xpath':
                    matches = selector.xpath(sel)
                else:
                    matches = selector.css(sel)
                    
                results[sel] = {
                    'found': len(matches) > 0,
                    'count': len(matches),
                    'texts': [m.get() for m in matches[:10]]  # Limit to first 10 matches
                }
            except Exception as e:
                results[sel] = {
                    'found': False,
                    'count': 0,
                    'error': str(e)
                }
                
        return results
        
    def benchmark(self, 
                 documents: List[str],
                 selectors: List[str],
                 selector_type: str = 'xpath',
                 runs: int = 5) -> Dict[str, Any]:
        """
        Benchmark GPU vs CPU performance
        
        Args:
            documents: Test documents
            selectors: Test selectors
            selector_type: 'xpath' or 'css'
            runs: Number of benchmark runs
            
        Returns:
            Benchmark results dictionary
        """
        if not self._initialized:
            self.initialize()
            
        results = {
            'gpu_times': [],
            'cpu_times': [],
            'speedups': [],
            'throughput_gpu': [],
            'throughput_cpu': []
        }
        
        # Warm up
        if self.use_gpu:
            self.parse_batch(documents[:10], selectors[:1], selector_type)
            
        for run in range(runs):
            # GPU benchmark
            if self.use_gpu:
                start = time.time()
                self.parse_batch(documents, selectors, selector_type)
                gpu_time = time.time() - start
                results['gpu_times'].append(gpu_time)
                results['throughput_gpu'].append(len(documents) / gpu_time)
                
            # CPU benchmark
            start = time.time()
            self._process_cpu_batch(documents, selectors, selector_type)
            cpu_time = time.time() - start
            results['cpu_times'].append(cpu_time)
            results['throughput_cpu'].append(len(documents) / cpu_time)
            
            if self.use_gpu:
                results['speedups'].append(cpu_time / gpu_time)
                
        # Calculate averages
        if self.use_gpu:
            results['avg_gpu_time'] = sum(results['gpu_times']) / runs
            results['avg_speedup'] = sum(results['speedups']) / runs
            results['avg_throughput_gpu'] = sum(results['throughput_gpu']) / runs
            
        results['avg_cpu_time'] = sum(results['cpu_times']) / runs
        results['avg_throughput_cpu'] = sum(results['throughput_cpu']) / runs
        results['total_documents'] = len(documents)
        results['backend'] = self.backend.device_name if self.backend else "CPU"
        
        return results
        
    def _calculate_speedup(self):
        """Calculate speedup factor"""
        if self.metrics.gpu_time > 0 and self.metrics.cpu_time > 0:
            # Estimate CPU time for GPU-processed documents
            cpu_rate = self.metrics.cpu_time / max(self.metrics.cpu_processed, 1)
            estimated_cpu_time = cpu_rate * self.metrics.gpu_processed
            self.metrics.speedup_factor = estimated_cpu_time / self.metrics.gpu_time
            
    def get_metrics(self) -> GPUMetrics:
        """Get performance metrics"""
        return self.metrics
        
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current backend"""
        if not self.backend:
            return {'available': False}
            
        return {
            'available': self.backend.available,
            'device_name': self.backend.device_name,
            'total_memory': self.backend.total_memory,
            'free_memory': self.backend.free_memory,
            'backend_type': type(self.backend).__name__,
            'use_gpu': self.use_gpu
        }
        
    def optimize_batch_size(self, 
                           sample_documents: List[str],
                           selectors: List[str],
                           selector_type: str = 'xpath') -> int:
        """
        Automatically determine optimal batch size
        
        Args:
            sample_documents: Sample documents for testing
            selectors: Test selectors
            selector_type: 'xpath' or 'css'
            
        Returns:
            Optimal batch size
        """
        if not self._initialized:
            self.initialize()
            
        test_sizes = [64, 128, 256, 512, 1024, 2048, 4096]
        best_size = self.batch_size
        best_throughput = 0
        
        for size in test_sizes:
            if size > len(sample_documents):
                break
                
            test_docs = sample_documents[:size]
            
            # Warm up
            self.parse_batch(test_docs[:10], selectors[:1], selector_type)
            
            # Benchmark
            start = time.time()
            self.parse_batch(test_docs, selectors, selector_type)
            elapsed = time.time() - start
            
            throughput = size / elapsed
            
            if throughput > best_throughput:
                best_throughput = throughput
                best_size = size
                
        self.batch_size = best_size
        logger.info(f"Optimal batch size determined: {best_size}")
        return best_size
        
    def clear_cache(self):
        """Clear any cached data"""
        # Clear LRU caches
        self._process_single_cpu.cache_clear()
        
    def shutdown(self):
        """Clean up resources"""
        if self.backend:
            self.backend.synchronize()
        self.clear_cache()


# Convenience functions for easy integration with Scrapy
def create_gpu_parser(settings: Optional[Dict[str, Any]] = None) -> GPUAcceleratedParser:
    """
    Create a GPU-accelerated parser from Scrapy settings
    
    Args:
        settings: Scrapy settings dictionary
        
    Returns:
        Configured GPUAcceleratedParser instance
    """
    config = {}
    
    if settings:
        config = {
            'use_gpu': settings.getbool('GPU_ACCELERATION_ENABLED', True),
            'backend': settings.get('GPU_BACKEND', 'auto'),
            'device_id': settings.getint('GPU_DEVICE_ID', 0),
            'batch_size': settings.getint('GPU_BATCH_SIZE', DEFAULT_BATCH_SIZE),
            'memory_limit': settings.getfloat('GPU_MEMORY_LIMIT', DEFAULT_GPU_MEMORY_LIMIT),
            'enable_profiling': settings.getbool('GPU_PROFILING', False)
        }
        
    parser = GPUAcceleratedParser(config)
    parser.initialize()
    return parser


def process_responses_gpu(responses: List[TextResponse],
                         selectors: List[str],
                         selector_type: str = 'xpath',
                         settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Process multiple responses using GPU acceleration
    
    Args:
        responses: List of Scrapy TextResponse objects
        selectors: List of XPath or CSS selectors
        selector_type: 'xpath' or 'css'
        settings: Optional Scrapy settings
        
    Returns:
        List of extraction results
    """
    parser = create_gpu_parser(settings)
    return parser.parse_batch(responses, selectors, selector_type)


# Integration with Scrapy's Selector
class GPUSelector(Selector):
    """GPU-accelerated Selector that extends Scrapy's Selector"""
    
    def __init__(self, *args, gpu_parser: Optional[GPUAcceleratedParser] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_parser = gpu_parser or GPUAcceleratedParser()
        self.gpu_parser.initialize()
        
    @classmethod
    def from_responses(cls, 
                      responses: List[TextResponse],
                      gpu_parser: Optional[GPUAcceleratedParser] = None) -> List['GPUSelector']:
        """
        Create GPUSelectors from multiple responses
        
        Args:
            responses: List of TextResponse objects
            gpu_parser: Optional GPU parser instance
            
        Returns:
            List of GPUSelector objects
        """
        parser = gpu_parser or GPUAcceleratedParser()
        parser.initialize()
        
        selectors = []
        for response in responses:
            selector = cls(text=response.text, gpu_parser=parser)
            selectors.append(selector)
            
        return selectors
        
    def xpath_batch(self, 
                   xpaths: List[str],
                   use_gpu: bool = True) -> List[Dict[str, Any]]:
        """
        Evaluate multiple XPath expressions in batch
        
        Args:
            xpaths: List of XPath expressions
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            List of results for each XPath
        """
        if use_gpu and self.gpu_parser.use_gpu:
            results = self.gpu_parser.parse_batch(
                [self.text], xpaths, 'xpath'
            )
            return results[0] if results else {}
        else:
            # Fallback to standard Scrapy Selector
            results = {}
            for xpath in xpaths:
                matches = self.xpath(xpath)
                results[xpath] = {
                    'found': len(matches) > 0,
                    'count': len(matches),
                    'texts': [m.get() for m in matches[:10]]
                }
            return results


# Context manager for automatic resource cleanup
class GPUAccelerationContext:
    """Context manager for GPU acceleration"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.parser = None
        
    def __enter__(self) -> GPUAcceleratedParser:
        self.parser = GPUAcceleratedParser(self.config)
        self.parser.initialize()
        return self.parser
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.parser:
            self.parser.shutdown()
        return False


# Performance monitoring decorator
def gpu_accelerated(func: Callable) -> Callable:
    """
    Decorator to automatically use GPU acceleration for batch processing
    
    Usage:
        @gpu_accelerated
        def process_documents(documents, selectors):
            # Your processing logic
            pass
    """
    def wrapper(documents: List[Any], 
                selectors: List[str],
                *args,
                gpu_config: Optional[Dict[str, Any]] = None,
                **kwargs) -> Any:
        
        with GPUAccelerationContext(gpu_config) as parser:
            # Check if documents are TextResponse objects
            if all(hasattr(doc, 'text') for doc in documents):
                return parser.parse_batch(documents, selectors, 'xpath')
            else:
                # Fall back to original function
                return func(documents, selectors, *args, **kwargs)
                
    return wrapper


# Auto-configuration for Scrapy
def configure_vex_for_gpu(settings):
    """
    Configure Scrapy settings for optimal GPU acceleration
    
    Args:
        settings: Scrapy settings object
    """
    # Increase concurrent requests for better GPU utilization
    settings.set('CONCURRENT_REQUESTS', 32)
    settings.set('CONCURRENT_REQUESTS_PER_DOMAIN', 16)
    
    # Enable AutoThrottle for adaptive request rates
    settings.set('AUTOTHROTTLE_ENABLED', True)
    settings.set('AUTOTHROTTLE_START_DELAY', 1)
    settings.set('AUTOTHROTTLE_MAX_DELAY', 10)
    
    # Enable HTTP caching to reduce network overhead
    settings.set('HTTPCACHE_ENABLED', True)
    settings.set('HTTPCACHE_EXPIRATION_SECS', 3600)
    
    # Set reasonable download timeout
    settings.set('DOWNLOAD_TIMEOUT', 30)
    
    logger.info("Scrapy configured for GPU acceleration")


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Test documents
    test_documents = [
        "<html><body><p>Test document 1</p><div class='content'>Content 1</div></body></html>",
        "<html><body><p>Test document 2</p><div class='content'>Content 2</div></body></html>",
        "<html><body><p>Test document 3</p><div class='content'>Content 3</div></body></html>"
    ]
    
    test_selectors = ["//p", "//div[@class='content']"]
    
    # Create parser
    parser = GPUAcceleratedParser({
        'use_gpu': True,
        'backend': 'auto',
        'batch_size': 1024
    })
    
    # Initialize
    if parser.initialize():
        print(f"Using backend: {parser.backend.device_name}")
        
        # Process documents
        results = parser.parse_batch(test_documents, test_selectors, 'xpath')
        
        # Print results
        for i, doc_results in enumerate(results):
            print(f"\nDocument {i+1}:")
            for selector, result in doc_results.items():
                print(f"  {selector}: {result['count']} matches")
                
        # Run benchmark
        print("\nRunning benchmark...")
        benchmark_results = parser.benchmark(
            test_documents * 100,  # Larger dataset
            test_selectors,
            runs=3
        )
        
        print(f"Average GPU time: {benchmark_results.get('avg_gpu_time', 'N/A'):.3f}s")
        print(f"Average CPU time: {benchmark_results['avg_cpu_time']:.3f}s")
        if 'avg_speedup' in benchmark_results:
            print(f"Average speedup: {benchmark_results['avg_speedup']:.2f}x")
            
        # Get metrics
        metrics = parser.get_metrics()
        print(f"\nMetrics:")
        print(f"  Total documents: {metrics.total_documents}")
        print(f"  GPU processed: {metrics.gpu_processed}")
        print(f"  CPU processed: {metrics.cpu_processed}")
        print(f"  Speedup factor: {metrics.speedup_factor:.2f}x")
        
        # Clean up
        parser.shutdown()
    else:
        print("Failed to initialize GPU parser", file=sys.stderr)
        sys.exit(1)