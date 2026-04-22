"""
Backend abstraction for CPU/GPU support
Automatically uses CuPy if available, falls back to NumPy
"""
import numpy as np

# Try to import CuPy for GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ GPU support enabled (CuPy detected)")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("⚠ GPU support disabled (CuPy not found, using CPU)")


class Backend:
    """Backend abstraction for array operations"""
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize backend
        
        Args:
            use_gpu: Whether to use GPU if available
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            self.xp = cp
            self.device = 'cuda'
            print(f"Using GPU backend (CuPy)")
        else:
            self.xp = np
            self.device = 'cpu'
            print(f"Using CPU backend (NumPy)")
    
    def array(self, data, dtype=np.float32):
        """Create array on appropriate device"""
        return self.xp.array(data, dtype=dtype)
    
    def zeros(self, shape, dtype=np.float32):
        """Create zeros array"""
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=np.float32):
        """Create ones array"""
        return self.xp.ones(shape, dtype=dtype)
    
    def zeros_like(self, arr):
        """Create zeros array with same shape"""
        return self.xp.zeros_like(arr)
    
    def ones_like(self, arr):
        """Create ones array with same shape"""
        return self.xp.ones_like(arr)
    
    def randn(self, *shape):
        """Create random normal array"""
        return self.xp.random.randn(*shape).astype(np.float32)
    
    def rand(self, *shape):
        """Create random uniform array"""
        return self.xp.random.rand(*shape).astype(np.float32)
    
    def to_numpy(self, arr):
        """Convert array to NumPy (for CPU compatibility)"""
        if self.use_gpu and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)
    
    def to_device(self, arr):
        """Move array to current device"""
        if self.use_gpu:
            if isinstance(arr, np.ndarray):
                return cp.asarray(arr)
            return arr
        else:
            if GPU_AVAILABLE and isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
            return np.asarray(arr)
    
    def synchronize(self):
        """Synchronize device (for GPU)"""
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()
    
    def get_memory_info(self):
        """Get memory usage information"""
        if self.use_gpu:
            mempool = cp.get_default_memory_pool()
            return {
                'used_bytes': mempool.used_bytes(),
                'total_bytes': mempool.total_bytes(),
                'device': 'GPU'
            }
        else:
            import psutil
            process = psutil.Process()
            return {
                'used_bytes': process.memory_info().rss,
                'device': 'CPU'
            }


# Global backend instance
_backend = None


def get_backend(use_gpu: bool = True):
    """Get or create global backend instance"""
    global _backend
    if _backend is None:
        _backend = Backend(use_gpu=use_gpu)
    return _backend


def set_backend(use_gpu: bool):
    """Set backend to use GPU or CPU"""
    global _backend
    _backend = Backend(use_gpu=use_gpu)
    return _backend


def is_gpu_available():
    """Check if GPU is available"""
    return GPU_AVAILABLE
