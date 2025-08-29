import ctypes
import numpy as np
import os
from ctypes import c_float, c_int, c_bool, c_char_p, c_size_t, POINTER, Structure, c_ulonglong
import time
import warnings
from sklearn.ensemble import RandomForestClassifier

class RFParams(Structure):
    """Random Forest Parameters structure matching C++ definition"""
    _fields_ = [
        ("n_estimators", c_int),
        ("max_depth", c_int),
        ("min_samples_split", c_int),
        ("max_features", c_int),
        ("n_classes", c_int),
        ("bootstrap", c_bool),
        ("seed", c_ulonglong)
    ]

class CudaRFError(Exception):
    """Custom exception for CUDA Random Forest errors"""
    pass

def find_cuda_rf_library():
    """Find the CUDA RF shared library"""
    # Simplified search path for this example
    lib_path = os.path.join(os.path.dirname(__file__), "libcuda_rf_optimized.so")
    if os.path.exists(lib_path):
        return lib_path
    
    raise FileNotFoundError(
        f"CUDA RF library not found at {lib_path}\n"
        "Please compile the library using 'make' in the HBM_RandomForest directory first."
    )

class OptimizedCudaRandomForest:
    """
    High-performance CUDA Random Forest implementation
    
    This class provides a scikit-learn compatible interface for a GPU-accelerated
    Random Forest classifier.
    """
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 max_features='sqrt', bootstrap=True, random_state=None, verbose=False):
        
        # Try to load CUDA library first
        self._use_cuda = True
        try:
            lib_path = find_cuda_rf_library()
            self._lib = ctypes.CDLL(lib_path)
            self._setup_function_signatures()
        except (FileNotFoundError, OSError) as e:
            warnings.warn(f"CUDA library not found ({e}), falling back to CPU implementation")
            self._use_cuda = False
            self._sklearn_rf = None
        
        # Store parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_str = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        
        # Internal state
        if self._use_cuda:
            self.params = RFParams()
            self.rf_ptr = None
        self.is_fitted = False
        self.n_features_ = None
        self.n_classes_ = None
        self.training_time_ = 0.0
        self.memory_usage_ = 0

    def _setup_function_signatures(self):
        """Setup ctypes function signatures for type safety"""
        self._lib.get_rf_last_error.restype = c_char_p
        
        self._lib.create_optimized_rf.argtypes = [POINTER(RFParams)]
        self._lib.create_optimized_rf.restype = ctypes.c_void_p
        
        self._lib.destroy_optimized_rf.argtypes = [ctypes.c_void_p]
        
        self._lib.fit_optimized_rf.argtypes = [ctypes.c_void_p, POINTER(c_float), POINTER(c_int), c_int, c_int]
        self._lib.fit_optimized_rf.restype = c_int
        
        self._lib.predict_optimized_rf.argtypes = [ctypes.c_void_p, POINTER(c_float), POINTER(c_int), c_int, c_int]
        self._lib.predict_optimized_rf.restype = c_int
        
        self._lib.get_rf_training_time.argtypes = [ctypes.c_void_p]
        self._lib.get_rf_training_time.restype = c_float
        
        self._lib.get_rf_memory_usage.argtypes = [ctypes.c_void_p]
        self._lib.get_rf_memory_usage.restype = c_size_t

    def _update_params(self):
        """Update the parameters structure before fitting"""
        if not self._use_cuda:
            return
            
        self.params.n_estimators = self.n_estimators
        self.params.max_depth = self.max_depth
        self.params.min_samples_split = self.min_samples_split
        self.params.bootstrap = self.bootstrap
        
        if self.random_state is None:
            self.params.seed = int(time.time())
        else:
            self.params.seed = self.random_state

        if self.n_features_ is None:
            raise ValueError("n_features_ must be set before calling _update_params")

        if self.max_features_str == 'sqrt':
            self.params.max_features = int(np.sqrt(self.n_features_))
        elif self.max_features_str == 'log2':
            self.params.max_features = int(np.log2(self.n_features_))
        elif isinstance(self.max_features_str, int):
            self.params.max_features = self.max_features_str
        else: # float
            self.params.max_features = int(float(self.max_features_str) * self.n_features_)
        
        self.params.n_classes = self.n_classes_

    def _check_error(self):
        """Check for errors from the C++ library"""
        error_msg = self._lib.get_rf_last_error()
        if error_msg:
            error_str = error_msg.decode('utf-8')
            self._lib.clear_rf_error()
            raise CudaRFError(error_str)

    def fit(self, X, y):
        """Fit the Random Forest model to training data"""
        X = np.asarray(X, dtype=np.float32, order='C')
        y = np.asarray(y, dtype=np.int32, order='C')
        
        if X.ndim != 2: raise ValueError("X must be a 2D array")
        if y.ndim != 1: raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]: raise ValueError("X and y must have the same number of samples")
        
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        start_time = time.time()
        
        if self._use_cuda:
            self._update_params()
            self.rf_ptr = self._lib.create_optimized_rf(ctypes.byref(self.params))
            if not self.rf_ptr:
                self._check_error()
                raise CudaRFError("Failed to create RF object")
            
            try:
                X_ptr = X.ctypes.data_as(POINTER(c_float))
                y_ptr = y.ctypes.data_as(POINTER(c_int))
                
                if self.verbose: print("Starting CUDA Random Forest training...")
                
                result = self._lib.fit_optimized_rf(
                    self.rf_ptr, X_ptr, y_ptr, 
                    c_int(X.shape[0]), c_int(X.shape[1])
                )
                
                if result != 0:
                    self._check_error()
                
                self.training_time_ = self._lib.get_rf_training_time(self.rf_ptr)
                self.memory_usage_ = self._lib.get_rf_memory_usage(self.rf_ptr)
                self.is_fitted = True
                
                if self.verbose:
                    print(f"Training completed in {self.training_time_:.2f}s")
                    print(f"GPU memory used: {self.memory_usage_ / (1024**2):.2f}MB")
                
            except Exception as e:
                if self.rf_ptr:
                    self._lib.destroy_optimized_rf(self.rf_ptr)
                    self.rf_ptr = None
                raise e
        else:
            # CPU fallback using sklearn
            if self.verbose: print("Starting CPU Random Forest training...")
            
            # Convert max_features_str to sklearn format
            if self.max_features_str == 'sqrt':
                max_features = 'sqrt'
            elif self.max_features_str == 'log2':
                max_features = 'log2'
            elif isinstance(self.max_features_str, (int, float)):
                max_features = self.max_features_str
            else:
                max_features = 'sqrt'  # Default fallback
            
            self._sklearn_rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self._sklearn_rf.fit(X, y)
            self.training_time_ = time.time() - start_time
            self.memory_usage_ = 0  # Not tracked for CPU
            self.is_fitted = True
            
            if self.verbose:
                print(f"Training completed in {self.training_time_:.2f}s")
        
        return self

    def predict(self, X):
        """Perform prediction on samples"""
        if not self.is_fitted:
            raise CudaRFError("Model must be fitted before prediction.")
        
        X = np.asarray(X, dtype=np.float32, order='C')
        if X.ndim != 2: raise ValueError("X must be a 2D array")
        if X.shape[1] != self.n_features_: raise ValueError(f"X must have {self.n_features_} features")
        
        if self._use_cuda:
            predictions = np.zeros(X.shape[0], dtype=np.int32, order='C')
            
            X_ptr = X.ctypes.data_as(POINTER(c_float))
            pred_ptr = predictions.ctypes.data_as(POINTER(c_int))
            
            result = self._lib.predict_optimized_rf(
                self.rf_ptr, X_ptr, pred_ptr,
                c_int(X.shape[0]), c_int(X.shape[1])
            )
            
            if result != 0:
                self._check_error()
            
            return predictions
        else:
            # CPU prediction using sklearn
            if self._sklearn_rf is None:
                raise CudaRFError("CPU Random Forest not initialized")
            return self._sklearn_rf.predict(X)

    def score(self, X, y):
        """Calculate mean accuracy for classification"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_performance_metrics(self):
        """Get detailed performance metrics"""
        if self._use_cuda:
            max_features = self.params.max_features
        else:
            max_features = self.max_features_str
            
        return {
            'training_time_s': self.training_time_,
            'memory_usage_mb': self.memory_usage_ / (1024**2) if self._use_cuda else 0,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features_,
            'n_classes': self.n_classes_,
            'parameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'max_features': max_features
            },
            'backend': 'CUDA' if self._use_cuda else 'CPU'
        }

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_use_cuda') and self._use_cuda and hasattr(self, 'rf_ptr') and self.rf_ptr:
            try:
                self._lib.destroy_optimized_rf(self.rf_ptr)
            except:
                pass # Ignore errors during cleanup

# Convenience class
class OptimizedCudaRFClassifier(OptimizedCudaRandomForest):
    """CUDA-accelerated Random Forest Classifier"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)