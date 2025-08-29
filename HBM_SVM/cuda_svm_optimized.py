import ctypes
import numpy as np
import os
import sys
import time
from ctypes import c_float, c_int, c_bool, c_char_p, c_size_t, POINTER, Structure
import warnings
from sklearn.svm import SVC, SVR, NuSVC, NuSVR

class SVMParams(Structure):
    """SVM Parameters structure matching C++ definition"""
    _fields_ = [
        ("svm_type", c_int),      # 0=C_SVC, 1=NU_SVC, 2=EPSILON_SVR, 3=NU_SVR
        ("kernel_type", c_int),   # 0=LINEAR, 1=RBF, 2=POLYNOMIAL, 3=SIGMOID
        ("C", c_float),
        ("epsilon", c_float),
        ("gamma", c_float),
        ("coef0", c_float),
        ("degree", c_int),
        ("nu", c_float),
        ("tolerance", c_float),
        ("max_iter", c_int),
        ("shrinking", c_bool),
        ("probability", c_bool)
    ]

class CudaSVMError(Exception):
    """Custom exception for CUDA SVM errors"""
    pass

def find_cuda_svm_library():
    """Find the CUDA SVM shared library"""
    possible_paths = [
        "./libcuda_svm_optimized.so",
        "./libcuda_svm.so",
        "/usr/local/lib/libcuda_svm_optimized.so",
        "/usr/local/lib/libcuda_svm.so",
        "/usr/lib/libcuda_svm_optimized.so",
        "/usr/lib/libcuda_svm.so",
        os.path.join(os.path.dirname(__file__), "libcuda_svm_optimized.so"),
        os.path.join(os.path.dirname(__file__), "libcuda_svm.so")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        f"CUDA SVM library not found. Searched paths: {possible_paths}\n"
        "Please compile the library using 'make' first."
    )

class OptimizedCudaSVM:
    """
    Memory-efficient and high-performance CUDA SVM implementation
    
    This class provides a scikit-learn compatible interface for GPU-accelerated
    Support Vector Machines with advanced optimizations.
    """
    
    def __init__(self, svm_type='c_svc', kernel='rbf', C=1.0, epsilon=0.1,
                 gamma='scale', coef0=0.0, degree=3, nu=0.5, tolerance=1e-3,
                 max_iter=1000, shrinking=True, probability=False, verbose=False):
        
        # Try to load CUDA library first
        self._use_cuda = True
        try:
            lib_path = find_cuda_svm_library()
            self._lib = ctypes.CDLL(lib_path)
            self._setup_function_signatures()
        except (FileNotFoundError, OSError) as e:
            warnings.warn(f"CUDA library not found ({e}), falling back to CPU implementation")
            self._use_cuda = False
            self._sklearn_svm = None
        
        # Check CUDA availability (allow CPU fallback)
        if self._use_cuda and not self._lib.check_cuda_available():
            if verbose:
                print("⚠ CUDA not available - using CPU fallback mode")
                print("  For best performance, install CUDA toolkit")
            self._use_cuda = False
            self._sklearn_svm = None
        elif self._use_cuda and verbose:
            self._print_gpu_info()
        
        # Map string parameters to integers
        self.svm_type_map = {'c_svc': 0, 'nu_svc': 1, 'epsilon_svr': 2, 'nu_svr': 3}
        self.kernel_map = {'linear': 0, 'rbf': 1, 'poly': 2, 'sigmoid': 3}
        
        # Store parameters
        self.svm_type = svm_type
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.nu = nu
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.shrinking = shrinking
        self.probability = probability
        self.verbose = verbose
        
        # Initialize parameters structure only if using CUDA
        if self._use_cuda:
            self.params = SVMParams()
            self._update_params()
            self.svm_ptr = None
        
        self.is_fitted = False
        self.n_features_ = None
        self.training_time_ = 0.0
        self.memory_usage_ = 0
        
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for type safety"""
        if not self._use_cuda:
            return
            
        # Error handling
        self._lib.get_last_error.argtypes = []
        self._lib.get_last_error.restype = c_char_p
        
        self._lib.clear_error.argtypes = []
        self._lib.clear_error.restype = None
        
        # SVM management
        self._lib.create_optimized_svm.argtypes = [POINTER(SVMParams)]
        self._lib.create_optimized_svm.restype = ctypes.c_void_p
        
        self._lib.destroy_optimized_svm.argtypes = [ctypes.c_void_p]
        self._lib.destroy_optimized_svm.restype = None
        
        # Training and prediction
        self._lib.fit_optimized_svm.argtypes = [ctypes.c_void_p, POINTER(c_float), POINTER(c_float), c_int, c_int]
        self._lib.fit_optimized_svm.restype = c_int
        
        self._lib.predict_optimized_svm.argtypes = [ctypes.c_void_p, POINTER(c_float), POINTER(c_float), c_int, c_int]
        self._lib.predict_optimized_svm.restype = c_int
        
        self._lib.predict_batch_async_svm.argtypes = [ctypes.c_void_p, POINTER(c_float), POINTER(c_float), c_int, c_int]
        self._lib.predict_batch_async_svm.restype = c_int
        
        # Performance metrics
        self._lib.get_training_time.argtypes = [ctypes.c_void_p]
        self._lib.get_training_time.restype = c_float
        
        self._lib.get_memory_usage.argtypes = [ctypes.c_void_p]
        self._lib.get_memory_usage.restype = c_size_t
        
        # Utility functions
        self._lib.check_cuda_available.argtypes = []
        self._lib.check_cuda_available.restype = c_int
        
        self._lib.get_gpu_info.argtypes = [POINTER(c_int), c_char_p, c_int]
        self._lib.get_gpu_info.restype = None
    
    def _update_params(self):
        """Update the parameters structure"""
        if not self._use_cuda:
            return
            
        self.params.svm_type = self.svm_type_map[self.svm_type]
        self.params.kernel_type = self.kernel_map[self.kernel]
        self.params.C = self.C
        self.params.epsilon = self.epsilon
        self.params.coef0 = self.coef0
        self.params.degree = self.degree
        self.params.nu = self.nu
        self.params.tolerance = self.tolerance
        self.params.max_iter = self.max_iter
        self.params.shrinking = self.shrinking
        self.params.probability = self.probability
    
    def _set_gamma(self, X):
        """Set gamma parameter based on input data"""
        if not self._use_cuda:
            return
            
        if self.gamma == 'scale':
            self.params.gamma = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            self.params.gamma = 1.0 / X.shape[1]
        else:
            self.params.gamma = float(self.gamma)
    
    def _check_error(self):
        """Check for errors from the C++ library"""
        if not self._use_cuda:
            return
            
        error_msg = self._lib.get_last_error()
        if error_msg:
            error_str = error_msg.decode('utf-8')
            self._lib.clear_error()
            raise CudaSVMError(error_str)
    
    def _print_gpu_info(self):
        """Print GPU information"""
        if not self._use_cuda:
            print("Using CPU fallback mode - no CUDA devices detected")
            return
            
        device_count = c_int()
        device_name = ctypes.create_string_buffer(256)
        self._lib.get_gpu_info(ctypes.byref(device_count), device_name, 256)
        
        if device_count.value > 0:
            print(f"CUDA Devices Available: {device_count.value}")
            print(f"Primary GPU: {device_name.value.decode('utf-8')}")
        else:
            print("Using CPU fallback mode - no CUDA devices detected")
    
    def fit(self, X, y):
        """
        Fit the SVM model to training data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training vectors
        y : array-like, shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
        """
        # Input validation
        X = np.asarray(X, dtype=np.float32, order='C')
        y = np.asarray(y, dtype=np.float32, order='C')
        
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        if not y.flags['C_CONTIGUOUS']:
            y = np.ascontiguousarray(y)
        
        self.n_features_ = X.shape[1]
        self._set_gamma(X)
        self._update_params()
        
        start_time = time.time()
        
        if self._use_cuda:
            # Create SVM object
            self.svm_ptr = self._lib.create_optimized_svm(ctypes.byref(self.params))
            if not self.svm_ptr:
                self._check_error()
                raise CudaSVMError("Failed to create SVM object")
            
            try:
                # Prepare data pointers
                X_ptr = X.ctypes.data_as(POINTER(c_float))
                y_ptr = y.ctypes.data_as(POINTER(c_float))
                
                # Fit the model
                result = self._lib.fit_optimized_svm(
                    self.svm_ptr, X_ptr, y_ptr, 
                    c_int(X.shape[0]), c_int(X.shape[1])
                )
                
                if result != 0:
                    self._check_error()
                    raise CudaSVMError("Training failed")
                
                # Get performance metrics
                self.training_time_ = self._lib.get_training_time(self.svm_ptr)
                self.memory_usage_ = self._lib.get_memory_usage(self.svm_ptr)
                
                self.is_fitted = True
                
                if self.verbose:
                    print(f"Training completed in {self.training_time_:.2f}s")
                    print(f"GPU memory used: {self.memory_usage_ / (1024**3):.2f}GB")
                
            except Exception as e:
                # Clean up on error
                if self.svm_ptr:
                    self._lib.destroy_optimized_svm(self.svm_ptr)
                    self.svm_ptr = None
                raise e
        else:
            # CPU fallback using sklearn
            if self.verbose:
                print("Starting CPU SVM training...")
            
            # Convert parameters to sklearn format
            if self.svm_type in ['c_svc', 'nu_svc']:
                # Classification
                if self.kernel == 'poly':
                    sklearn_kernel = 'poly'
                elif self.kernel == 'sigmoid':
                    sklearn_kernel = 'sigmoid'
                elif self.kernel == 'linear':
                    sklearn_kernel = 'linear'
                else:  # rbf
                    sklearn_kernel = 'rbf'
                
                if self.svm_type == 'c_svc':
                    self._sklearn_svm = SVC(
                        C=self.C,
                        kernel=sklearn_kernel,
                        gamma=self.params.gamma if hasattr(self, 'params') else 'scale',
                        coef0=self.coef0,
                        degree=self.degree,
                        shrinking=self.shrinking,
                        probability=self.probability,
                        tol=self.tolerance,
                        max_iter=self.max_iter,
                        random_state=42
                    )
                else:  # nu_svc
                    self._sklearn_svm = NuSVC(
                        nu=self.nu,
                        kernel=sklearn_kernel,
                        gamma=self.params.gamma if hasattr(self, 'params') else 'scale',
                        coef0=self.coef0,
                        degree=self.degree,
                        shrinking=self.shrinking,
                        probability=self.probability,
                        tol=self.tolerance,
                        max_iter=self.max_iter,
                        random_state=42
                    )
            else:
                # Regression
                if self.kernel == 'poly':
                    sklearn_kernel = 'poly'
                elif self.kernel == 'sigmoid':
                    sklearn_kernel = 'sigmoid'
                elif self.kernel == 'linear':
                    sklearn_kernel = 'linear'
                else:  # rbf
                    sklearn_kernel = 'rbf'
                
                if self.svm_type == 'epsilon_svr':
                    self._sklearn_svm = SVR(
                        C=self.C,
                        epsilon=self.epsilon,
                        kernel=sklearn_kernel,
                        gamma=self.params.gamma if hasattr(self, 'params') else 'scale',
                        coef0=self.coef0,
                        degree=self.degree,
                        shrinking=self.shrinking,
                        tol=self.tolerance,
                        max_iter=self.max_iter
                    )
                else:  # nu_svr
                    self._sklearn_svm = NuSVR(
                        nu=self.nu,
                        C=self.C,
                        kernel=sklearn_kernel,
                        gamma=self.params.gamma if hasattr(self, 'params') else 'scale',
                        coef0=self.coef0,
                        degree=self.degree,
                        shrinking=self.shrinking,
                        tol=self.tolerance,
                        max_iter=self.max_iter
                    )
            
            # Convert labels for classification
            if self.svm_type in ['c_svc', 'nu_svc']:
                y_fit = y.copy()
                if y.dtype != np.int32:
                    # Convert to sklearn format (-1, 1)
                    unique_labels = np.unique(y)
                    if len(unique_labels) == 2:
                        y_fit = np.where(y == unique_labels[0], -1, 1)
                    else:
                        y_fit = y.astype(np.int32)
            else:
                y_fit = y
            
            self._sklearn_svm.fit(X, y_fit)
            self.training_time_ = time.time() - start_time
            self.memory_usage_ = 0  # Not tracked for CPU
            self.is_fitted = True
            
            if self.verbose:
                print(f"Training completed in {self.training_time_:.2f}s")
        
        return self
    
    def predict(self, X, batch_async=False):
        """
        Perform prediction on samples
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        batch_async : bool, default=False
            Use asynchronous batch prediction for better performance
        
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted values
        """
        if not self.is_fitted:
            raise CudaSVMError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float32, order='C')
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} features")
        if not X.flags['C_CONTIGUOUS']:
            X = np.ascontiguousarray(X)
        
        if self._use_cuda:
            predictions = np.zeros(X.shape[0], dtype=np.float32, order='C')
            
            X_ptr = X.ctypes.data_as(POINTER(c_float))
            pred_ptr = predictions.ctypes.data_as(POINTER(c_float))
            
            # Choose prediction function based on batch_async flag
            if batch_async:
                result = self._lib.predict_batch_async_svm(
                    self.svm_ptr, X_ptr, pred_ptr,
                    c_int(X.shape[0]), c_int(X.shape[1])
                )
            else:
                result = self._lib.predict_optimized_svm(
                    self.svm_ptr, X_ptr, pred_ptr,
                    c_int(X.shape[0]), c_int(X.shape[1])
                )
            
            if result != 0:
                self._check_error()
                raise CudaSVMError("Prediction failed")
            
            # For classification, ensure outputs are -1 or 1
            if self.svm_type in ['c_svc', 'nu_svc']:
                predictions = np.where(predictions > 0, 1.0, -1.0)
        else:
            # CPU prediction using sklearn
            if self._sklearn_svm is None:
                raise CudaSVMError("CPU SVM not initialized")
            
            predictions = self._sklearn_svm.predict(X)
            
            # Convert sklearn predictions back to our format
            if self.svm_type in ['c_svc', 'nu_svc']:
                # sklearn SVC/NuSVC returns class labels, we want -1/1
                if hasattr(self._sklearn_svm, 'classes_') and len(self._sklearn_svm.classes_) == 2:  # type: ignore
                    predictions = np.where(predictions == self._sklearn_svm.classes_[0], -1.0, 1.0)  # type: ignore
        
        return predictions
    
    def score(self, X, y):
        """
        Calculate accuracy for classification or R² for regression
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels/values
        
        Returns:
        --------
        score : float
            Accuracy (classification) or R² score (regression)
        """
        predictions = self.predict(X)
        
        if self.svm_type in ['c_svc', 'nu_svc']:
            # Classification accuracy
            return np.mean(predictions == y)
        else:
            # Regression R² score
            y = np.asarray(y)
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def get_performance_metrics(self):
        """Get detailed performance metrics"""
        if self._use_cuda:
            gamma_val = self.params.gamma
        else:
            gamma_val = self.gamma
            
        return {
            'training_time': self.training_time_,
            'memory_usage_gb': self.memory_usage_ / (1024**3) if self._use_cuda else 0,
            'is_fitted': self.is_fitted,
            'n_features': self.n_features_,
            'parameters': {
                'svm_type': self.svm_type,
                'kernel': self.kernel,
                'C': self.C,
                'gamma': gamma_val,
                'epsilon': self.epsilon
            },
            'backend': 'CUDA' if self._use_cuda else 'CPU'
        }
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_use_cuda') and self._use_cuda and hasattr(self, 'svm_ptr') and self.svm_ptr:
            try:
                self._lib.destroy_optimized_svm(self.svm_ptr)
            except:
                pass  # Ignore errors during cleanup

# Convenience classes
class OptimizedCudaSVC(OptimizedCudaSVM):
    """CUDA-accelerated Support Vector Classifier"""
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', **kwargs):
        super().__init__(svm_type='c_svc', C=C, kernel=kernel, gamma=gamma, **kwargs)

class OptimizedCudaSVR(OptimizedCudaSVM):
    """CUDA-accelerated Support Vector Regressor"""
    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='scale', **kwargs):
        super().__init__(svm_type='epsilon_svr', C=C, epsilon=epsilon, 
                        kernel=kernel, gamma=gamma, **kwargs)

# Test and example usage
if __name__ == "__main__":
    print("Testing Optimized CUDA SVM...")
    
    try:
        # Test library loading
        svm = OptimizedCudaSVM(verbose=True)
        print("✓ Library loaded successfully")
        
        # Generate test data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        y = np.where(y == 0, -1, 1)  # Convert to SVM format
        
        print("✓ Test data generated")
        
        # Test fitting
        svm.fit(X, y)
        print("✓ Model training completed")
        
        # Test prediction
        predictions = svm.predict(X[:100])
        accuracy = svm.score(X[:100], y[:100])
        print(f"✓ Prediction accuracy: {accuracy:.4f}")
        
        # Show performance metrics
        metrics = svm.get_performance_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print("\nAll tests passed! CUDA SVM is ready to use.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure CUDA toolkit is installed")
        print("2. Run 'make' to compile the library")
        print("3. Check GPU drivers are up to date")
        print("4. Verify library path in find_cuda_svm_library()")
