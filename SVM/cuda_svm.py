import ctypes
import numpy as np
from ctypes import c_float, c_int, c_bool, POINTER, Structure
import os

# Get the directory where this Python file is located
_current_dir = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(_current_dir, "libcuda_svm.so")

if not os.path.exists(lib_path):
    raise RuntimeError(f"CUDA SVM library not found at {lib_path}. Please run 'make' in the SVM directory to build it.")

cuda_svm_lib = ctypes.CDLL(lib_path)

class SVMParams(Structure):
    _fields_ = [
        ("svm_type", c_int),
        ("kernel_type", c_int), 
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

# Define C++ interface functions
cuda_svm_lib.create_svm.argtypes = [POINTER(SVMParams)]
cuda_svm_lib.create_svm.restype = ctypes.c_void_p

cuda_svm_lib.destroy_svm.argtypes = [ctypes.c_void_p]
cuda_svm_lib.destroy_svm.restype = None

cuda_svm_lib.fit_svm.argtypes = [ctypes.c_void_p, POINTER(c_float), POINTER(c_float), c_int, c_int]
cuda_svm_lib.fit_svm.restype = None

cuda_svm_lib.predict_svm.argtypes = [ctypes.c_void_p, POINTER(c_float), POINTER(c_float), c_int, c_int]
cuda_svm_lib.predict_svm.restype = None

cuda_svm_lib.predict_proba_svm.argtypes = [ctypes.c_void_p, POINTER(c_float), POINTER(c_float), c_int, c_int]
cuda_svm_lib.predict_proba_svm.restype = None

class CudaSVM:
    """
    Advanced CUDA-accelerated Support Vector Machine for Classification and Regression
    
    Parameters:
    -----------
    svm_type : str, default='c_svc'
        Type of SVM ('c_svc', 'nu_svc', 'epsilon_svr', 'nu_svr')
    kernel : str, default='rbf'
        Kernel type ('linear', 'rbf', 'poly', 'polynomial', 'sigmoid')
    C : float, default=1.0
        Regularization parameter
    epsilon : float, default=0.1
        Epsilon parameter for regression
    gamma : float, default='scale'
        Kernel coefficient for rbf, poly and sigmoid
    coef0 : float, default=0.0
        Independent term in poly and sigmoid kernels
    degree : int, default=3
        Degree of polynomial kernel
    nu : float, default=0.5
        Nu parameter for nu-SVM
    tolerance : float, default=1e-3
        Tolerance for stopping criterion
    max_iter : int, default=1000
        Maximum number of iterations
    shrinking : bool, default=True
        Whether to use shrinking heuristic
    probability : bool, default=False
        Whether to enable probability estimates
    """
    
    def __init__(self, svm_type='c_svc', kernel='rbf', C=1.0, epsilon=0.1,
                 gamma='scale', coef0=0.0, degree=3, nu=0.5, tolerance=1e-3,
                 max_iter=1000, shrinking=True, probability=False):
        
        # Map string parameters to integers
        svm_type_map = {'c_svc': 0, 'nu_svc': 1, 'epsilon_svr': 2, 'nu_svr': 3}
        kernel_map = {'linear': 0, 'rbf': 1, 'poly': 2, 'polynomial': 2, 'sigmoid': 3}
        
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
        
        # Initialize parameters structure
        self.params = SVMParams()
        self.params.svm_type = svm_type_map[svm_type]
        self.params.kernel_type = kernel_map[kernel]
        self.params.C = C
        self.params.epsilon = epsilon
        self.params.coef0 = coef0
        self.params.degree = degree
        self.params.nu = nu
        self.params.tolerance = tolerance
        self.params.max_iter = max_iter
        self.params.shrinking = shrinking
        self.params.probability = probability
        
        self.svm_ptr = None
        self.is_fitted = False
        self.n_features_ = None
        
    def _set_gamma(self, X):
        """Set gamma parameter based on input data"""
        if self.gamma == 'scale':
            self.params.gamma = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            self.params.gamma = 1.0 / X.shape[1]
        else:
            self.params.gamma = float(self.gamma)
    
    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training vectors
        y : array-like, shape (n_samples,)
            Target values (class labels for classification, real numbers for regression)
        
        Returns:
        --------
        self : object
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        self.n_features_ = X.shape[1]
        self._set_gamma(X)
        
        # Create SVM object
        self.svm_ptr = cuda_svm_lib.create_svm(ctypes.byref(self.params))
        if not self.svm_ptr:
            raise RuntimeError("Failed to create CUDA SVM object")
        
        # Prepare data for C interface
        X_ptr = X.ctypes.data_as(POINTER(c_float))
        y_ptr = y.ctypes.data_as(POINTER(c_float))
        
        # Fit the model
        cuda_svm_lib.fit_svm(self.svm_ptr, X_ptr, y_ptr, 
                            c_int(X.shape[0]), c_int(X.shape[1]))
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Perform classification or regression on samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
        
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Class labels (classification) or predicted values (regression)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} features")
        
        predictions = np.zeros(X.shape[0], dtype=np.float32)
        
        X_ptr = X.ctypes.data_as(POINTER(c_float))
        pred_ptr = predictions.ctypes.data_as(POINTER(c_float))
        
        cuda_svm_lib.predict_svm(self.svm_ptr, X_ptr, pred_ptr,
                                c_int(X.shape[0]), c_int(X.shape[1]))
        
        return predictions
    
    def predict_proba(self, X):
        """
        Compute probabilities of possible outcomes for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples
        
        Returns:
        --------
        probabilities : array, shape (n_samples,)
            Probability estimates
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        if not self.probability:
            raise RuntimeError("Probability estimation not enabled. Set probability=True")
        
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} features")
        
        probabilities = np.zeros(X.shape[0], dtype=np.float32)
        
        X_ptr = X.ctypes.data_as(POINTER(c_float))
        prob_ptr = probabilities.ctypes.data_as(POINTER(c_float))
        
        cuda_svm_lib.predict_proba_svm(self.svm_ptr, X_ptr, prob_ptr,
                                      c_int(X.shape[0]), c_int(X.shape[1]))
        
        return probabilities
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels for X
        
        Returns:
        --------
        score : float
            Mean accuracy
        """
        predictions = self.predict(X)
        
        if self.svm_type in ['c_svc', 'nu_svc']:
            # Classification accuracy
            return np.mean(predictions == y)
        else:
            # Regression R² score
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot)
    
    def __del__(self):
        """Cleanup CUDA resources"""
        if hasattr(self, 'svm_ptr') and self.svm_ptr:
            cuda_svm_lib.destroy_svm(self.svm_ptr)

# Convenience classes for specific use cases
class CudaSVC(CudaSVM):
    """CUDA-accelerated Support Vector Classifier"""
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', **kwargs):
        super().__init__(svm_type='c_svc', C=C, kernel=kernel, gamma=gamma, **kwargs)

class CudaSVR(CudaSVM):
    """CUDA-accelerated Support Vector Regressor"""
    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='scale', **kwargs):
        super().__init__(svm_type='epsilon_svr', C=C, epsilon=epsilon, 
                        kernel=kernel, gamma=gamma, **kwargs)
