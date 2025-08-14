import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from SVM.cuda_svm import CudaSVC, CudaSVR

# Classification Example
print("=== CUDA SVM Classification ===")
X_cls, y_cls = make_classification(n_samples=10000, n_features=20, 
                                  n_informative=15, n_redundant=5, 
                                  n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to -1, 1
y_train_svm = np.where(y_train == 0, -1, 1)
y_test_svm = np.where(y_test == 0, -1, 1)

# Create and train CUDA SVM classifier
svc = CudaSVC(C=1.0, kernel='rbf', gamma='scale', probability=True)
svc.fit(X_train_scaled, y_train_svm)

# Make predictions
y_pred = svc.predict(X_test_scaled)
y_proba = svc.predict_proba(X_test_scaled)

# Calculate accuracy
accuracy = svc.score(X_test_scaled, y_test_svm)
print(f"Classification Accuracy: {accuracy:.4f}")

print("\n=== CUDA SVM Regression ===")
# Regression Example
X_reg, y_reg = make_regression(n_samples=10000, n_features=20, 
                              n_informative=15, noise=0.1, 
                              random_state=42)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Standardize features and targets
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_r_scaled = scaler_X.fit_transform(X_train_r)
X_test_r_scaled = scaler_X.transform(X_test_r)
y_train_r_scaled = scaler_y.fit_transform(y_train_r.reshape(-1, 1)).ravel()
y_test_r_scaled = scaler_y.transform(y_test_r.reshape(-1, 1)).ravel()

# Create and train CUDA SVM regressor
svr = CudaSVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale')
svr.fit(X_train_r_scaled, y_train_r_scaled)

# Make predictions
y_pred_r = svr.predict(X_test_r_scaled)

# Calculate R² score
r2_score = svr.score(X_test_r_scaled, y_test_r_scaled)
print(f"Regression R² Score: {r2_score:.4f}")

# Performance comparison example
import time
from sklearn.svm import SVC, SVR

print("\n=== Performance Comparison ===")
# Compare with sklearn SVM
start_time = time.time()
sklearn_svc = SVC(C=1.0, kernel='rbf', gamma='scale')
sklearn_svc.fit(X_train_scaled, y_train_svm)
sklearn_pred = sklearn_svc.predict(X_test_scaled)
sklearn_time = time.time() - start_time

start_time = time.time()
cuda_svc = CudaSVC(C=1.0, kernel='rbf', gamma='scale')
cuda_svc.fit(X_train_scaled, y_train_svm)
cuda_pred = cuda_svc.predict(X_test_scaled)
cuda_time = time.time() - start_time

print(f"Scikit-learn SVM Time: {sklearn_time:.2f}s")
print(f"CUDA SVM Time: {cuda_time:.2f}s")
print(f"Speedup: {sklearn_time/cuda_time:.2f}x")

#----------------------------------------------------------------------------------------

# Advanced usage with hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

class CudaSVMGridSearch(BaseEstimator, ClassifierMixin):
    """Wrapper for CUDA SVM to work with sklearn GridSearchCV"""
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', **kwargs):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.kwargs = kwargs
        
    def fit(self, X, y):
        self.svm_ = CudaSVC(C=self.C, kernel=self.kernel, gamma=self.gamma, **self.kwargs)
        self.svm_.fit(X, y)
        return self
        
    def predict(self, X):
        return self.svm_.predict(X)
        
    def score(self, X, y):
        return self.svm_.score(X, y)

# Hyperparameter tuning
print("\n=== Hyperparameter Tuning ===")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

grid_search = GridSearchCV(CudaSVMGridSearch(), param_grid, cv=5, 
                          scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train_svm)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Test with best parameters
best_svm = grid_search.best_estimator_
best_accuracy = best_svm.score(X_test_scaled, y_test_svm)
print(f"Test accuracy with best parameters: {best_accuracy:.4f}")
