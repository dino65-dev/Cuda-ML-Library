"""
Example usage of Optimized CUDA SVM
"""

import numpy as np
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

# Import our optimized CUDA SVM
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'HBM_SVM'))
from cuda_svm_optimized import OptimizedCudaSVC, OptimizedCudaSVR

def classification_example():
    """Classification example with performance comparison"""
    print("=== CUDA SVM Classification Example ===")
    
    # Generate dataset
    X, y = make_classification(
        n_samples=10000, n_features=50, 
        n_informative=30, n_redundant=10,
        n_classes=2, random_state=42
    )
    
    # Convert labels to SVM format (-1, 1)
    y = np.where(y == 0, -1, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train CUDA SVM
    print("Training CUDA SVM...")
    start_time = time.time()
    
    cuda_svm = OptimizedCudaSVC(
        C=1.0, kernel='rbf', gamma='scale', 
        verbose=True
    )
    cuda_svm.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = cuda_svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nResults:")
    print(f"Training time: {training_time:.2f}s")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Show performance metrics
    metrics = cuda_svm.get_performance_metrics()
    print(f"GPU memory used: {metrics['memory_usage_gb']:.2f}GB")
    
    return cuda_svm, accuracy

def regression_example():
    """Regression example"""
    print("\n=== CUDA SVM Regression Example ===")
    
    # Generate dataset
    X, y = make_regression(
        n_samples=5000, n_features=30,
        n_informative=20, noise=0.1,
        random_state=42
    )
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    
    # Train CUDA SVR
    print("Training CUDA SVR...")
    cuda_svr = OptimizedCudaSVR(
        C=1.0, epsilon=0.1, kernel='rbf', 
        gamma='scale', verbose=True
    )
    cuda_svr.fit(X_train_scaled, y_train_scaled)
    
    # Make predictions
    y_pred_scaled = cuda_svr.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2:.4f}")
    
    return cuda_svr, r2

def batch_prediction_example():
    """Example of batch prediction for high throughput"""
    print("\n=== Batch Prediction Example ===")
    
    # Generate large test dataset
    X_large, y_large = make_classification(
        n_samples=50000, n_features=100,
        random_state=42
    )
    y_large = np.where(y_large == 0, -1, 1)
    
    # Train a small model first
    X_train = X_large[:1000]
    y_train = y_large[:1000]
    
    svm = OptimizedCudaSVC(kernel='rbf', gamma='scale')
    svm.fit(X_train, y_train)
    
    # Batch prediction
    print("Performing batch prediction...")
    start_time = time.time()
    
    # Regular prediction
    pred_regular = svm.predict(X_large)
    regular_time = time.time() - start_time
    
    start_time = time.time()
    # Async batch prediction
    pred_async = svm.predict(X_large, batch_async=True)
    async_time = time.time() - start_time
    
    print(f"Regular prediction time: {regular_time:.2f}s")
    print(f"Async prediction time: {async_time:.2f}s")
    print(f"Speedup: {regular_time/async_time:.2f}x")
    
    # Verify results are the same
    print(f"Results match: {np.allclose(pred_regular, pred_async)}")

if __name__ == "__main__":
    try:
        # Run examples
        classification_example()
        regression_example()
        batch_prediction_example()
        
        print("\n✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("\nMake sure to compile the CUDA library first:")
        print("  make clean && make")
