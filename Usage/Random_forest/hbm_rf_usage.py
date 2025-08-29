import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as SklearnRF

# Import our optimized CUDA Random Forest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'HBM_RandomForest'))
from Random_forest.cuda_rf_optimized import OptimizedCudaRFClassifier, CudaRFError

def classification_example():
    """Classification example with performance comparison"""
    print("=== CUDA Random Forest Classification Example ===")
    
    # Generate dataset
    X, y = make_classification(
        n_samples=20000, n_features=100, 
        n_informative=40, n_redundant=20,
        n_classes=5, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # It's good practice to scale features, though less critical for RF
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Train CUDA Random Forest ---
    print("\nTraining with Optimized CUDA Random Forest...")
    cuda_rf = OptimizedCudaRFClassifier(
        n_estimators=200,
        max_depth=12,
        verbose=True,
        random_state=42
    )
    cuda_rf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_cuda = cuda_rf.predict(X_test_scaled)
    accuracy_cuda = accuracy_score(y_test, y_pred_cuda)
    
    print(f"\n--- CUDA RF Results ---")
    print(f"Accuracy: {accuracy_cuda:.4f}")
    metrics = cuda_rf.get_performance_metrics()
    print(f"Training time: {metrics['training_time_s']:.2f}s")
    print(f"GPU memory used: {metrics['memory_usage_mb']:.2f}MB")

    # --- Compare with Scikit-learn ---
    print("\nTraining with Scikit-learn Random Forest...")
    start_time = time.time()
    sklearn_rf = SklearnRF(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    sklearn_rf.fit(X_train_scaled, y_train)
    sklearn_time = time.time() - start_time
    y_pred_sklearn = sklearn_rf.predict(X_test_scaled)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"\n--- Scikit-learn RF Results ---")
    print(f"Accuracy: {accuracy_sklearn:.4f}")
    print(f"Training time: {sklearn_time:.2f}s")
    print(f"\nSpeedup: {sklearn_time / metrics['training_time_s']:.2f}x")

if __name__ == "__main__":
    try:
        classification_example()
        print("\n✓ Example completed successfully!")
    except (CudaRFError, FileNotFoundError) as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure the 'HBM_RandomForest' library is compiled with 'make'.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")