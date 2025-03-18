# utils.py
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n_samples, n_features):
    """
    Generate synthetic 2D data points and labels for classification.

    Parameters:
    - n_samples: Number of data points.
    - n_features: Number of features (should be 2 for plotting in 2D).

    Returns:
    - X: Data points (features).
    - y: Class labels.
    """
    np.random.seed(42)
    X_class0 = np.random.randn(50, 2) * 0.8 + np.array([2, 2])  # Class 0
    X_class1 = np.random.randn(50, 2) * 0.8 + np.array([6, 6])  # Class 1
    X = np.vstack((X_class0, X_class1))
    y = np.hstack((-1 * np.ones(50), np.ones(50)))  # Convert labels to -1 and 1
    
    return X, y

# Plot decision boundary
def plot_svm_boundary(w, b, X, y, title=""):
    plt.figure(figsize=(8,6))
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', label="Class -1")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label="Class 1")

    # Compute decision boundary
    x_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    y_vals = -(w[0] * x_vals + b) / w[1]

    # Plot decision boundary and margins
    plt.plot(x_vals, y_vals, 'k-', label="SVM Decision Boundary with: " + title)
    plt.plot(x_vals, y_vals + 1/np.linalg.norm(w), 'g--', label="Margin")
    plt.plot(x_vals, y_vals - 1/np.linalg.norm(w), 'g--')

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.title("SVM Decision Boundary with: " + title)
