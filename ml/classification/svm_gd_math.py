import numpy as np
import matplotlib.pyplot as plt

from utils import generate_data, plot_svm_boundary

# Generate data
n_samples, n_features = 100, 2
X, y = generate_data(n_samples, n_features)

def hinge_loss(y, X, w, b):
    margins = y * (np.dot(X, w) + b)
    losses = np.maximum(0, 1 - margins)
    return np.mean(losses) + 0.5 * np.dot(w, w)  # Regularization term added

def compute_gradients(y, X, w, b, C):
    margins = y * (np.dot(X, w) + b)
    indicator = (margins < 1).astype(float)  # Indicator function for misclassified points
    grad_w = w - C * np.dot((y * indicator), X)  # Gradient for w (weights)
    grad_b = -C * np.sum(y * indicator)  # Gradient for b (bias)
    return grad_w, grad_b

def svm_gradient_descent(X, y, w, b, C, learning_rate, num_iterations):
    for _ in range(num_iterations):
        grad_w, grad_b = compute_gradients(y, X, w, b, C)
        w -= learning_rate * grad_w  # Update weights
        b -= learning_rate * grad_b  # Update bias
    return w, b

w = np.zeros(X.shape[1])
b = 0

# Set hyperparameters
C = 1.0  # Regularization parameter
learning_rate = 0.01
num_iterations = 1000

# Train the SVM using gradient descent
w, b = svm_gradient_descent(X, y, w, b, C, learning_rate, num_iterations)

print("Trained weights:", w)
print("Trained bias:", b)

print(f"Final weights: {w}, Bias: {b}")

plot_svm_boundary(w, b, X, y, title="scikit-learn")
plt.show()