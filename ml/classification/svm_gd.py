import numpy as np
import matplotlib.pyplot as plt
from utils import generate_data, plot_svm_boundary

# Generate data
n_samples, n_features = 100, 2
X, y = generate_data(n_samples, n_features)

# SVM Parameters
learning_rate = 0.001
C = 1.0  # Regularization term
epochs = 500

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0

# Training loop
for epoch in range(epochs):
    for i in range(len(y)):
        # Compute margin
        margin = y[i] * (np.dot(w, X[i]) + b)
        
        # Hinge loss condition: If correctly classified, only minimize w
        if margin >= 1:
            w -= learning_rate * w  # Only weight decay (minimize ||w||^2)
        else:
            # Update weights and bias for misclassified points
            w -= learning_rate * (w - C * y[i] * X[i])
            b -= learning_rate * (-C * y[i])
    
    # Optional: Print loss every 100 epochs
    if epoch % 100 == 0:
        loss = (1/2) * np.dot(w, w) + C * np.sum(np.maximum(0, 1 - y * (np.dot(X, w) + b)))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print(f"Final weights: {w}, Bias: {b}")

# Show the plot with both data points and decision boundary
plot_svm_boundary(w, b, X, y, title="scikit-learn")
plt.show()
