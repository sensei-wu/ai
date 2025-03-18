import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Generate linearly separable data
X, y = make_classification(n_samples=20, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, random_state=42)
y = 2 * y - 1  # Convert labels from (0,1) to (-1,1) for SVM formulation

# Train an SVM with linear kernel
clf = SVC(kernel='linear', C=1e6)  # High C to enforce hard margin
clf.fit(X, y)

# Extract the normal vector w and bias b
w = clf.coef_[0]
b = clf.intercept_[0]

# Compute decision boundary
x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y_vals = -(w[0] * x_vals + b) / w[1]

# Compute margins
margin = 1 / np.linalg.norm(w)
y_vals_margin_up = y_vals + margin
y_vals_margin_down = y_vals - margin

# Shrink w artificially (incorrect scaling)
k = 0.3  # Scaling factor
w_shrunk = k * w
b_shrunk = k * b

# Compute new decision boundary after shrinking w
y_vals_shrunk = -(w_shrunk[0] * x_vals + b_shrunk) / w_shrunk[1]

# Plot the original decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', marker='o', label="Class +1")
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='blue', marker='s', label="Class -1")
plt.plot(x_vals, y_vals, 'k-', label="Optimal Decision Boundary")
plt.plot(x_vals, y_vals_margin_up, 'k--', label="Margin Boundaries")
plt.plot(x_vals, y_vals_margin_down, 'k--')

# Plot the incorrect decision boundary after shrinking w
plt.plot(x_vals, y_vals_shrunk, 'g-', label="Shrunken w Boundary")

plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Effect of Shrinking w on SVM Decision Boundary")
plt.show()
