import numpy as np
import matplotlib.pyplot as plt
from utils import generate_data, plot_svm_boundary
from sklearn import svm

# Generate data
n_samples, n_features = 100, 2
X, y = generate_data(n_samples, n_features)

# Train an SVM model
clf = svm.SVC(kernel='linear', C=1)
clf.fit(X, y)

# Get the weights and bias from the trained model
w = clf.coef_.flatten()
b = clf.intercept_[0]

# Create a decision boundary
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(w[0] * x_vals + b) / w[1]

# Show the plot with both data points and decision boundary
plot_svm_boundary(w, b, X, y, title="scikit-learn")
plt.show()
