import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.optimize import minimize 

from quadraic_fn import get_SPD_matrix, get_vector

num_iterations = 1000
initial_guess = np.array([0.0, 0.0, 0.0])  
tolerance = 1e-4  # Stopping criterion

Q = np.array([[13, 12, -2],
             [12, 17, 6],
              [-2, 6, 12]])
b = np.array([-22, -14.5, 13])

print("Q:", Q)
print("b:", b)

eigenvalues, eigenvectors = np.linalg.eig(Q)

print("Eigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

fixed_eta = 1 /np.max(eigenvalues)  # Use max eigenvalue
print("Fixed learning rate:", fixed_eta)

def gradient_descent_fixed(lr, num_iterations=num_iterations):
    x = initial_guess.copy()
    history = []
    for i in range(num_iterations):
        gr = compute_gradient(x)
        if np.linalg.norm(gr) < tolerance:  
            break
        x -= lr * gr
        history.append(quadratic_function(x) - f_opt)  # Store suboptimality
    return x, history, i+1 

# gradient for the function 1/2 x^TQx + q^Tx
def compute_gradient(x):
    return np.dot(Q, x) + b

def quadratic_function(x):
    return 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(b, x)

# Compute the optimal value using scipy.optimize.minimize for comparison
result = minimize(quadratic_function, initial_guess)
x_opt = result.x
f_opt = quadratic_function(x_opt)
print("GD using scipy.optimize.minimize at:", x_opt)

x, history_fixed, steps_fixed = gradient_descent_fixed(fixed_eta)
print("GD using fixed learning rate at:", x, "in", steps_fixed, "steps")

def line_search(x, direction, alpha=0.5, beta=0.5):
    """
    Perform backtracking line search to find the step size.
    Args:
        x: Current point.
        direction: Descent direction.
        alpha: Initial step size.
        beta: Step size reduction factor (0 < beta < 1).
    Returns:
        Step size satisfying the Armijo condition.
    """
    step_size = alpha
    while quadratic_function(x + step_size * direction) > quadratic_function(x) + alpha * step_size * np.dot(compute_gradient(x), direction):
        step_size *= beta
    return step_size

def gradient_descent_with_line_search(num_iterations=num_iterations):
    x = initial_guess.copy()
    history = []
    for i in range(num_iterations):
        gr = compute_gradient(x)
        if np.linalg.norm(gr) < tolerance:
            break
        direction = -gr  
        step_size = line_search(x, direction)
        x += step_size * direction
        history.append(quadratic_function(x) - f_opt)  # Store suboptimality
    return x, history, i+1

x_ls, history_ls, steps_ls = gradient_descent_with_line_search()
print("GD using line search at:", x_ls, "in", steps_ls, "steps")

plt.plot(history_fixed, label=f'Fixed LR ({steps_fixed} steps)')
plt.plot(history_ls, label=f'Line Search ({steps_ls} steps)')
plt.xlabel('Iteration')
plt.ylabel('Suboptimality (f(x) - f(x_opt))')
plt.yscale('log')  # Use a logarithmic scale for better visualization
plt.legend()
plt.title('Convergence of Gradient Descent Algorithms (Suboptimality)')
plt.show()