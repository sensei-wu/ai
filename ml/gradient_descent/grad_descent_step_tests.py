import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from scipy.optimize import minimize  # Import minimize from scipy.optimize

from quadraic_fn import get_SPD_matrix, get_vector

num_iterations = 1000
initial_guess = np.array([3.0, 3.0, 3.0])  
tolerance = 1e-6  # Stopping criterion

# Generate Q and b once
Q = get_SPD_matrix(3)
b = get_vector(3)

print("Q:", Q)
print("b:", b)

def gradient_descent_fixed(lr, num_iterations=num_iterations):
    x = initial_guess.copy()
    history = []
    for i in range(num_iterations):
        gr = compute_gradient(x)
        if np.linalg.norm(gr) < tolerance:  
            break
        x -= lr * gr
        history.append(np.linalg.norm(x))  # Store the norm of x
    return x, history, i+1 

def gradient_descent_1byt(num_iterations=num_iterations):
    x = initial_guess.copy()
    t = 1.0
    history = []
    for i in range(num_iterations):
        gr = compute_gradient(x)
        if np.linalg.norm(gr) < tolerance:  
            break
        x -= (1/t) * gr
        t += 1
        history.append(np.linalg.norm(x))  # Store the norm of x
    return x, history, i+1  

def gradient_descent_1bySqrtOft(num_iterations=num_iterations):
    x = initial_guess.copy()
    t = 1.0
    history = []
    epsilon = 1e-8  # Small value to avoid division by zero
    for i in range(num_iterations):
        gr = compute_gradient(x)
        if np.linalg.norm(gr) < tolerance:  
            break
        x -= (1/(sqrt(t) + epsilon)) * gr  # Add epsilon to avoid division by zero
        t += 1
        history.append(np.linalg.norm(x))  # Store the norm of x
    return x, history, i+1  

# gradient for the function 1/2 x^TQx + q^Tx
def compute_gradient(x):
    return np.dot(Q, x) + b

# Define the quadratic function
def quadratic_function(x):
    return 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(b, x)

x, history_fixed, steps_fixed = gradient_descent_fixed(0.01)
print("GD using fixed learning rate at:", x, "in", steps_fixed, "steps")

x, history_fixed1, steps_fixed1 = gradient_descent_fixed(0.1)
print("GD using fixed learning rate at:", x, "in", steps_fixed1, "steps")

x, history_1byt, steps_1byt = gradient_descent_1byt(3000)
print("GD using 1/t learning rate at:", x, "in", steps_1byt, "steps")

x, history_1bySqrtOft, steps_1bySqrtOft = gradient_descent_1bySqrtOft(10000)
print("GD using 1/sqrt(t) learning rate at:", x, "in", steps_1bySqrtOft, "steps")

# Verify the minimum using scipy.optimize.minimize
result = minimize(quadratic_function, initial_guess)
print("GD using scipy.optimize.minimize at:", result.x)

# Plotting the convergence
plt.plot(history_fixed, label=f'Fixed Learning Rate ({steps_fixed} steps)')
plt.plot(history_fixed1, label=f'Fixed Learning Rate 1 ({steps_fixed1} steps)')
plt.plot(history_1byt, label=f'1/t Learning Rate ({steps_1byt} steps)')
plt.plot(history_1bySqrtOft, label=f'1/sqrt(t) Learning Rate ({steps_1bySqrtOft} steps)')
plt.xlabel('Iteration')
plt.ylabel('Norm of x')
plt.legend()
plt.title('Convergence of Gradient Descent Algorithms')
plt.show()