import numpy as np
import matplotlib.pyplot as plt
from quadraic_fn import get_SPD_matrix, get_vector

# Generate SPD matrix Q and vector b
n = 3  # Dimensionality
Q = get_SPD_matrix(n)
b = get_vector(n)

print("Q:", Q)
print("b:", b)

# Gradient function
def compute_gradient(x):
    return np.dot(Q, x) + b

# Quadratic function value
def quadratic_function(x):
    return 0.5 * np.dot(x.T, np.dot(Q, x)) + np.dot(b, x)

# Gradient Descent with early stopping
def gradient_descent(start_x, alpha_func, num_iters, tol=1e-6):
    x = start_x
    function_values = [quadratic_function(x)]
    
    for t in range(1, num_iters + 1):  
        alpha = alpha_func(t)  
        x = x - alpha * compute_gradient(x)
        function_values.append(quadratic_function(x))

        # Convergence check: Stop if change is below tolerance
        if abs(function_values[-1] - function_values[-2]) < tol:
            print(f"Converged for {alpha_func.__name__} in {t} steps with value {function_values[-1]:.6f}")
            return function_values[:t + 1], t, function_values[-1]
    
    print(f"Max iterations reached for {alpha_func.__name__} with final value {function_values[-1]:.6f}")
    return function_values, num_iters, function_values[-1]

# Learning rate functions
def fixed_alpha_01(t): return 0.01
def fixed_alpha_1(t): return 0.1
def inverse_decay(t): return 1.0 / t  # α_t = 1/t
def sqrt_decay(t): return 1.0 / np.sqrt(t)  # α_t = 1/sqrt(t)

# Learning rate strategies
learning_rates = {
    "Fixed α=0.01": fixed_alpha_01,
    "Fixed α=0.1": fixed_alpha_1,
    "α=1/t": inverse_decay,
    "α=1/sqrt(t)": sqrt_decay
}

num_iters = 200  
start_x = np.random.randn(n)  

# Run gradient descent for each learning rate strategy
plt.figure(figsize=(8, 6))
convergence_data = {}

for label, alpha_func in learning_rates.items():
    function_values, steps_taken, final_value = gradient_descent(start_x, alpha_func, num_iters)
    plt.plot(range(len(function_values)), function_values, marker="o", linestyle="--", label=f"{label} ({steps_taken} steps)")
    convergence_data[label] = (steps_taken, final_value)

# Plot settings
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.yscale("log")  
plt.legend()
plt.title("Gradient Descent with Different Learning Rate Strategies")
plt.show()

print("\nSummary of Results:")
for label, (steps, final_val) in convergence_data.items():
    print(f"{label}: Converged in {steps} steps, final function value = {final_val:.6f}")
