import numpy as np
import matplotlib.pyplot as plt

num_iterations = 1000
initial_guess = 3.0
tolerance = 1e-6  # Stopping criterion

def gradient_descent_fixed(lr, num_iterations=num_iterations):
    x = initial_guess
    history = []
    for i in range(num_iterations):
        gr = compute_gradient(x)
        if abs(gr) < tolerance:
            break
        x -= lr * gr
        history.append(x)
    return x, history, i+1 

def gradient_descent_2minust(num_iterations=num_iterations):
    x = initial_guess
    t = 1/2
    history = []
    for i in range(num_iterations):
        gr = compute_gradient(x)
        if abs(gr) < tolerance:
            break
        x -= t * gr
        t /= 2
        history.append(x)
    return x, history, i+1  

# gradient for the function 3x^2-3x+1
def compute_gradient(x):
    return 6 * x - 3

x, history_fixed, steps_fixed = gradient_descent_fixed(0.01)
print("Found minimum using fixed learning rate at:", x, "in", steps_fixed, "steps")

x, history_2minust, steps_2minust = gradient_descent_2minust(3000)
print("Found minimum using 2^-t learning rate at:", x, "in", steps_2minust, "steps")

# Plotting the convergence
plt.plot(history_fixed, label=f'Fixed Learning Rate ({steps_fixed} steps)')
plt.plot(history_2minust, label=f'2^-t Learning Rate ({steps_2minust} steps)')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.legend()
plt.title('Convergence of Gradient Descent Algorithms')
plt.show()