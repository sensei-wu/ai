from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt

x_init = np.array([0.25, 0.25, 10.0]) 

def compute_f(x):
    return 100 * x[0]**2 + x[1] ** 2 + (x[2] - 20) ** 2

def compute_gradient(x):
    return np.array([200 * x[0], 2 * x[1], 2 * (x[2] - 20)])

def constraint_set():
    # Equality constraint: x1 + x2 + x3/20 = 1
    return {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] / 20 - 1}

from scipy.optimize import linprog

def solve_linear_minimization(gradient):
    """
    Solve the linear minimization problem using a standard LP solver (linprog).
    Minimize: gradient ⋅ s
    Subject to: x1 + x2 + x3/20 = 1 and x1, x2, x3 >= 0
    """
    # Coefficients for the objective function (dot product with gradient)
    c = gradient

    # Equality constraint: x1 + x2 + x3/20 = 1
    A_eq = [[1, 1, 1 / 20]]
    b_eq = [1]

    # Bounds: x1, x2, x3 >= 0
    bounds = [(0, None), (0, None), (0, None)]

    # Solve the linear program
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if result.success:
        return result.x
    else:
        raise ValueError("Linear minimization problem did not converge")

def frank_wolfe():

    x = x_init
    tol=1e-6

    trajectory = [x.copy()]

    for i in range (1, 100):
        # Solve the linear minimization problem
        gradient = compute_gradient(x)
        s_next = solve_linear_minimization(gradient)

        alpha = 2/(i+2)
        x_next = (1 - alpha) * x + alpha * (s_next)
        trajectory.append(x_next.copy())

        # Check for convergence
        if np.linalg.norm(x_next - x) < tol:
            break

        x = x_next

    return x_next, trajectory

def projected_gradient_descent(f, grad_f, x0, domain, lr=0.1, max_iter=500, tol=1e-6):
    """
    Projected Gradient Descent algorithm for convex optimization.

    Parameters:
    - f: function to minimize
    - grad_f: gradient of the function
    - x0: initial point
    - domain: function that projects onto the feasible domain
    - lr: learning rate
    - max_iter: maximum number of iterations
    - tol: tolerance for convergence

    Returns:
    - x: the point that minimizes f
    - trajectory: list of points visited during optimization
    """
    x = x0
    trajectory = [x0.copy()]
    for i in range(max_iter):
        grad = grad_f(x)
        x_new = domain(x - lr * grad)  # Gradient descent step followed by projection
        trajectory.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x, trajectory

# Define the domain projection (e.g., onto the unit simplex)
def domain(v):
    """
    Project onto the feasible domain defined by:
    x1 + x2 + x3/20 = 1 and x1, x2, x3 >= 0.
    """
    # Solve the projection using a simple heuristic for the constraints
    x = np.maximum(v, 0)  # Ensure non-negativity
    x[2] = 20 * (1 - x[0] - x[1])  # Enforce x1 + x2 + x3/20 = 1
    if x[2] < 0:  # If x3 becomes negative, adjust x1 and x2 proportionally
        x[2] = 0
        total = x[0] + x[1]
        if total > 0:
            x[0] *= 1 / total
            x[1] *= 1 / total
    return x

result, trajectory = frank_wolfe()
print("Optimal point (Frank-Wolfe with trajectory):", result)
print("Optimal value (Frank-Wolfe with trajectory):", compute_f(result))

# Run Projected Gradient Descent with trajectory tracking
result_pgd, trajectory_pgd = projected_gradient_descent(compute_f, compute_gradient, np.array([0.5, 0.5, 10.0])  
, domain, lr=0.05)
print("Optimal point (PGD):", result_pgd)
print("Optimal value (PGD):", compute_f(result_pgd))

# Generate level set plot with trajectory for Frank-Wolfe
x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 100 * X**2 + Y**2 + (20 - 20)**2  # Simplified for visualization

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# Plot the Frank-Wolfe trajectory
trajectory = np.array(trajectory)
ax.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='red', label='Frank-Wolfe Trajectory')
ax.scatter(trajectory[0, 0], trajectory[0, 1], color='blue', label='Start (FW)', zorder=5)
ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='green', label='End (FW)', zorder=5)

# Add labels
ax.set_title('Level Sets and Trajectory of Frank-Wolfe Algorithm')
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.legend()
plt.show()

 # Generate level set plot with trajectory for PGD
x_vals = np.linspace(-0.5, 1.5, 100)
y_vals = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 100 * X**2 + Y**2 + (20 - 20)**2  # Simplified for visualization

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# Plot the PGD trajectory
trajectory_pgd = np.array(trajectory_pgd)
ax.plot(trajectory_pgd[:, 0], trajectory_pgd[:, 1], marker='o', color='red', label='PGD Trajectory')
ax.scatter(trajectory_pgd[0, 0], trajectory_pgd[0, 1], color='blue', label='Start (PGD)', zorder=5)
ax.scatter(trajectory_pgd[-1, 0], trajectory_pgd[-1, 1], color='green', label='End (PGD)', zorder=5)

# Add labels
ax.set_title('Level Sets and Trajectory of Projected Gradient Descent')
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.legend()
plt.show()