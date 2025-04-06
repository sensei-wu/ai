import numpy as np
import matplotlib.pyplot as plt

# Create a grid of x and y values
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Define the function f(x, y) = |x + y|
Z = np.abs(X + Y)

# Create the plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Add labels
ax.set_title('f(x, y) = |x + y|')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')

# Show the plot
plt.show()

# Create the contour plot (level sets)
fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contour(X, Y, Z, levels=20, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

# Add labels
ax.set_title('Level Sets of f(x, y) = |x + y|')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Show the plot
plt.show()
