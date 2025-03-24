import numpy as np
import matplotlib.pyplot as plt

def get_SPD_matrix(n):
    A = np.random.randint(0, 6, size=(n, n))  # Generate random integers between 0 and 5
    return np.dot(A, A.T)  # Matrix multiplied by its transpose is SPD

def get_vector(n):
    return np.random.randint(0, 6, size=n)


