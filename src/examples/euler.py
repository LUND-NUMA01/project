import numpy as np
import matplotlib.pyplot as plt

from algorithms import explicit_euler

T = 2 * np.pi
N = 24

func = lambda t, u: -u*np.cos(t)

t, u = explicit_euler(func, T, N, 1)

plt.scatter(t, u)
plt.show()
