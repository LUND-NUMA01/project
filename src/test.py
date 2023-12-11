import numpy as np
import matplotlib.pyplot as plt

def explicit_euler(f, T, N, y0):
    """
    Solve the ODE using explicit Euler method over t ∈ [0, T] using N steps.

    Parameters:
        - f: The function representing the derivative dy/dt.
        - T: The endpoint of the interval
        - N: The number of iterations
        - y0: The initial value

    Returns:
        - t: List of time points.
        - u: List of corresponding approximated y values.
    """
    # step size
    h = T / N

    # pre-initialize zeroed arrays
    t = np.zeros(N)
    u = np.zeros(N)

    # initial values
    t[0] = 0
    u[0] = y0

    # N iterations
    for i in range(1, N):
        t[i] = i * h
        u[i] = u[i-1] + h * f(t[i-1], u[i-1])
    
    # return bot the arrays t_i and u_i
    return (t, u)


T = 2 * np.pi
N = 24

def fun(t, y):
    return -y*np.cos(t)

(t, u) = explicit_euler(fun, T, N, 1)

plt.scatter(t, u)
plt.show()
