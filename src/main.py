import numpy as np
import matplotlib.pyplot as plt

# some facts:
d = 0.24 # m
m = 0.6 # kg
cw = 0.45
p = 1.23 # kg/m^3
xB = 2 # m
yB = 3.05 # m
g = 9.81 # m/s^2

# intial conditions:
x0 = 0  # m
y0 = 1.75 # m
s0 = 9  # m/s

# ---------------------------------------------------------- #
#                           Task 1                           #
# ---------------------------------------------------------- #

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
    t = np.zeros(N, dtype=object)
    u = np.zeros(N, dtype=object)

    # initial values
    t[0] = 0
    u[0] = y0

    # N iterations
    for i in range(1, N):
        t[i] = i * h
        u[i] = u[i-1] + h * f(t[i-1], u[i-1])
    
    # return both the arrays t_i and u_i
    return (t, u)

# (x, y) = explicit_euler(lambda t, u: u, 5, 20, 2)
# plt.scatter(x, y)
# plt.show()

# ---------------------------------------------------------- #
#                           Task 2                           #
# ---------------------------------------------------------- #

class BoundaryValueProblem:
    def __init__():
        ()

# ---------------------------------------------------------- #
#                           Task 3                           #
# ---------------------------------------------------------- #

def plot_ball_trajectory():
    return

# ---------------------------------------------------------- #
#                           Task 4                           #
# ---------------------------------------------------------- #

def nonlinear_newtons_method(f, j, a0, z0):
    return

# ---------------------------------------------------------- #
#                           Task 5                           #
# ---------------------------------------------------------- #

def find_optimal_angle():
    f = () - xB, yB # function
    j = ()  # jacobian / derivative
    # initial guesses
    a0 = 1.4
    z0 = 0.8

    a, z = nonlinear_newtons_method(f, j, a0, z0)
    return

# ---------------------------------------------------------- #
#                           Tests                            #
# ---------------------------------------------------------- #

if __name__ == "__main__":
    (x, y) = explicit_euler(test, 10, 100, y0)
    plt.plot(x, y)
    plt.show()

