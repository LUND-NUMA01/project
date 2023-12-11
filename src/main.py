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

# ---------------------------------------------------------- #
#                           Task 2                           #
# ---------------------------------------------------------- #

def air_resistance(s, p=1.23, cw=0.45, d=0.24):
    return 0.5 * p * cw * np.pi / 4 * d**2 * s**2

class BoundaryValueProblem:
    def solve_with_euler(self, z0, a0, s0=9, x0=0, y0=1.75):
        T = 1
        N = 16
        
        fn_x = lambda t, x: ___ / z0
        fn_y = lambda t, y: ___ / z0

        # computing x
        (_, ux) = explicit_euler(fn_x, T, N, x0)

        # computing y
        (_, uy) = explicit_euler(fn_y, T, N, y0)
        return (ux, uy)

# ---------------------------------------------------------- #
#                           Task 3                           #
# ---------------------------------------------------------- #

def plot_ball_trajectory():
    x, y = BoundaryValueProblem().solve_with_euler(1, np.pi / 3)
    plt.scatter(x, y)
    plt.show()

# ---------------------------------------------------------- #
#                           Task 4                           #
# ---------------------------------------------------------- #

def nonlinear_newtons_method():
    return

# ---------------------------------------------------------- #
#                           Task 5                           #
# ---------------------------------------------------------- #

def find_optimal_angle():
    return

# ---------------------------------------------------------- #
#                           Tests                            #
# ---------------------------------------------------------- #

def test(t, y):
    return y

if __name__ == "__main__":
    (x, y) = explicit_euler(test, 10, 100, y0)
    plt.plot(x, y)
    plt.show()


