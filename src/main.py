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

def air_resistance(s, p=1.23, cw=0.45, d=0.24):
    return 0.5 * p * cw * np.pi / 4 * d**2 * s**2

B = 1.23*0.45*0.24*np.pi/8

C = -0.5*p*cw*np.pi/4*d**2/m

z = 0.3

def func_velocity(t, u):
    s = np.sqrt(u[0]**2 + u[1]**2)
    ax = C * s * u[0] * z
    ay = C * s * u[1] * z**2 - g
    return np.array([ax,ay])

def func_positions(velocities):
    return lambda t, u: ()

class BoundaryValueProblem:
    def solve_with_euler(self, z0, a0, s0=9, x0=0, y0=1.75):
        T = 2
        N = 16

        vx0 = s0 * np.cos(a0)
        vy0 = s0 * np.sin(a0)

        initial_velocity = np.array([vx0, vy0])
        initial_positions = np.array([x0, y0])

        (vt, vu) = explicit_euler(func_velocity, T, N, initial_velocity)

        print(vu)

        h = T / N

        values = np.empty(N, dtype=object)

        values[0] = initial_positions

        for i in range(1, N):
            values[i] = (values[i-1] + vu[i-1]*h*z)

        x = np.zeros(N)
        y = np.zeros(N)

        for i in range(N):
            x[i] = values[i][0]
            y[i] = values[i][1]

        return x, y

# ---------------------------------------------------------- #
#                           Task 3                           #
# ---------------------------------------------------------- #

def plot_ball_trajectory():
    x, y = BoundaryValueProblem().solve_with_euler(1, np.pi / 4)
    plt.scatter(x, y)
    plt.show()

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
    plot_ball_trajectory()
