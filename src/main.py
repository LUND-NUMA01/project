import numpy as np
import matplotlib.pyplot as plt

from algorithms import explicit_euler, newton

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

# constant combining all other constants for
# the air resistance and acceleration
C = -0.5*p*cw*np.pi/4*d**2/m

def solve_with_euler(z0, a0, s0=9, x0=0, y0=1.75):
    T = 1
    N = 32

    # compute initial x,y velocities
    vx0 = s0 * np.cos(a0)
    vy0 = s0 * np.sin(a0)
    
    # create array for the initial values
    initial_values = np.array([x0, y0, vx0, vy0])
    
    def func(_, u):
        # u[0] = x-position
        # u[1] = y-position
        # u[2] = x-velocity
        # u[3] = y-velocity
        s = np.sqrt(u[2]**2 + u[3]**2)
        ax = C * s * u[2] * z0
        ay = C * s * u[3] * z0**2 - g
        vx = u[2] * z0
        vy = u[3] * z0
        return np.array([vx, vy, ax, ay])

    _, u = explicit_euler(func, T, N, initial_values)

    # extract x and y values from u
    x = np.zeros(N)
    y = np.zeros(N)

    for i in range(N):
        x[i] = u[i][0]
        y[i] = u[i][1]

    # return x and y values
    return x, y

# ---------------------------------------------------------- #
#                           Task 3                           #
# ---------------------------------------------------------- #

def plot_ball_trajectory():
    z0 = 0.5
    a0 = 1
    x, y = solve_with_euler(z0, a0)
    plt.xlim([0, 2.5])
    plt.scatter(x, y)
    plt.show()

# ---------------------------------------------------------- #
#                           Task 5                           #
# ---------------------------------------------------------- #

def function(p, q):
    x, y = solve_with_euler(p, q)
    return np.array([x[-1] - xB, y[-1] - yB])

def numerical_jacobian(func, x, epsilon=1e-6):
    """
    Compute the numerical approximation of the Jacobian matrix of a function.

    Parameters:
    - func: Callable function for which the Jacobian is computed. It should take a NumPy array as input.
    - x: Point at which the Jacobian is evaluated.
    - epsilon: Perturbation value for finite differences. Default is 1e-6.

    Returns:
    - numpy.ndarray: Numerical approximation of the Jacobian matrix.
    """
    n = len(x)
    m = len(func(*x))
    J = np.zeros((m, n))

    # print(f'n: {n}, m: {m}')
    # print(f'initial guess: {x}')

    # print(f'test: {func(*x)}')

    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += epsilon

        J[:, i] = (func(*x_perturbed) - func(*x)) / epsilon

    return J

def find_optimal_angle():
    # initial guesses
    z0 = 0.5
    a0 = 1

    jacobian = lambda x, y: numerical_jacobian(function, [x, y])

    return newton(function, jacobian, [z0, a0])

# ---------------------------------------------------------- #
#                           Tests                            #
# ---------------------------------------------------------- #

if __name__ == "__main__":
    # plot_ball_trajectory()
    z, a = find_optimal_angle()
    print(f'time: {1/z}, angle: {a}')
