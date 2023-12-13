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
        ax = C * s * u[2] * z0**2
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
    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += epsilon
        J[:, i] = (func(*x_perturbed) - func(*x)) / epsilon
    return J

def find_optimal_angle(z_0, a_0, max_iter=1.e3):  # initial guesses
    jacobian = lambda x, y: numerical_jacobian(function, [x, y])
    return newton(function, jacobian, [z_0, a_0], max_iterations=max_iter)


def plot_intermediate_trajectory(z_0, a_0):
    [z, a], set_interations = find_optimal_angle(z_0, a_0) # determines how many intermediate trajectories we should plot
    # z0 = 1 # initial guess, an idea for the reader
    # a0 = 1.5 
    first_iter_z = z_0
    first_iter_a = a_0
    x, y = solve_with_euler(first_iter_z, first_iter_a)
    plt.figure(figsize=(6,6))
    plt.plot(x,y, color="black", label="The initial guess trajectory" )

    for i in range(1, set_interations+1):
        [z_1, a_1], _ = find_optimal_angle(first_iter_z, first_iter_a, i)
        first_iter_z = z_1
        first_iter_a = a_1  #update
        x, y = solve_with_euler(first_iter_z, first_iter_a)
        plt.plot(x,y, label=f"Intermediate trajectory by Newton method iteration number: {i}")

    plt.scatter(xB,yB, s= 40, label="Final position") # point where the hoop is
    plt.scatter(x0, y0, s =40, color="Black", label="Starting position")# ball starting position
    plt.xlabel("The horizontal distance in meters between the ball player and the hoop")
    plt.ylabel("The verticle distance in meters between the ball and the hoop")
    plt.title("Final and intermediate trajectory of the basketball")
    plt.legend(loc='lower right', fontsize=6)
    plt.xlim([x0-0.2, xB+0.5])
    plt.ylim([y0-0.5, yB+0.5])
    return plt.show()  #tolerance can be adjusted for fewer graphs of good enough approximation.

# ---------------------------------------------------------- #
#                           Tests                            #
# ---------------------------------------------------------- #

if __name__ == "__main__":
    # plot_ball_trajectory()
    #[z, a], count = find_optimal_angle()
    #print(f'time: {1/z}, angle: {a}, iterations: {count}')
    plot_intermediate_trajectory(0.4, 1.05)
