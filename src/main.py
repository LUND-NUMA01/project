import numpy as np
import matplotlib.pyplot as plt

from algorithms import explicit_euler, nonlinear_newton

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

z = 0.5

def func_velocity(t, u):
    s = np.sqrt(u[0]**2 + u[1]**2)
    ax = C * s * u[0] * z
    ay = C * s * u[1] * z**2 - g
    return np.array([ax,ay])

def solve_with_euler(z0, a0, s0=9, x0=0, y0=1.75):
    T = 1
    N = 128

    # compute initial x,y velocities
    vx0 = s0 * np.cos(a0)
    vy0 = s0 * np.sin(a0)
    
    # create arrays for the initial values
    initial_velocity = np.array([vx0, vy0])
    initial_positions = np.array([x0, y0])

    # compute all velocities
    (vt, vu) = explicit_euler(func_velocity, T, N, initial_velocity)

    # do some sketchy stuff
    h = T / N
    values = np.empty(N + 1, dtype=object)
    values[0] = initial_positions
    # compute positions using the velocities
    for i in range(1, N + 1):
        values[i] = (values[i-1] + vu[i-1]*h*z)
    
    # extract x and y values

    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        x[i] = values[i][0]
        y[i] = values[i][1]
    
    # return x and y values
    return x, y

# ---------------------------------------------------------- #
#                           Task 3                           #
# ---------------------------------------------------------- #

def plot_ball_trajectory():
    z0 = 1
    a0 = 0.8
    x, y = solve_with_euler(z0, a0)
    plt.xlim([0, 2.5])
    plt.scatter(x, y)
    plt.show()

# ---------------------------------------------------------- #
#                           Task 5                           #
# ---------------------------------------------------------- #

def find_optimal_angle():
    f = () - xB, yB # function
    j = ()  # jacobian / derivative
    # initial guesses
    a0 = 1.4
    z0 = 0.8

    # a, z = nonlinear_newtons_method(f, j, a0, z0)
    return

# ---------------------------------------------------------- #
#                           Tests                            #
# ---------------------------------------------------------- #

if __name__ == "__main__":
    plot_ball_trajectory()
