import numpy as np
import matplotlib.pyplot as plt

from algorithms import explicit_euler, newton, numerical_jacobian

# ------------------------------------------------------- #

# constants:
d = 0.24  # diameter of the ball [m]
m = 0.6   # mass of the ball [kg]
cw = 0.45 # air resistance coefficient [ ]
p = 1.23  # air density [kg/m^3]
g = 9.81  # gravitation constant [m/s^2]

# positions:
x0 = 0    # initial x position [m]
y0 = 1.75 # initial y position [m]
xB = 2    # final x position [m]
yB = 3.05 # final y position [m]

# ------------------------------------------------------- #

def approx_positions(z0, a0, s0=9, x0=0, y0=1.75):
    T = 1
    N = 32

    z_t = 1 / z0

    # compute initial x,y velocities
    vx0 = s0 * np.cos(a0)
    vy0 = s0 * np.sin(a0)
    
    # create array for the initial values
    initial_values = np.array([x0, y0, vx0, vy0])
    
    # constant combining all other constants for
    # the air resistance and acceleration to
    # reduce computations
    C = -0.5*p*cw*np.pi/4*d**2/m

    # define inline function which has access 
    # to z0. An external function would not be
    # able to access the z0 value because the
    # function has to take two arguments (t and u)
    # to work with the explicit euler method
    def func(_, u):
        """
        The derivative of the function
        
        Parameters:
        - _: the t value is ignored
        - u[0]: x-position
        - u[1]: y-position
        - u[2]: x-velocity
        - u[3]: y-velocity

        Returns:
        - the derivates of the positions (velocities) and 
          the velocities (acceleration) as an numpy array
        """
        s = np.sqrt(u[2]**2 + u[3]**2)
        ax = C * s * u[2] * z_t**2
        ay = C * s * u[3] * z_t**2 - g
        vx = u[2] * z_t
        vy = u[3] * z_t
        return np.array([vx, vy, ax, ay])

    # approximate the positions and velocities
    # u is an array of length N with arrays of
    # length 4 as elements (positions + velocities)
    _, u = explicit_euler(func, T, N, initial_values)

    # extract x and y values from u
    x = np.zeros(N)
    y = np.zeros(N)

    for i in range(N):
        x[i] = u[i][0]
        y[i] = u[i][1]

    return x, y

# ------------------------------------------------------- #

def function(p, q):
    x, y = approx_positions(p, q)
    return np.array([x[-1] - xB, y[-1] - yB])

def approx_optimal_angle(z0, a0, max_iter=1e3):
    """
    This function approximates the optimal angle for the
    basket ball throw, so it hits the basket

    Parameters:
    - z0: initial guess for z
    - a0: initial guess for the angle
    - max_iter: maximum number of iterations for
                the approximation (default = 1000)

    Returns:
    - The optimal z
    - The optimal angle a
    """
    jacobian = lambda x, y: numerical_jacobian(function, [x, y])
    return newton(function, jacobian, [z0, a0], max_iter=max_iter)

# ------------------------------------------------------- #

# this function just plots the approximated
# positions for the given z and a values
def plot_ball_trajectory(z, a):
    x, y = approx_positions(z, a)
    plt.xlim([0, 2.5])
    plt.scatter(x, y)
    plt.show()

# ------------------------------------------------------- #

def plot_intermediate_trajectory(z_0, a_0, x0, y0, xB, yB):
    """
    initial guess, an idea for the reader
        z0 = 1
        a0 = 1.5

    tolerance can be adjusted for fewer graphs of good enough approximation.
    """
    # determine how many intermediate trajectories we should plot
    _, set_interations = approx_optimal_angle(z_0, a_0, max_iter=1000)

    iter_z = z_0
    iter_a = a_0
    x, y = approx_positions(iter_z, iter_a)
    plt.figure(figsize=(6,6))
    plt.plot(x, y, color="black", label=f"The initial guess trajectory (z={z_0}, a={a_0})" )

    for i in range(set_interations):
        [iter_z, iter_a], _ = approx_optimal_angle(iter_z, iter_a, 1)
        x, y = approx_positions(iter_z, iter_a)
        plt.plot(x,y, label=f"Intermediate trajectory by Newton method iteration number: {i+1}")

    plt.scatter(xB, yB, s=40, label="Final position") # point where the hoop is
    plt.scatter(x0, y0, s=40, color="Black", label="Starting position")# ball starting position
    plt.xlabel("The horizontal distance in meters between the ball player and the hoop")
    plt.ylabel("The verticle distance in meters between the ball and the hoop")
    plt.title("Final and intermediate trajectory of the basketball")
    plt.legend(loc='lower right', fontsize=6)
    plt.xlim([x0-0.2, xB+0.5])
    plt.ylim([y0-0.5, yB+0.5])
    plt.show()

# ------------------------------------------------------- #

if __name__ == "__main__":
    # plot_ball_trajectory()
    # [z, a], count = approx_optimal_angle(2, 1.4)
    # print(f'time: {z}, angle: {a}, iterations: {count}')
    plot_intermediate_trajectory(2, 1.4, x0, y0, xB, yB)
    # _, set_interations = approx_optimal_angle(1, 1.4, max_iter=2)
