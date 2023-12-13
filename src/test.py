import numpy as np
import matplotlib.pyplot as plt

# from algorithms import explicit_euler

# # some facts:
# d = 0.24 # m
# m = 0.6 # kg
# cw = 0.45
# p = 1.23 # kg/m^3
# xB = 2 # m
# yB = 3.05 # m
# g = 9.81 # m/s^2

# # intial conditions:
# x0 = 0  # m
# y0 = 1.75 # m
# s0 = 9  # m/s

# # constant combining all other constants for
# # the air resistance and acceleration
# C = -0.5*p*cw*np.pi/4*d**2/m

# def solve_with_euler(z0, a0, s0=9, x0=0, y0=1.75):
#     T = 1
#     N = 32

#     # compute initial x,y velocities
#     vx0 = s0 * np.cos(a0)
#     vy0 = s0 * np.sin(a0)
    
#     # create array for the initial values
#     initial_values = np.array([x0, y0, vx0, vy0])
    
#     def func(_, u):
#         # u[0] = x-position
#         # u[1] = y-position
#         # u[2] = x-velocity
#         # u[3] = y-velocity
#         s = np.sqrt(u[2]**2 + u[3]**2)
#         ax = C * s * u[2] * z0
#         ay = C * s * u[3] * z0**2 - g
#         vx = u[2] * z0
#         vy = u[3] * z0
#         return np.array([vx, vy, ax, ay])

#     _, u = explicit_euler(func, T, N, initial_values)

#     # extract x and y values from u
#     x = np.zeros(N)
#     y = np.zeros(N)

#     for i in range(N):
#         x[i] = u[i][0]
#         y[i] = u[i][1]

#     # return x and y values
#     return x, y

# ---------------------------------------------------------- #
#                           Task 3                           #
# ---------------------------------------------------------- #

# z0 = 0.5
# a0 = 0.9
# x, y = solve_with_euler(z0, a0)
# # plt.xlim([0, 2.5])
# plt.scatter(x, y)
# plt.show()

def newton(function, jacobian, x, iter=64, tolerance=1e-4):
    x = np.array(x, dtype=float)
    for i in range(iter):
        J = function(*x) # calculate jacobian J = df(X)/dY(X) 
        Y = jacobian(*x) # calculate function Y = f(X)
        dX = np.linalg.solve(J, Y) # solve for increment from JdX = Y 
        x -= dX # step X by dX 
        if np.linalg.norm(dX) < tolerance: # break if converged
            print('converged.')
            break
    return x

# def newton(F, J, x, eps=1e-12):
#     """
#     Solve nonlinear system F=0 by Newton's method.
#     J is the Jacobian of F. Both F and J must be functions of x.
#     At input, x holds the start value. The iteration continues
#     until ||F|| < eps.
#     """
#     iterations = 1000
#     for i in range(iterations):
#         F_value = F(x)
#         F_norm = np.linalg.norm(F_value, ord=2)  # l2 norm of vector
#         if abs(F_norm) < eps:
#             break
#         x += np.linalg.solve(J(x), -F_value)

#     return x

def newton(system_function, jacobian_function, initial_guess, tolerance=1e-20, max_iterations=1e3):
    current_guess = np.array(initial_guess, dtype=float)

    for _ in range(int(max_iterations)):
        # Calculate the Jacobian matrix anf function value
        J = jacobian_function(*current_guess)
        F = system_function(*current_guess)

        # Calculate the update using the inverse of the Jacobian matrix
        update = np.linalg.solve(J, -F)

        # Update the guess
        current_guess += update

        # Check for convergence
        if np.linalg.norm(update) < tolerance:
            break

    return current_guess

def function(x, y, z):
    return np.array([x+y+z-3, (x**2)+(y**2)+(z**2)-5, (np.exp(x))+(x*y)-(x*z)-1])

def jacobian(x, y, z):
    return np.array([[1, 1, 1], [2*x, 2*y, 2*z], [np.exp(x), x, -x]])

print(newton(function, jacobian, [1, 2, 3]))

def function(x, y):
    return np.array([x+2*y-2,x**2+4*y**2-4])

def jacobian(x, y):
    return np.array([[1,2],[2*x,8*y]])

print(newton(function, jacobian, [1, 2]))

from scipy.optimize import fsolve
def function(xy):
    x, y = xy
    return np.array([x+2*y-2,x**2+4*y**2-4])

def jacobian(xy):
    x, y = xy
    return np.array([[1,2],[2*x,8*y]])

x0 = [1, 2]
sol = fsolve(function, x0, fprime=jacobian, full_output=1)
print('solution exercice fsolve:', sol)
