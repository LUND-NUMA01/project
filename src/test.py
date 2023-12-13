import numpy as np

from algorithms import newton

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
