import numpy as np
import unittest

from algorithms import explicit_euler, newton, numerical_jacobian

class TestExplicitEulerMethod(unittest.TestCase):
    pass

class TestNewtonMethod(unittest.TestCase):
    pass

class TestJacobian(unittest.TestCase):
    pass

if __name__ == "__main__":
    unittest.main()

# from scipy.optimize import fsolve
# def function(xy):
#     x, y = xy
#     return np.array([x+2*y-2,x**2+4*y**2-4])

# def jacobian(xy):
#     x, y = xy
#     return np.array([[1,2],[2*x,8*y]])

# x0 = [1, 2]
# sol = fsolve(function, x0, fprime=jacobian, full_output=1)
# print('solution exercice fsolve:', sol)
