import numpy as np
import unittest

from algorithms import newton

class TestNewtonMethod(unittest.TestCase):
    def test_linear_system(self):
        # Linear system: f1(x, y) = x + y - 3, f2(x, y) = 2x - y + 1

        def linear_system(x, y):
            return np.array([x + y - 3, 2 * x - y + 1])

        # Jacobian matrix of the linear system
        def linear_jacobian(x, y):
            return np.array([[1, 1], [2, -1]])

        # Initial guess
        initial_guess = np.array([1.0, 2.0])

        # Use Newton method
        result, _ = newton(linear_system, linear_jacobian, initial_guess)

        # Assert the result is close to the actual solution (2/3, 7/3)
        expected_solution = np.array([2/3, 7/3])
        self.assertTrue(np.allclose(result, expected_solution))

    def test_nonlinear_system(self):
        # Nonlinear system: f1(x, y) = x^2 + y^2 - 5, f2(x, y) = xy - 1
        def nonlinear_system(x, y):
            return np.array([x**2 + y**2 - 5, x * y - 1])

        # Jacobian matrix of the nonlinear system
        def nonlinear_jacobian(x, y):
            return np.array([[2 * x, 2 * y], [y, x]])

        initial_guess = np.array([2.0, 1.0])
        result, _ = newton(nonlinear_system, nonlinear_jacobian, initial_guess)

        expected_solution = np.array([2.189, 0.457])
        self.assertTrue(np.allclose(result, expected_solution, atol=0.01))

    def test_system_of_equations(self):
        # Define a system of equations:
        # f1(x, y) = x + y - 3
        # f2(x, y) = x^2 + y^2 - 5

        def system_of_equations(x, y):
            return np.array([x + y - 3, x**2 + y**2 - 5])

        # Define the Jacobian matrix of the system
        def jacobian_matrix(x, y):
            # return np.shape((2, 2), 1, 1, 2*x, 2*y)
            return np.array([[1, 1], [2 * x, 2 * y]])

        initial_guess = [3.0, 1.0]
        result, _ = newton(system_of_equations, jacobian_matrix, initial_guess)

        # Assert the result is close to the actual solution (2, 1)
        expected_solution = np.array([2, 1])
        self.assertTrue(np.allclose(result, expected_solution))

if __name__ == '__main__':
    unittest.main()
