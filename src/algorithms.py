import numpy as np

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
    # pre-initialize zeroed arrays
    # t = np.zeros(N, dtype=object)
    t = np.linspace(0, T, N)
    u = np.zeros(N, dtype=object)

    # initial values
    u[0] = y0

    # step size
    h = t[1] - t[0]

    # N iterations
    for i in range(1, N):
        u[i] = u[i-1] + h * f(t[i-1], u[i-1])
    
    # return both the arrays t_i and u_i
    return (t, u)

# -----------------------------------------------------------

def newton(function, jacobian, initial_guess, tolerance=1e-2000, max_iterations=1e3):
    """
    Apply the Newton-Raphson method to solve a system of nonlinear equations.

    Parameters:
    - function: Callable function representing the system of equations. It takes the current guess as arguments.
    - jacobian: Callable function representing the Jacobian matrix of the system. It also takes the current guess as arguments.
    - initial_guess: Initial guess for the solution (list).
    - tolerance: Convergence criterion, the method stops when the update is smaller than this value. Default is 1e-20.
    - max_iterations: Maximum number of iterations. Default is 1e3.

    Returns:
    - numpy.ndarray: The solution to the system of equations.
    
    Note:
    The functions 'function' and 'jacobian' should be defined to take the current guess as arguments and return NumPy arrays.
    """
    current_guess = np.array(initial_guess, dtype=float)

    for _ in range(int(max_iterations)):
        # Calculate the Jacobian matrix anf function value
        J = jacobian(*current_guess)
        F = function(*current_guess)

        # Calculate the update using the inverse of the Jacobian matrix
        update = np.linalg.solve(J, -F)

        # Update the guess
        current_guess += update

        # Check for convergence
        if np.linalg.norm(update) < tolerance:
            break

    return current_guess

# -----------------------------------------------------------

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
        x_shifted = x.copy()
        x_shifted[i] += epsilon

        # assigning the entire column
        J[:, i] = (func(*x_shifted) - func(*x)) / epsilon

    return J
