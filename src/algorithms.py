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

def newton(function, jacobian, initial_guess, tolerance=1e-5, max_iter=1e3):
    count = max_iter
    current_guess = np.array(initial_guess, dtype=float)

    for i in range(int(max_iter)):
        # Calculate the Jacobian matrix anf function value
        J = jacobian(*current_guess)
        F = function(*current_guess)

        # Calculate the update
        update = np.linalg.solve(J, F)

        # Update the guess
        current_guess -= update
        
        # Check for convergence
        if np.linalg.norm(update) < tolerance:
            count = i
            break

    return current_guess, count

# -----------------------------------------------------------

def numerical_jacobian(func, x, epsilon=1e-6):
    n = len(x)
    m = len(func(*x))
    j = np.zeros((m, n))

    for i in range(n):
        x_shifted = x.copy()
        x_shifted[i] += epsilon

        # assigning the entire column
        j[:, i] = (func(*x_shifted) - func(*x)) / epsilon

    return j
