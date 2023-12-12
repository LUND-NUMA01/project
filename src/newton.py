import numpy as np

def iter_newton(X,function,jacobian,imax = 1e6,tol = 1e-5):
    for i in range(int(imax)):
        J = jacobian(X) # calculate jacobian J = df(X)/dY(X) 
        Y = function(X) # calculate function Y = f(X)
        dX = np.linalg.solve(J,Y) # solve for increment from JdX = Y 
        X -= dX # step X by dX 
        if np.linalg.norm(dX)<tol: # break if converged
            print('converged.')
            break
    return X
