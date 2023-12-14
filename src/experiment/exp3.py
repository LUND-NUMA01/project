import numpy as np
import matplotlib.pyplot as plt


d = 0.24 # m
m = 0.6 # kg
cw = 0.45 # air resistance coeff.
p = 1.23 # kg/m^3
xB = 2 # m
yB = 3.05 # m
g = 9.81 # m/s^2

def force(s, p=1.23, cw=0.45, d=0.24):
    return p*cw*np.pi*s**2*d**2/8

def explicit_euler(f, T, N, y0, z_0=1):
    #normalize the interval from 0-1
    interval = np.linspace(0, T, N)
    interval = interval/z_0
    # step size
    h = abs(interval[0] - interval[1])
    # pre-initialize zeroed arrays
    u = np.zeros(N)
    # initial value
    u[0] = y0
    # N iterations
    for i in range(1, N):
        u[i] = u[i-1] + h * f(interval[i-1], u[i-1])*z_0
    # return both the arrays interval and u_i
    return (interval, u)
