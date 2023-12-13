import numpy as np
import matplotlib.pyplot as plt

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

z = 1

def force(speed, p=1.23, cw=0.45, d=0.24):
    return 0.5 * p * cw * np.pi / 4 * d**2 * speed**2

def x_acceleration(angle, speed, m=0.6):
    return (-force(speed) * np.cos(angle)) / m

def y_acceleration(angle, speed, m=0.6, g=9.81):
    return (-force(speed) * np.sin(angle) - m * g) / m

a0 = np.pi / 4
s0 = 9
x0 = 0
y0 = 1.75
vx0 = s0 * np.cos(a0)
vy0 = s0 * np.sin(a0)

initial_state = [x0, y0, vx0, vy0, a0]

T = 1
N = 100

states = np.empty(N, dtype=object)
states[0] = initial_state

h = T / N

for i in range(1, N):
    #    x              + v * dx
    xi = states[i-1][0] + states[i-1][2] * h
    yi = states[i-1][1] + states[i-1][3] * h

    old_speed = np.sqrt(states[i-1][2]**2 + states[i-1][3]**2)
    old_angle = states[i-1][4]

    xacc = x_acceleration(old_angle, old_speed)
    yacc = y_acceleration(old_angle, old_speed)

    vxi = (states[i-1][2] + xacc * h) * z
    vyi = (states[i-1][3] + yacc * h) * z
    speed = np.sqrt(vxi**2 + vyi**2)
    ai = np.arccos(vxi / speed)

    states[i] = [xi, yi, vxi, vyi, ai]

# ------------------------------------------------------------

x = [i[0] for i in states]
y = [i[1] for i in states]

degree = 2
coefficients = np.polyfit(x, y, degree)

# Generate the fitted curve
x_fit = np.linspace(min(x), max(x), N)
y_fit = np.polyval(coefficients, x_fit)

diffy = [abs(y[i] - y_fit[i]) for i in range(N)]

plt.plot(x, y)
# plt.plot(x_fit, y_fit, label=f'Polynomial Fit (Degree {degree})', color='red')
# plt.plot(x, diffy)
plt.show()
