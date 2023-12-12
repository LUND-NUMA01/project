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

B = p*cw*np.pi**d**2/8

def force(s, p=1.23, cw=0.45, d=0.24):
    return B*s**2

# intial conditions:
x0 = 0  # m
y0 = 1.75 # m
s0 = 9  # m/s
a0 = 1.41

T = 1
N = 100
h = T / N

states = np.empty((N, 6))

states[0, 0] = x0
states[0, 1] = y0
states[0, 2] = s0 * np.cos(a0)
states[0, 3] = s0 * np.sin(a0)
states[0, 4] = a0
states[0, 5] = s0

z = 1

for i in range(1, N):
    old_x = states[i-1, 0]
    old_y = states[i-1, 1]
    old_vx = states[i-1, 2]
    old_vy = states[i-1, 3]
    old_a = states[i-1, 4]
    old_s = states[i-1, 5]

    x = old_x + old_vx * h
    y = old_y + old_vy * h

    fx = -z * force(old_s) * np.cos(old_a)
    fy = -z * (force(old_s) * np.sin(old_a) + m * g)

    vx = (old_vx + fx / m * h) * z
    vy = (old_vy + fy / m * h) * z

    s = np.sqrt(vx**2 + vy**2)
    a = np.arccos(vx / s)

    states[i, 0] = x
    states[i, 1] = y
    states[i, 2] = vx
    states[i, 3] = vy
    states[i, 4] = a
    states[i, 5] = s

# x = [states[i, 0] for i in range(N)]
# # x = np.linspace(0, 1, N)
# y = [states[i, 1] for i in range(N)]

x = states[:, 0]
y = states[:, 1]

print(y[-1])

plt.plot(x, y)
plt.show()
