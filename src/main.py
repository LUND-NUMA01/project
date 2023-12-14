import numpy as np
import matplotlib.pyplot as plt

from typing import Literal
from algorithms import explicit_euler, newton, numerical_jacobian, adams_bashforth

class Basketball:
    def __init__(self, x0, y0, xB, yB, s0, d=0.24, m=0.6, cw=0.45, p=1.23, g=9.81):
        self.x0 = x0
        self.y0 = y0
        self.xB = xB
        self.yB = yB
        self.s0 = s0
        self.d = d
        self.m = m
        self.cw = cw
        self.p = p
        self.g = g

    def approx_postion(self, z, a, algorithm: Literal['euler','adams']='euler'):
        T = 1
        N = 64

        match algorithm:
            case 'euler': fn_algorithm = explicit_euler
            case 'adams': fn_algorithm = adams_bashforth
            case _: raise TypeError(f"unknown algorithm: {algorithm}")

        # compute initial x,y velocities
        vx0 = self.s0 * np.cos(a)
        vy0 = self.s0 * np.sin(a)

        # constant function z(tau)
        z_t = 1/z

        # create array for the initial values
        initial_values = np.array([self.x0, self.y0, vx0, vy0])

        # constant combining all other constants for the air
        # resistance and acceleration to reduce computations
        C = -0.5*self.p*self.cw*np.pi/4*self.d**2/self.m

        def derivative(_, u):
            """
            The derivative of the function

            Parameters:
            - _: the t value is ignored
            - u[0]: x-position
            - u[1]: y-position
            - u[2]: x-velocity
            - u[3]: y-velocity

            Returns:
            - the derivates of the positions (velocities) and
              the velocities (acceleration) as a numpy array
            """
            s = np.sqrt(u[2]**2 + u[3]**2)
            ax = C * s * u[2] * z_t**2
            ay = C * s * u[3] * z_t**2 - self.g
            vx = u[2] * z_t
            vy = u[3] * z_t
            return np.array([vx, vy, ax, ay])

        # approximate the positions and velocities
        # u is an array of length N with arrays of
        # length 4 as elements (positions + velocities)
        _, u = fn_algorithm(derivative, T, N, initial_values)

        # extract x and y values from u
        x = np.zeros(N)
        y = np.zeros(N)

        for i in range(N):
            x[i] = u[i][0]
            y[i] = u[i][1]

        # return x and y values
        return x, y

    def approx_optimal_angle(self, z0, a0, algorithm='euler', max_iter=1.e3):
        # This is the function G(z0, a0)
        def function(p, q):
            x, y = self.approx_postion(p, q, algorithm=algorithm)
            return np.array([x[-1] - self.xB, y[-1] - self.yB])
        jacobian = lambda x, y: numerical_jacobian(function, [x, y])
        return newton(function, jacobian, [z0, a0], max_iter=max_iter)

    def plot_trajectory(self, z, a, algorithm='euler'):
        """
        this function just plots the approximated
        positions for the given z and a values
        """
        x, y = self.approx_postion(z, a, algorithm=algorithm)
        plt.scatter(self.xB, self.yB, s=40, label="Final position") # point where the hoop is
        plt.scatter(self.x0, self.y0, s=40, color="Black", label="Starting position") # ball starting position
        plt.xlim([0, 2.5])
        plt.plot(x, y)
        return plt.show()

    def plot_intermediate_trajectories(self, z, a, algorithm='euler'):
        # determines how many intermediate trajectories we should plot
        _, set_iterations = self.approx_optimal_angle(z, a, algorithm=algorithm)
        iter_z = z
        iter_a = a
        x, y = self.approx_postion(iter_z, iter_a, algorithm=algorithm)
        plt.figure(figsize=(6,6))
        plt.plot(x, y, color="black", label="The initial guess trajectory" )

        for i in range(set_iterations-1):
            [iter_z, iter_a], _ = self.approx_optimal_angle(iter_z, iter_a, algorithm=algorithm, max_iter=1)
            x, y = self.approx_postion(iter_z, iter_a, algorithm=algorithm)
            plt.plot(x,y, label=f"Intermediate trajectory by Newton method iteration number: {i+1}")

        [iter_z, iter_a], _ = self.approx_optimal_angle(iter_z, iter_a, algorithm=algorithm, max_iter=1)
        x, y = self.approx_postion(iter_z, iter_a, algorithm=algorithm)
        plt.plot(x,y, label=f"Final trajectory. Time taken: {iter_z:.2f}s, Angle: {iter_a/np.pi*180:.2f}˚, Final iteration number: {set_iterations}")

        plt.scatter(self.xB, self.yB, s=40, label="Final position") # point where the hoop is
        plt.scatter(self.x0, self.y0, s=40, color="Black", label="Starting position") # ball starting position
        plt.xlabel("The horizontal distance in meters between the ball player and the hoop")
        plt.ylabel("The verticle distance in meters between the ball and the hoop")
        plt.title("Final and intermediate trajectory of the basketball")
        plt.legend(loc='lower right', fontsize=6)
        plt.xlim([self.x0-0.5, self.xB+0.5])
        plt.ylim([self.y0-0.5, self.yB+0.5])
        return plt.show() # tolerance can be adjusted for fewer graphs of good enough approximation.

if __name__ == "__main__":
    ball = Basketball(0, 1.75, 2, 3.05, 9)
    ball.plot_trajectory(2, 1.2, algorithm='euler')
    ball.plot_trajectory(2, 1.2, algorithm='adams')
    ball.plot_intermediate_trajectories(2, 1.4, algorithm='euler')
