# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:14:15 2024

@author: robin
"""

from math import e
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython import display

def init(line):
    line[0].set_data([], [])
    line[1].set_data([], [])
    return line

def fe(x, t, c):
    return np.select([(t >= 4/c)*(t < 12/c)], [-0.5*(f((-4-x)-c*(4/c - t)) + f((4-x)+c*(4/c - t)))], 0.5*(f(x+c*t) + f(x-c*t)))

def pw(x):
    return -x**2 + 1

def f(x):
    return np.piecewise(x, [x <= 1.0, x >= -1.0, x > 1.0, x < -1.0], [pw, pw, 0, 0])

def F(x):
    return 0

def p(t):
    return 0

def r(t):
    return 0


def FiniteDiffPDE_Wave(x, t, f, F, p, r, c, stepsX):
    delX = (x[1] - x[0])/ stepsX
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(c*(t[1] - t[0])/(delX)))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((stepsT + 1, stepsX + 1))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    #Fill in boudary values
    u[:,0] = p(t_points)
    u[:,-1] = r(t_points)
    #Fill initial values for first two time steps using initial position and velocity
    u[0,:] = f(x_points)
    u[1,:] = u[0,:] + F(x_points)*delT

    for i in range(1, stepsT):#Already know values at time 0
        for j in range(1, stepsX):#Already know values at x = 0 and x = L 
            u[i + 1, j] = 2*u[i, j] - u[i - 1, j]+ (((c**2)*(delT**2))/(delX**2)) * (u[i, j + 1] - 2*u[i, j] + u[i, j - 1])
    
    return x_points, t_points, u

L = [-4,4]
time = [0, 80]
c = 2**(1/2)
fig1 = plt.figure(1, figsize=(12,14))
ax = plt.axes(projection = '3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 50)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Wave Eqn Solution x-steps = 50')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x,t)')
plt.show()

fig2 = plt.figure(2, figsize=(12,14))
ax = plt.axes(projection = '3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 100)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Wave Eqn Solution x-steps = 100')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x,t)')
plt.show()

#Animation
fig3 = plt.figure(3)
ax = plt.axes(xlim=(-4,4),ylim=(-1.1,1.1))
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('U_tt = (c^2)U_xx')

line = [[],[]]
line[0], = ax.plot([], [], linewidth=2.0,color='red', label='Numerical Solution')
line[1], = ax.plot([], [], linewidth=2.0,color='black', label='Initial Condition')

def draw_wave(T):
    line[0].set_data(x,u[T,:])
    #line[1].set_data(x,u[0,:])

anim = animation.FuncAnimation(fig3, draw_wave, frames=len(u[:,0]))    

#Exact
fig4 = plt.figure(6, figsize=(12,14))
ax = plt.axes(projection = '3d')
x = np.linspace(L[0], L[1], 100)
t = np.linspace(time[0], time[1], 100)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, fe(X,T,c), cmap='gist_heat', alpha=0.8)
ax.set_title('u(x,t) = (1/2)(f(x + ct) + f(x-ct))')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
plt.show()

#Error
#time = [0, 2]
fig5 = plt.figure(7, figsize=(12,14))
ax = fig5.add_subplot(1, 3, 1, projection='3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 50)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u - fe(X,T,c), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x-steps = 50')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u_est(x,t) - u_exact(x,t)')
ax = fig5.add_subplot(1, 3, 2, projection='3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 100)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u - fe(X,T,c), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x-steps = 100')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u_est(x,t) - u_exact(x,t)')