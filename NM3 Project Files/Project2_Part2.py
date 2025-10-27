# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:27:55 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def f(x):
    return np.sin(x)

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
    for i in range(stepsT+1):
        u[i, 0] = p(delT*i + t[0])
        u[i, -1] = r(delT*i + t[0])
    #Fill initial values for first two time steps using initial position and velocity
    for j in range(stepsX+1):
        u[0, j] = f(j*delX + x[0])
        u[1, j] = u[0,j] + F(j*delX + x[0])*delT #Next position = prevPos + velocity*delta-t
    
    for i in range(1, stepsT):#Already know values at time 0
        for j in range(1, stepsX):#Already know values at x = 0 and x = L 
            u[i + 1, j] = 2*u[i, j] - u[i - 1, j]+ (((c**2)*(delT**2))/(delX**2)) * (u[i, j + 1] - 2*u[i, j] + u[i, j - 1])
    
    return x_points, t_points, u

L = [0,np.pi]
time = [0, 12]
c = 2**(1/2)
fig1 = plt.figure(1, figsize=(12,14))
ax = plt.axes(projection = '3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 5)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Wave Eqn Solution x-steps = 5')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x,t)')
plt.show()

fig2 = plt.figure(2, figsize=(12,14))
ax = plt.axes(projection = '3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 10)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Wave Eqn Solution x-steps = 10')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x,t)')
plt.show()

fig3 = plt.figure(3, figsize=(12,14))
ax = plt.axes(projection = '3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 20)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Wave Eqn Solution x-steps = 20')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x,t)')
plt.show()

#Exact
fig4 = plt.figure(4, figsize=(12,14))
ax = plt.axes(projection = '3d')
x = np.linspace(L[0], L[1], 100)
t = np.linspace(time[0], time[1], 100)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, np.cos(c*T)*np.sin(X), cmap='gist_heat', alpha=0.8)
ax.set_title('u(x,t) = cos(sqrt(2)*t)*sin(x)')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
plt.show()

#Error
fig5 = plt.figure(5, figsize=(12,14))
ax = fig5.add_subplot(1, 3, 1, projection='3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 5)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u - (np.cos(c*T)*np.sin(X)), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x-steps = 5')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u_est(x,t) - u_exact(x,t)')
ax = fig5.add_subplot(1, 3, 2, projection='3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 10)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u - (np.cos(c*T)*np.sin(X)), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x-steps = 10')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u_est(x,t) - u_exact(x,t)')
ax = fig5.add_subplot(1, 3, 3, projection='3d')
x, t, u = FiniteDiffPDE_Wave(L, time, f, F, p, r, c, 20)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u - (np.cos(c*T)*np.sin(X)), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x-steps = 20')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u_est(x,t) - u_exact(x,t)')