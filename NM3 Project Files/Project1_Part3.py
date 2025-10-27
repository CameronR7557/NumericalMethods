# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:12:16 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def f(x):
    return np.sin(np.pi*x)*(1 + 2*np.cos(np.pi*x))

def g(t):
    return 0

def h(t):
    return 0


def FiniteDiffPDE(x, t, f, g, h, k, stepsX):
    delX = (x[1] - x[0])/ stepsX
    # (tn - t0)/stepsT <= h^2 / (2*alpha) --> (tn - t0)*((2.1*alpha)/h^2) = stepsT
    stepsT = int(np.ceil((t[1] - t[0]) / ((delX**2)/(2.1*k))))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((stepsT + 1, stepsX + 1))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    #Fill in boudary values
    for i in range(stepsT+1):
        u[i, 0] = g(delT*i)
        u[i, -1] = h(delT*i)
    #Fill initial value (t = 0 for all x)
    for j in range(stepsX+1):
        u[0, j] = f(j*delX)
        
    for i in range(1, stepsT+1):#Already know values at time 0 (f(x))
        for j in range(1, stepsX-1):#Already know values at x = 0 and x = L (g and h)
            u[i, j] = u[i - 1, j] + ((k*delT)/(delX**2)) * (u[i - 1, j + 1] - 2*u[i - 1, j] + u[i - 1, j - 1])
    
    return x_points, t_points, u

L = [0,1]
time = [0, 0.25]

fig1 = plt.figure(1, figsize=(12,14))
ax = plt.axes(projection = '3d')
x, t, u = FiniteDiffPDE(L, time, f, g, h, 1, 5)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Heat Eqn Solution x-steps = 5')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x,t)')
plt.show()

fig2 = plt.figure(2, figsize=(12,14))
ax = plt.axes(projection = '3d')
x, t, u = FiniteDiffPDE(L, time, f, g, h, 1, 10)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Heat Eqn Solution x-steps = 10')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x,t)')
plt.show()

fig3 = plt.figure(3, figsize=(12,14))
ax = plt.axes(projection = '3d')
x, t, u = FiniteDiffPDE(L, time, f, g, h, 1, 20)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Heat Eqn Solution x-steps = 20')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U(x,t)')
plt.show()

#Exact
fig4 = plt.figure(4, figsize=(12,14))
ax = plt.axes(projection = '3d')
x = np.linspace(0, 1, 100)
t = np.linspace(0, 0.25, 100)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, np.sin(np.pi * X)*np.e**(-((np.pi)**2)*T) + np.sin(2 * np.pi * X)*np.e**(-4*((np.pi)**2)*T), cmap='gist_heat', alpha=0.8)
ax.set_title('u(x,t) = sin(pi*x)*e^((-pi^2)*t) + np.sin(2*pi*x)*e^(-4*(pi^2)*t)')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
plt.show()

#Error
fig5 = plt.figure(5, figsize=(12,14))
ax = fig5.add_subplot(1, 3, 1, projection='3d')
x, t, u = FiniteDiffPDE(L, time, f, g, h, 1, 5)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u - (np.sin(np.pi * X)*np.e**(-((np.pi)**2)*T) + np.sin(2 * np.pi * X)*np.e**(-4*((np.pi)**2)*T)), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x-steps = 5')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u_est(x,t) - u_exact(x,t)')
ax = fig5.add_subplot(1, 3, 2, projection='3d')
x, t, u = FiniteDiffPDE(L, time, f, g, h, 1, 10)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u - (np.sin(np.pi * X)*np.e**(-((np.pi)**2)*T) + np.sin(2 * np.pi * X)*np.e**(-4*((np.pi)**2)*T)), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x-steps = 10')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u_est(x,t) - u_exact(x,t)')
ax = fig5.add_subplot(1, 3, 3, projection='3d')
x, t, u = FiniteDiffPDE(L, time, f, g, h, 1, 20)
X, T = np.meshgrid(x, t)
ax.plot_surface(X, T, u - (np.sin(np.pi * X)*np.e**(-((np.pi)**2)*T) + np.sin(2 * np.pi * X)*np.e**(-4*((np.pi)**2)*T)), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x-steps = 20')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u_est(x,t) - u_exact(x,t)')