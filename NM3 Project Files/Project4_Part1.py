# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:59:43 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from matplotlib import animation
from IPython import display

def f(x, y):
    return np.select([(x**2 + y**2 < 3)], [e**(-4*(x**2 + y**2))], 0)

def F(x, y):
    return 0

def p(x, t):
    return 0

def r(x, t):
    return 0

def g(y, t):
    return 0

def h(y, t):
    return 0

def FiniteDiff2D_Wave(x, y, t, f, F, p, r, g, h, c, steps):
    delX = (x[1] - x[0])/ steps
    delY = (y[1] - y[0])/ steps
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(2*c*(t[1] - t[0])/((delX))))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((steps + 1, steps + 1, stepsT + 1))
    x_points = np.linspace(x[0], x[-1], steps + 1)
    y_points = np.linspace(y[0], y[-1], steps + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    
    X,Y = np.meshgrid(x_points, y_points)
    u[:,:,0] = f(X, Y)
    u[:,:,1] = u[:,:,0] + F(X, Y)*delT
    
    for j in range(1, stepsT):
        for k in range(1, steps):
            for i in range(1, steps):
                if((delX*i + x[0])**2 + (delY*k + y[0])**2 >= 9):
                    u[i,k,j] = 0
                u[i, k, j+1] = 2*u[i,k,j] - u[i,k,j-1] + ((c**2)*(delT**2))*((u[i+1,k,j] - 2*u[i,k,j] + u[i-1,k,j])/(delX**2) + (u[i,k+1,j] - 2*u[i,k,j] + u[i,k-1,j])/(delY**2))
                
    return x_points, y_points, t_points, u

L = [-3.2, 3.2]
W = [-3.2, 3.2]
time = [0, 6]
c = 2

x, y, t, u = FiniteDiff2D_Wave(L, W, time, f, F, p, r, g, h, c, 50)
X, Y= np.meshgrid(x, y)

#Animation
fig1 = plt.figure(1, figsize=(12,14))
ax = plt.axes(projection = '3d')
line = [ax.plot_surface(X, Y, u[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_xlim3d([L[0],L[-1]])
ax.set_ylim3d([W[0],W[-1]])
ax.set_zlim3d([-0.25, 0.25])
ax.set_title('u_tt = (c^2)(u_xx + u_yy)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

def draw_wave(num, u, line):
    line[0].remove()
    line[0] = ax.plot_surface(X, Y, u[:,:,num], cmap='gist_heat', alpha=0.8)

anim = animation.FuncAnimation(fig1, draw_wave, len(u[0,0,:]), fargs=(u, line))    
# afile = r"Proj3P5Numerical50.gif" 
# writergif = animation.PillowWriter(fps=4) 
plt.show()
