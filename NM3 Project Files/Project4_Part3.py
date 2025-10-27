# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:19:14 2024

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

def initial_v(x):
    return 0
def initial_p(x):
    return np.select([(x >= 8)*(x <= 12)], [2], 0)
def F(x, delX):
    dx = np.zeros((len(x)))
    dx[0] = ((-3/2)*x[0] + 2*x[1] - 0.5*x[2]) / delX
    for i in range(1, len(x)-1):
        dx[i] = (x[i+1] - x[i-1])/(2*delX)
    dx[-1] = ((3/2)*x[-1] - 2*x[-2] + 0.5*x[-3]) / delX
    return dx

def FiniteDiffPDE_Wave(x, t, ip, iv, rho, k, c, stepsX):
    delX = (x[1] - x[0])/ stepsX
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(1.01*c*(t[1] - t[0])/(delX)))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((stepsT + 1, stepsX + 3, 2))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    #Fill initial values for first two time steps using initial position and velocity
    u[0, 1:-1, 0] = ip(x_points)
    u[0, 1:-1, 1] = iv(x_points)
    u[1, 1:-1, 0] = u[0,1:-1,0] - k*F(u[0, 1:-1, 1], delX)*delT
    u[1, 1:-1, 1] = u[0,1:-1,1] - (1/rho)*F(u[0, 1:-1, 0], delX)*delT
    
    for i in range(1, len(u[:,0,0]) - 1):
        u[i+1,0,0] = u[i,1,0]
        u[i+1,-1,0] = u[i,-2,0]
        u[i+1,0,1] = u[i,1,1]
        u[i+1,-1,1] = u[i,-2,1]
        for j in range(1, len(u[0,:,0]) - 1):
            u[i + 1, j,0] = 2*u[i, j,0] - u[i - 1, j,0]+ (((c**2)*(delT**2))/(delX**2)) * (u[i, j + 1,0] - 2*u[i, j,0] + u[i, j - 1,0])
            u[i + 1, j,1] = 2*u[i, j,1] - u[i - 1, j,1]+ (((c**2)*(delT**2))/(delX**2)) * (u[i, j + 1,1] - 2*u[i, j,1] + u[i, j - 1,1])
    return x_points, t_points, u[:,1:-1,:]

L = [0,20]
time = [0, 6]
k = 4
rho = 1
c = (k/rho)**(1/2)
fig1 = plt.figure(1, figsize=(12,14))
ax = plt.axes(projection = '3d')
x, t, u = FiniteDiffPDE_Wave(L, time, initial_p, initial_v, rho, k, c, 100)
X, T = np.meshgrid(x, t)

#Animation
fig1 = plt.figure(1)
ax = plt.axes(xlim=(L[0],L[1]),ylim=(-2.1,2.1))
ax.set_xlabel('x')
ax.set_ylabel('P')
ax.set_title('P_tt = (c^2)P_xx')

line = [[],[]]
line[0], = ax.plot([], [], linewidth=2.0,color='red', label='Pressure')
line[1], = ax.plot([], [], linewidth=2.0,color='black', label='Velocity')

def draw_wave(T):
    line[0].set_data(x,u[T,:,0])
    line[1].set_data(x,u[T,:,1])

anim = animation.FuncAnimation(fig1, draw_wave, frames=len(u[:,0,0])) 