# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:32:07 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
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

def rho(x):
    return 1
def k(x):
    return 4
def speed(x):
    return np.sqrt(rho(x)*k(x))

def Wave1D_FiniteVolumes(x, t, k, rho, c, ip, iv, stepsX, c_max):
    delX = (x[1] - x[0])/ stepsX
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(c_max*(t[1] - t[0])/(delX)))
    delT = (t[1] - t[0])/ stepsT
    
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((stepsT + 1, stepsX + 3, 2))#+2 in x for ghost cells
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    xm_points = np.linspace(x[0] - delX/2, x[-1] + delX/2, stepsX + 2)#Midpoints
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    
    #ICs
    u[0, 1:-1, 0] = ip(x_points)
    u[0, 1:-1, 1] = iv(x_points)
    
    for n in range(len(t_points)-1):
        #Set Ghost Cells
        u[n, 0, 0] = u[n, 1, 0] 
        u[n, -1, 0] = u[n, -2, 0]
        u[n, 0, 1] = u[n, 1, 1]
        u[n, -1, 1] = u[n, -2, 1]
        for i in range(len(x_points)): 
            pd = u[n, i+1, 0] - u[n, i, 0]
            vd = u[n, i+1, 1] - u[n, i, 1] 
            #May be wrong, but does not matter for this problem since they are constants
            ci = c(xm_points[i])
            zi = rho(xm_points[i]) * ci
            ci1 = c(xm_points[i+1])
            zi1 = rho(xm_points[i+1]) * ci1
            #print(str(ci) + ", " + str(ci1) + ", " + str(zi) + ", " + str(zi1))
            a = (-pd + vd*zi1)/(zi + zi1)
            #My issue was taking a and B from the same point, when B should have been from i-1/2, not i+1/2 like a
            B = ((u[n, i, 0] - u[n, i-1, 0]) + (u[n, i, 1] - u[n, i-1, 1])*zi)/(zi + zi1)
            u[n+1, i, 0] = u[n, i, 0] - (delT/delX)*(ci*B*zi1 + ci*a*zi)
            u[n+1, i, 1] = u[n, i, 1] - (delT/delX)*(ci*B - ci*a)
    return x_points, t_points, u[:,1:-1,:]

steps = 100
L = [0, 20]
time = [0, 6]
c_max = 2

x, t, u = Wave1D_FiniteVolumes(L, time, k, rho, speed, initial_p, initial_v, steps, c_max)

#Animation
fig3 = plt.figure(3)
ax = plt.axes(xlim=(L[0],L[1]),ylim=(-2.1,2.1))
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_title('U_tt = (c^2)U_xx')

line = [[],[]]
line[0], = ax.plot([], [], linewidth=2.0,color='red', label='Numerical Solution')
line[1], = ax.plot([], [], linewidth=2.0,color='black', label='Initial Condition')

def draw_wave(T):
    line[0].set_data(x,u[T, :, 0])
    line[1].set_data(x,u[T,:,1])

anim = animation.FuncAnimation(fig3, draw_wave, frames=len(u[:,0,0]))    

