# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:43:34 2024

@author: robin
"""

from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from matplotlib import animation
from IPython import display

def f(r):
    return np.select([abs(r) <= 3],[e**(-4*r**2)],0)

def F(x, y):
    return 0

def p(x, t):
    return 0

def q(x, t):
    return 0

def g(y, t):
    return 0

def h(y, t):
    return 0


def FiniteDiff2D_Wave_Polar(r, o, t, f, F, p, q, g, h, c, steps):
    delR = (r[1] - r[0])/ steps
    delO = (o[1] - o[0])/ steps
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(2*c*(t[1] - t[0])/((delR))))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((steps + 1, steps + 1, stepsT + 1))
    r_points = np.linspace(r[0], r[-1], steps + 1)
    o_points = np.linspace(o[0], o[-1], steps + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    
    R,O = np.meshgrid(r_points, o_points)
    u[:,:,0] = f(R)
    u[:,:,1] = u[:,:,0] + F(R, O)*delT
    #Fill in boundary values
    u[0, :, :] = g(o_points, t_points)
    u[-1, :, :] = h(o_points, t_points)
    u[:, 0, :] = p(r_points, t_points)
    u[:, -1, :] = q(r_points, t_points)
    
    for j in range(1, stepsT):
        for i in range(1, steps):
            for k in range(1, steps):
                rp = r_points[k]
                if(abs(rp) <=  0.0000000000001):#Do not calculate at r = 0
                    rp = 0.0000000000001
                #Ignores theta componenet since radially symmetric
                u[i, k, j+1] = 2*u[i,k,j] - u[i,k,j-1] + ((c**2)*(delT**2))*((u[i, k+1, j] -u[i, k-1, j])/(2*rp*delR) + (u[i,k+1,j] - 2*u[i,k,j] + u[i,k-1,j])/(delR**2))
    return r_points, o_points, t_points, u

radius = [-3, 3]
theta = [0, np.pi * 2]
time = [0, 6]
c = 2

r, o, t, u = FiniteDiff2D_Wave_Polar(radius, theta, time, f, F, p, q, g, h, c, 50)
R, O= np.meshgrid(r, o)
x = R*np.cos(O)
y = R*np.sin(O)

#Animation
fig1 = plt.figure(1, figsize=(12,14))
ax = plt.axes(projection = '3d')
line = [ax.plot_surface(x, y, u[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_xlim3d([-3,3])
ax.set_ylim3d([-3,3])
ax.set_zlim3d([-0.25,0.25])
ax.set_title('u_tt = (c^2)(u_xx + u_yy)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

def draw_wave(num, u, line):
    line[0].remove()
    line[0] = ax.plot_surface(x, y, u[:,:,num], cmap='gist_heat', alpha=0.8)

anim = animation.FuncAnimation(fig1, draw_wave, len(u[0,0,:]), fargs=(u, line))     
plt.show()
