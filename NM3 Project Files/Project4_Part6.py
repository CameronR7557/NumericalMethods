# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:59:54 2024

@author: robin
"""

from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from matplotlib import animation
from IPython import display

def f(x, y):
    return np.select([y <= x], [100], 0)

def p(x, t):
    return 0

def r(x, t):
    return 0

def g(y, t):
    return 0

def h(y, t):
    return 0


def FTCS_2DParabolicPDE(x, y, t, f, p, r, g, h, a, steps):
    delX = (x[1] - x[0])/ steps
    delY = (y[1] - y[0])/ steps
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(2*a*(t[1] - t[0])*(1/(delX**2) + 1/(delY**2))))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((steps + 1, steps + 1, stepsT + 1))
    x_points = np.linspace(x[0], x[-1], steps + 1)
    y_points = np.linspace(y[0], y[-1], steps + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    
    X,Y = np.meshgrid(x_points, y_points)
    u[:,:,0] = f(X, Y)
    #Fill in boundary values
    u[0, :, :] = g(y_points, t_points)
    u[-1, :, :] = h(y_points, t_points)
    u[:, 0, :] = p(x_points, t_points)
    u[:, -1, :] = r(x_points, t_points)
    
    for j in range(0, stepsT-1):
        for k in range(1, steps):
            for i in range(1, steps):
                u[i, k, j+1] = u[i,k,j] + (delT)*((u[i+1,k,j] - 2*u[i,k,j] + u[i-1,k,j])/(delX**2) + (u[i,k+1,j] - 2*u[i,k,j] + u[i,k-1,j])/(delY**2))
    return x_points, y_points, t_points, u

L = [0, 1]
W = [0, 1]
time = [0, 0.1]
a = 1

x, y, t, u = FTCS_2DParabolicPDE(L, W, time, f, p, r, g, h, a, 40)
X, Y= np.meshgrid(x, y)

#Animation
fig1 = plt.figure(1, figsize=(12,14))
ax = plt.axes(projection = '3d')
line = [ax.plot_surface(X, Y, u[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_xlim3d([L[0],L[-1]])
ax.set_ylim3d([W[0],W[-1]])
ax.set_zlim3d([0,105])
ax.set_title('u_t = u_xx + u_yy')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
def draw_wave(num, u, line):
    line[0].remove()
    line[0] = ax.plot_surface(X, Y, u[:,:,num], cmap='gist_heat', alpha=0.8)

# anim = animation.FuncAnimation(fig1, draw_wave, len(u[0,0,:]), fargs=(u, line))    
# plt.show()

nn=10
stepsX = 40
stepsY = 40
delX = (L[1] - L[0])/stepsX
delY = (W[1] - W[0])/stepsY
stepsT = int(np.ceil(2*a*(time[1] - time[0])*(1/(delX**2) + 1/(delY**2))))
X = np.linspace(L[0],L[1],stepsX + 1)
Y = np.linspace(W[0],W[1],stepsY + 1)
T = np.linspace(time[0],time[1],stepsT + 1)

x,y,t = np.meshgrid(X,Y,T)

ue=np.zeros((stepsY + 1, stepsX + 1, stepsT + 1))
ve=np.zeros((stepsY + 1, stepsX + 1, stepsT + 1))

for n in range (1,nn):
    for m in range(1,nn):
        if(n != m):
            C = 4*(((-1)**n)*(((-1)**m) - 1)*n**2 + (((-1)**n) - 1)*m**2)/(n*m*(n**2 - m**2)*(np.pi)**2)
        else:
            C = (2*(((-1)**n) - 1)**2)/((n**2)*(np.pi)**2)
        ve = C*np.sin(n*np.pi*x)*np.sin(m*np.pi*y)*e**(-((np.pi)**2)*(n**2 + m**2)*t) + ve
ue = 100 * ve
X1, Y1 = np.meshgrid(X,Y)
fig2 = plt.figure(2, figsize=(12,14))
ax = plt.axes(projection = '3d')
line2 = [ax.plot_surface(X1, Y1, ue[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_xlim3d([L[0],L[-1]])
ax.set_ylim3d([W[0],W[-1]])
ax.set_zlim3d([0,105])
ax.set_title('u_t = u_xx + u_yy')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

def draw_wave2(num, ue, line):
    line2[0].remove()
    line2[0] = ax.plot_surface(X1, Y1, ue[:,:,num], cmap='gist_heat', alpha=0.8)

anim = animation.FuncAnimation(fig2, draw_wave2, len(ue[0,0,:]), fargs=(ue, line2))    
plt.show()

#Error
#Error is initially very large, but it seems that the IC of the exact is not the same.
# err = u - ue
# fig3 = plt.figure(3, figsize=(12,14))
# ax = plt.axes(projection = '3d')
# line3 = [ax.plot_surface(X1, Y1, err[:,:,0], color='0.75', rstride=1, cstride=1)]
# ax.set_xlim3d([L[0],L[-1]])
# ax.set_ylim3d([W[0],W[-1]])
# ax.set_zlim3d([0,4])
# ax.set_title('Error')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('u')

# def draw_wave3(num, err, line3):
#     line3[0].remove()
#     line3[0] = ax.plot_surface(X1, Y1, err[:,:,num], cmap='gist_heat', alpha=0.8)

# anim = animation.FuncAnimation(fig3, draw_wave3, len(err[0,0,:]), fargs=(err, line3))    
# plt.show()