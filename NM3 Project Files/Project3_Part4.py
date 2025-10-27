# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:37:39 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from matplotlib import animation
from IPython import display

# def init(line):
#     line[0].set_data([], [], [])
#     line[1].set_data([], [], [])
#     return line

def f(x, y):
    return (np.sin(2*x)*np.sin(2*y))

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
    #Fill in boundary values
    u[0, :, :] = g(y_points, t_points)
    u[-1, :, :] = h(y_points, t_points)
    u[:, 0, :] = p(x_points, t_points)
    u[:, -1, :] = r(x_points, t_points)
    
    for j in range(1, stepsT):
        for k in range(1, steps):
            for i in range(1, steps):
                u[i, k, j+1] = 2*u[i,k,j] - u[i,k,j-1] + ((c**2)*(delT**2))*((u[i+1,k,j] - 2*u[i,k,j] + u[i-1,k,j])/(delX**2) + (u[i,k+1,j] - 2*u[i,k,j] + u[i,k-1,j])/(delY**2))
    return x_points, y_points, t_points, u

L = [0,np.pi]
W = [0, np.pi]
time = [0, 6]
c = 1
x, y, t, u = FiniteDiff2D_Wave(L, W, time, f, F, p, r, g, h, c, 50)
X, Y= np.meshgrid(x, y)

#Animation
fig1 = plt.figure(1, figsize=(12,14))
ax = plt.axes(projection = '3d')
line = [ax.plot_surface(X, Y, u[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_xlim3d([L[0],L[-1]])
ax.set_ylim3d([W[0],W[-1]])
ax.set_zlim3d([-1,1])
ax.set_title('u_tt = (c^2)(u_xx + u_yy)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
def draw_wave(num, u, line):
    line[0].remove()
    line[0] = ax.plot_surface(X, Y, u[:,:,num], cmap='gist_heat', alpha=0.8)

anim = animation.FuncAnimation(fig1, draw_wave, len(u[0,0,:]), fargs=(u, line))    
plt.show()
ue = np.zeros((len(u[0,:,0]), len(u[:,0,0]), len(u[0,0,:])))
x = np.linspace(L[0], L[-1], len(ue[:,0,0]))
y = np.linspace(W[0], W[-1], len(ue[0,:,0]))
t = np.linspace(time[0], time[-1], len(ue[0,0,:]))
X2,Y2,T = np.meshgrid(x, y, t)
ue[:,:,:] = np.cos((2*(2)**0.5) * T) * np.sin(2*X2) * np.sin(2*Y2)
fig2 = plt.figure(2, figsize=(12,14))
ax = plt.axes(projection = '3d')
line2 = [ax.plot_surface(X, Y, ue[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_xlim3d([L[0],L[-1]])
ax.set_ylim3d([W[0],W[-1]])
ax.set_zlim3d([-1,1])
ax.set_title('u(x,y,t) = cos(2*sqrt(2)*t)*sin(2x)*sin(2y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
def draw_wave2(num, ue, line):
    line2[0].remove()
    line2[0] = ax.plot_surface(X, Y, ue[:,:,num], cmap='gist_heat', alpha=0.8)
anim = animation.FuncAnimation(fig2, draw_wave2, len(ue[0,0,:]), fargs=(ue, line2))    
plt.show()

#Error
err = u - ue
fig3 = plt.figure(3, figsize=(12,14))
ax = plt.axes(projection = '3d')
line3 = [ax.plot_surface(X, Y, err[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_xlim3d([L[0],L[-1]])
ax.set_ylim3d([W[0],W[-1]])
ax.set_zlim3d([-0.1,0.1])
ax.set_title('Error = u(x,y,t) - u_exact(x,y,t), 50 Steps in X and Y')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
def draw_wave3(num, err, line3):
    line3[0].remove()
    line3[0] = ax.plot_surface(X, Y, err[:,:,num], cmap='gist_heat', alpha=0.8)
anim = animation.FuncAnimation(fig3, draw_wave3, len(err[0,0,:]), fargs=(err, line3))    
plt.show()

#Error 2 - 25 steps
x1, y1, t1, u2 = FiniteDiff2D_Wave(L, W, time, f, F, p, r, g, h, c, 25)
ue2 = np.zeros((len(u2[0,:,0]), len(u2[:,0,0]), len(u2[0,0,:])))
x1 = np.linspace(L[0], L[-1], len(ue2[:,0,0]))
y1 = np.linspace(W[0], W[-1], len(ue2[0,:,0]))
t1 = np.linspace(t[0], t[-1], len(ue2[0,0,:]))
X3,Y3,T2 = np.meshgrid(x1, y1, t1)
X4, Y4= np.meshgrid(x1, y1)
ue2[:,:,:] = np.cos((2*(2)**0.5) * T2) * np.sin(2*X3) * np.sin(2*Y3)
err2 = u2 - ue2
fig4 = plt.figure(4, figsize=(12,14))
ax = plt.axes(projection = '3d')
line4 = [ax.plot_surface(X4, Y4, err2[:,:,0], color='0.75', rstride=1, cstride=1)]
ax.set_xlim3d([L[0],L[-1]])
ax.set_ylim3d([W[0],W[-1]])
ax.set_zlim3d([-0.1,0.1])
ax.set_title('Error = u(x,y,t) - u_exact(x,y,t), 25 Steps in X and Y')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
def draw_wave4(num, err2, line4):
    line4[0].remove()
    line4[0] = ax.plot_surface(X4, Y4, err2[:,:,num], cmap='gist_heat', alpha=0.8)
anim = animation.FuncAnimation(fig4, draw_wave4, len(err2[0,0,:]), fargs=(err2, line4))    
plt.show()