# -*- coding: utf-8 -*-
"""
Created on Thu Apr  25 2:35:45 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def f(y):
    return 0
def g(y):
    return 0
def p(x):
    return 0
def r(x):
    return 100


def CFD_Rectangle_Hole(x, y, f, g, p, r, stepsX, stepsY, hole_width):
    delX = (x[1] - x[0])/ stepsX
    delY = (y[1] - y[0])/ stepsY
    stepsH = int(np.ceil((stepsX/x[1]) * hole_width))
    i_h = int((stepsY - stepsH)/2)
    j_h = int((stepsX - stepsH)/2)
    #Set up solution array. Rows are the stencil points, cols are the equations for individual points
    A = np.zeros(((stepsY - 1) * (stepsX - 1), (stepsY - 1) * (stepsX - 1)))
    b = np.zeros(((stepsY - 1) * (stepsX - 1), 1))
    u = np.zeros(((stepsX + 1), (stepsY + 1)))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    y_points = np.linspace(y[0], y[-1], stepsY + 1)
    for i in range(0, len(u[:,0])):
        u[i, 0] = f(delY * i + y[0])  #Left BCs
        u[i, -1] = g(delY * i + y[0]) #Right BCs
    for i in range(0, len(u[0,:])):
        u[0, i] = p(delX * i + x[0])  #Lower BCs
        u[-1, i] = r(delX * i + x[0]) #Upper BCs
    for i in range(stepsH+1):
        u[stepsY - i_h, i + j_h] = 100
    i = 0
    j = 0
    curRow = 0
    stepsX -= 2
    stepsY -= 2
    for i in range(0, stepsY + 1):
        for j in range(0, stepsX + 1):
            curRow = i*(stepsX + 1) + j
            A[curRow, curRow] = 4
            if(i == 0):#Bottom BC, p(x)
                b[curRow, 0] += p(delX *(j+1) + x[0])
            else:
                A[curRow, curRow - stepsX - 1] = -1
            
            if(i == stepsY):#Upper BC, r(x)
                b[curRow, 0] += r(delX *(j+1) + x[0]) 
            else:
                A[curRow, curRow + stepsX + 1] = -1
            
            if(j == 0):#Left bound is BC, f(y)
                b[curRow, 0] += f(delY *(i+1) + y[0])
            else:
                A[curRow, curRow - 1] = -1
            
            if(j == stepsX):#Right bound is BC, g(y)
                b[curRow, 0] += g(delY *(i+1) + y[0])
            else:
                A[curRow, curRow + 1] = -1
            
            if(i == stepsY - i_h + 1 and (j >= j_h-1 and j <= stepsX - j_h+1)):
                A[curRow, curRow + 1] = 0
                A[curRow, curRow - 1] = 0
                A[curRow, curRow + stepsX + 1] = 0
                A[curRow, curRow - stepsX - 1] = 0
                A[curRow, curRow] = 1
                b[curRow, 0] = 100
            elif((j >= j_h-1 and j <= stepsX - j_h+1) and (i >= i_h-1 and i <= stepsY - i_h+1)):
                A[curRow, curRow + 1] = 0
                A[curRow, curRow - 1] = 0
                A[curRow, curRow + stepsX + 1] = 0
                A[curRow, curRow - stepsX - 1] = 0
                b[curRow, 0] = 0
            
    temps = np.linalg.solve(A, b)
    for i in range(1, len(u[0, :]) - 1):
        u[i, 1:-1] = temps[(stepsX+1)*(i-1):((stepsX+1)*(i-1) + stepsX + 1), 0].T
    return x_points, y_points, u


L = [0, 10]
W = [0, 10]
fig1 = plt.figure(1, figsize=(12,7))
ax = plt.axes(projection = '3d')
x, y, u = CFD_Rectangle_Hole(L, W, f, g, p, r, 50, 50, 4)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Laplace\'s Eqn Solution x and y steps = 50')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
plt.show()
