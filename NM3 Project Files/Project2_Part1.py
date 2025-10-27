# -*- coding: utf-8 -*-
"""
Created on Thu Apr  22 5:26:45 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def f(y):
    return 4*y

def g(y):
    return 4*y

def p(x):
    return 0

def r(x):
    return 4


def CFD_Rectangle(x, y, f, g, p, r, stepsX, stepsY):
    delX = (x[1] - x[0])/ stepsX
    delY = (y[1] - y[0])/ stepsY
    #Set up solution array. Rows are the stencil points, cols are the equations for individual points
    A = np.zeros(((stepsY - 1) * (stepsX - 1), (stepsY - 1) * (stepsX - 1)))
    b = np.zeros(((stepsY - 1) * (stepsX - 1), 1))
    u = np.zeros(((stepsY + 1), (stepsX + 1)))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    y_points = np.linspace(y[0], y[-1], stepsY + 1)
    for i in range(0, len(u[:,0])):
        u[0, i] = p(delX * i + x[0])  #Lower BCs
        u[-1, i] = r(delX * i + x[0]) #Upper BCs
        u[i, 0] = f(delY * i + y[0])  #Left BCs
        u[i, -1] = g(delY * i + y[0]) #Right BCs
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
    temps = np.linalg.solve(A, b)
    for i in range(1, len(u[0, :]) - 1):
        u[i, 1:-1] = temps[(stepsX+1)*(i-1):((stepsX+1)*(i-1) + stepsX + 1), 0].T
    return x_points, y_points, u


L = [0, 1]
W = [0, 1]

fig1 = plt.figure(1, figsize=(12,7))
ax = plt.axes(projection = '3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, 5, 5)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Laplace\'s Eqn Solution x and y steps = 5')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
plt.show()

fig2 = plt.figure(2, figsize=(12,7))
ax = plt.axes(projection = '3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, 10, 10)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Laplace\'s Eqn Solution x and y steps = 10')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
plt.show()

fig3 = plt.figure(3, figsize=(12,7))
ax = plt.axes(projection = '3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, 20, 20)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='gist_heat', alpha=0.8)
ax.set_title('Laplace\'s Eqn Solution x and y steps = 20')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
plt.show()

# #Exact
fig4 = plt.figure(4, figsize=(12,7))
ax = plt.axes(projection = '3d')
x = np.linspace(L[0], L[1], 100)
y = np.linspace(W[0], W[1], 100)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, 4*Y, cmap='gist_heat', alpha=0.8)
ax.set_title('u(x,y) = 4y')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u(x,y)')
plt.show()

#Error
fig5 = plt.figure(5, figsize=(12,8))
ax = fig5.add_subplot(1, 3, 1, projection='3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, 5, 5)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, (u - (4*Y)), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x and y steps = 5')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u_est(x,y) - u_exact(x,y)')
ax = fig5.add_subplot(1, 3, 2, projection='3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, 10, 10)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, (u - (4*Y)), cmap='gist_heat', alpha=0.8)
print("Numerical:")
print(u)
print("Exact:")
print(4*Y)
ax.set_title('Error x and y steps = 10')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u_est(x,y) - u_exact(x,y)')
ax = fig5.add_subplot(1, 3, 3, projection='3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, 20, 20)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, (u - (4*Y)), cmap='gist_heat', alpha=0.8)
ax.set_title('Error x and y steps = 20')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u_est(x,y) - u_exact(x,y)')