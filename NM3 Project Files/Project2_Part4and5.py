# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:06:02 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def f(y):
    return 150

def g(y):
    return 150

def p(x):
    return 400

def r(x):
    return 250


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
        u[i, 0] = f(delY * i + y[0])  #Left BCs
        u[i, -1] = g(delY * i + y[0]) #Right BCs
    for i in range(0, len(u[0,:])):
        u[0, i] = p(delX * i + x[0])  #Lower BCs
        u[-1, i] = r(delX * i + x[0]) #Upper BCs
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
                b[curRow, 0] += p(delX *(j+1) + x[0])#Check these. First make sure x and y values are correct
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
    for i in range(1, len(u[:, 0]) - 1):
        u[i, 1:-1] = temps[(stepsX+1)*(i-1):((stepsX+1)*(i-1) + stepsX + 1), 0]
    return x_points, y_points, u


L = [0, 10]
W = [0, 6]
stepsX = 10
stepsY = 6

#Exact Solution from Prof. Fogarty
#nn=L[-1]*W[-1]
nn=300
a=L[0]
b=L[-1]
c=W[0]
d=W[-1]
X = np.linspace(a,b,stepsX + 1)
Y = np.linspace(c,d,stepsY + 1)
h=(b-a)/stepsX

x,y = np.meshgrid(X,Y)

ue=np.zeros((stepsY + 1,stepsX + 1))
ve=np.zeros((stepsY + 1,stepsX + 1))
we=np.zeros((stepsY + 1,stepsX + 1))

for i in range (0,nn):
    bn=(2/((i+1)*np.pi*np.sinh((i+1)*np.pi*6/10)))*(1-np.cos((i+1)*np.pi))
    Bn=(2/((i+1)*np.pi*np.sinh((i+1)*np.pi*6/10)))*(1-np.cos((i+1)*np.pi))
    ve=bn*np.sin((i+1)*np.pi*x/10)*np.sinh((i+1)*np.pi*y/10)+ve
    we=Bn*np.sin((i+1)*np.pi*x/10)*np.sinh((i+1)*np.pi*(6-y)/10)+we
    
ue=100*ve+250*we+150
u1=ue

u1[:,0]=f(Y)
u1[:,stepsX]=g(Y)
u1[0,:]=p(X)
u1[stepsY,:]=r(X)

fig5 = plt.figure(5, figsize=(12,7))
ax = plt.axes(projection = '3d')
ax.plot_surface(x, y, u1, cmap='gray', alpha=0.4)
CS = ax.contour(x, y, u1, levels=[300], cmap='autumn')
ax.clabel(CS, inline=1, fontsize=10)
CS.collections[0].set_label('300 deg F Isothermal Curve')
CS = ax.contour(x, y, u1, levels=[200], cmap='coolwarm')
CS.collections[0].set_label('200 deg F Contour')
ax.set_title('10x6 ft Steel Plate Exact Solution. 300 deg F isothernal and 200 deg F contour')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
ax.legend()
plt.show()

fig6 = plt.figure(6, figsize=(12,7))
ax = plt.axes(projection = '3d')
ax.plot_surface(x, y, u1, cmap='gist_heat', alpha=0.8)
ax.set_title('10x6 ft Steel Plate Exact Solution')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
ax.legend()
plt.show()

#Numerical Solutions
fig1 = plt.figure(1, figsize=(12,7))
ax = plt.axes(projection = '3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, stepsX, stepsY)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='gray', alpha=0.4)
CS = ax.contour(X, Y, u, levels=[300], cmap='autumn')
ax.clabel(CS, inline=1, fontsize=10)
CS.collections[0].set_label('300 deg F Isothermal Curve')
CS = ax.contour(X, Y, u, levels=[200], cmap='coolwarm')
CS.collections[0].set_label('200 deg F Contour')
ax.set_title('10x6 ft Steel Plate x and y steps = 1ft. 300 deg F isothernal and 200 deg F contour')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
ax.legend()
plt.show()

fig2 = plt.figure(2, figsize=(12,7))
ax = plt.axes(projection = '3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, stepsX, stepsY)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='gist_heat', alpha=0.9)
ax.set_title('10x6 ft Steel Plate (Laplace\'s Eqn Solution) x and y steps = 1ft')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
plt.show()

fig7 = plt.figure(7, figsize=(12,7))
ax = plt.axes(projection = '3d')
ax.plot_surface(X, Y, u - u1, cmap='gist_heat', alpha=0.9)
ax.set_title('Error Graph (10 x-steps, 6 y-steps)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u_est - u_exact')
plt.show()

stepsX = 60
stepsY =36

fig3 = plt.figure(3, figsize=(12,7))
ax = plt.axes(projection = '3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, stepsX, stepsY)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='gray', alpha=0.4)
CS = ax.contour(X, Y, u, levels=[300], cmap='autumn')
ax.clabel(CS, inline=1, fontsize=10)
CS.collections[0].set_label('300 deg F Isothermal Curve')
CS = ax.contour(X, Y, u, levels=[200], cmap='coolwarm')
CS.collections[0].set_label('200 deg F Contour')
ax.set_title('10x6 ft Steel Plate x and y steps = 2in. 300 deg F isothermal and 200 deg F contour')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
ax.legend()
plt.show()

fig4 = plt.figure(4, figsize=(12,7))
ax = plt.axes(projection = '3d')
x, y, u = CFD_Rectangle(L, W, f, g, p, r, stepsX, stepsY)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='gist_heat', alpha=0.9)
ax.set_title('10x6 ft Steel Plate (Laplace\'s Eqn Solution) x and y steps = 2in')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U(x,y)')
plt.show()