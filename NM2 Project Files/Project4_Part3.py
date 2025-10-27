# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:58:14 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from scipy.integrate import solve_bvp

def y1_sys(x, y):
    return np.vstack((y[1], y[1] + 2*y[0] + np.cos(x)))

def y2_sys(x, y):
    return np.vstack((y[1], y[1]*(1/x) + y[0]*(1/(x**2)) + np.log(x)))

def bc1(a, b):
    return np.array([a[0] + 0.3, b[0] + 0.1])

def bc2(a, b):
    return np.array([a[0], b[0] + 2])

def p1(x):
    return -1
def q1(x):
    return -2
def r1(x):
    return np.cos(x)

def p2(x):
    return -1/x
def q2(x):
    return 1/x**2
def r2(x):
    return np.log(x)/(x**2)

def finiteDifferenceBVP(p, q, r, A, B, bounds, h):
    x_points = [bounds[0]]
    y_points = [A]
    dim = int(((bounds[1] - bounds[0])/h) - 1)
    #Create Matrices
    A_matrix = np.zeros((int(dim), int(dim)))
    b_matrix = np.zeros((int(dim), 1))
    a = bounds[0]
    b = bounds[1]
    col = 0
    
    #Fill in first row
    A_matrix[0,:2] = [(-2/(h**2) + q(a + h)), (1/(h**2) + p(a + h)/(2*h))]
    b_matrix[0] = [r(a + h) - A/(h**2) + (p(a + h)*A)/(2*h)]
    a = a + h
    x_points.append(a)
    #Fill in middle rows of matrices
    for row in range(1, dim-1):
        a = a + h
        A_matrix[row, col:(col+3)] = [(1/(h**2) - p(a)/(2*h)), (q(a) - 2/(h**2)), (1/(h**2) + p(a)/(2*h))]
        b_matrix[row] = [r(a)]
        x_points.append(a)
        col = col + 1
    #Fill in last row
    A_matrix[dim - 1, dim-2:] = [(1/(h**2) - p(b - h)/(2*h)), (-2/(h**2) + q(b - h))]
    b_matrix[dim-1] = [r(b - h) - B/(h**2) - (p(b - h)*B)/(2*h)]
    x_points.append(a + h)
    #Solve Matrix
    y_matrix = np.linalg.solve(A_matrix, b_matrix)
    #Get points for y from y_matrix
    for i in range(0, len(y_matrix[:,0])):
        y_points.append(y_matrix[i, 0])
    #Append right bound
    x_points.append(bounds[1])
    y_points.append(B)
    return x_points, y_points

x1_linspace = np.linspace(0,np.pi/2,200)
steps = 3
colors1 = ['r', 'g', 'm']
colors2 = ['y', 'c', 'k']
fig1 = plt.figure(1, figsize=(12,14))
fig2 = plt.figure(2, figsize=(12,14))
#Graph Functions for Y1 and Y2 with three h values
for i in range(3):
    fig1 = plt.figure(1, figsize=(12,14))
    #Solve Y1 finite diff BVP
    x1, y1 =  finiteDifferenceBVP(p1, q1, r1, -0.3, -0.1, [0, np.pi/2], (np.pi/2)/steps)
    plt.subplot(2,1,1)
    #Graph Y1
    plt.plot(x1, y1, color = colors1[i], linewidth=1, label = 'Finite Difference, h = ' + str((np.pi/2)/steps)[:6]) 
    fig2 = plt.figure(2, figsize=(12,14))
    #Solve Y2 finite diff BVP
    x1, y1 =  finiteDifferenceBVP(p2, q2, r2, 0, -2, [1, 2], 1/steps)
    plt.subplot(2,1,1)
    #Graph Y2
    plt.plot(x1, y1, color = colors1[i], linewidth=1, label = 'Finite Difference, h = ' + str((np.pi/2)/steps)[:6]) 
    steps *= 2


fig1 = plt.figure(1, figsize=(12,14))
plt.subplot(2,1,1)
plt.grid()
plt.plot(x1_linspace, -0.3*np.cos(x1_linspace) - 0.1*np.sin(x1_linspace), color = 'b', label = 'actual', linewidth=1) #Graph Exact
plt.title('Solution Graphs for y1(x) = -0.3*cos(x) - 0.1*sin(x)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Graph solve_bvp
x_a = np.linspace(0, np.pi/2, 20)
y_a = np.zeros((2, x_a.size))
sol = solve_bvp(y1_sys, bc1, x_a, y_a)
plt.subplot(2,1,2)
plt.grid()
plt.plot(x_a, sol.y[0], color = 'r', linewidth=1, label = 'solve_bvp estimate')
plt.plot(x1_linspace, -0.3*np.cos(x1_linspace) - 0.1*np.sin(x1_linspace), color = 'b', label = 'actual', linewidth=1) #Graph Exact
plt.title('solve_bvp Result Graph')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#-------------------- EQN 2 ------------------------
x2_linspace = np.linspace(1,2,200)
fig2 = plt.figure(2, figsize=(12,14))
plt.subplot(2,1,1)
plt.grid()
plt.plot(x2_linspace, -2*x2_linspace -0.5*x2_linspace*np.log(x2_linspace) + np.log(x2_linspace) + 2, color = 'b', label = 'actual', linewidth=1) #Graph Exact Y2
plt.title('Solution Graphs for y2(t) = −2x −0.5x*ln(x) + ln(x) + 2')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph solve_bvp
x_b = np.linspace(1, 2, 20)
y_b = np.zeros((2, x_b.size))
sol = solve_bvp(y2_sys, bc2, x_b, y_b)
plt.subplot(2,1,2)
plt.grid()
plt.plot(x_b, sol.y[0], color = 'r', linewidth=1, label = 'solve_bvp estimate')
plt.plot(x2_linspace, -2*x2_linspace -0.5*x2_linspace*np.log(x2_linspace) + np.log(x2_linspace) + 2, color = 'b', label = 'actual', linewidth=1) #Graph Exact
plt.title('solve_bvp Result Graph')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
