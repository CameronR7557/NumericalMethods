# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:15:26 2024

@author: robin
"""
from math import e
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.integrate import solve_bvp, solve_ivp

#First ODE Functions
def y1_2(x, y1, y2):
    return x*(y2**2)
def y1_1(x, y1, y2):
    return y2

def y1_sys(x, y):
    return np.vstack((y[1], x*(y[1])**2))

def bc1(a, b):
    return np.array([a[0] - np.pi/2, b[0] - np.pi/4])

#Second ODE Functions
def y2_2(x, y1, y2):
    return y1*(np.cos(x))**2 - np.sin(x)*e**(np.sin(x))

def y2_1(x, y1, y2):
    return y2

def y2_sys(x, y):
    return np.vstack((y[1], y[0]*(np.cos(x))**2 - np.sin(x)*e**(np.sin(x))))

def bc2(a, b):
    return np.array([a[0] - 1, b[0] - 1])

def EulersMethod(f1, f2, bounds, y10, y20, h):
    steps = int((bounds[1] - bounds[0]) / h)
    y1 = [y10]
    y2 = [y20]
    t = [bounds[0]]
    tempVal = 0
    for i in range(steps):
        tempVal = y1[-1] + h*f1(t[-1], y1[-1], y2[-1])
        y2.append(y2[-1] + h*f2(t[-1], y1[-1], y2[-1]))
        y1.append(tempVal)
        t.append(t[-1] + h)
    return t, y1, y2

def ShootingMethod(f1, f2, A, B, bounds, h):
    t1, y1, y2 = EulersMethod(f1, f2, bounds, A, -0.2, h)#Get an initial guess
    #prev_x, cur_x, next_x are the guesses for the initial value of y2. Used in root finding
    prev_x = -0.2
    cur_x = -0.1
    g = y2[-1] #g is the most recent estimate of B
    while(abs(B - y1[-1]) > 0.00000001):
        t1, y1, y2 = EulersMethod(f1, f2, bounds, A, cur_x, h)#Get next guess
        #Secant Method: guess values are x, (g - B) and (y22[-1] - b) are f(x) (errors)
        next_x = (prev_x * (y1[-1] - B) - cur_x * (g - B)) / (y1[-1] - g)
        prev_x = cur_x
        cur_x = next_x
        g = y1[-1]
    return t1, y1 ,y2

colors1 = ['r', 'g', 'm']
colors2 = ['y', 'c', 'k']
fig1 = plt.figure(1, figsize=(12,14))
plt.subplot(2,1,1)

x1 = np.linspace(0, 2, 200)
steps = 10
x, y1, y2= ShootingMethod(y1_1, y1_2, np.pi/2, np.pi/4, [0,2], 2/steps)
#Plot Shooting Method Estimate
plt.plot(x, y1, color = colors1[0], linewidth=1, label = 'Shooting Method, h = ' + str(2/steps)[:6])
#Plot with 20 steps
steps = 20
x, y1, y2= ShootingMethod(y1_1, y1_2, np.pi/2, np.pi/4, [0,2], 2/steps)
plt.plot(x, y1, color = colors1[1], linewidth=1, label = 'Shooting Method, h = ' + str(2/steps)[:6])
#Plot with 40 steps
steps = 40
x, y1, y2= ShootingMethod(y1_1, y1_2, np.pi/2, np.pi/4, [0,2], 2/steps)
plt.plot(x, y1, color = colors1[2], linewidth=1, label = 'Shooting Method, h = ' + str(2/steps)[:6])

#Plot Actual
cotf = []
for i in x1:
    cotf.append(sp.acot(i/2))#Does not like graphing arccot directly
plt.plot(x1, cotf, color = 'b', linewidth=1, label = 'Actual') 
plt.grid()
plt.title('Solution Graphs for y(x) = arccot(x/2)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Graph solve_bvp
x_a = np.linspace(0, 2, 20)
y_a = np.zeros((2, x_a.size))
sol = solve_bvp(y1_sys, bc1, x_a, y_a)
plt.subplot(2,1,2)
plt.grid()
plt.plot(x_a, sol.y[0], color = 'r', linewidth=1, label = 'solve_bvp estimate')
plt.plot(x1, cotf, color = colors1[2], linewidth=1, label = 'Actual')  #Graph Exact
plt.title('solve_bvp Result Graph')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

fig2 = plt.figure(2, figsize=(12,14))
plt.subplot(2,1,1)
x2 = np.linspace(0, np.pi, 200)
steps = 10
x, y1, y2= ShootingMethod(y2_1, y2_2, 1, 1, [0,np.pi], np.pi/steps)
#Plot Shooting Method Estimate
plt.plot(x, y1, color = colors1[0], linewidth=1, label = 'Shooting Method, h = ' + str(np.pi/steps)[:6])
#Plot with 20 steps 
steps = 20
x, y1, y2= ShootingMethod(y2_1, y2_2, 1, 1, [0,np.pi], np.pi/steps)
plt.plot(x, y1, color = colors1[1], linewidth=1, label = 'Shooting Method, h = ' + str(np.pi/steps)[:6])
#Plot with 40 steps
steps = 40
x, y1, y2= ShootingMethod(y2_1, y2_2, 1, 1, [0,np.pi], np.pi/steps)
plt.plot(x, y1, color = colors1[2], linewidth=1, label = 'Shooting Method, h = ' + str(np.pi/steps)[:6])
plt.plot(x2, e**(np.sin(x2)), color = 'b', linewidth=1, label = 'Actual')  #Graph Exact
plt.grid()
plt.title('Solution Graphs for y(x) = e^(sin(x))')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Graph solve_bvp
x_a = np.linspace(0, np.pi, 20)
y_a = np.zeros((2, x_a.size))
sol = solve_bvp(y2_sys, bc2, x_a, y_a)
plt.subplot(2,1,2)
plt.grid()
plt.plot(x_a, sol.y[0], color = 'r', linewidth=1, label = 'solve_bvp estimate')
plt.plot(x2, e**(np.sin(x2)), color = 'b', linewidth=1, label = 'Actual')  #Graph Exact
plt.title('solve_bvp Result Graph')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

