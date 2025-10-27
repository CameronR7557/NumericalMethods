# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 09:59:35 2024

@author: Cameron Robinson
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def f1(t,y):# Sol: np.cos(t) + e**(-t)
    return np.cos(t) - np.sin(t) - y

def f2(t,y):#Sol: 1 + 4*t + 0.25*(t**2)
    return 2 + (y - 2*t + 3)**(0.5)

def RK4(f, bounds, y0, h):
    totalIterations = 0
    if bounds[0] > bounds[1]:
        temp = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = temp
    y_points = [y0]                                                       #Initialize Y values list
    t_points = [bounds[0]]                                                #Initialize T values list
    while t_points[-1] < (bounds[1] - 0.0000000000001):                   #Account for any rounding error  
        if (t_points[-1] + h > bounds[1]):                                #Clamp t in case h does not divide (b - a)
            h = abs(bounds[1] - t_points[-1])
        k1 = f(t_points[-1], y_points[-1])
        k2 = f(t_points[-1] + 0.5*h, y_points[-1] + 0.5*h*k1)
        k3 = f(t_points[-1] + 0.5*h, y_points[-1] + 0.5*h*k2)
        k4 = f(t_points[-1] + h, y_points[-1] + h*k3)
        y_points.append(y_points[-1] + h*(k1 + 2*k2 + 2*k3 + k4)/6)       #Gets most recent y-value and adds the derivative at most recent t-value to it
        t_points.append(t_points[-1] + h)                                 #Gets most recent t-value and increments it by h
        totalIterations += 1
    print("RK4 (h = " + str(h)[:6] + "): " + str(totalIterations))
    return t_points, y_points    

def EulerPC(f, bounds, y0, h, tolerance):
    y_points = [y0]
    t_points = [bounds[0]]
    numIters = 0
    totalIterations = 0
    prevError = -1
    while t_points[-1] < (bounds[1] - 0.0000000000001):                   #Account for any rounding error  
        tolerance_flag = False
        if (t_points[-1] + h > bounds[1]):                                #Clamp t in case h does not divide (b - a)
            h = abs(bounds[1] - t_points[-1])
        #Predict: (Fwd Euler)
        y_points.append(y_points[-1] + h*f(t_points[-1], y_points[-1]))
        t_points.append(t_points[-1] + h)
        #Correct:
        while not tolerance_flag:   #Correct until tolerance flag is set to True (by if-statement below)
            totalIterations += 1
            numIters += 1
            #Implicit Euler, no root finding
            correct = y_points[-2] + h*f(t_points[-1], y_points[-1])
            #Stop correcting if the increase was <= tolerance, if the maximum number of iterations is reached, or if the change in error is >= 0
            if(not(abs(correct - y_points[-1]) > tolerance) or numIters > 1000 or abs(correct - y_points[-1]) >= prevError):
                tolerance_flag = True
            prevError = abs(correct - y_points[-1])
            y_points[-1] = correct
            
    print("EulerPC (h = " + str(h)[:6] + "): " + str(totalIterations))
    return t_points, y_points


#-------------------- EQN 1 ------------------------

t1_linspace = np.linspace(0,10,200)
steps = 13
colors1 = ['r', 'g', 'm']
colors2 = ['y', 'c', 'k']
fig1 = plt.figure(1, figsize=(12,14))
fig2 = plt.figure(2, figsize=(12,14))
#Graph Functions for Y1 and Y2 with three h values
for i in range(3):
    fig1 = plt.figure(1, figsize=(12,14))
    #Solve Y1 Euler PC
    t1, y1 =  EulerPC(f1, [0,10], 2, 3/steps, 0.0001)
    plt.subplot(2,1,1)
    #Graph Y1 Euler PC
    plt.plot(t1, y1, color = colors1[i], linewidth=1, label = 'Y1 Euler PC estimate, h = ' + str(10/steps)[:6]) 
    #Solve Y1 RK4
    t1, y1 =  RK4(f1, [0,10], 2, 3/steps)
    #Graph Y1 RK4
    plt.plot(t1, y1, color = colors2[i], linewidth=1, label = 'Y1 RK4 estimate, h = ' + str(10/steps)[:6])
    #Solve Y2 Euler PC
    t2, y2 =  EulerPC(f2, [0,1.5], 1, 2/steps, 0.0001)
    fig2 = plt.figure(2, figsize=(12,14))
    plt.subplot(2,1,1)
    #Graph Y2 Euler PC
    plt.plot(t2, y2, color = colors1[i], linewidth=1, label = 'Y2 Euler PC estimate h = ' + str(1.5/steps)[:6])
    #Solve Y2 RK4
    t2, y2 =  RK4(f2, [0,1.5], 1, 2/steps)
    #Graph Y2 RK4
    plt.plot(t2, y2, color = colors2[i], linewidth=1, label = 'Y2 RK4 estimate h = ' + str(1.5/steps)[:6])
    steps *= 2


fig1 = plt.figure(1, figsize=(12,14))
plt.subplot(2,1,1)
plt.grid()
plt.plot(t1_linspace, np.cos(t1_linspace) + e**(-t1_linspace), color = 'b', label = 'actual', linewidth=1) #Graph Exact Y1
plt.title('Solution Graphs for y1(t) = cos(t) + e^(-t)')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint
sol = odeint(f1, 2, t1_linspace, tfirst=True)
plt.subplot(2,1,2)
plt.grid()
plt.plot(t1_linspace, sol, color = 'r', linewidth=1, label = 'odeint estimate')
plt.plot(t1_linspace, np.cos(t1_linspace) + e**(-t1_linspace), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#-------------------- EQN 2 ------------------------
t2_linspace = np.linspace(0,1.5,200)
fig2 = plt.figure(2, figsize=(12,14))
plt.subplot(2,1,1)
plt.grid()
plt.plot(t2_linspace, 1 + 4*t2_linspace + 0.25*(t2_linspace**2), color = 'b', label = 'actual', linewidth=1) #Graph Exact Y2
plt.title('Solution Graphs for y2(t) = 1 + 4*t + 0.25*(t^2)')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint
sol = odeint(f2, 1, t2_linspace, tfirst=True)
plt.subplot(2,1,2)
plt.grid()
plt.plot(t2_linspace, sol, color = 'r', linewidth=1, label = 'odeint estimate')
plt.plot(t2_linspace, 1 + 4*t2_linspace + 0.25*(t2_linspace**2), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()