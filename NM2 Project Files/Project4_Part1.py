# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:00:57 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def v_prime(t,v):
    return 32 - 2.2*v

def RK2_Midpoint(f, bounds, y0, h):
    if bounds[0] > bounds[1]:
        temp = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = temp
    y_points = [y0]        #Initialize Y values list
    t_points = [bounds[0]] #Initialize T values list
    while t_points[-1] < (bounds[1] - 0.0000000000001): #Account for any rounding error     
        k1 = f(t_points[-1], y_points[-1])
        k2 = f(t_points[-1] + 0.5*h, y_points[-1] + h*k1*0.5)
        y_points.append(y_points[-1] + h*k2)
        t_points.append(t_points[-1] + h)#Gets most recent t-value and increments it by h
    print("RK2 Solution: " + str(y_points[-1]))
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
    print("Solution: " + str(y_points[-1]))
    return t_points, y_points

t1_linspace = np.linspace(0,3,200)
steps = 13
colors1 = ['r', 'g', 'm']
colors2 = ['y', 'c', 'k']
fig1 = plt.figure(1, figsize=(12,14))
#Graph Functions for Y1 and Y2 with three h values
for i in range(3):
    fig1 = plt.figure(1, figsize=(12,14))
    #Solve Y1 Euler PC
    t1, y1 =  EulerPC(v_prime, [0,3], 0, 3/steps, 0.0001)
    plt.subplot(2,1,1)
    #Graph Y1 Euler PC
    plt.plot(t1, y1, color = colors1[i], linewidth=1, label = 'Euler PC, h = ' + str(3/steps)[:6]) 
    #Solve Y1 RK2
    t1, y1 =  RK2_Midpoint(v_prime, [0,3], 0, 3/steps)
    #Graph Y1 RK2
    plt.plot(t1, y1, color = colors2[i], linewidth=1, label = 'RK2 Midpoint, h = ' + str(3/steps)[:6])
    steps *= 2


fig1 = plt.figure(1, figsize=(12,14))
plt.subplot(2,1,1)
plt.grid()
plt.plot(t1_linspace, (32/2.2) * (1 - e**(-2.2*t1_linspace)), color = 'b', label = 'actual', linewidth=1) #Graph Exact
plt.title('Solution Graphs for v(t) = (g/2.2) * (1 - e^(-2.2t))')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint
sol = odeint(v_prime, 0, t1_linspace, tfirst=True)
plt.subplot(2,1,2)
plt.grid()
plt.plot(t1_linspace, sol, color = 'r', linewidth=1, label = 'odeint estimate')
plt.plot(t1_linspace, (32/2.2) * (1 - e**(-2.2*t1_linspace)), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()
