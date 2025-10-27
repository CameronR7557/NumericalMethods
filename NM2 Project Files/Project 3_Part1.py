# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:37:29 2024

@author: Cameron Robinson
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def f1(t,y):
    return (1 - t) * y

def f2(t,y):
    return t*(y**2)

def TrapezoidPredictorCorrector(func, bounds, y0, h, tolerance):
    tolerance_flag = False
    t_points = [bounds[0]]
    y_points = [y0]
    totalIterations = 0
    prevError = -1
    while t_points[-1] < (bounds[1] - 0.0000000000001): #Account for any rounding error 
        numIters = 0
        tolerance_flag = False                              
        if (t_points[-1] + h > bounds[1]):#Clamp t in case h does not divide (b - a)
            h = abs(bounds[1] - t_points[-1])
        predict = y_points[-1] + h * func(t_points[-1], y_points[-1])
        y_points.append(predict)
        t_points.append(t_points[-1] + h)
        while not tolerance_flag:   #Correct until tolerance flag is set to True (by if-statement below)
            totalIterations += 1
            numIters = numIters + 1
            correct = y_points[-2] + (h/2) * (func(t_points[-1], y_points[-1]) + func(t_points[-2], y_points[-2]))
            #Stop correcting if the increase was <= tolerance, if the maximum number of iterations is reached, or if the change in error is zero
            if(not(abs(correct - y_points[-1]) > tolerance) or numIters > 1000 or abs(correct - y_points[-1]) == prevError):
                tolerance_flag = True
            prevError = abs(correct - y_points[-1])
            y_points[-1] = correct
    print("Trapezoid PC (h = " + str(h)[:6] + "): " + str(totalIterations))
    return t_points, y_points

#-------------------- EQN 1 ------------------------
t1_linspace = np.linspace(0,3,200)
steps = 13
colors = ['r', 'g', 'm', 'y', 'c']
fig1 = plt.figure(1, figsize=(12,14))
fig2 = plt.figure(2, figsize=(12,14))
#Graph Functions for Y1 and Y2 with different h values
for i in range(3):
    fig1 = plt.figure(1, figsize=(12,14))
    t1, y1 = TrapezoidPredictorCorrector(f1, [0,3], 3, 3/steps, 0.0001) #Solve Y1
    plt.subplot(2,1,1)
    plt.plot(t1, y1, color = colors[i], linewidth=1, label = 'Y1 estimate, h = ' + str(2/steps)[:6])
    t2, y2 = TrapezoidPredictorCorrector(f2, [0,2], 0.4, 2/steps, 0.0001) #Solve Y2
    fig2 = plt.figure(2, figsize=(12,14))
    plt.subplot(2,1,1)
    plt.plot(t2, y2, color = colors[i], linewidth=1, label = ' Y2 estimate h = ' + str(3/steps)[:6])
    steps *= 2

#Graph Acutal Solution to Y1
fig1 = plt.figure(1, figsize=(12,14))
plt.subplot(2,1,1)
plt.grid()
plt.plot(t1_linspace, 3*e**(t1_linspace - 0.5*t1_linspace**2), color = 'b', label = 'actual', linewidth=1)
plt.title('Trapezoid PC Method for y1(t) = 3e^(t - 0.5*t^2)')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint for Y1
sol = odeint(f1, 3, t1_linspace, tfirst=True)
plt.subplot(2,1,2)
plt.grid()
plt.plot(t1_linspace, sol, color = 'r', linewidth=1, label = 'odeint() estimate')
plt.plot(t1_linspace, 3*e**(t1_linspace - 0.5*t1_linspace**2), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#-------------------- EQN 2 ------------------------
#Graph Acutal Solution to Y2
t2_linspace = np.linspace(0,2,200)
fig2 = plt.figure(2, figsize=(12,14))
plt.subplot(2,1,1)
plt.grid()
plt.plot(t2_linspace, -1/(0.5*t2_linspace**2 - 2.5), color = 'b', label = 'actual', linewidth=1)
plt.title('Trapezoid PC Method for y2(t) = -1/(0.5*t^2 - 2.5')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint for Y2
sol = odeint(f2, 0.4, t2_linspace, tfirst=True)
plt.subplot(2,1,2)
plt.grid()
plt.plot(t2_linspace, sol, color = 'r', linewidth=1, label = 'odeint() estimate')
plt.plot(t2_linspace, -1/(0.5*t2_linspace**2 - 2.5), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()