# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:44:39 2024

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

def AdamsMoultonPredictorCorrector(func, bounds, y0, h, tolerance):
    tolerance_flag = False
    t_points = [bounds[0]]
    y_points = [y0]
    totalIterations = 0
    prevError = -1
    count = 0
    while t_points[-1] < (bounds[1] - 0.0000000000001): #Account for any rounding error 
        numIters = 0
        tolerance_flag = False                              
        if (t_points[-1] + h > bounds[1]):#Clamp t in case h does not divide (b - a)
            h = abs(bounds[1] - t_points[-1])
        #Use RK4 for the first three predictions then use AB4 - AM4 predictor corrector for remaining points
        if(count < 3):                                                    #Compute RK4 for first 4 points
            k1 = func(t_points[-1], y_points[-1])
            k2 = func(t_points[-1] + 0.5*(h), y_points[-1] + 0.5*(h)*k1)#Use h/3 for the first three points since RK4 is explicit
            k3 = func(t_points[-1] + 0.5*(h), y_points[-1] + 0.5*(h)*k2)
            k4 = func(t_points[-1] + (h), y_points[-1] + (h)*k3)
            y_points.append(y_points[-1] + (h)*(k1 + 2*k2 + 2*k3 + k4)/6)
            t_points.append((t_points[-1] + h))
        else: #Use AB4 for the rest of the points
            y_points.append(y_points[-1] + (h/24)*(55*func(t_points[-1], y_points[-1]) - 59*func(t_points[-2], y_points[-2]) + 37*func(t_points[-3], y_points[-3]) - 9*func(t_points[-4], y_points[-4])))
            t_points.append(t_points[-1] + h)
            while not tolerance_flag:   #Correct until tolerance flag is set to True (by if-statement below)
                totalIterations += 1
                numIters = numIters + 1
                correct = y_points[-2] + (h/24) * (9*func(t_points[-1], y_points[-1]) + 19*func(t_points[-2], y_points[-2]) - 5*func(t_points[-3], y_points[-3]) + func(t_points[-4], y_points[-4]))
                #Stop correcting if the increase was <= tolerance, if the maximum number of iterations is reached, or if the change in error is zero
                if(not(abs(correct - y_points[-1]) > tolerance) or numIters > 1000 or abs(correct - y_points[-1]) == prevError):
                    tolerance_flag = True
                prevError = abs(correct - y_points[-1])
                y_points[-1] = correct
        count += 1
    print("AM4 PC (h = " + str(h)[:6] + "): " + str(totalIterations))
    return t_points, y_points

#-------------------- EQN 1 ------------------------

t1_linspace = np.linspace(0,3,200)
steps = 13
colors = ['r', 'g', 'm', 'y', 'c']
fig1 = plt.figure(1, figsize=(12,14))
fig2 = plt.figure(2, figsize=(12,14))

#Graph Functions for Y1 and Y2 with three h values
for i in range(3):
    fig1 = plt.figure(1, figsize=(12,14))
    t1, y1 =  AdamsMoultonPredictorCorrector(f1, [0,3], 3, 3/steps, 0.0001) #Solve Y1
    plt.subplot(2,1,1)
    plt.plot(t1, y1, color = colors[i], linewidth=1, label = 'Y1 estimate, h = ' + str(3/steps)[:6]) #Graph Y1
    t2, y2 =  AdamsMoultonPredictorCorrector(f2, [0,2], 0.4, 2/steps, 0.0001) #Solve Y2
    fig2 = plt.figure(2, figsize=(12,14))
    plt.subplot(2,1,1)
    plt.plot(t2, y2, color = colors[i], linewidth=1, label = ' Y2 estimate h = ' + str(2/steps)[:6]) #Graph Y2
    steps *= 2


fig1 = plt.figure(1, figsize=(12,14))
plt.subplot(2,1,1)
plt.grid()
plt.plot(t1_linspace, 3*e**(t1_linspace - 0.5*t1_linspace**2), color = 'b', label = 'actual', linewidth=1) #Graph Exact Y1
plt.title('AM4 PC Method for y1(t) = 3e^(t - 0.5*t^2)')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint
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
t2_linspace = np.linspace(0,2,200)
fig2 = plt.figure(2, figsize=(12,14))
plt.subplot(2,1,1)
plt.grid()
plt.plot(t2_linspace, -1/(0.5*t2_linspace**2 - 2.5), color = 'b', label = 'actual', linewidth=1) #Graph Exact Y2
plt.title('AM4 PC Method for y2(t) = -1/(0.5*t^2 - 2.5')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint
sol = odeint(f2, 0.4, t2_linspace, tfirst=True)
plt.subplot(2,1,2)
plt.grid()
plt.plot(t2_linspace, sol, color = 'r', linewidth=1, label = 'odeint() estimate')
plt.plot(t2_linspace, -1/(0.5*t2_linspace**2 - 2.5), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()