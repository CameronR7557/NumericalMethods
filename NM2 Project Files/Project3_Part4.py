# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:47:51 2024

@author: Cameron Robinson
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def y1_prime(y1, y2):
    return 3*y1 - 37*y2

def y2_prime(y1, y2):
    return 5*y1 - 39*y2

def y_prime_sys(y, t = 0):
    y1_prime = 3*y[0] - 37*y[1]
    y2_prime = 5*y[0] - 39*y[1]
    return [y1_prime, y2_prime]

def EulerPC(f1, f2, bounds, y10, y20, h, tolerance):
    y1_points = [y10]
    y2_points = [y20]
    y1 = 0
    y2 = 0
    t_points = [bounds[0]]
    numIters = 0
    totalIterations = 0
    prevError1 = -1
    prevError2 = -1
    while t_points[-1] < (bounds[1] - 0.0000000000001):                   #Account for any rounding error  
        tolerance_flag = False
        if (t_points[-1] + h > bounds[1]):                                #Clamp t in case h does not divide (b - a)
            h = abs(bounds[1] - t_points[-1])
        #Predict: (Fwd Euler)
        y1 = (y1_points[-1] + h*f1(y1_points[-1], y2_points[-1]))
        y2 = (y2_points[-1] + h*f2(y1_points[-1], y2_points[-1]))
        y1_points.append(y1)
        y2_points.append(y2)
        
        t_points.append(t_points[-1] + h)
        
        #Get initial prediction with implicit Euler
        y1 = y1_points[-2] + h*f1(y1_points[-1], y2_points[-1])
        y2 = y2_points[-2] + h*f2(y1_points[-1], y2_points[-1])
        
        while not tolerance_flag:   #Correct until tolerance flag is set to True (by if-statement below)
            totalIterations += 1
            numIters += 1
            #Improve implicit Euler prediction with Secant method on g(y) = y(t+h) - y(t) - h*f(y1(t+h),y2(t+h))
            correct1 = (y1_points[-1] * (y1 - y1_points[-1] - h * f1(y1, y2)) - y1 * (y1_points[-1] - y1_points[-2] - h * f1(y1, y2))) / \
                        ((y1 - y1_points[-1] - h * f1(y1, y2)) - (y1_points[-1] - y1_points[-2] - h * f1(y1, y2)))
            correct2 = (y2_points[-1] * (y2 - y2_points[-1] - h * f2(y1, y2)) - y2 * (y2_points[-1] - y2_points[-2] - h * f2(y1, y2))) / \
                        ((y2 - y2_points[-1] - h * f2(y1, y2)) - (y2_points[-1] - y2_points[-2] - h * f2(y1, y2)))
            
            #Stop correcting if the increase was <= tolerance, if the maximum number of iterations is reached, or if the change in error is >= 0
            if(((not(abs(correct1 - y1_points[-1]) > tolerance) or abs(correct1 - y1_points[-1]) >= prevError1) or \
                (not(abs(correct2 - y2_points[-1]) > tolerance) or abs(correct2 - y2_points[-1]) >= prevError2)) or numIters > 50 ):
                tolerance_flag = True
            
            prevError1 = abs(correct1 - y1_points[-1])
            prevError2 = abs(correct2 - y2_points[-1])
            y1_points[-1] = correct1
            y2_points[-1] = correct2
            
    print("EulerPC (h = " + str(h)[:6] + "): " + str(totalIterations))
    return t_points, y1_points, y2_points

colors = ['r', 'g', 'm', 'y', 'c']
steps = 60
t = np.linspace(0,2,200)
fig1 = plt.figure(1, figsize=(10,14))
y1 = 0
y2 = 0

for i in range(3):
    t1, y1, y2 = EulerPC(y1_prime, y2_prime, [0,2], 16, -16, 2/steps, 0.0001) #Solve with Euler PC
    plt.subplot(3,1,1)
    plt.plot(t1, y1, color = colors[i], linewidth=1, label = 'Y1 Euler PC estimate, h = ' + str(2/steps)[:6]) #Plot y1
    plt.subplot(3,1,2)
    plt.plot(t1, y2, color = colors[i], linewidth=1, label = ' Y2 Euler PC estimate h = ' + str(2/steps)[:6]) #Plot y2
    steps *= 2

plt.subplot(3,1,1)
plt.plot(t, 37*e**(-2*t) - 21*e**(-34*t), color = 'b', label = 'actual', linewidth=1) #Plot exact y1
plt.grid()
plt.title('y1(t) = 37*e^(-2*t) - 21*e^(-34*t)')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()
    
plt.subplot(3,1,2)
plt.grid()
plt.plot(t, 5*e**(-2*t) - 21*e**(-34*t), color = 'b', label = 'actual', linewidth=1) #Plot exact y2
plt.title('y2(t) = 5*e^(-2*t) - 21*e^(-34*t)')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Plot odeint solution
sol = odeint(y_prime_sys, [16, -16], t)
plt.subplot(3,1,3)
plt.grid()
plt.plot(t, sol[:,0], color = 'r', linewidth=1, label = 'Y1 odeint estimate')
plt.plot(t, sol[:,1], color = 'm', linewidth=1, label = 'Y2 odeint estimate')
plt.plot(t, 5*e**(-2*t) - 21*e**(-34*t), color = 'b', label = 'Y2 actual', linewidth=1)
plt.plot(t, 37*e**(-2*t) - 21*e**(-34*t), color = 'g', label = 'Y1 actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()
