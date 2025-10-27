# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 13:17:40 2024

@author: Cameron Robinson
"""

from math import e, log
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

"""
Function: f_1
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point
"""
def f_1(t, y):
    return y + 2*t*e**(2*t)                 #Return slope of differential equation

"""
Function: f_2
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point
"""
def f_2(t, y):
    return 2*t - 3*y + 1                 #Return slope of differential equation

"""
Function: f_3
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point
"""
def f_3(t, y):
    return -y*e**t               #Return slope of differential equation

"""
Function: f_4
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point
"""
def f_4(t, y):
    return -np.cos(t)*y               #Return slope of differential equation

"""
Function: AB4
Input: dydt: function
       bounds: list
       y0: int or float
       h: int or float
Return: Estimated value of f at xf
Purpose: Use AB4 method to estimate an ODE and return the x and y points
"""
def AB4(f, bounds, y0, h):
    count = 0
    if bounds[0] > bounds[1]:
        temp = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = temp
    y_points = [y0]                                                       #Initialize Y values list
    t_points = [bounds[0]]                                                #Initialize T values list
    while t_points[-1] < (bounds[1] - 0.0000000000001):                              #Account for any rounding error  
        if (t_points[-1] + h > bounds[1]):                                               #Clamp t in case h does not divide (b - a)
            h = abs(bounds[1] - t_points[-1])
            
        if(count < 3):                                                    #Compute RK4 for first 4 points
            k1 = f(t_points[-1], y_points[-1])
            k2 = f(t_points[-1] + 0.5*h, y_points[-1] + 0.5*h*k1)
            k3 = f(t_points[-1] + 0.5*h, y_points[-1] + 0.5*h*k2)
            k4 = f(t_points[-1] + h, y_points[-1] + h*k3)
            y_points.append(y_points[-1] + h*(k1 + 2*k2 + 2*k3 + k4)/6)
        else: #Used AB4 for the rest of the points
            print(t_points[-1] + h)#Make sure it is actually using AB4
            y_points.append(y_points[-1] + (h/24)*(55*f(t_points[-1], y_points[-1]) - 59*f(t_points[-2], y_points[-2]) + 37*f(t_points[-3], y_points[-3]) - 9*f(t_points[-4], y_points[-4])))
        t_points.append(t_points[-1] + h)                                 #Gets most recent t-value and increments it by h
        count = count + 1
    return t_points, y_points                                             #Return the estimated values


#-------------------- EQN 1 ------------------------
t = np.linspace(0,1,50)
fig1 = plt.figure(1, figsize=(10,10))
t1, y1 = AB4(f_1, [0,1], 3, 0.2)
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'AB4 estimate')
plt.plot(t, (2*t*e**(2*t) - 2*e**(2*t) + 5*e**t), color = 'b', label = 'actual', linewidth=1)
plt.title('AB4 Method h = 0.2')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

t1, y1 = AB4(f_1, [0,1], 3, 0.1)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'AB4 estimate')
plt.plot(t, (2*t*e**(2*t) - 2*e**(2*t) + 5*e**t), color = 'b', label = 'actual', linewidth=1)
plt.title('AB4 Method h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint
sol = odeint(f_1, 3, t, tfirst=True)
plt.subplot(3,1,3)
plt.grid()
plt.plot(t, sol, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (2*t*e**(2*t) - 2*e**(2*t) + 5*e**t), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#-------------------- EQN 2 ------------------------
fig2 = plt.figure(2, figsize=(10,10))
t = np.linspace(1,2,50)
t1, y1 = AB4(f_2, [1,2], 5, 0.2)
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'AB4 estimate')
plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
plt.title('AB4 Method h = 0.2')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

t1, y1 = AB4(f_2, [1,2], 5, 0.1)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'AB4 estimate')
plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
plt.title('AB4 Method h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint
sol = odeint(f_2, 5, t, tfirst=True)
plt.subplot(3,1,3)
plt.grid()
plt.plot(t, sol, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()
#-------------------- EQN 3 ------------------------
fig3 = plt.figure(3, figsize=(10,10))
t = np.linspace(0,1,50)
t1, y1 = AB4(f_3, [0,1], 3, 0.2)
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'AB4 estimate')
plt.plot(t, 3*e**(-e**t + 1), color = 'b', label = 'actual', linewidth=1)
plt.title('AB4 Method h = 0.2')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

t1, y1 = AB4(f_3, [0,1], 3, 0.1)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'AB4 estimate')
plt.plot(t, 3*e**(-e**t + 1), color = 'b', label = 'actual', linewidth=1)
plt.title('AB4 Method h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint
sol = odeint(f_3, 3, t, tfirst=True)
plt.subplot(3,1,3)
plt.grid()
plt.plot(t, sol, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, 3*e**(-e**t + 1), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()
#-------------------- EQN 4 ------------------------
fig4 = plt.figure(4, figsize=(10,10))
t = np.linspace(0,np.pi/3,50)
t1, y1 = AB4(f_4, [0,np.pi/3], 2, 0.2)
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'AB4 estimate')
plt.plot(t, 2*e**(-np.sin(t)), color = 'b', label = 'actual', linewidth=1)
plt.title('AB4 Method h = 0.2')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

t1, y1 = AB4(f_4, [0,np.pi/3], 2, 0.1)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'AB4 estimate')
plt.plot(t, 2*e**(-np.sin(t)), color = 'b', label = 'actual', linewidth=1)
plt.title('AB4 Method h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph odeint
sol = odeint(f_4, 2, t, tfirst=True)
plt.subplot(3,1,3)
plt.grid()
plt.plot(t, sol, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, 2*e**(-np.sin(t)), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()
