# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:08:05 2024

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
Function: RK2_ModifiedEuler
Input: dydt: function
       bounds: list
       y0: int or float
       h: int or float
Return: Estimated value of f at xf
Purpose: Use RK2_ModifiedEuler method to estimate an ODE and return the x and y points
"""
def RK2_ModifiedEuler(f, bounds, y0, h):
    if bounds[0] > bounds[1]:
        temp = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = temp
    y_points = [y0]                                                       #Initialize Y values list
    t_points = [bounds[0]]                                                #Initialize T values list
    t = bounds[0]
    while t < (bounds[1] - 0.0000000000001):                              #Account for any rounding error  
        t = t + h
        if (t > bounds[1]):                                               #Clamp t in case h does not divide (b - a)
            t = bounds[1]    
        k1 = f(t_points[-1], y_points[-1])
        k2 = f(t_points[-1] + h, y_points[-1] + h*k1)
        y_points.append(y_points[-1] + 0.5*h*(k1 + k2))#Gets most recent y-value and adds the derivative at most recent t-value to it
        t_points.append(t_points[-1] + h)                                 #Gets most recent t-value and increments it by h
    return t_points, y_points                                             #Return the estimated values

"""
Function: RK2_Midpoint
Input: dydt: function
       bounds: list
       y0: int or float
       h: int or float
Return: Estimated value of f at xf
Purpose: Use RK2_Midpoint method to estimate an ODE and return the x and y points
"""
def RK2_Midpoint(f, bounds, y0, h):
    if bounds[0] > bounds[1]:
        temp = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = temp
    y_points = [y0]                                                       #Initialize Y values list
    t_points = [bounds[0]]                                                #Initialize T values list
    while t_points[-1] < (bounds[1] - 0.0000000000001):                              #Account for any rounding error     
        k1 = f(t_points[-1], y_points[-1])
        k2 = f(t_points[-1] + 0.5*h, y_points[-1] + h*k1*0.5)
        y_points.append(y_points[-1] + h*k2)
        t_points.append(t_points[-1] + h)                                 #Gets most recent t-value and increments it by h
    return t_points, y_points     

t = np.linspace(0,1,50)
fig1 = plt.figure(1, figsize=(10,10))
t1, y1 = RK2_ModifiedEuler(f_1, [0,1], 3, 0.5)
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'RK2 Modified Euler estimate')
t1, y1 = RK2_Midpoint(f_1, [0,1], 3, 0.5)
plt.plot(t1, y1, color = 'g', linewidth=1, label = 'RK2 Midpoint estimate')
plt.plot(t, (2*t*e**(2*t) - 2*e**(2*t) + 5*e**t), color = 'b', label = 'actual', linewidth=1)
plt.title('RK2 Methods h = 0.5')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

t1, y1 = RK2_ModifiedEuler(f_1, [0,1], 3, 0.1)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'm', linewidth=1, label = 'RK2 Modified Euler estimate')
t1, y1 = RK2_Midpoint(f_1, [0,1], 3, 0.1)
plt.plot(t1, y1, color = 'g', linewidth=1, label = 'RK2 Midpoint estimate')
plt.plot(t, (2*t*e**(2*t) - 2*e**(2*t) + 5*e**t), color = 'b', label = 'actual', linewidth=1)
plt.title('RK2 Methods h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

# t1, y1 = RK2_ModifiedEuler(f_1, [0,1], 3, 0.01)
# plt.subplot(4,1,3)
# plt.grid()
# plt.plot(t1, y1, color = 'r', linewidth=1, label = 'RK2 Modified Euler estimate')
# t1, y1 = RK2_Midpoint(f_1, [0,1], 3, 0.01)
# plt.plot(t1, y1, color = 'g', linewidth=1, label = 'RK2 Midpoint estimate')
# plt.plot(t, (2*t*e**(2*t) - 2*e**(2*t) + 5*e**t), color = 'b', label = 'actual', linewidth=1)
# plt.title('RK2 Methods h = 0.01')
# plt.xlabel('T')
# plt.ylabel('Y')
# plt.legend()

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

#-------------------- Function 2 -------------------------
fig2 = plt.figure(2, figsize=(10,10))
t = np.linspace(1,2,50)
t1, y1 = RK2_ModifiedEuler(f_2, [1,2], 5, 0.5)
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'RK2 Modified Euler estimate')
t1, y1 = RK2_Midpoint(f_2, [1,2], 5, 0.5)
plt.plot(t1, y1, color = 'g', linewidth=1, label = 'RK2 Midpoint estimate')
plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
plt.title('RK2 Methods h = 0.5')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

t1, y1 = RK2_ModifiedEuler(f_2, [1,2], 5, 0.1)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'm', linewidth=1, label = 'RK2 Modified Euler estimate')
t1, y1 = RK2_Midpoint(f_2, [1,2], 5, 0.1)
plt.plot(t1, y1, color = 'g', linewidth=1, label = 'RK2 Midpoint estimate')
plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
plt.title('RK2 Methods h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

# t1, y1 = RK2_ModifiedEuler(f_2, [1,2], 5, 0.01)
# plt.subplot(4,1,3)
# plt.grid()
# plt.plot(t1, y1, color = 'r', linewidth=1, label = 'RK2 Modified Euler estimate')
# t1, y1 = RK2_Midpoint(f_2, [1,2], 5, 0.01)
# plt.plot(t1, y1, color = 'g', linewidth=1, label = 'RK2 Midpoint estimate')
# plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
# plt.title('RK2 Methods h = 0.01')
# plt.xlabel('T')
# plt.ylabel('Y')
# plt.legend()

#Graph odeint
sol = odeint(f_2, 5, t, tfirst=True)
plt.subplot(4,1,4)
plt.grid()
plt.plot(t, sol, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
plt.title('odeint Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#-------------------- Function 3 -------------------------
fig3 = plt.figure(3, figsize=(10,10))
t = np.linspace(0,1,50)
t1, y1 = RK2_ModifiedEuler(f_3, [0,1], 3, 0.5)
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'RK2 Modified Euler estimate')
t1, y1 = RK2_Midpoint(f_3, [0,1], 3, 0.5)
plt.plot(t1, y1, color = 'g', linewidth=1, label = 'RK2 Midpoint estimate')
plt.plot(t, 3*e**(-e**t + 1), color = 'b', label = 'actual', linewidth=1)
plt.title('RK2 Methods h = 0.5')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

t1, y1 = RK2_ModifiedEuler(f_3, [0,1], 3, 0.1)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'm', linewidth=1, label = 'RK2 Modified Euler estimate')
t1, y1 = RK2_Midpoint(f_3, [0,1], 3, 0.1)
plt.plot(t1, y1, color = 'g', linewidth=1, label = 'RK2 Midpoint estimate')
plt.plot(t, 3*e**(-e**t + 1), color = 'b', label = 'actual', linewidth=1)
plt.title('RK2 Methods h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

# t1, y1 = RK2_ModifiedEuler(f_3, [0,1], 3, 0.01)
# plt.subplot(4,1,3)
# plt.grid()
# plt.plot(t1, y1, color = 'r', linewidth=1, label = 'RK2 Modified Euler estimate')
# t1, y1 = RK2_Midpoint(f_3, [0,1], 3, 0.01)
# plt.plot(t1, y1, color = 'g', linewidth=1, label = 'RK2 Midpoint estimate')
# plt.plot(t, 3*e**(-e**t + 1), color = 'b', label = 'actual', linewidth=1)
# plt.title('RK2 Methods h = 0.01')
# plt.xlabel('T')
# plt.ylabel('Y')
# plt.legend()

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