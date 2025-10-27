# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:10:11 2024

@author: Cameron Robinson
"""

from math import e, log
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.integrate import odeint

"""
Function: f_1
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point for Taylor's Method
"""
def f_1(t, y):
    return y + 2*t*e**(2*t)                 #Return slope of differential equation

"""
Function: f_prime1
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point for Taylor's Method
"""
def f_prime1(t, y):
    return y + 6*t*e**(2*t) + 2*e**(2*t)                    #Return slope of differential equation

"""
Function: f_2prime1
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point for Taylor's Method
"""
def f_2prime1(t, y):
    return y + 14*t*e**(2*t) + 6*e**(2*t)                    #Return slope of differential equation

"""
Function: TaylorsMethod'
Input: f: function
       fprime: function
       f2prime: function
       bounds: list
       y0: int or float
       h: int or float
Return: Estimated value of f at xf
Purpose: Use Taylors's method to estimate an ODE and return the x and y points
"""
def TaylorsMethod(f, fprime, f2prime, bounds, y0, h):
    if bounds[0] > bounds[1]:
        temp = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = temp
    y_points = [y0]                                                       #Initialize Y values list
    t_points = [bounds[0]]                                                #Initialize T values list
    while t_points[-1] < (bounds[1] - 0.0000000000001):                              #Account for any rounding error    
        y_points.append(y_points[-1] + h*f(t_points[-1], y_points[-1]) + (h**2)*0.5*fprime(t_points[-1], y_points[-1]) + ((h**3)*f2prime(t_points[-1], y_points[-1]))/6)#Gets most recent y-value and adds the derivative at most recent t-value to it
        t_points.append(t_points[-1] + h)                                 #Gets most recent t-value and increments it by h
    return t_points, y_points                                             #Return the estimated values

#Graph Taylor's Method with h = 0.5
t1, y1 = TaylorsMethod(f_1, f_prime1, f_2prime1, [0,1], 3, 0.5)
t = np.linspace(0,1,50)

fig1 = plt.figure(1, figsize=(10,10))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (2*t*e**(2*t) - 2*e**(2*t) + 5*e**t), color = 'b', label = 'actual', linewidth=1)
plt.title('Taylor\'s Method h = 0.5')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph Taylor's Method with h = 0.1
t1, y1 = TaylorsMethod(f_1, f_prime1, f_2prime1, [0,1], 3, 0.1)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (2*t*e**(2*t) - 2*e**(2*t) + 5*e**t), color = 'b', label = 'actual', linewidth=1)
plt.title('Taylor\'s Method h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph Taylor's Method with h = 0.01
# t1, y1 = TaylorsMethod(f_1, f_prime1, f_2prime1, [0,1], 3, 0.01)
# plt.subplot(4,1,3)
# plt.grid()
# plt.plot(t1, y1, color = 'r', linewidth=1, label = 'estimate')
# plt.plot(t, (2*t*e**(2*t) - 2*e**(2*t) + 5*e**t), color = 'b', label = 'actual', linewidth=1)
# plt.title('Taylor\'s Method h = 0.01')
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

#-------------------------------------------------------------------
#                              PART B
#-------------------------------------------------------------------

"""
Function: f_2
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point for Taylor's Method'
"""
def f_2(t, y):
    dydt = 2*t - 3*y + 1
    return dydt               #Return slope of differential equation

"""
Function: f_prime2
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point for Taylor's Method
"""
def f_prime2(t, y):
    return 9*y - 6*t -1                    #Return slope of differential equation

"""
Function: f_2prime2
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point for Taylor's Method
"""
def f_2prime2(t, y):
    return -27*y + 18*t + 3                    #Return slope of differential equation


#Graphing for part b
#Graph Taylor's Method with h = 0.5
t1, y1 = TaylorsMethod(f_2, f_prime2, f_2prime2, [1,2], 5, 0.5)
t = np.linspace(1,2,50)
fig1 = plt.figure(2, figsize=(10,10))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
plt.title('Taylor\'s Method h = 0.5')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph Taylor's Method with h = 0.2
t1, y1 = TaylorsMethod(f_2, f_prime2, f_2prime2, [1,2], 5, 0.1)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
plt.title('Taylor\'s Method h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

# #Graph Taylor's Method with h = 0.01
# t1, y1 = TaylorsMethod(f_2, f_prime2, f_2prime2, [1,2], 5, 0.01)
# plt.subplot(4,1,3)
# plt.grid()
# plt.plot(t1, y1, color = 'r', linewidth=1, label = 'estimate')
# plt.plot(t, (1/9) + (2/3)*t + (38/9)*e**(3 - 3*t), color = 'b', label = 'actual', linewidth=1)
# plt.title('Taylor\'s Method h = 0.01')
# plt.xlabel('T')
# plt.ylabel('Y')
# plt.legend()

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