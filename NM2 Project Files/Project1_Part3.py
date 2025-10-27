# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:10:11 2024

@author: Cameron Robinson
"""

from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

"""
Function: dxdy
Input: t: int or float
       y: int or float
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point
"""
def dydt(t, y):
    return -3*y*np.sin(t)                    #Return slope of differential equation

"""
Function: fun
Input: t: int or float
       y: list
Return: Returns the y-value of an ODE for a specified t and y
Purpose: Calculate value of an ODE at a specified point
"""
def fun(t, y):
    y1 = [-3*y[0]*np.sin(t)]
    return y1                    #Return slope of differential equation

"""
Function: EulersMethod'
Input: dydt: function
       bounds: list
       y0: int or float
       h: int or float
Return: Estimated value of f at xf
Purpose: Use Euler's method to estimate an ODE and return the x and y points
"""
def EulersMethod(dydt, bounds, y0, h):
    if bounds[0] > bounds[1]:
        temp = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = temp
    y_points = [y0]                                                       #Initialize Y values list
    t_points = [bounds[0]]                                                #Initialize T values list
    t = bounds[0]
    while t < bounds[1]:   
        t = t + h
        if t > bounds[1]:                                                 #Clamp t in case h does not divide (b - a)
            t = bounds[1]                
        y_points.append(y_points[-1] + h*dydt(t_points[-1], y_points[-1]))#Gets most recent y-value and adds the derivative at most recent t-value to it
        t_points.append(t_points[-1] + h)                                 #Gets most recent t-value and increments it by h
    return t_points, y_points                                             #Return the estimated values


#Graph Euler's method h = 0.1 with actual solution
t1, y1 = EulersMethod(dydt, [0,2], 0.5, 0.1)
t = np.linspace(0,2,50)
fig1 = plt.figure(1, figsize=(10,10))
plt.subplot(3,1,1)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (1/(2*e**3) * e**(3*np.cos(t))), color = 'b', label = 'actual', linewidth=1)
plt.title('Euler\'s Method h = 0.1')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph Euler's method h = 0.01 with actual solution
t1, y1 = EulersMethod(dydt, [2,0], 0.5, 0.01)
plt.subplot(3,1,2)
plt.grid()
plt.plot(t1, y1, color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (1/(2*e**3) * e**(3*np.cos(t))), color = 'b', label = 'actual', linewidth=1)
plt.title('Euler\'s Method h = 0.01')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Graph solve_ivp estimate of ODE solution along with actual solution
sol = solve_ivp(fun, [0,2], [0.5])
plt.subplot(3,1,3)
plt.grid()
plt.plot(sol.t, sol.y[0], color = 'r', linewidth=1, label = 'estimate')
plt.plot(t, (1/(2*e**3) * e**(3*np.cos(t))), color = 'b', label = 'actual', linewidth=1)
plt.title('solve_ivp Result Graph')
plt.xlabel('T')
plt.ylabel('Y')
plt.legend()

#Calculate Errors
t1, y1 = EulersMethod(dydt, [0,2], 0.5, 0.1)
t2, y2 = EulersMethod(dydt, [0,2], 0.5, 0.01)
err1 = 0
err2 = 0
print("h = 0.1 Errors:")
for i in range(len(t1)):
    err1 = ((1/(2*e**3) * e**(3*np.cos(t1[i]))) - y1[i])
    if(not(i % 2)):     #Print every 2nd error
        print(str(err1))

print("h = 0.01 Errors:")
for i in range(len(t2)):
    err2 = ((1/(2*e**3) * e**(3*np.cos(t2[i]))) - y2[i])
    if(not(i % 20)):   #Print every 20th error 
        print(str(err2))
