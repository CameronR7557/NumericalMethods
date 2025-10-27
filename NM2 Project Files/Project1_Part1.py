# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:13:56 2024

@author: Cameron Robinson
"""
from math import e, sin, cos, log, pi
import numpy as np
import matplotlib.pyplot as plt 

"""
Function: y1
Input: x: float or int
Return: function value at x
Purpose: Get value of e^(x^2) at x
"""
def y1(x):
    return e**(x**2)

"""
Function: y2
Input: x: float or int
Return: function value at x
Purpose: Get value of sin(x) at x
"""
def y2(x):
    return sin(x)

"""
Function: y3
Input: x: float or int
Return: function value at x
Purpose: Get value of ln(x^2 + x) at x
"""
def y3(x):
    return log((x**2 + x), e)

"""
Function: FOCFD
Input: func: function
       lower: int or float
       upper: int or float
       steps: int
Return: x and y-value lists for derivative of passed function
Purpose: Calculate the first derivative of a function using center finite difference
"""
def FOCFD(func, lower, upper, steps):   
    x_list = []                      
    y_list = []
    if lower > upper: #Check bounds
        temp = lower
        lower = upper
        upper = temp
    h = (upper - lower) / steps
    x = lower + h
    y_list.append(((-3/2)*func(lower) + 2*func(lower + h) - 0.5*func(lower + 2*h))/h) #Forward finite difference for left bound
    x_list.append(lower)
    while x < upper:
        y_list.append((func(x + h) - func(x - h)) / (2*h)) #Calc centered finite difference for middle points
        x_list.append(x)
        x = x + h        
    y_list.append(((3/2)*func(upper) - 2*func(upper - h) + 0.5*func(upper - 2*h))/h)#Backward finite difference for right bound 
    x_list.append(x)
    return x_list, y_list

"""
Function: RichardsonMethod
Input: func: function
       lower: int or float
       upper: int or float
       steps: int
Return: x and y-value lists for derivative of passed function
Purpose: Calculate the first derivative of a function using Richardson's Method
"""
def RichardsonMethod(func, lower, upper, steps):
    x_list = []                      
    y_list = []
    if lower > upper: #Check bounds
        temp = lower
        lower = upper
        upper = temp
    h = (upper - lower) / steps
    x = lower + h
    y_list.append(((-3/2)*func(lower) + 2*func(lower + h) - 0.5*func(lower + 2*h))/h) #Forward finite difference for left bound
    x_list.append(lower)
    while x < upper:                           #Loop until a step before final point and calc first derivative
        y_list.append((4/3) * (((func(x + (h/2)) - func(x - (h/2))) / h)) - (1/3)*((func(x + h) - func(x - h)) / (2*h)))
        x_list.append(x)
        x = x + h
    y_list.append(((3/2)*func(upper) - 2*func(upper - h) + 0.5*func(upper - 2*h))/h)#Backward finite difference for right bound 
    x_list.append(x)
    return x_list, y_list

"""
Function: SOCFD
Input: func: function
       lower: int or float
       upper: int or float
       steps: int
Return: x and y-value lists for 2nd derivative of passed function
Purpose: Calculate the second derivative of a function using center finite difference
"""
def SOCFD(func, lower, upper, steps):
    x_list = []                       
    y_list = []
    if lower > upper: #Check bounds
        temp = lower
        lower = upper
        upper = temp
    h = (upper - lower) / steps
    x = lower + h
    y_list.append((2*func(lower) - 5*func(lower + h) + 4*func(lower + 2*h) - func(lower + 3*h))/h**2)#Forward difference for first point
    x_list.append(lower)
    while x < upper:                                                 #Loop until one step before final point
        y_list.append((func(x + h) + func(x - h) - 2*func(x)) / h**2)#Calc 2nd derivative for x
        x_list.append(x)
        x = x + h  
    y_list.append((2*func(upper) - 5*func(upper - h) + 4*func(upper - 2*h) - func(upper - 3*h))/h**2)#Backward difference for last point
    x_list.append(x)
    return x_list, y_list

"""
Function: DoubleDerivative
Input: func: function
       lower: int or float
       upper: int or float
       steps: int
Return: x and y-value lists for second derivative of passed function
Purpose: Calculate the 2nd derivative of a function by performing two first-order derivative methods.
         Performs Richardson's for first derivative, and then FOCFD for 2nd derivative
"""
def DoubleDerivative(func, lower, upper, steps):
    first_y = []
    x_list = []                       
    y_list = []
    if lower > upper: #Check bounds
        temp = lower
        lower = upper
        upper = temp
    h = (upper - lower) / steps
    x_list, first_y = RichardsonMethod(func, lower, upper, steps) #Calc first derivative
    #Calc second derivative
    y_list.append(((-3/2)*first_y[0] + 2*first_y[1] - 0.5*first_y[2])/h)#Forward difference for first point
    for i in range(1, (len(first_y) - 1)):
        y_list.append(((first_y[i + 1] - first_y[i - 1]) / (2*h)))
    y_list.append(((3/2)*first_y[-1] - 2*first_y[-2] + 0.5*first_y[-3])/h)#Backward difference for last point
    return x_list, y_list

#Initialize graphing varaibles
x1 = np.linspace(-1.5,1.5,50)
y1_points = []

x2 = np.linspace(pi/2,7*pi/2,50)
y2_points = []

x3 = np.linspace(1,5*e,50)
y3_points = []

#The Rest of the code is copy-paste graphing for first and second derivatives of y1, y2, and y3
#Figures 1 - 3 are first derivatives and Figures 4 - 6 are second derivatives
#---------------------- FIRST-ORDER ------------------------------------------------------------------------------
#Calculate derivatives for y1 with steps of 10, 30, and 100
x1_points, y1_points = FOCFD(y1, -1.5, 1.5, 10)

x2_points, y2_points = FOCFD(y1, -1.5, 1.5, 30)

x3_points, y3_points = FOCFD(y1, -1.5, 1.5, 100)

fig1 = plt.figure(1, figsize=(10,10))   #Create figure for y1 graphs

#Subplot for y1 with 10 steps
plt.subplot(3,1,1)                                                              
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'FOCFD estimate')
plt.plot(x1, 2*x1*(e**(x1**2)), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = RichardsonMethod(y1, -1.5, 1.5, 10)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'Richardson estimate')
plt.title('y1 Derivative Estimate Graph (Steps = 10)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y1 with 30 steps
plt.subplot(3,1,2)
plt.grid()
plt.plot(x2_points, y2_points, color = 'r', linewidth=1, label = 'FOCFD estimate')
plt.plot(x1, 2*x1*(e**(x1**2)), color = 'b', label = 'Actual', linewidth=1)
x2_points, y2_points = RichardsonMethod(y1, -1.5, 1.5, 30)
plt.plot(x2_points, y2_points, color = 'g', linewidth=1, label = 'Richardson estimate')
plt.title('y1 Derivative Estimate Graph (Steps = 30)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y1 with 100 steps
plt.subplot(3,1,3)
plt.grid()
plt.plot(x3_points, y3_points, color = 'r', linewidth=1, label = 'FOCFD estimate')
plt.plot(x1, 2*x1*(e**(x1**2)), color = 'b', label = 'Actual', linewidth=1)
x3_points, y3_points = RichardsonMethod(y1, -1.5, 1.5, 100)
plt.plot(x3_points, y3_points, color = 'g', linewidth=1, label = 'Richardson estimate')
plt.title('y1 Derivative Estimate Graph (Steps = 100)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#--------------------------------------------------------------------------------------------

#Calculate derivatives for y2 with steps of 10, 30, and 100
x1_points, y1_points = FOCFD(y2, pi/2, (7*pi)/2, 10)

x2_points, y2_points = FOCFD(y2, pi/2, (7*pi)/2, 30)

x3_points, y3_points = FOCFD(y2, pi/2, (7*pi)/2, 100)

fig2 = plt.figure(2, figsize=(10,10))   #Create figure for y2 graphs

#Subplot for y2 with 10 steps
plt.subplot(3,1,1)
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'FOCFD estimate')
plt.plot(x2, np.cos(x2), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = RichardsonMethod(y2, pi/2, (7*pi)/2, 10)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'Richardson estimate')
plt.title('y2 Derivative Estimate Graph (Steps = 10)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y2 with 30 steps
plt.subplot(3,1,2)
plt.grid()
plt.plot(x2_points, y2_points, color = 'r', linewidth=1, label = 'FOCFD estimate')
plt.plot(x2, np.cos(x2), color = 'b', label = 'Actual', linewidth=1)
x2_points, y2_points = RichardsonMethod(y2, pi/2, (7*pi)/2, 30)
plt.plot(x2_points, y2_points, color = 'g', linewidth=1, label = 'Richardson estimate')
plt.title('y2 Derivative Estimate Graph (Steps = 30)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y2 with 100 steps
plt.subplot(3,1,3)
plt.grid()
plt.plot(x3_points, y3_points, color = 'r', linewidth=1, label = 'FOCFD estimate')
plt.plot(x2, np.cos(x2), color = 'b', label = 'Actual', linewidth=1)
x3_points, y3_points = RichardsonMethod(y2, pi/2, (7*pi)/2, 100)
plt.plot(x3_points, y3_points, color = 'g', linewidth=1, label = 'Richardson estimate')
plt.title('y2 Derivative Estimate Graph (Steps = 100)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#--------------------------------------------------------------------------------------------

#Calculate derivative for y3 with 10, 30 and 100 steps
x1_points, y1_points = FOCFD(y3, 1, 5*e, 10)

x2_points, y2_points = FOCFD(y3, 1, 5*e, 30)

x3_points, y3_points = FOCFD(y3, 1, 5*e, 100)

fig3 = plt.figure(3, figsize=(10,10)) #Create figure for y3 graphs

#Subplot for y3 with 10 steps
plt.subplot(3,1,1)
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'FOCFD estimate')
plt.plot(x3, (2*x3 + 1) / (x3**2 + x3), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = RichardsonMethod(y3, 1, 5*e, 10)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'Richardson estimate')
plt.title('y3 Derivative Estimate Graph (Steps = 10)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y3 with 30 steps
plt.subplot(3,1,2)
plt.grid()
plt.plot(x2_points, y2_points, color = 'r', linewidth=1, label = 'FOCFD estimate')
plt.plot(x3, (2*x3 + 1) / (x3**2 + x3), color = 'b', label = 'Actual', linewidth=1)
x2_points, y2_points = RichardsonMethod(y3, 1, 5*e, 30)
plt.plot(x2_points, y2_points, color = 'g', linewidth=1, label = 'Richardson estimate')
plt.title('y3 Derivative Estimate Graph (Steps = 30)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y3 with 100 steps
plt.subplot(3,1,3)
plt.grid()
plt.plot(x3_points, y3_points, color = 'r', linewidth=1, label = 'FOCFD estimate')
plt.plot(x3, (2*x3 + 1) / (x3**2 + x3), color = 'b', label = 'Actual', linewidth=1)
x3_points, y3_points = RichardsonMethod(y3, 1, 5*e, 100)
plt.plot(x3_points, y3_points, color = 'g', linewidth=1, label = 'Richardson estimate')
plt.title('y3 Derivative Estimate Graph (Steps = 100)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#---------------------- SECOND ORDER -------------------------------------------------------------------------------
fig4 = plt.figure(4, figsize=(10,10))   #Create figure for y1 graphs

#Calculate derivatives for y1 with steps of 10, 30, and 100
x1_points, y1_points = SOCFD(y1, -1.5, 1.5, 10)

x2_points, y2_points = SOCFD(y1, -1.5, 1.5, 30)

x3_points, y3_points = SOCFD(y1, -1.5, 1.5, 100)

#Subplot for y1'' with 10 steps
plt.subplot(3,1,1)                                                              
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'SOCFD estimate')
plt.plot(x1, 2*(e**(x1**2)) * (2*(x1**2) + 1), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = DoubleDerivative(y1, -1.5, 1.5, 10)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'DoubleDerivative estimate')
plt.title('y1 2nd Derivative Estimate Graph (Steps = 10)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y1'' with 30 steps
x1_points, y1_points = SOCFD(y1, -1.5, 1.5, 30)
plt.subplot(3,1,2)                                                              
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'SOCFD estimate')
plt.plot(x1, 2*(e**(x1**2)) * (2*(x1**2) + 1), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = DoubleDerivative(y1, -1.5, 1.5, 30)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'DoubleDerivative estimate')
plt.title('y1 2nd Derivative Estimate Graph (Steps = 30)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y1'' with 100 steps
x1_points, y1_points = SOCFD(y1, -1.5, 1.5, 100)
plt.subplot(3,1,3)                                                              
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'SOCFD estimate')
plt.plot(x1, 2*(e**(x1**2)) * (2*(x1**2) + 1), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = DoubleDerivative(y1, -1.5, 1.5, 100)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'DoubleDerivative estimate')
plt.title('y1 2nd Derivative Estimate Graph (Steps = 100)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#--------------------------------------------------------------------------------------------

#Plotting 2nd Derivative of y2
fig5 = plt.figure(5, figsize=(10,10))   #Create figure for y2'' graphs
#Subplot for y2'' with 10 steps
x1_points, y1_points = SOCFD(y2, pi/2, (7*pi)/2, 10)
plt.subplot(3,1,1)
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'SOCFD estimate')
plt.plot(x2, -np.sin(x2), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = DoubleDerivative(y2, pi/2, (7*pi)/2, 10)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'DoubleDerivative estimate')
plt.title('y2 2nd Derivative Estimate Graph (Steps = 10)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y2'' with 30 steps
x1_points, y1_points = SOCFD(y2, pi/2, (7*pi)/2, 30)
plt.subplot(3,1,2)
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'SOCFD estimate')
plt.plot(x2, -np.sin(x2), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = DoubleDerivative(y2, pi/2, (7*pi)/2, 30)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'DoubleDerivative estimate')
plt.title('y2 2nd Derivative Estimate Graph (Steps = 30)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y2'' with 100 steps
x1_points, y1_points = SOCFD(y2, pi/2, (7*pi)/2, 100)
plt.subplot(3,1,3)
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'SOCFD estimate')
plt.plot(x2, -np.sin(x2), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = DoubleDerivative(y2, pi/2, (7*pi)/2, 100)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'DoubleDerivative estimate')
plt.title('y2 2nd Derivative Estimate Graph (Steps = 100)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#--------------------------------------------------------------------------------------------
#Plotting 2nd Derivative for y3
fig6 = plt.figure(6, figsize=(10,10))   #Create figure for y3'' graphs

#Subplot for y3'' with 10 steps
x1_points, y1_points = SOCFD(y3, 1, 5*e, 10)
plt.subplot(3,1,1)
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'SOCFD estimate')
plt.plot(x3, (-2*x3**2 - 2*x3 - 1)/((x3**2 + x3)**2), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = DoubleDerivative(y3, 1, 5*e, 10)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'DoubleDerivative estimate')
plt.title('y3 2nd Derivative Estimate Graph (Steps = 10)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y3'' with 30 steps
x1_points, y1_points = SOCFD(y3, 1, 5*e, 30)
plt.subplot(3,1,2)
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'SOCFD estimate')
plt.plot(x3, (-2*x3**2 - 2*x3 - 1)/((x3**2 + x3)**2), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = DoubleDerivative(y3, 1, 5*e, 30)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'DoubleDerivative estimate')
plt.title('y3 2nd Derivative Estimate Graph (Steps = 30)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#Subplot for y3'' with 100 steps
x1_points, y1_points = SOCFD(y3, 1, 5*e, 100)
plt.subplot(3,1,3)
plt.grid()
plt.plot(x1_points, y1_points, color = 'r', linewidth=1, label = 'SOCFD estimate')
plt.plot(x3, (-2*x3**2 - 2*x3 - 1)/((x3**2 + x3)**2), color = 'b', label = 'Actual', linewidth=1)
x1_points, y1_points = DoubleDerivative(y3, 1, 5*e, 100)
plt.plot(x1_points, y1_points, color = 'g', linewidth=1, label = 'DoubleDerivative estimate')
plt.title('y3 2nd Derivative Estimate Graph (Steps = 100)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

#x1_points, y1_points = SOCFD(y2, pi/2, (7*pi)/2, 10)
#x2_points, y2_points = SOCFD(y2, pi/2, (7*pi)/2, 20)

#For checking error:
# print("steps = 10:")
# for i in range(0, len(y1_points)):
#     print(str(-np.sin(x1_points[i]) - y1_points[i]))
# print("steps = 20:")
# for i in range(0, len(y2_points)):
#     print(str(-np.sin(x2_points[i]) - y2_points[i]))


