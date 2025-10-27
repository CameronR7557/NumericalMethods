# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:35:25 2024

@author: Cameron Robinson
"""
from math import e, sin, cos, log, pi
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import quad
from scipy.special import erfi

def y1(x):
    return e**(x**2)

def y2(x):
    return sin(x)

def y3(x):
    return log((x**2 + x), e)

def TrapezoidIntegralMethod(func, lower, upper, steps):
    if lower > upper:                                       #Switch bounds if out of order
        temp = lower
        lower = upper
        upper = temp
    h = (upper - lower) / steps                             #Calculate step size (change in x)
    x = lower + h
    area = 0
    while x < upper:                                        #Sum heights of trapezoids    
        area = area + (func(x) + func(x + h))/2
        x = x + h
    area = area * h                                         #Multiply sum of trapezoid heights by delta-x
    area = area + (h/2) * (func(lower) + func(upper))       #Add area of endpoints
    return area

def MidpointIntegralMethod(func, lower, upper, steps):
    if lower > upper:                                       #Switch bounds if out of order
        temp = lower
        lower = upper
        upper = temp
    h = (upper - lower) / steps                             #Calculate step size (change in x)
    x = lower + h/2
    area = 0
    while x < (upper - h/2):                                #Loop through all midpoints and sum heights
        area = area + func(x)
        x = x + h
    area = area * h                                         #Multiply by width to get area
    return area

#Actual Integrals 
Y1Area = 0.5*((pi**(0.5)) * erfi(1.5) - (pi**(0.5)) * erfi(-1.5))
Y2Area = 0
Y3Area = 5*e*log(25*e**2 + 5*e,e) - 10*e + log(5*e + 1,e) - 2*log(2,e) + 2 

#Y1
#Calc integrals for trapezoid, midpoint, and an internal method
areaTrapezoid = TrapezoidIntegralMethod(y1, -1.5, 1.5, 1000000)
areaMidpoint = MidpointIntegralMethod(y1, -1.5, 1.5, 1000000)
internalMethod = quad(y1, -1.5, 1.5)
#Print calculated values and error
print("y1 Integrals:")
print("Actual: " + str(Y1Area))
print("Trapezoid: " + str(areaTrapezoid) + "\r\nMidpoint: " + str(areaMidpoint) + "\r\nInternal: " + str(internalMethod[0]))
print("Error Trapezoid: " + str(Y1Area - areaTrapezoid) + "\r\nError Midpoint: " + str(Y1Area - areaMidpoint) + "\r\nError Internal: " + str(Y1Area - internalMethod[0]))

#Y2
#Calc integrals for trapezoid, midpoint, and an internal method
areaTrapezoid = TrapezoidIntegralMethod(y2, pi/2, (7*pi)/2, 100000000)
areaMidpoint = MidpointIntegralMethod(y2, pi/2, (7*pi)/2, 100000)
internalMethod = quad(y2, pi/2, (7*pi)/2)
#Print calculated values and error
print("\r\ny2 Integrals:")
print("Actual: " + str(Y2Area))
print("Trapezoid: " + str(areaTrapezoid) + "\r\nMidpoint: " + str(areaMidpoint) + "\r\nInternal: " + str(internalMethod[0]))
print("Error Trapezoid: " + str(Y2Area - areaTrapezoid) + "\r\nError Midpoint: " + str(Y2Area - areaMidpoint) + "\r\nError Internal: " + str(Y2Area - internalMethod[0]))

#Y3
#Calc integrals for trapezoid, midpoint, and an internal method
areaTrapezoid = TrapezoidIntegralMethod(y3, 1, 5*e, 100000000)
areaMidpoint = MidpointIntegralMethod(y3, 1, 5*e, 1000000)
internalMethod = quad(y3, 1, 5*e)
#Print calculated values and error
print("\r\ny3 Integrals:")
print("Actual: " + str(Y3Area))
print("Trapezoid: " + str(areaTrapezoid) + "\r\nMidpoint: " + str(areaMidpoint) + "\r\nInternal: " + str(internalMethod[0]))
print("Error Trapezoid: " + str(Y3Area - areaTrapezoid) + "\r\nError Midpoint: " + str(Y3Area - areaMidpoint) + "\r\nError Internal: " + str(Y3Area - internalMethod[0]))
