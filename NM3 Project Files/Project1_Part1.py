# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:14:47 2024

@author: robin
"""
from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint

def PowerMethod(A, x0, tol, max_steps):
    steps = 0
    prevEigenVal = 0
    eigenVal = 0
    #Repeat while improvement is larger than the tolerance, and the steps is less than the max steps
    while (steps == 0 or ((abs(prevEigenVal - eigenVal) > tol) and (steps < max_steps))):
        #Multiply A and current eigenvector
        Ax = np.dot(A, x0)              #x = Av
        x0 = Ax / np.linalg.norm(Ax)    #v = x/|x|
        prevEigenVal = eigenVal
        eigenVal = np.dot(Ax.T, x0)
        steps += 1
    return eigenVal, x0

A = [[2, 1, -1, 3],
     [1, 7, 0, -1],
     [-1, 0, 4, 2],
     [3, -1, -2, 1]]

x = [[1], [0], [0], [0]]
A_inv = np.linalg.inv(A)
print("Built-In: ")
eigs1, eVecs = np.linalg.eig(A)
print(eigs1[-1])
print(eVecs[:, 3])
print("\nEstimated: ")
eigs, eVecs = PowerMethod(A, x, 0.00001, 100)
print(eigs)
print(eVecs)
print("Error: " + str(abs(eigs1[-1] - eigs)[0,0]))

# print(np.linalg.eig(A_inv))
# print(PowerMethod(A_inv, x, 0.00001, 100))