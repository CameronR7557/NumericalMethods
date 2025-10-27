from math import e
import numpy as np
from scipy.integrate import odeint

"""
Method to iteratively solving for largest eigenvalue and its corresponding eigenvector
"""
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