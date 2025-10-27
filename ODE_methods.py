from math import e
import numpy as np

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

def EulersMethod(f1, f2, bounds, y10, y20, h):
    steps = int((bounds[1] - bounds[0]) / h)
    y1 = [y10]
    y2 = [y20]
    t = [bounds[0]]
    tempVal = 0
    for i in range(steps):
        tempVal = y1[-1] + h*f1(t[-1], y1[-1], y2[-1])
        y2.append(y2[-1] + h*f2(t[-1], y1[-1], y2[-1]))
        y1.append(tempVal)
        t.append(t[-1] + h)
    return t, y1, y2

def ShootingMethod(f1, f2, A, B, bounds, h):
    t1, y1, y2 = EulersMethod(f1, f2, bounds, A, -0.2, h)#Get an initial guess
    #prev_x, cur_x, next_x are the guesses for the initial value of y2. Used in root finding
    prev_x = -0.2
    cur_x = -0.1
    g = y2[-1] #g is the most recent estimate of B
    while(abs(B - y1[-1]) > 0.00000001):
        t1, y1, y2 = EulersMethod(f1, f2, bounds, A, cur_x, h)#Get next guess
        #Secant Method: guess values are x, (g - B) and (y22[-1] - b) are f(x) (errors)
        next_x = (prev_x * (y1[-1] - B) - cur_x * (g - B)) / (y1[-1] - g)
        prev_x = cur_x
        cur_x = next_x
        g = y1[-1]
    return t1, y1 ,y2

"""Implicit Euler's method"""
def EulerPC(f, bounds, y0, h, tolerance):
    y_points = [y0]
    t_points = [bounds[0]]
    numIters = 0
    totalIterations = 0
    prevError = -1
    while t_points[-1] < (bounds[1] - 0.0000000000001):                   #Account for any rounding error  
        tolerance_flag = False
        if (t_points[-1] + h > bounds[1]):                                #Clamp t in case h does not divide (b - a)
            h = abs(bounds[1] - t_points[-1])
        #Predict: (Fwd Euler)
        y_points.append(y_points[-1] + h*f(t_points[-1], y_points[-1]))
        t_points.append(t_points[-1] + h)
        #Correct:
        while not tolerance_flag:   #Correct until tolerance flag is set to True (by if-statement below)
            totalIterations += 1
            numIters += 1
            #Implicit Euler, no root finding
            correct = y_points[-2] + h*f(t_points[-1], y_points[-1])
            #Stop correcting if the increase was <= tolerance, if the maximum number of iterations is reached, or if the change in error is >= 0
            if(not(abs(correct - y_points[-1]) > tolerance) or numIters > 1000 or abs(correct - y_points[-1]) >= prevError):
                tolerance_flag = True
            prevError = abs(correct - y_points[-1])
            y_points[-1] = correct
            
    print("EulerPC (h = " + str(h)[:6] + "): " + str(totalIterations))
    print("Solution: " + str(y_points[-1]))
    return t_points, y_points

"""Runge-Kutta 2 Method"""
def RK2_Midpoint(f, bounds, y0, h):
    if bounds[0] > bounds[1]:
        temp = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = temp
    y_points = [y0]        #Initialize Y values list
    t_points = [bounds[0]] #Initialize T values list
    while t_points[-1] < (bounds[1] - 0.0000000000001): #Account for any rounding error     
        k1 = f(t_points[-1], y_points[-1])
        k2 = f(t_points[-1] + 0.5*h, y_points[-1] + h*k1*0.5)
        y_points.append(y_points[-1] + h*k2)
        t_points.append(t_points[-1] + h)#Gets most recent t-value and increments it by h
    #print("RK2 Solution: " + str(y_points[-1]))
    return t_points, y_points  

def RK4(f, bounds, y0, h):
    totalIterations = 0
    if bounds[0] > bounds[1]:
        temp = bounds[0]
        bounds[0] = bounds[1]
        bounds[1] = temp
    y_points = [y0]                                                       #Initialize Y values list
    t_points = [bounds[0]]                                                #Initialize T values list
    while t_points[-1] < (bounds[1] - 0.0000000000001):                   #Account for any rounding error  
        if (t_points[-1] + h > bounds[1]):                                #Clamp t in case h does not divide (b - a)
            h = abs(bounds[1] - t_points[-1])
        k1 = f(t_points[-1], y_points[-1])
        k2 = f(t_points[-1] + 0.5*h, y_points[-1] + 0.5*h*k1)
        k3 = f(t_points[-1] + 0.5*h, y_points[-1] + 0.5*h*k2)
        k4 = f(t_points[-1] + h, y_points[-1] + h*k3)
        y_points.append(y_points[-1] + h*(k1 + 2*k2 + 2*k3 + k4)/6)       #Gets most recent y-value and adds the derivative at most recent t-value to it
        t_points.append(t_points[-1] + h)                                 #Gets most recent t-value and increments it by h
        totalIterations += 1
    print("RK4 (h = " + str(h)[:6] + "): " + str(totalIterations))
    return t_points, y_points    

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

"""
Function: AB4 - Adams Bashforth 4
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
    while t_points[-1] < (bounds[1] - 0.0000000000001):                   #Account for any rounding error  
        if (t_points[-1] + h > bounds[1]):                                #Clamp t in case h does not divide (b - a)
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

def finiteDifferenceBVP(p, q, r, A, B, bounds, h):
    x_points = [bounds[0]]
    y_points = [A]
    dim = int(((bounds[1] - bounds[0])/h) - 1)
    #Create Matrices
    A_matrix = np.zeros((int(dim), int(dim)))
    b_matrix = np.zeros((int(dim), 1))
    a = bounds[0]
    b = bounds[1]
    col = 0
    
    #Fill in first row
    A_matrix[0,:2] = [(-2/(h**2) + q(a + h)), (1/(h**2) + p(a + h)/(2*h))]
    b_matrix[0] = [r(a + h) - A/(h**2) + (p(a + h)*A)/(2*h)]
    a = a + h
    x_points.append(a)
    #Fill in middle rows of matrices
    for row in range(1, dim-1):
        a = a + h
        A_matrix[row, col:(col+3)] = [(1/(h**2) - p(a)/(2*h)), (q(a) - 2/(h**2)), (1/(h**2) + p(a)/(2*h))]
        b_matrix[row] = [r(a)]
        x_points.append(a)
        col = col + 1
    #Fill in last row
    A_matrix[dim - 1, dim-2:] = [(1/(h**2) - p(b - h)/(2*h)), (-2/(h**2) + q(b - h))]
    b_matrix[dim-1] = [r(b - h) - B/(h**2) - (p(b - h)*B)/(2*h)]
    x_points.append(a + h)
    #Solve Matrix
    y_matrix = np.linalg.solve(A_matrix, b_matrix)
    #Get points for y from y_matrix
    for i in range(0, len(y_matrix[:,0])):
        y_points.append(y_matrix[i, 0])
    #Append right bound
    x_points.append(bounds[1])
    y_points.append(B)
    return x_points, y_points

def TrapezoidPredictorCorrector(func, bounds, y0, h, tolerance):
    tolerance_flag = False
    t_points = [bounds[0]]
    y_points = [y0]
    totalIterations = 0
    prevError = -1
    while t_points[-1] < (bounds[1] - 0.0000000000001): #Account for any rounding error 
        numIters = 0
        tolerance_flag = False                              
        if (t_points[-1] + h > bounds[1]):#Clamp t in case h does not divide (b - a)
            h = abs(bounds[1] - t_points[-1])
        predict = y_points[-1] + h * func(t_points[-1], y_points[-1])
        y_points.append(predict)
        t_points.append(t_points[-1] + h)
        while not tolerance_flag:   #Correct until tolerance flag is set to True (by if-statement below)
            totalIterations += 1
            numIters = numIters + 1
            correct = y_points[-2] + (h/2) * (func(t_points[-1], y_points[-1]) + func(t_points[-2], y_points[-2]))
            #Stop correcting if the increase was <= tolerance, if the maximum number of iterations is reached, or if the change in error is zero
            if(not(abs(correct - y_points[-1]) > tolerance) or numIters > 1000 or abs(correct - y_points[-1]) == prevError):
                tolerance_flag = True
            prevError = abs(correct - y_points[-1])
            y_points[-1] = correct
    print("Trapezoid PC (h = " + str(h)[:6] + "): " + str(totalIterations))
    return t_points, y_points