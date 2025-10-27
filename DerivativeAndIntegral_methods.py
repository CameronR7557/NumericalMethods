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