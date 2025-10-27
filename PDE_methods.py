from math import e
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import odeint


def FiniteDiffPDE(x, t, f, g, h, k, stepsX):
    delX = (x[1] - x[0])/ stepsX
    # (tn - t0)/stepsT <= h^2 / (2*alpha) --> (tn - t0)*((2.1*alpha)/h^2) = stepsT
    stepsT = int(np.ceil((t[1] - t[0]) / ((delX**2)/(2.1*k))))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((stepsT + 1, stepsX + 1))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    #Fill in boudary values
    for i in range(stepsT+1):
        u[i, 0] = g(delT*i)
        u[i, -1] = h(delT*i)
    #Fill initial value (t = 0 for all x)
    for j in range(stepsX+1):
        u[0, j] = f(j*delX)
        
    for i in range(1, stepsT+1):#Already know values at time 0 (f(x))
        for j in range(1, stepsX-1):#Already know values at x = 0 and x = L (g and h)
            u[i, j] = u[i - 1, j] + ((k*delT)/(delX**2)) * (u[i - 1, j + 1] - 2*u[i - 1, j] + u[i - 1, j - 1])
    
    return x_points, t_points, u

def FiniteDiffPDE_Wave(x, t, f, F, p, r, c, stepsX):
    delX = (x[1] - x[0])/ stepsX
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(c*(t[1] - t[0])/(delX)))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((stepsT + 1, stepsX + 1))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    #Fill in boudary values
    for i in range(stepsT+1):
        u[i, 0] = p(delT*i + t[0])
        u[i, -1] = r(delT*i + t[0])
    #Fill initial values for first two time steps using initial position and velocity
    for j in range(stepsX+1):
        u[0, j] = f(j*delX + x[0])
        u[1, j] = u[0,j] + F(j*delX + x[0])*delT #Next position = prevPos + velocity*delta-t
    
    for i in range(1, stepsT):#Already know values at time 0
        for j in range(1, stepsX):#Already know values at x = 0 and x = L 
            u[i + 1, j] = 2*u[i, j] - u[i - 1, j]+ (((c**2)*(delT**2))/(delX**2)) * (u[i, j + 1] - 2*u[i, j] + u[i, j - 1])
    
    return x_points, t_points, u

# def F(x, delX):
#     dx = np.zeros((len(x)))
#     dx[0] = ((-3/2)*x[0] + 2*x[1] - 0.5*x[2]) / delX
#     for i in range(1, len(x)-1):
#         dx[i] = (x[i+1] - x[i-1])/(2*delX)
#     dx[-1] = ((3/2)*x[-1] - 2*x[-2] + 0.5*x[-3]) / delX
#     return dx

# def FiniteDiffPDE_Wave(x, t, ip, iv, rho, k, c, stepsX):
#     delX = (x[1] - x[0])/ stepsX
#     #Solve for time-step restraint (CFL condition)
#     stepsT = int(np.ceil(1.01*c*(t[1] - t[0])/(delX)))
#     delT = (t[1] - t[0])/ stepsT
#     #Set up solution array. Columns are delta-x, rows are delta-t
#     u = np.zeros((stepsT + 1, stepsX + 3, 2))
#     x_points = np.linspace(x[0], x[-1], stepsX + 1)
#     t_points = np.linspace(t[0], t[-1], stepsT + 1)
#     #Fill initial values for first two time steps using initial position and velocity
#     u[0, 1:-1, 0] = ip(x_points)
#     u[0, 1:-1, 1] = iv(x_points)
#     u[1, 1:-1, 0] = u[0,1:-1,0] - k*F(u[0, 1:-1, 1], delX)*delT
#     u[1, 1:-1, 1] = u[0,1:-1,1] - (1/rho)*F(u[0, 1:-1, 0], delX)*delT
    
#     for i in range(1, len(u[:,0,0]) - 1):
#         u[i+1,0,0] = u[i,1,0]
#         u[i+1,-1,0] = u[i,-2,0]
#         u[i+1,0,1] = u[i,1,1]
#         u[i+1,-1,1] = u[i,-2,1]
#         for j in range(1, len(u[0,:,0]) - 1):
#             u[i + 1, j,0] = 2*u[i, j,0] - u[i - 1, j,0]+ (((c**2)*(delT**2))/(delX**2)) * (u[i, j + 1,0] - 2*u[i, j,0] + u[i, j - 1,0])
#             u[i + 1, j,1] = 2*u[i, j,1] - u[i - 1, j,1]+ (((c**2)*(delT**2))/(delX**2)) * (u[i, j + 1,1] - 2*u[i, j,1] + u[i, j - 1,1])
#     return x_points, t_points, u[:,1:-1,:]

def FiniteDiff2D_Wave(x, y, t, f, F, p, r, g, h, c, steps):
    delX = (x[1] - x[0])/ steps
    delY = (y[1] - y[0])/ steps
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(2*c*(t[1] - t[0])/((delX))))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((steps + 1, steps + 1, stepsT + 1))
    x_points = np.linspace(x[0], x[-1], steps + 1)
    y_points = np.linspace(y[0], y[-1], steps + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    
    X,Y = np.meshgrid(x_points, y_points)
    u[:,:,0] = f(X, Y)
    u[:,:,1] = u[:,:,0] + F(X, Y)*delT
    #Fill in boundary values
    u[0, :, :] = g(y_points, t_points)
    u[-1, :, :] = h(y_points, t_points)
    u[:, 0, :] = p(x_points, t_points)
    u[:, -1, :] = r(x_points, t_points)
    
    for j in range(1, stepsT):
        for k in range(1, steps):
            for i in range(1, steps):
                u[i, k, j+1] = 2*u[i,k,j] - u[i,k,j-1] + ((c**2)*(delT**2))*((u[i+1,k,j] - 2*u[i,k,j] + u[i-1,k,j])/(delX**2) + (u[i,k+1,j] - 2*u[i,k,j] + u[i,k-1,j])/(delY**2))
    return x_points, y_points, t_points, u

def FiniteDiff2D_Wave_Polar(r, o, t, f, F, p, q, g, h, c, steps):
    delR = (r[1] - r[0])/ steps
    delO = (o[1] - o[0])/ steps
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(2*c*(t[1] - t[0])/((delR))))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((steps + 1, steps + 1, stepsT + 1))
    r_points = np.linspace(r[0], r[-1], steps + 1)
    o_points = np.linspace(o[0], o[-1], steps + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    
    R,O = np.meshgrid(r_points, o_points)
    u[:,:,0] = f(R)
    u[:,:,1] = u[:,:,0] + F(R, O)*delT
    #Fill in boundary values
    u[0, :, :] = g(o_points, t_points)
    u[-1, :, :] = h(o_points, t_points)
    u[:, 0, :] = p(r_points, t_points)
    u[:, -1, :] = q(r_points, t_points)
    
    for j in range(1, stepsT):
        for i in range(1, steps):
            for k in range(1, steps):
                rp = r_points[k]
                if(abs(rp) <=  0.0000000000001):#Do not calculate at r = 0
                    rp = 0.0000000000001
                #Ignores theta componenet since radially symmetric
                u[i, k, j+1] = 2*u[i,k,j] - u[i,k,j-1] + ((c**2)*(delT**2))*((u[i, k+1, j] -u[i, k-1, j])/(2*rp*delR) + (u[i,k+1,j] - 2*u[i,k,j] + u[i,k-1,j])/(delR**2))
    return r_points, o_points, t_points, u

def CFD_Rectangle(x, y, f, g, p, r, stepsX, stepsY):
    delX = (x[1] - x[0])/ stepsX
    delY = (y[1] - y[0])/ stepsY
    #Set up solution array. Rows are the stencil points, cols are the equations for individual points
    A = np.zeros(((stepsY - 1) * (stepsX - 1), (stepsY - 1) * (stepsX - 1)))
    b = np.zeros(((stepsY - 1) * (stepsX - 1), 1))
    u = np.zeros(((stepsY + 1), (stepsX + 1)))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    y_points = np.linspace(y[0], y[-1], stepsY + 1)
    for i in range(0, len(u[:,0])):
        u[0, i] = p(delX * i + x[0])  #Lower BCs
        u[-1, i] = r(delX * i + x[0]) #Upper BCs
        u[i, 0] = f(delY * i + y[0])  #Left BCs
        u[i, -1] = g(delY * i + y[0]) #Right BCs
    i = 0
    j = 0
    curRow = 0
    stepsX -= 2
    stepsY -= 2
    for i in range(0, stepsY + 1):
        for j in range(0, stepsX + 1):
            curRow = i*(stepsX + 1) + j
            A[curRow, curRow] = 4
            if(i == 0):#Bottom BC, p(x)
                b[curRow, 0] += p(delX *(j+1) + x[0])
            else:
                A[curRow, curRow - stepsX - 1] = -1
            if(i == stepsY):#Upper BC, r(x)
                b[curRow, 0] += r(delX *(j+1) + x[0]) 
            else:
                A[curRow, curRow + stepsX + 1] = -1
            if(j == 0):#Left bound is BC, f(y)
                b[curRow, 0] += f(delY *(i+1) + y[0])
            else:
                A[curRow, curRow - 1] = -1
            if(j == stepsX):#Right bound is BC, g(y)
                b[curRow, 0] += g(delY *(i+1) + y[0])
            else:
                A[curRow, curRow + 1] = -1
    temps = np.linalg.solve(A, b)
    for i in range(1, len(u[0, :]) - 1):
        u[i, 1:-1] = temps[(stepsX+1)*(i-1):((stepsX+1)*(i-1) + stepsX + 1), 0].T
    return x_points, y_points, u

"""Centered-finite-difference with a hole in center"""
def CFD_Rectangle_Hole(x, y, f, g, p, r, stepsX, stepsY, hole_width):
    delX = (x[1] - x[0])/ stepsX
    delY = (y[1] - y[0])/ stepsY
    stepsH = int(np.ceil((stepsX/x[1]) * hole_width))
    i_h = int((stepsY - stepsH)/2)
    j_h = int((stepsX - stepsH)/2)
    #Set up solution array. Rows are the stencil points, cols are the equations for individual points
    A = np.zeros(((stepsY - 1) * (stepsX - 1), (stepsY - 1) * (stepsX - 1)))
    b = np.zeros(((stepsY - 1) * (stepsX - 1), 1))
    u = np.zeros(((stepsX + 1), (stepsY + 1)))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    y_points = np.linspace(y[0], y[-1], stepsY + 1)
    for i in range(0, len(u[:,0])):
        u[i, 0] = f(delY * i + y[0])  #Left BCs
        u[i, -1] = g(delY * i + y[0]) #Right BCs
    for i in range(0, len(u[0,:])):
        u[0, i] = p(delX * i + x[0])  #Lower BCs
        u[-1, i] = r(delX * i + x[0]) #Upper BCs
    for i in range(stepsH+1):
        u[stepsY - i_h, i + j_h] = 100
    i = 0
    j = 0
    curRow = 0
    stepsX -= 2
    stepsY -= 2
    for i in range(0, stepsY + 1):
        for j in range(0, stepsX + 1):
            curRow = i*(stepsX + 1) + j
            A[curRow, curRow] = 4
            if(i == 0):#Bottom BC, p(x)
                b[curRow, 0] += p(delX *(j+1) + x[0])
            else:
                A[curRow, curRow - stepsX - 1] = -1
            
            if(i == stepsY):#Upper BC, r(x)
                b[curRow, 0] += r(delX *(j+1) + x[0]) 
            else:
                A[curRow, curRow + stepsX + 1] = -1
            
            if(j == 0):#Left bound is BC, f(y)
                b[curRow, 0] += f(delY *(i+1) + y[0])
            else:
                A[curRow, curRow - 1] = -1
            
            if(j == stepsX):#Right bound is BC, g(y)
                b[curRow, 0] += g(delY *(i+1) + y[0])
            else:
                A[curRow, curRow + 1] = -1
            
            if(i == stepsY - i_h + 1 and (j >= j_h-1 and j <= stepsX - j_h+1)):
                A[curRow, curRow + 1] = 0
                A[curRow, curRow - 1] = 0
                A[curRow, curRow + stepsX + 1] = 0
                A[curRow, curRow - stepsX - 1] = 0
                A[curRow, curRow] = 1
                b[curRow, 0] = 100
            elif((j >= j_h-1 and j <= stepsX - j_h+1) and (i >= i_h-1 and i <= stepsY - i_h+1)):
                A[curRow, curRow + 1] = 0
                A[curRow, curRow - 1] = 0
                A[curRow, curRow + stepsX + 1] = 0
                A[curRow, curRow - stepsX - 1] = 0
                b[curRow, 0] = 0
            
    temps = np.linalg.solve(A, b)
    for i in range(1, len(u[0, :]) - 1):
        u[i, 1:-1] = temps[(stepsX+1)*(i-1):((stepsX+1)*(i-1) + stepsX + 1), 0].T
    return x_points, y_points, u

def CrankNicolson(x, t, f, g, h, k, stepsX, stepsT):
    delX = (x[1] - x[0])/ stepsX
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((stepsT + 1, stepsX + 1))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    #Fill in boudary values
    for i in range(stepsT+1):
        u[i, 0] = g(delT*i)
        u[i, -1] = h(delT*i)
    #Fill initial value (t = 0 for all x)
    for j in range(stepsX+1):
        u[0, j] = f(j*delX)
    
    r = (k*delT)/(2*(delX)**2) #Likely need to include k in this calc
    #Create A matrix
    A = np.zeros((stepsX + 1, stepsX + 1))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, stepsX): #Fill in A matrix rows
        A[i, (i-1):(i+2)] = [-r, 1 + 2*r, -r]
    
    for i in range(1, stepsT+1):#Already know values at time 0 (f(x))
        for j in range(1, stepsX-1):#Already know values at x = 0 and x = L (g and h)
            #Calc b in Ax = b
            u[i, j] = r*u[i - 1, j + 1] + (1-2*r)*u[i - 1, j] + r*u[i - 1, j - 1]
        u[i, :] = np.linalg.solve(A, u[i, :]) #Solve Ax = b for unknown rows in u (temp along rod at a certain time)
    
    return x_points, t_points, u

"""Backward time, centered space"""
def BTCS_SolvePDE(x, t, f, g, h, k, stepsX, stepsT):
    delX = (x[1] - x[0])/ stepsX
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((stepsT + 1, stepsX + 1))
    x_points = np.linspace(x[0], x[-1], stepsX + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    #Fill in boudary values
    for i in range(stepsT+1):
        u[i, 0] = g(delT*i)
        u[i, -1] = h(delT*i)
    #Fill initial value (t = 0 for all x)
    for j in range(stepsX+1):
        u[0, j] = f(j*delX)
    
    r = -delT*k/(delX)**2
    #Create A matrix
    A = np.zeros((stepsX + 1, stepsX + 1))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, stepsX): #Fill in A matrix rows
        A[i, (i-1):(i+2)] = [r, 1 - 2*r, r]
    
    for i in range(1, stepsT+1):#Already know values at time 0 (f(x))
        u[i, :] = np.linalg.solve(A, u[i - 1, :]) #Solve Ax = b for unknown rows in u (temp along rod at a certain time)
    
    return x_points, t_points, u

"""Forward time, centered space"""
def FTCS_2DParabolicPDE(x, y, t, f, p, r, g, h, a, steps):
    delX = (x[1] - x[0])/ steps
    delY = (y[1] - y[0])/ steps
    #Solve for time-step restraint (CFL condition)
    stepsT = int(np.ceil(2*a*(t[1] - t[0])*(1/(delX**2) + 1/(delY**2))))
    delT = (t[1] - t[0])/ stepsT
    #Set up solution array. Columns are delta-x, rows are delta-t
    u = np.zeros((steps + 1, steps + 1, stepsT + 1))
    x_points = np.linspace(x[0], x[-1], steps + 1)
    y_points = np.linspace(y[0], y[-1], steps + 1)
    t_points = np.linspace(t[0], t[-1], stepsT + 1)
    
    X,Y = np.meshgrid(x_points, y_points)
    u[:,:,0] = f(X, Y)
    #Fill in boundary values
    u[0, :, :] = g(y_points, t_points)
    u[-1, :, :] = h(y_points, t_points)
    u[:, 0, :] = p(x_points, t_points)
    u[:, -1, :] = r(x_points, t_points)
    
    for j in range(0, stepsT-1):
        for k in range(1, steps):
            for i in range(1, steps):
                u[i, k, j+1] = u[i,k,j] + (delT)*((u[i+1,k,j] - 2*u[i,k,j] + u[i-1,k,j])/(delX**2) + (u[i,k+1,j] - 2*u[i,k,j] + u[i,k-1,j])/(delY**2))
    return x_points, y_points, t_points, u