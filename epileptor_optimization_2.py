# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:00:41 2021

@author: João Angelo Ferres Brogin
"""
''

import numpy as np
from matplotlib import rc

rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#%% Simulation parameters:
Fs = 100    # Sampling frequency
dt = 1/Fs   # Time step
nd = 6      # Number of differential equations   
N = 200000  # Total number of points
t0 = 0      
tf = N*dt
t = np.linspace(t0, tf, N)
X = np.zeros((nd, N))

#%% Parameters of the model:
tau1 = 1
tau2 = 10
tau0 = 2857
gamma = 0.01
I1 = 3.1     
I2 = 0.45    
x0 = -1.6
y0 = 1

#%% Differential equations in matrix notation:
def __dXdt__( Xd ):
    global h1, h1z, c1, c3, h2, c2
    x1 = Xd[0]
    y1 = Xd[1]
    z = Xd[2]
    x2 = Xd[3]
    y2 = Xd[4]
    u = Xd[5]
    X_ = np.array([x1,y1,z,x2,y2,u])
    
    if x1 < 0:
        h1 = x1**2 - 3*x1
    elif x1 >= 0:
        h1 = x2 - 0.6*z**2 + 4.8*z - 9.6
    if x2 < -0.25:
        h2 = 0
        c3 = 0
    elif x2 >= -0.25:
        h2 = 6
        c3 = 1.5
              
    # Dynamic matrix A:
    A = np.array([  [         -h1,         1,              - 1,                0,            0,            0], \
                    [  -5*x1/tau1,        -1/tau1,           0,                0,            0,            0], \
                    [      4/tau0,         0,           -1/tau0,               0,            0,            0], \
                    [           0,         0,              -0.3,    ( 1 - x2**2),           -1,            2], \
                    [           0,         0,                 0,         h2/tau2,      -1/tau2,            0], \
                    [   0.1*gamma,         0,                 0,               0,            0,       -gamma] ]) 
    
    # Matrix B:
    b = np.array([ I1/tau1, \
                  y0/tau1, \
                  -4*x0/tau0, \
                  I2/tau1 + (3.5*0.3)/tau1, \
                  c3/tau2, \
                  0])    
    
    # Each variable obtained separately:
    aux = A.dot(X_)
    t1 = float(aux[0] + b[0] )
    t2 = float(aux[1] + b[1] )
    t3 = float(aux[2] + b[2] )
    t4 = float(aux[3] + b[3] )
    t5 = float(aux[4] + b[4] )
    t6 = float(aux[5] + b[5] )
    sol1 = np.array([t1, t2, t3, t4, t5, t6])
           
    return sol1
    
#%% Fourth-order Runge-Kutta Numerical Integrator:     
x_transl = np.array([-1.98,-18.66,4.15,-0.88,0,-0.035]) # Seizure-free condition

x10 = x_transl[0]
y10 = x_transl[1]
z0 = x_transl[2]
x20 = x_transl[3]
y20 = x_transl[4]
u0 = x_transl[5]
X[:,0] = [x10, y10, z0, x20, y20, u0]
    
for k in range(0, N-1):
    k1 = __dXdt__( X[:,k] )
    k2 = __dXdt__( X[:,k] + k1*(dt/2) )
    k3 = __dXdt__( X[:,k] + k2*(dt/2) )
    k4 = __dXdt__( X[:,k] + k3*dt )   
    X[:,k+1] = X[:,k] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) 
              
#%% Solution in the time domain (local field potentials):
LFP1 = X[0,:]  
LFP2 = X[3,:] 
LFP = -LFP1 + LFP2

Y = X
