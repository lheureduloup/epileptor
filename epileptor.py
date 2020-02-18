# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:59:27 2019
Edited on Mon Feb 17 2020

@author: João Angelo Ferres Brogin

Note: The following code has been adapted considering dt = 0.01 for speed 
      purposes. Once the reader runs it, it takes ~2 to ~3 minutes until the
      curves are plotted, depending on the computer you're using.

"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#%% Simulation parameters:
Fs = 100        # Sampling frequency
dt = 1/Fs       # Time step
nd = 6          # Number of variables 
N = 500000      # Total number of points

# Parameters for the time vector:
t0 = 0   
tf = N*dt
t = np.linspace(t0, tf, N)
X = np.zeros((nd, N))

# Scaling factor for adding noise to the input:
sf1 = 0.002

#%% Parameters of the Epileptor model:
tau1 = 1
tau2 = 10   
tau0 = 2857.0
gamma = 0.01
I1 = 3.1     
I2 = 0.45    
x0 = -1.6
y0 = 1

#%% Set of differential equations in matrix form:
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
        c1 = 0
    elif x1 >= 0:
        h1 = x2 - 0.6*z**2 + 4.8*z - 9.6
        c1 = 0
    if x2 < -0.25:
        h2 = 0
        c3 = 0
    elif x2 >= -0.25:
        h2 = 6
        c3 = 1.5

    # Matrix A:
    A = np.array([ [      -h1,         1,          -1,              c1,            0,            0], \
                   [    -5*x1,        -1,           0,               0,            0,            0], \
                   [   4/tau0,         0,     -1/tau0,               0,            0,            0], \
                   [        0,         0,        -0.3,     (1 - x2**2),           -1,            2], \
                   [        0,         0,           0,         h2/tau2,      -1/tau2,            0], \
                   [0.1*gamma,         0,           0,               0,            0,       -gamma]  ]) 
    
    # Matrix B:
    b = np.array([I1, y0, -4*x0/tau0, I2 + (3.5*0.3), c3/tau2, 0])    
    
    # Solution in the time domain:
    aux = A.dot(X_)
    t1 = float(aux[0] + b[0])
    t2 = float(aux[1] + b[1])
    t3 = float(aux[2] + b[2])
    t4 = float(aux[3] + b[3])
    t5 = float(aux[4] + b[4] + float(sf1*np.random.normal(0,1,1))) # Adds noise to y2 to simulate a more stochastic behavior
    t6 = float(aux[5] + b[5])
    sol = np.array([t1, t2, t3, t4, t5, t6])
    
    return sol
    
#%% 4th Order Runge-Kutta integrator:  
    
# Initial conditions:
x10 = 0
y10 = -5
z0 = 3
x20 = 0
y20 = 0
u0 = 0
X[:,0] = [x10, y10, z0, x20, y20, u0]
    
# Time required for the simulation:
time0 = time.time()

for k in range(0, N-1):
    print('Iteration number: ' + str(k))
    k1 = __dXdt__( X[:,k] )
    k2 = __dXdt__( X[:,k] + k1*(dt/2) )
    k3 = __dXdt__( X[:,k] + k2*(dt/2) )
    k4 = __dXdt__( X[:,k] + k3*dt )
    X[:,k+1] = X[:,k] + (k1 + 2*k2 + 2*k3 + k4)*(dt/6)
        
time_elapsed = time.time() - time0
print('Time elapsed: ' + str(round(time_elapsed/60, 2)) + ' min')

#%% Local Field Potentials (LFPs): -x1(t) + x2(t)
LFP1 = X[0,:]  
LFP2 = X[3,:] 
LFP = -LFP1 + LFP2 

# RMS value of the LFP:
RMS = np.sqrt(np.mean(LFP**2))

# Adding noise to the output (measurement noise):
sf2 = 0.01  # Scaling factor
LFP = LFP + sf2*RMS*np.random.normal(0,1,len(LFP))

#%% Local field potentials over time:
plt.figure(1)
plt.subplot(211)
ax=plt.plot(t, LFP, 'k', linewidth=1)  
plt.ylabel('$LFP$',fontsize=20)
plt.xlim(0,t[-1])
plt.ylim(-3,2)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.subplot(212)
ax=plt.plot(t, X[2,:], 'r', linewidth=2)  
plt.xlabel('$t$ $[s]$',fontsize=20)
plt.ylabel('$z(t)$',fontsize=20)
plt.xlim(0,t[-1])
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

#%% Phase portraits for each subsystem:
plt.figure(2)
plt.subplot(121)
plt.plot(X[0,:],X[1,:],'b')
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$y_1$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.show()

plt.subplot(122)
plt.plot(X[3,:],X[4,:],'b')
plt.xlabel('$x_2$', fontsize=20)
plt.ylabel('$y_2$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.show()

#%% Plot for z vs x1:
plt.figure(3)
plt.plot(X[2,100000:300000], X[0,100000:300000], 'b')
plt.xlabel('$z$', fontsize=20)
plt.ylabel('$x_1$', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.show()