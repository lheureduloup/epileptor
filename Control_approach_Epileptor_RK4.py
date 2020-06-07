# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:59:27 2019

@author: João Angelo Ferres Brogin
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

time0 = time.time()

# Use this line to import the gain matrix Gx and matrix P from the optimization procress:
from LMIs_Epileptor import X, Gx

P = X.I
G = Gx.dot(P)

#%% Input force matrix:
B = np.identity(6)
B[2][2] = 0.05

#%% Simulation parameters:
Fs = 100
dt = 1/Fs
nd = 6       # Number of differential equations
N = 300000   # Number of points
t0 = 0
tf = N*dt
t = np.linspace(t0, tf, N)
X = np.zeros((nd, N))

#%% Parameters of the model:
tau1 = 1
tau2 = 10   
tau0 = 2857.0
gamma = 0.01
I1 = 3.1     
I2 = 0.45    
x0 = -1.6
y0 = 1

#%% Vector for the membership functions:
F1pp = []
F2pp = []
F3pp = []
F4pp = []
F5pp = []
F6pp = []

#%% Input forces for u(t) and b * u(t):
ut1 = [None]*N
ut2 = [None]*N
ut3 = [None]*N
ut4 = [None]*N
ut5 = [None]*N
ut6 = [None]*N

Fc1 = [None]*N
Fc2 = [None]*N
Fc3 = [None]*N
Fc4 = [None]*N
Fc5 = [None]*N
Fc6 = [None]*N

U = [None]*N
V = [None]*N
FC = [None]*N
TRCK = []

#%% Set of differential equations in matrix form:
def __dXdt__( Xd, g ):
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
        c1 = 0
    elif x2 >= -0.25:
        h2 = 6
        c1 = 1.5
 
    # States to which the controller drives the system after it is on:
    x_transl = np.array([-1.98,-18.66,4.15,-0.88,0,-0.035])
    
    # Variable for tracking the trajectory of the controller and verify whether x_transl is reached:
    X_ = X_.reshape(-1,1)
    x_transl = x_transl.reshape(-1,1)
    tracking = np.sqrt(sum((X_ - x_transl)**2))
    
    # Vector u(t) = -G*[x(t) - xh]:
    UT = -g * G * (X_ - x_transl)

    # Fc = B.dot(UT.T):
    Fc = -g * B * G * (X_ - x_transl)

    # Input forces u(t) for each state:
    ut1 = UT[0]
    ut2 = UT[1]
    ut3 = UT[2]
    ut4 = UT[3]
    ut5 = UT[4]
    ut6 = UT[5]
    Uc = np.array([ut1, ut2, ut3, ut4, ut5, ut6])
    
    # Matrix A:
    A = np.array([  [       -h1,         1,          -1,              0,            0,            0], \
                    [     -5*x1,        -1,           0,              0,            0,            0], \
                    [    4/tau0,         0,     -1/tau0,              0,            0,            0], \
                    [         0,         0,        -0.3,    (1 - x2**2),           -1,            2], \
                    [         0,         0,           0,        h2/tau2,      -1/tau2,            0], \
                    [ 0.1*gamma,         0,           0,              0,            0,       -gamma] ]) 
    
    # Matrix B:
    b = np.array([ I1, \
                   y0, \
                  -4*x0/tau0, \
                  I2 + (3.5*0.3), \
                  c1/tau2, \
                  0])    
    
    # Each variable is obtained separately and summed together.
    # To add noise to the variables, change add_noise to 1.
    # sf = scaling factor (change its value to increase or decrease the intensity of noise)
    sf = 0.0
    add_noise = 0
    aux = A.dot(X_)
    t1 = float(aux[0] + Fc[0] + b[0] + add_noise*sf*aux[0]*np.random.normal(0,1,1) )
    t2 = float(aux[1] + Fc[1] + b[1] + add_noise*sf*aux[1]*np.random.normal(0,1,1) )
    t3 = float(aux[2] + Fc[2] + b[2] + add_noise*sf*aux[2]*np.random.normal(0,1,1) )
    t4 = float(aux[3] + Fc[3] + b[3] + add_noise*sf*aux[3]*np.random.normal(0,1,1) )
    t5 = float(aux[4] + Fc[4] + b[4] + add_noise*sf*aux[4]*np.random.normal(0,1,1) )
    t6 = float(aux[5] + Fc[5] + b[5] + add_noise*sf*aux[5]*np.random.normal(0,1,1) )
    sol1 = np.array([t1, t2, t3, t4, t5, t6])
           
    # Variables for the membership functions:
    F1p = -h1
    F2p = -1
    F3p = -5*x1
    F4p = 1 - x2**2 
    F5p = h2/tau2
    F6p = c1
    
    return sol1, F1p, F2p, F3p, F4p, F5p, F6p, Uc, tracking, Fc
    
#%% Fourth-order Runge-Kutta:     
n_start = 180000
n_end = n_start + 1000 # The controller is on for 10 seconds of simulation
x_transl = np.array([-1.98,-18.66,4.15,-0.88,0,-0.035])

# Initial Conditions:
# During a seizure:
x10 = 0.0
y10 = -5
z0 = 3
x20 = 0
y20 = 0
u0 = 0

# Healthy:
# x10 = x_transl[0]
# y10 = x_transl[1]
# z0 = x_transl[2]
# x20 = x_transl[3]
# y20 = x_transl[4]
# u0 = x_transl[5]

X[:,0] = [x10, y10, z0, x20, y20, u0]
    
for k in range(0, N-1):
    if k <= n_start:
        g = 0
    if k > n_start and k <= n_end:  # Control is on if g = 1 here
        g = 1
    if k > n_end:
        g = 0
    print('Iteration number: ' + str(k))
    k1, F1p, F2p, F3p, F4p, F5p, F6p, Uc, tracking, wu = __dXdt__( X[:,k], g )
    k2, F1p, F2p, F3p, F4p, F5p, F6p, Uc, tracking, wu = __dXdt__( X[:,k] + k1*(dt/2), g )
    k3, F1p, F2p, F3p, F4p, F5p, F6p, Uc, tracking, wu = __dXdt__( X[:,k] + k2*(dt/2), g )
    k4, F1p, F2p, F3p, F4p, F5p, F6p, Uc, tracking, wu = __dXdt__( X[:,k] + k3*dt, g )   
    X[:,k+1] = X[:,k] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) 
        
    # State vector after translation to the seizure-free point:
    XX = np.array([ [X[0,k] - x_transl[0]] ,
                    [X[1,k] - x_transl[1]] ,
                    [X[2,k] - x_transl[2]] ,
                    [X[3,k] - x_transl[3]] ,
                    [X[4,k] - x_transl[4]] ,
                    [X[5,k] - x_transl[5]] ])
    
    # Lyapunov function:
    V[k] = float(XX.T * P * XX)
        
    # Nonlinear functions and Euclidean distance (tracking variable):
    F1pp.append(F1p)
    F2pp.append(F2p)
    F3pp.append(F3p)
    F4pp.append(F4p)        
    F5pp.append(F5p)
    F6pp.append(F6p)
    TRCK.append(float(tracking))

    # Input forces u(t) for each state:
    ut1[k] = float(Uc[0])
    ut2[k] = float(Uc[1])
    ut3[k] = float(Uc[2])
    ut4[k] = float(Uc[3])
    ut5[k] = float(Uc[4])
    ut6[k] = float(Uc[5])
        
    # Input forces BG[x(t) - xh]:
    Fc1[k] = float(wu[0])
    Fc2[k] = float(wu[1])
    Fc3[k] = float(wu[2])
    Fc4[k] = float(wu[3])
    Fc5[k] = float(wu[4])
    Fc6[k] = float(wu[5])
        
    # Euclidean norm of the input forces:
    U[k] = np.sqrt( ut1[k]**2 + ut2[k]**2 + ut3[k]**2 + ut4[k]**2 + ut5[k]**2 + ut6[k]**2 )
    FC[k] = np.sqrt( Fc1[k]**2 + Fc2[k]**2 + Fc3[k]**2 + Fc4[k]**2 + Fc5[k]**2 + Fc6[k]**2 )
    
elapsed_time = time.time() - time0
print('Elapsed time: ' + str(round(elapsed_time/60,2)) + ' [min]')
        
#%% Local Field Potentials (LFPs):
LFP1 = X[0,:]  
LFP2 = X[3,:] 
LFP = -LFP1 + LFP2

#%% Epileptor's behavior in the time domain:
plt.figure(1)
plt.subplot(211)  
ax=plt.plot(t[: n_start], LFP[:n_start], 'k', linewidth=2)  
plt.plot(t[n_start:n_end], LFP[n_start:n_end], 'r-.', linewidth=2)
plt.plot(t[n_end:], LFP[n_end:], 'k', linewidth=2)
plt.plot(t[n_start]*np.ones(len(LFP)), np.linspace(-3,3,len(LFP)), '--k', linewidth=1.5)
plt.plot(t, LFP, 'k', linewidth=2)
plt.show()
plt.ylabel('$LFP$',fontsize=35)
plt.xlim(0,2700)
plt.ylim(-3,2)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.grid()

plt.subplot(212)  
plt.plot(t[:n_start], X[2,:][:n_start], 'k', linewidth=2)  
plt.plot(t[n_start:n_end], X[2,:][n_start:n_end], 'r-.', linewidth=2)
plt.plot(t[n_end:], X[2,:][n_end:], 'k', linewidth=2)
plt.plot(t, max(X[2,0:150000])*np.ones(len(LFP)), '--b', linewidth=2)
plt.plot(t, min(X[2,:])*np.ones(len(LFP)), '--b', linewidth=2)
plt.show()
plt.xlabel('$t$ $[s]$',fontsize=35)
plt.ylabel('$z(t)$',fontsize=35)
plt.xlim(0,2700)
plt.ylim(2.8,4.2)    
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=25)

#%% Phase portraits for both subsystems:
plt.figure(2)
plt.subplot(121)
plt.plot(X[0,:][:n_start], X[1,:][:n_start], 'b')
plt.plot(X[0,:][n_start:n_end], X[1,:][n_start:n_end], 'r-.', linewidth=3)
plt.plot(X[0,:][n_end:], X[1,:][n_end:], 'b')
plt.xlabel('$x_{1}$',fontsize=35)
plt.ylabel('$y_{1}$',fontsize=35)    
plt.show()
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=25)
   
plt.subplot(122)
plt.plot(X[3,:][:n_start], X[4,:][:n_start], 'b')
plt.plot(X[3,:][n_start:250000], X[4,:][n_start:250000], 'r-.', linewidth=3)
plt.plot(X[3,:][n_end:], X[4,:][n_end:], 'b')
plt.show()
plt.grid()
plt.xlabel('$x_{2}$',fontsize=35)
plt.ylabel('$y_{2}$',fontsize=35)
plt.tick_params(axis='both', which='major', labelsize=25)
    
max_f1p = (max(F1pp))
min_f1p = (min(F1pp))

max_f2p = (max(F2pp))
min_f2p = (min(F2pp))
    
max_f3p = (max(F3pp))
min_f3p = (min(F3pp))
    
max_f4p = (max(F4pp))
min_f4p = (min(F4pp))
    
max_f5p = (max(F5pp))
min_f5p = (min(F5pp))

#%% Nonlinear functions: 
plt.figure(3)
plt.subplot(221)
plt.plot(t[:-1], F1pp, 'k', linewidth=1.5)
plt.plot(t[:-1], max_f1p*(np.ones(len(t[:-1]))), '--r', linewidth=3)
plt.plot(t[:-1], min_f1p*(np.ones(len(t[:-1]))), '--r', linewidth=3)
plt.show()
plt.xlabel('$t$ $[s]$',fontsize=35)
plt.ylabel('$f_{1}(\mathbf{x}(t))$',fontsize=35)
plt.xlim(t[0], t[-1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=25)
   
plt.subplot(223)
plt.plot(t[:-1], F3pp, 'k', linewidth=2)
plt.plot(t[:-1], max_f3p*(np.ones(len(t[:-1]))), '--r', linewidth=3)
plt.plot(t[:-1], min_f3p*(np.ones(len(t[:-1]))), '--r', linewidth=3)
plt.show()
plt.xlabel('$t$ $[s]$',fontsize=18)
plt.ylabel('$f_{2}(\mathbf{x}(t))$',fontsize=18)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=25)

plt.subplot(224)
plt.plot(t[:-1], F4pp, 'k', linewidth=2)
plt.plot(t[:-1], max_f4p*(np.ones(len(t[:-1]))), '--r', linewidth=3)
plt.plot(t[:-1], min_f4p*(np.ones(len(t[:-1]))), '--r', linewidth=3)
plt.show()
plt.xlabel('$t$ $[s]$',fontsize=18)
plt.ylabel('$f_{3}(\mathbf{x}(t))$',fontsize=18)
plt.grid()    
plt.tick_params(axis='both', which='major', labelsize=25)

plt.subplot(222)
plt.plot(t[:-1], F5pp, 'k', linewidth=2)
plt.plot(t[:-1], max_f5p*(np.ones(len(t[:-1]))), '--r', linewidth=3)
plt.plot(t[:-1], min_f5p*(np.ones(len(t[:-1]))), '--r', linewidth=3)
plt.show()
plt.ylabel('$f_{4}(\mathbf{x}(t))$',fontsize=18)
plt.xlabel('$t$ $[s]$',fontsize=18)
plt.grid() 
plt.tick_params(axis='both', which='major', labelsize=25)
   
#%% Input forces:    
plt.figure(4)
plt.plot(t[:n_start], U[:n_start], 'k')
plt.plot(t[n_start:n_end], U[n_start:n_end], 'r-.', linewidth=3)
plt.plot(t[n_end:], U[n_end:], 'k')
plt.show()
plt.xlabel('$t$ $[s]$',fontsize=35)
plt.ylabel('$\parallel \mathbf{u}(t) \parallel$',fontsize=35)
plt.ylim(0,60)
plt.xlim(t[179000], t[182000])
plt.grid()      
plt.tick_params(axis='both', which='major', labelsize=25)

#%% Tracking variable:
plt.figure(6)
plt.plot(t[:n_start], TRCK[:n_start], 'k', linewidth=2)
plt.plot(t[n_start:n_end], TRCK[n_start:n_end], 'r-.', linewidth=3)
plt.plot(t[n_end: -1], TRCK[n_end:], 'k', linewidth=2)
plt.xlabel('$t$ $[s]$',fontsize=35)
plt.ylabel('$\parallel \mathbf{x}(t)-\mathbf{x}_{h} \parallel$',fontsize=35)
plt.grid()
plt.xlim(0,t[-1])
plt.plot(t, max(TRCK)*np.ones(len(LFP)), '--b', linewidth=2)
plt.plot(t, np.zeros(len(LFP)), '--b', linewidth=2)
plt.tick_params(axis='both', which='major', labelsize=25)

#%% Lyapunov function:
plt.figure(7)
plt.plot(t[n_start:n_end], V[n_start:n_end], 'k', linewidth=3)
plt.xlabel('$t$ $[s]$',fontsize=35)
plt.ylabel('$V(\mathbf{x}(t)-\mathbf{x}_{h})$',fontsize=35)
plt.grid()
plt.xlim(t[n_start], t[n_end])
plt.tick_params(axis='both', which='major', labelsize=25)

#%% Lyapunov function's derivative:
plt.figure(8)
plt.plot(t[n_start:n_end], np.gradient(V[n_start:n_end]), 'r', linewidth=3)
plt.xlabel('$t$ $[s]$',fontsize=35)
plt.ylabel('$\dot{V}(\mathbf{x}(t)-\mathbf{x}_{h})$',fontsize=35)
plt.grid()
plt.xlim(t[n_start], t[n_end])
plt.ylim(-27,0)
plt.tick_params(axis='both', which='major', labelsize=25)   

#%% Input forces (Fc):
plt.plot(9)
plt.plot(t, Fc1, 'k', linewidth=3, label='$F_{c1}$')
plt.plot(t, Fc2, 'r--', linewidth=3, label='$F_{c2}$')
plt.plot(t, Fc3, 'b-.', linewidth=3, label='$F_{c3}$')
plt.plot(t, Fc4, '-go', linewidth=1.5, label='$F_{c4}$')
plt.plot(t, Fc5, '-+', linewidth=3, label='$F_{c5}$')
plt.plot(t, Fc6, 'm:', linewidth=3, label='$F_{c6}$')
plt.show()
plt.grid()
plt.xlim(1800,1810)
plt.ylim(-25,5)
plt.xlabel('$t$ $[s]$',fontsize=35)
plt.ylabel('$F_{c}$',fontsize=35)
plt.tick_params(axis='both', which='major', labelsize=25)   

#%% Input forces (separately):
plt.plot(10)
plt.subplot(321)
plt.plot(t, Fc1, 'k', linewidth=3, label='$F_{c1}$')
plt.grid()
plt.xlim(1795,1805)
plt.ylabel('$F_{c1}$',fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)   

plt.subplot(322)
plt.plot(t, Fc2, 'r--', linewidth=3, label='$F_{c2}$')
plt.grid()
plt.xlim(1795,1805)
plt.ylabel('$F_{c2}$',fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20) 

plt.subplot(323)
plt.plot(t, Fc3, 'b-.', linewidth=3, label='$F_{c3}$')
plt.grid()
plt.xlim(1795,1805)
plt.ylabel('$F_{c3}$',fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20) 

plt.subplot(324)
plt.plot(t, Fc4, '-g^', linewidth=1.5, label='$F_{c4}$')
plt.grid()
plt.xlim(1795,1805)
plt.ylabel('$F_{c4}$',fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)   

plt.subplot(325)
plt.plot(t, Fc5, '-C1+', linewidth=3, label='$F_{c5}$')
plt.grid()
plt.xlim(1795,1805)
plt.xlabel('$t$ $[s]$',fontsize=25)
plt.ylabel('$F_{c5}$',fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20) 

plt.subplot(326)
plt.plot(t, Fc6, 'm:', linewidth=3, label='$F_{c6}$')
plt.grid()
plt.xlim(1795,1805)
plt.xlabel('$t$ $[s]$',fontsize=25)
plt.ylabel('$F_{c6}$',fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)   

