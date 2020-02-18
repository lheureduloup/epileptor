# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 09:04:11 2019
Edited on Mon Feb 17 2020

@author: João Angelo Ferres Brogin

Note: In this case, a different version of the code is implemented.
      All of the equations are defined separately, not in matrix form.
      By doing so, it is possible to have more control over the parameters.
      
Note: The following code has been adapted considering dt = 0.01 for speed
      purposes. Once the reader runs it, it takes ~1 to ~2 minutes until
      the curves are plotted, depending on the computer you're using.
      If other sampling rates are used, new parameters must be set.

"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#%% Parameters of the adapted model:
tau1 = 0.3
tau2 = 2.0
tau0 = 2857.0
gamma = 0.01
I1 = 3.2   
I2 = 0.45    
x0 = -1.6
y0 = 1

#%% Set of differential equations:
def dx1dt(x1,x2,y1,z):
    if x1 < 0:
        f1 = x1**3 - 3*x1**2
    if x1>=0:
        f1 = (0.95*x2*x1 - 0.21*x1*z)/0.9
    dx1dt = (1/tau1)*(y1 - f1 - z + I1) 
    return dx1dt

def dy1dt(x1,y1):
    dy1dt = (2/tau1)*(y0 - 5.5*x1**2 - y1)
    return dy1dt

def dzdt(x1,z):
    dzdt = (1/tau0)*(4*(x1 - x0) - z)
    return dzdt

def dudt(u,x1):
    dudt = -gamma*(u - 0.1*x1)
    return dudt

def dx2dt(y2,x2,u,z,par_ctr):
    dx2dt = 6.67*(-par_ctr*y2 + x2 - x2**3 + I2 + 2*u - 0.15*(z - 3.5)) 
    return dx2dt

def dy2dt(y2,x1,x2):
    if x2 < -0.25:
        f2 = 0
    if x2 >= -0.25:
        f2 = 6*(x2 + 0.25)
    dy2dt = (1/tau2)*(-y2 + f2)    
    return dy2dt

#%% Simulation parameters:
Fs = 100       # Sampling frequency
h = 1/Fs       # Time step
N = 500000     # Total number of points

# Range in which par_ctrl is changed to incresae the amplitude of the limit
# cycle (x2,y2):
n_start = 135000
n_end = 266000
n_step = 0.000035

# Scaling factor for adding noise to the input:
sf1 = 0.002

# Pre-defined vectors:
t = [None]*N
x1 = [None]*N
x2 = [None]*N
y1 = [None]*N
y2 = [None]*N
u = [None]*N
z = [None]*N

# Initial conditions:
t[0] = 0
x1[0] = 0
y1[0] = -5
z[0] = 3
x2[0] = 0
y2[0] = 0
u[0] = 0

# Initial value for par_ctrl:
par_ctr = 1
PC = []

#%% 4th Order Runge-Kutta integrator:

# Time required for the simulation:
time0 = time.time()

for k in range(0,N-1):
    print('Iteration number: ' + str(k))
    # dx1dt:
    dx1dt_k1 = dx1dt(x1[k],x2[k],y1[k],z[k])
    dx1dt_k2 = dx1dt(x1[k] + h*dx1dt_k1/2,x2[k] + h*dx1dt_k1/2,y1[k] + h*dx1dt_k1/2,z[k] + h*dx1dt_k1/2)
    dx1dt_k3 = dx1dt(x1[k] + h*dx1dt_k2/2,x2[k] + h*dx1dt_k2/2,y1[k] + h*dx1dt_k2/2,z[k] + h*dx1dt_k2/2)
    dx1dt_k4 = dx1dt(x1[k] + h*dx1dt_k3,x2[k] + h*dx1dt_k3,y1[k] + h*dx1dt_k3,z[k] + h*dx1dt_k3)
    x1[k+1] = x1[k] + h*(dx1dt_k1 + 2*dx1dt_k2 + 2*dx1dt_k3 + dx1dt_k4)/6
    
    # dy1dt:
    dy1dt_k1 = dy1dt(x1[k],y1[k])
    dy1dt_k2 = dy1dt(x1[k] + h*dy1dt_k1/2,y1[k] + h*dy1dt_k1/2)
    dy1dt_k3 = dy1dt(x1[k] + h*dy1dt_k2/2,y1[k] + h*dy1dt_k2/2)
    dy1dt_k4 = dy1dt(x1[k] + h*dy1dt_k3,y1[k] + h*dy1dt_k3)
    y1[k+1] = y1[k] + h*(dy1dt_k1 + 2*dy1dt_k2 + 2*dy1dt_k3 + dy1dt_k4)/6

    # dzdt:
    dzdt_k1 = dzdt(x1[k],z[k])
    dzdt_k2 = dzdt(x1[k] + h*dzdt_k1/2,z[k] + h*dzdt_k1/2)
    dzdt_k3 = dzdt(x1[k] + h*dzdt_k2/2,z[k] + h*dzdt_k2/2)
    dzdt_k4 = dzdt(x1[k] + h*dzdt_k3,z[k] + h*dzdt_k3)
    z[k+1] = z[k] + h*(dzdt_k1 + 2*dzdt_k2 + 2*dzdt_k3 + dzdt_k4)/6   

    # dudt:
    dudt_k1 = dudt(u[k],x1[k])
    dudt_k2 = dudt(u[k] + h*dudt_k1/2,x1[k] + h*dudt_k1/2)
    dudt_k3 = dudt(u[k] + h*dudt_k2/2,x1[k] + h*dudt_k2/2)
    dudt_k4 = dudt(u[k] + h*dudt_k3,x1[k] + h*dudt_k3)
    u[k+1] = u[k] + h*(dudt_k1 + 2*dudt_k2 + 2*dudt_k3 + dudt_k4)/6

    # dx2dt:
    dx2dt_k1 = dx2dt(y2[k],x2[k],u[k],z[k], par_ctr)
    dx2dt_k2 = dx2dt(y2[k] + h*dx2dt_k1/2,x2[k] + h*dx2dt_k1/2,u[k] + h*dx2dt_k1/2,z[k] + h*dx2dt_k1/2, par_ctr)
    dx2dt_k3 = dx2dt(y2[k] + h*dx2dt_k2/2,x2[k] + h*dx2dt_k2/2,u[k] + h*dx2dt_k2/2,z[k] + h*dx2dt_k2/2, par_ctr)
    dx2dt_k4 = dx2dt(y2[k] + h*dx2dt_k3,x2[k] + h*dx2dt_k3,u[k] + h*dx2dt_k3,z[k] + h*dx2dt_k3, par_ctr)
    x2[k+1] = x2[k] + h*(dx2dt_k1 + 2*dx2dt_k2 + 2*dx2dt_k3 + dx2dt_k4)/6
    
    # Changing the parameter par_ctr increases the amplitude of the limit cycle (x2,y2):
    if k == n_start:
        par_ctr = 5.1
    if k >= n_start and k < n_end:
        par_ctr = par_ctr - n_step
        PC.append(par_ctr)
    if k >= n_end:
        par_ctr = 1
        
    # dy2dt:
    dy2dt_k1 = dy2dt(y2[k],x1[k],x2[k])
    dy2dt_k2 = dy2dt(y2[k] + h*dy2dt_k1/2,x1[k] + h*dy2dt_k1/2,x2[k] + h*dy2dt_k1/2)
    dy2dt_k3 = dy2dt(y2[k] + h*dy2dt_k2/2,x1[k] + h*dy2dt_k2/2,x2[k] + h*dy2dt_k2/2)
    dy2dt_k4 = dy2dt(y2[k] + h*dy2dt_k3,x1[k] + h*dy2dt_k3,x2[k] + h*dy2dt_k3)
    y2[k+1] = y2[k] + h*(dy2dt_k1 + dy2dt_k2 + dy2dt_k3 + dy2dt_k4)/6 + float(sf1*np.random.normal(0,1,1) )
       
    # Time vector:
    t[k+1] = t[k] + h

time_elapsed = time.time() - time0
print('Time elapsed: ' + str(round(time_elapsed/60,2)) + ' min')

#%% Local field potential: s1*x1 + s2*x2
s1 = 0.1
s2 = 0.8
LFP = s1*np.asarray(x1) + s2*np.asarray(x2)

# RMS value of the LFP:
RMS = np.sqrt(np.mean(LFP**2))

# Adding noise to the output (measurement noise):
sf2 = 0.01   # Scaling factor
Xp = LFP + sf2*RMS*np.random.normal(0,1,len(LFP)) 

#%% Local field potentials over time:
plt.figure(1)
plt.subplot(211)
plt.plot(np.asarray(t[n_start-70000:n_end+110000])/60,Xp[n_start-70000:n_end+110000],'k',linewidth=1)
plt.xlabel('$Time$ $(Mins)$', fontsize=20)
plt.ylabel('$LFP$', fontsize=20)
plt.xlim(t[n_start-70000]/60, t[n_end+110000]/60)
plt.ylim(min(Xp),max(Xp))
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

plt.subplot(212)
plt.plot(np.asarray(t[n_start-70000:n_end+110000])/60,z[n_start-70000:n_end+110000],'r',linewidth=2)
plt.xlabel('$Time$ $(Mins)$', fontsize=20)
plt.ylabel('$z(t)$', fontsize=20)
plt.xlim(t[n_start-70000]/60, t[n_end+110000]/60)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()

#%% Phase portraits for each subsystem:
plt.figure(2)
plt.subplot(121)
plt.plot(x1,y1,'b')
plt.xlabel('$x_1$',fontsize=20)
plt.ylabel('$y_1$',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.show()
        
plt.subplot(122)
plt.plot(x2,y2,'b')
plt.xlabel('$x_2$',fontsize=15)
plt.ylabel('$y_2$',fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()
plt.show()

#%% Plot for z vs x1:
plt.figure(3)
plt.plot(z[n_start-70000:n_end+110000], x1[n_start-70000:n_end+110000], 'b')
plt.xlabel('$z$',fontsize=20)
plt.ylabel('$x_1$',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.grid()