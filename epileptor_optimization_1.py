# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:17:01 2021

@author: João Angelo Ferres Brogin
"""

import time
import numpy as np
import scipy.optimize
from matplotlib import rc
import random

rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from epileptor_optimization_2 import Y

#%% Simulation parameters::
ident_par_list = []
min_func_list = []
cell = 250              # Number of points for each cell
Fs = 100                # Sampling frequency
dt = 1/Fs               # Time step
nd = 6                  # Number of differential equations
N = cell      
t0 = 0
tf = N*dt
t = np.linspace(t0, tf, N)
X = np.zeros((nd, N))
N_ref = 200000
n_windows = int(N_ref/cell)

#%% Main loop (windows over the reference signal):
for outer_loop in range(0, n_windows):
    inic = outer_loop
    final = outer_loop + 1
    LFP1_ref = Y[0, inic * cell : final * cell]
    LFP2_ref = Y[3, inic * cell : final * cell] 
    LFP_ref = -LFP1_ref + LFP2_ref

#%% Differential equations in matrix notation:
    def __dXdt__( Xd, p ):
       global h1, h1z, c1, c3, h2, c2
       x1 = Xd[0]
       y1 = Xd[1]
       z = Xd[2]
       x2 = Xd[3]
       y2 = Xd[4]
       u = Xd[5]
       X_ = np.array([x1,y1,z,x2,y2,u])
    
       # Estimated parameters:
       x00 = p[0]
       tau2 = p[1]
       gamma = p[2]
       I1 = p[3]
       I2 = p[4]
       y0 = p[5]
       tau0 = p[6]
       tau1 = p[7]
    
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
              
       # Dynamic Matrix A:
       A = np.array([[       -h1,         1,       -1,              0,            0,            0], \
                     [-5*x1/tau1,   -1/tau1,        0,              0,            0,            0], \
                     [    4/tau0,         0,  -1/tau0,              0,            0,            0], \
                     [         0,         0,     -0.3,    (1 - x2**2),           -1,            2], \
                     [         0,         0,        0,        h2/tau2,      -1/tau2,            0], \
                     [ 0.1*gamma,         0,        0,              0,            0,       -gamma] ]) 
    
        # Matrix B:
       b = np.array([ I1, \
                  y0/tau1, \
                  -4*x00/tau0, \
                  I2 + (3.5*0.3), \
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
    
#%% Fourth-Order Runge-Kutta Integrator:     

    x_transl = Y[:,inic * cell]

    x10 = x_transl[0]
    y10 = x_transl[1]
    z0 = x_transl[2]
    x20 = x_transl[3]
    y20 = x_transl[4]
    u0 = x_transl[5]
    X[:,0] = [x10, y10, z0, x20, y20, u0]

    func_eval = []
    
    # Objective function:
    def __MSEObjetive__( x0 ):
        for k in range(0, N-1):
            k1 = __dXdt__( X[:,k], x0 )
            k2 = __dXdt__( X[:,k] + k1*(dt/2), x0 )
            k3 = __dXdt__( X[:,k] + k2*(dt/2), x0 )
            k4 = __dXdt__( X[:,k] + k3*dt, x0 )   
            X[:,k+1] = X[:,k] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) 

        LFP1 = X[0,:]  
        LFP2 = X[3,:] 
        LFP = -LFP1 + LFP2
        p1 = np.sqrt( np.mean( (LFP_ref - LFP)**2 ) )
        p2 = np.sqrt( ( max(LFP_ref) - min(LFP_ref) )**2  )
        aux = p1/p2      
        func_eval.append(aux)
        print(aux)

        return aux

    time0 = time.time()

#%% Optimization process:
    ref_par = np.array([-1.6, 10, 0.01, 3.1, 0.45, 1, 2857, 1])

    pctg = 0.25 # Uncertainty factor
    v_pms = [random.randint(-100,100) for i in range(0,8)] # Random initial guesses
    v_pms = np.asarray(v_pms)
    v_pms = pctg * v_pms/(max(abs(v_pms))) # Normalized random initial guesses
    ref_par_pctg = ref_par + ref_par * v_pms # New initial guesses
    initial_guess = ref_par_pctg

    obj_func = lambda x0: __MSEObjetive__( x0 )
    LFP_opt = scipy.optimize.fmin(func = obj_func, x0 = initial_guess, args=(), xtol=1e-07, ftol=1e-07, maxiter=None, maxfun=2000, full_output=0, disp=1, retall=0, callback=None, initial_simplex=None)

    minval = obj_func(LFP_opt)
    ident_par_list.append( LFP_opt )
    min_func_list.append( minval )

    elapsed_time = time.time() - time0
    print('Elapsed time: ' + str(round(elapsed_time/60,2)) + ' [min]')
    print('################################## ITERATION No: ' + str(outer_loop + 1) +  ' ##################################')
