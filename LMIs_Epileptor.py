# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:24:21 2019

@author: João Angelo Ferres Brogin
"""

import time

time0 = time.time()

import numpy as np
import picos as pic
import cvxopt as cvx

#%% Parameters of the model:
tau1 = 1
tau2 = 10.0
tau0 = 2857.0
gamma = 0.01
I1 = 3.1     
I2 = 0.45    
x0 = -1.6
y0 = 1

#%% Maximum and minimum values of the nonlinear functions:
f1max = 1.9851
f1min = -9.9536

f3max = 9.9668
f3min = -7.9107

f4max = 1.0000
f4min = -0.5976

f5max = 0.6
f5min = 0

vetor_max_min = [ [f1min,f1max], [f3min,f3max], [f4min,f4max], [f5min,f5max] ]

Nnl = len(vetor_max_min)
Nsub = 2**Nnl

#%% Set of sub-models:

A1 = np.array([[            f1max,         1,          -1,                                      0,            0,            0], \
               [            f3max,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4max,           -1,            2], \
               [                0,         0,           0,                                  f5max,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A2 = np.array([[            f1max,         1,          -1,                                      0,            0,            0], \
               [            f3max,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4max,           -1,            2], \
               [                0,         0,           0,                                  f5min,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A3 = np.array([[            f1max,         1,          -1,                                      0,            0,            0], \
               [            f3max,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4min,           -1,            2], \
               [                0,         0,           0,                                  f5max,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A4 = np.array([[            f1max,         1,          -1,                                      0,            0,            0], \
               [            f3max,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4min,           -1,            2], \
               [                0,         0,           0,                                  f5min,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A5 = np.array([[            f1max,         1,          -1,                                      0,            0,            0], \
               [            f3min,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4max,           -1,            2], \
               [                0,         0,           0,                                  f5max,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A6 = np.array([[            f1max,         1,          -1,                                      0,            0,            0], \
               [            f3min,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4max,           -1,            2], \
               [                0,         0,           0,                                  f5min,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A7 = np.array([[            f1max,         1,          -1,                                      0,            0,            0], \
               [            f3min,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4min,           -1,            2], \
               [                0,         0,           0,                                  f5max,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A8 = np.array([[            f1max,         1,          -1,                                      0,            0,            0], \
               [            f3min,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4min,           -1,            2], \
               [                0,         0,           0,                                  f5min,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A9 = np.array([[            f1min,         1,          -1,                                      0,            0,            0], \
               [            f3max,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4max,           -1,            2], \
               [                0,         0,           0,                                  f5max,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A10= np.array([[            f1min,         1,          -1,                                      0,            0,            0], \
               [            f3max,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                   f4max,           -1,            2], \
               [                0,         0,           0,                                   f5min,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A11= np.array([[            f1min,         1,          -1,                                      0,            0,            0], \
               [            f3max,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                   f4min,           -1,            2], \
               [                0,         0,           0,                                   f5max,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                       0,            0,       -gamma] ])

A12= np.array([[            f1min,         1,          -1,                                      0,            0,            0], \
               [            f3max,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                   f4min,           -1,            2], \
               [                0,         0,           0,                                   f5min,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A13= np.array([[            f1min,         1,          -1,                                      0,            0,            0], \
               [            f3min,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                   f4max,           -1,            2], \
               [                0,         0,           0,                                   f5max,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                       0,            0,       -gamma] ])

A14= np.array([[            f1min,         1,          -1,                                      0,            0,            0], \
               [            f3min,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4max,           -1,            2], \
               [                0,         0,           0,                                  f5min,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A15 = np.array([[           f1min,         1,          -1,                                      0,            0,            0], \
               [            f3min,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4min,           -1,            2], \
               [                0,         0,           0,                                  f5max,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

A16= np.array([[            f1min,         1,          -1,                                      0,            0,            0], \
               [            f3min,        -1,           0,                                      0,            0,            0], \
               [           4/tau0,         0,     -1/tau0,                                      0,            0,            0], \
               [                0,         0,        -0.3,                                  f4min,           -1,            2], \
               [                0,         0,           0,                                  f5min,      -1/tau2,            0], \
               [        0.1*gamma,         0,           0,                                      0,            0,       -gamma] ])

#%% Input matrices:
B = np.identity(6)
B[2][2] = 0.05

#$$ Constraints on the input:
#mu = 30

# Some initial conditions to test using the constraints on the input:
# c1 = np.array([ 0.41910121, -1.43561464,  3.19707359, -1.24743669,  1.22565394,   0.00804828]) #120000
# c2 = np.array([ 0.73325955, -1.35203863,  3.29575959, -1.25241954,  1.27535408,   0.01771379]) #320000
# c3 = np.array([ 0.35813301,  0.45798644,  3.39428489,  0.57979664,  1.27802522,   0.02338059]) #520000
# c4 = np.array([ 0.3354754 , -1.27151392,  3.19723849, -1.24479871,  1.21345848,   0.00807792]) #120010
# c5 = np.array([ 0.88822892, -1.09502667,  3.19490066, -1.18725144,  1.35455709,   0.00737479]) #119900
# c6 = np.array([-0.11501808,  0.02625421,  3.19830817, -1.21672605,  1.10901755,   0.00804856]) #120100

# x10 = c5[0]
# y10 = c5[1]
# z0 = c5[2]
# x20 = c5[3]
# y20 = c5[4]
# u0 = c5[5]

# # Vector for the inintial conditions:
# X0 = np.array([x10, y10, z0, x20, y20, u0])
# X0T = np.array([[ x10],
#                 [ y10],
#                 [  z0],
#                 [ x20],
#                 [ y20],
#                 [  u0]])

#%% Optimization problem:
prob = pic.Problem()

# Sub-models and parameters:
# Uncomment the following lines if constraints on the input are applied:
# X_0 = pic.new_param('X0', cvx.matrix(X0))
# X_0_T = pic.new_param('X0T', cvx.matrix(X0T))

BB = pic.new_param('B', cvx.matrix(B))
AA1 = pic.new_param('A1', cvx.matrix(A1) )    
AA2 = pic.new_param('A2', cvx.matrix(A2) )   
AA3 = pic.new_param('A3', cvx.matrix(A3) )   
AA4 = pic.new_param('A4', cvx.matrix(A4) )   
AA5 = pic.new_param('A5', cvx.matrix(A5) )   
AA6 = pic.new_param('A6', cvx.matrix(A6) )   
AA7 = pic.new_param('A7', cvx.matrix(A7) )   
AA8 = pic.new_param('A8', cvx.matrix(A8) )   
AA9 = pic.new_param('A9', cvx.matrix(A9) )   
AA10 = pic.new_param('A10', cvx.matrix(A10) )   
AA11 = pic.new_param('A11', cvx.matrix(A11) )   
AA12 = pic.new_param('A12', cvx.matrix(A12) )   
AA13 = pic.new_param('A13', cvx.matrix(A13) )   
AA14 = pic.new_param('A14', cvx.matrix(A14) )   
AA15 = pic.new_param('A15', cvx.matrix(A15) )   
AA16 = pic.new_param('A16', cvx.matrix(A16) )   
                                  
Gx = prob.add_variable('Gx', (6,6), vtype='symmetric')
X = prob.add_variable('X', (6,6), vtype='symmetric')

# Uncomment the following line if constraints on the onput are applied:
# mu = prob.add_variable('mu')

# Decay rate:
sigma = 0

# Restrictions for the sub-models:
prob.add_constraint(X *  AA1.T - Gx.T * BB.T +  AA1 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X *  AA2.T - Gx.T * BB.T +  AA2 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X *  AA3.T - Gx.T * BB.T +  AA3 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X *  AA4.T - Gx.T * BB.T +  AA4 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X *  AA5.T - Gx.T * BB.T +  AA5 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X *  AA6.T - Gx.T * BB.T +  AA6 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X *  AA7.T - Gx.T * BB.T +  AA7 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X *  AA8.T - Gx.T * BB.T +  AA8 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X *  AA9.T - Gx.T * BB.T +  AA9 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X * AA10.T - Gx.T * BB.T + AA10 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X * AA11.T - Gx.T * BB.T + AA11 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X * AA12.T - Gx.T * BB.T + AA12 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X * AA13.T - Gx.T * BB.T + AA13 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X * AA14.T - Gx.T * BB.T + AA14 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X * AA15.T - Gx.T * BB.T + AA15 * X - BB * Gx + 2 * sigma * X << 0 )
prob.add_constraint(X * AA16.T - Gx.T * BB.T + AA16 * X - BB * Gx + 2 * sigma * X << 0 )
    
# Constraints on the input:
# prob.add_constraint( ( (1 & X_0.T )//(X_0 & X) ) >> 0 )
# prob.add_constraint( ( (X & Gx.T )//(Gx & (mu**2)*I) ) >> 0 )

# Positiveness:
prob.add_constraint(X >> 0)

#%% Solve the optimization process:
# Solver:
prob.solve(verbose=1)
print('Status: ' + prob.status)

X = np.matrix(X.value)
Gx = np.matrix(Gx.value)

P = X.I
G = Gx.dot(P)

