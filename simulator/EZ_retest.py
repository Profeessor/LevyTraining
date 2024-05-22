#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:47:09 2023

@author: ebrahimi
"""
from numba import jit, prange,njit
import ctypes
from numba.extending import get_cython_function_address
import tensorflow as tf
import numpy as np


# Get a pointer to the C function diffusion.c
addr_diffusion = get_cython_function_address("diffusion", "diffusion_trial")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, 
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_int)

diffusion_trial = functype(addr_diffusion)



@njit
def simulate_diffusion_condition(n_trials,v,sv,zr,szr, a, ndt,sndt,alpha, dt=0.001, max_steps=15000):
    """Simulates a diffusion process over an entire condition."""
    
    x = np.empty(n_trials)
    for i in range(n_trials):
        x[i] = diffusion_trial(v,sv,zr,szr, a, ndt,sndt,alpha, dt, max_steps)
    return x


@njit
def simulate_diffusion_2_conds(theta, n_trials, dt=0.001, max_steps=15000):
    """Simulates a diffusion process for 2 conditions with 7 parameters (v1, v2, a1, a2, ndt1, ndt2, zr=0.5)"""
    
    n_trials_c1 = n_trials[0]
    n_trials_c2 = n_trials[1]
    
    
    v1,v2, a, ndt,alpha = theta
    rt_c1 = simulate_diffusion_condition(n_trials_c1,v1,0.0,0.5,0.0, a, ndt,0.0,alpha, dt, max_steps)
    rt_c2 = simulate_diffusion_condition(n_trials_c2, v2,0.0,0.5,0.0, a, ndt,0.0,alpha, dt, max_steps)
    rts = np.concatenate((rt_c1, rt_c2))
    
    # Assign conditions
    cond_arr = np.concatenate((np.zeros(n_trials_c1), np.ones(n_trials_c2)))
    x = np.stack((rts, cond_arr), axis=1)
    
    return x

def data_generator( n_obs=None, to_tensor=True, n_obs_min=60, 
                   n_obs_max=60, include_criterion=True):
    """
    Runs the forward model 'batch_size' times by first sampling fromt the prior
    theta ~ p(theta) and running x ~ p(x|theta) with the specified n_obs. If 
    n_obs is None, random number of trials for each condition are generated.
    ----------
    
    Arguments:
    batch_size : int -- the number of samples to draw from the prior
    n_obs      : tuple (int, int) or None -- the numebr of observations to draw from p(x|theta)
                                  for each condition
    n_obs_min  : int -- the minimum number of observations per condition
    n_obs_max  : int -- the maximum number of observations per condition
    to_tensor  : boolean -- converts theta and x to tensors if True
    include_criterion : boolean -- whether to set RTs < 0.3 or RT > 10 to missing (0), as in IAT analysis
    ----------
    
    Output:
    theta : tf.Tensor or np.ndarray of shape (batch_size, theta_dim) - the data gen parameters 
    x     : tf.Tensor of np.ndarray of shape (batch_size, n_obs, x_dim)  - the generated data
    """
    
    # Sample from prior
    # theta is a np.array of shape (batch_size, theta_dim)
    theta = draw_prior()
    
    # Fixed or random number of DM samples
    if n_obs is None:
        n_obs = np.random.randint(n_obs_min, n_obs_max+1, 2)
    
    # Generate data
    # x is a np.ndarray of shape (batch_size x n_obs, x_dim)
    x = np.apply_along_axis(simulate_diffusion_2_conds, axis=1, arr=theta, n_trials=n_obs)
    
    # Mark bullshit data with 0
    if include_criterion:
        x[np.abs(x) < 0.3] = 0.
        x[np.abs(x) > 10] = 0.
    
    # Assign conditions
    cond_arr = np.stack( [np.concatenate((np.zeros(n_obs[0]), np.ones(n_obs[1])))] )
    x = np.stack((x, cond_arr), axis=-1)
    
    # Convert to tensor, if specified 
    if to_tensor:
        theta = tf.convert_to_tensor(theta, dtype=tf.float32)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
    return {'theta': theta, 'x': x}