import numpy as np
from scipy.stats import norm
import random

def feat2_function(z):
    return z



def function_zero(x):
    return 0


def feat0_function(z,x):
    temp =  [1]
    temp.extend(z)
    temp.append(x)
    return temp

def feat1_function(z):
    return z

def init_mu(val,length):
    return [val]* length

def eta(x):
    return 1

def py_c_func(argone,argtwo):
    if type(argone)!=list:
        temp=[argone]
        if type(argtwo)!=list:
        
            temp.append(argtwo)
        else:
            temp.extend(argtwo)
    else:
        temp = [i for i in argone]
        if type(argtwo)!=list:
            temp.append(argtwo)
        else:
            temp.extend(argtwo)
    return temp
        
    

def calculate_prob(z, x, mu, Sigma, eta, initial_params):
    
    
    pos_mean = np.dot(feat2_function(z,x),mu)
 
    pos_var = np.dot(np.dot(np.transpose(feat2_function(z,x)),Sigma),feat2_function(z,x))
    pos_var = max(0,pos_var)
    
    margin = eta (x) * initial_params.xi
    
    pit_zero = norm.cdf((pos_mean-margin)/(pos_var)**.5)
    

    

    
    prob =  min(py_c_func(initial_params.pi_max, max(py_c_func(initial_params.pi_min, pit_zero))))
    
    return prob


def policy_update(batch,initial_params,proxy):
    
    pass
    


