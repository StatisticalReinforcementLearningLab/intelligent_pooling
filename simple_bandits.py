import gpflow
import numpy as np
import math
import tensorflow as tf
import sys
import pickle
import pandas as pd
from sklearn import preprocessing
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.simplefilter('ignore')
import os
import random
import sim_functions as sf


def get_inv_term(cov,X_dim,noise_term):
    noise = noise_term * np.eye(X_dim)
    middle_term = np.add(cov,noise)

    return np.linalg.inv(middle_term)



def get_theta(dim_baseline):
    m = np.eye(dim_baseline)

    return m




def get_sigma_u(u1,u2,rho):
    off_diagaonal_term = u1**.5*u2**.5*(rho-1)
    return np.array([[u1,off_diagaonal_term],[off_diagaonal_term,u2]])




def create_H(num_baseline_features,num_responsivity_features,psi_indices):
  
    
    random_effect_one = [1]
    random_effect_two = [1]
    
    column_one = [1]
    column_one = column_one+[0]*num_baseline_features
    column_one = column_one+[0]
    column_one = column_one+[0]*num_responsivity_features
    column_one = column_one+[0]
    column_one = column_one+[0]*num_responsivity_features
    
    
    column_two = [0]
    column_two = column_two+[0]*num_baseline_features
    column_two = column_two+[int(i in psi_indices) for i in range(2*num_responsivity_features+2)]

    
    return np.transpose(np.array([column_one,column_two]))







def get_M(global_params,user_id,user_study_day,history):
  
  
    day_id =user_study_day

    M = [[] for i in range(history.shape[0])]

    H = create_H(global_params.num_baseline_features,global_params.num_responsivity_features,global_params.psi_indices)

    for x_old_i in range(history.shape[0]):
        x_old = history[x_old_i]
        old_user_id = x_old[global_params.user_id_index]
        old_day_id = x_old[global_params.user_day_index]
        
     
        phi = np.array([x_old[i] for i in global_params.baseline_indices])
        
        t_one = np.dot(np.transpose(phi),global_params.sigma_theta)

        
        temp = np.dot(H,global_params.sigma_u)
        temp = np.dot(temp,H.T)
        temp = np.dot(np.transpose(phi),temp)
        temp = float(old_user_id==user_id)*temp
        t_two = temp

        temp = np.dot(H,global_params.sigma_v.reshape(2,2))
        temp = np.dot(temp,H.T)
        temp = np.dot(np.transpose(phi),temp)
        temp = rbf_custom_np(user_study_day,old_day_id)*temp
        t_three = temp

        
     
        term = np.add(t_one,t_two)
        
        term = np.add(term,t_three)
     
        M[x_old_i]=term

    return np.array(M)

def get_RT(y,X,sigma_theta,x_dim):
    
    to_return = [y[i]-np.dot(X[i][0:x_dim],sigma_theta) for i in range(len(X))]
    return np.array([i[0] for i in to_return])



def get_M_faster(global_params,user_id,user_study_day,history,users,sigma_u):
    
    
    day_id =user_study_day
    #print(history)
    M = [[] for i in range(history.shape[0])]
    
    H = create_H(global_params.num_baseline_features,global_params.num_responsivity_features,global_params.psi_indices)
    
    phi = history[:,global_params.baseline_indices]
    
    t_one = np.dot(phi,global_params.sigma_theta)

    temp = np.dot(H,global_params.sigma_u)
    
    temp = np.dot(temp,H.T)
    temp = np.dot(phi,temp)
    
    user_ids =users
   

    my_days = np.ma.masked_where(user_ids==user_id, user_ids).mask.astype(float)
    
    if type(my_days)!=np.ndarray:
        my_days = np.zeros(history.shape[0])
    user_matrix = np.diag(my_days)

    t_two = np.matmul(user_matrix,temp)
  
    term = np.add(t_one,t_two)
    
    
    return term


def get_M_faster_four(global_params,user_id,user_study_day,history,users,sigma_u):
    
    
    day_id =user_study_day
 
    M = [[] for i in range(history.shape[0])]
    
    H = create_H_four(global_params.num_baseline_features,global_params.num_responsivity_features,global_params.psi_indices)
    
    phi = history[:,global_params.baseline_indices]
   
    t_one = np.dot(phi,global_params.sigma_theta)

    temp = np.dot(H,sigma_u)
    
    temp = np.dot(temp,H.T)
    temp = np.dot(phi,temp)
    
    user_ids =users
 
    
    my_days = np.ma.masked_where(user_ids==user_id, user_ids).mask.astype(float)
    
    if type(my_days)!=np.ndarray:
        my_days = np.zeros(history.shape[0])
    user_matrix = np.diag(my_days)

    t_two = np.matmul(user_matrix,temp)

    term = np.add(t_one,t_two)
    
    
    return term





def calculate_posterior_current(global_params,user_id,user_study_day,X,users,y):
    sigma_u =get_sigma_umore(global_params)
   
    H = create_H_four(global_params.num_baseline_features,global_params.num_responsivity_features,global_params.psi_indices)
    
    M = get_M_faster_four(global_params,user_id,user_study_day,X,users,sigma_u)
    
    adjusted_rewards =get_RT(y,X,global_params.mu_theta,global_params.theta_dim)
    
    mu = get_middle_term(X.shape[0],global_params.cov,global_params.noise_term,M,adjusted_rewards,global_params.mu_theta,global_params.inv_term)

    sigma = get_post_sigma(H,global_params.cov,sigma_u,None,global_params.noise_term,M,X.shape[0],global_params.sigma_theta,global_params.inv_term)
    
    return mu[-(global_params.num_responsivity_features+1):],[j[-(global_params.num_responsivity_features+1):] for j in sigma[-(global_params.num_responsivity_features+1):]]


def calculate_posterior_time_effects(global_params,user_id,user_study_day,X,users,days,y):
    
    H = create_H(global_params.num_baseline_features,global_params.num_responsivity_features,global_params.psi_indices)
    
    M = get_M_faster_time_effects(global_params,user_id,user_study_day,X,users,global_params.sigma_u,days,global_params.sigma_v)
  
    adjusted_rewards =get_RT(y,X,global_params.mu_theta,global_params.theta_dim)
    
    
    mu = get_middle_term(X.shape[0],global_params.cov,global_params.noise_term,M,adjusted_rewards,global_params.mu_theta,global_params.inv_term)
 

    sigma = get_post_sigma_time(H,global_params.cov,global_params.sigma_u.reshape(2,2),global_params.sigma_v,global_params.noise_term,M,X.shape[0],global_params.sigma_theta,global_params.inv_term)
    
    return mu[-(global_params.num_responsivity_features+1):],[j[-(global_params.num_responsivity_features+1):] for j in sigma[-(global_params.num_responsivity_features+1):]]



def calculate_posterior(global_params,user_id,user_study_day,X,y):
    H = create_H(global_params.num_baseline_features,global_params.num_responsivity_features,global_params.psi_indices)
   
    M = get_M(global_params,user_id,user_study_day,X)
 
    adjusted_rewards =get_RT(y,X,global_params.mu_theta,global_params.theta_dim)
   
    mu = get_middle_term(X.shape[0],global_params.cov,global_params.noise_term,M,adjusted_rewards,global_params.mu_theta,global_params.inv_term)

    sigma = get_post_sigma(H,global_params.cov,global_params.sigma_u.reshape(2,2),global_params.sigma_v.reshape(2,2),global_params.noise_term,M,X.shape[0],global_params.sigma_theta)

    return mu[-(global_params.num_responsivity_features+1):],[j[-(global_params.num_responsivity_features+1):] for j in sigma[-(global_params.num_responsivity_features+1):]]




def get_middle_term(X_dim,cov,noise_term,M,adjusted_rewards,mu_theta,inv_term):
  
    middle_term = np.matmul(M.T,inv_term)
   
    middle_term = np.matmul(middle_term,adjusted_rewards)

    return np.add(mu_theta,middle_term)

def get_post_sigma(H,cov,sigma_u,sigma_v,noise_term,M,x_dim,sigma_theta,inv_term):
   
    first_term = sigma_u
    
    first_term = np.dot(H,first_term)
   
    first_term = np.dot(first_term,H.T)
   
    middle_term = np.dot(M.T,inv_term)

    middle_term = np.dot(middle_term,M)

    last = np.add(sigma_theta,first_term)
    last = np.subtract(last,middle_term)
    
    return last

