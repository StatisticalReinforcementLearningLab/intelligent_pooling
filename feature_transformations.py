import pandas as pd
import numpy as np
import pickle

import os
import math

import random
from datetime import datetime

from sklearn import preprocessing

class feature_transformation:


        def __init__(self,root):


            with open('{}label_to_temperature_value_stats.pkl'.format(root),'rb') as f:
                self.weather_label_to_val = pickle.load(f)

            with open('{}loc_label_to_intervention_label.pkl'.format(root),'rb') as f:
                self.loc_label_to_val = pickle.load(f)


            with open('{}baseline_sufficient_statistics.pkl'.format(root),'rb') as f:
                self.dists = pickle.load(f)


            with open('{}anti_sedentary_sufficient_statistics.pkl'.format(root),'rb') as f:
                self.dists_intervention_anti_sedentary = pickle.load(f)



            with open('{}activity_suggestion_sufficient_statistics.pkl'.format(root),'rb') as f:
                self.dists_intervention_activity_suggestion = pickle.load(f)




            with open('{}initial_location_distributions.pkl'.format(root),'rb') as f:
                self.loc_lookup_prior = pickle.load(f)

            with open('{}location_conditiononed_on_last_location.pkl'.format(root),'rb') as f:
                self.loc_lookup_transition = pickle.load(f)

            with open('{}initial_temperature_distributions.pkl'.format(root),'rb') as f:
                self.temperature_lookup_prior = pickle.load(f)

            with open('{}temperature_conditiononed_on_last_temperature.pkl'.format(root),'rb') as f:
                self.temperature_lookup_transition = pickle.load(f)

            with open('{}hour_to_id.pkl'.format(root),'rb') as f:
                        self.hour_lookup = pickle.load(f)

            with open('{}day_to_id.pkl'.format(root),'rb') as f:
                        self.day_lookup = pickle.load(f)


        

        def history_to_matrix_for_gp(self,which_history = None):
            pass
        
        def get_time_of_day(self,an_index):
            
            return self.hour_lookup[an_index.hour]

        def get_day_of_week(self,an_index):
            
            return self.day_lookup[an_index.dayofweek]
    
        def get_pretreatment(self,steps):
 
            return int(steps>4.83)
        
        def get_add_no_action(self,state_vector,beta,Z):
            if Z is None:
                Z=0
            return np.dot(beta,state_vector)+Z
        
    
        def add_intercept(self,data):
            return [[1]+x for x in data]
        
        def get_history_decision_time_avail(self,exp,last_update_time):
            to_return = {}
                            
            for userid,data in exp.population.items():
                to_return[userid]= {k:v for k,v in data.history.items() if k<last_update_time and v['avail'] and v['decision_time']}
            
            return to_return
            
        def get_history_decision_time_avail_single(self,history_dict,last_update_time):
            to_return = {}
            
            for userid,data in history_dict.items():
                to_return[userid]= {k:v for k,v in data.items() if k<last_update_time and v['avail'] and v['decision_time']}
            
            return to_return
        
       
        def history_semi_continuous(self,history_dict,glob):
            probs ={}
            actions = {}
            users = {}
            baseline_data = {}
            responsivity_data = {}
            steps = {}

            history_count = 0
            for user_id,history in history_dict.items():

                for hk,h in history.items():
                    base_line = [h[b] for b in glob.baseline_features]
                    responsivity_vec =[h[b] for b in glob.responsivity_keys]
                   
                    baseline_data[history_count]=base_line
                    responsivity_data[history_count]=responsivity_vec
                    probs[history_count]=h['prob']
                    actions[history_count]=h['action']
                    users[history_count]=user_id
                    steps[history_count]=h['steps']
                    history_count = history_count+1
            if glob.standardize:
                baseline_data = self.standardize(baseline_data)
            return {'base':baseline_data,'resp':responsivity_data,'probs':probs,'actions':actions,'users':users,'steps':steps}

        def get_RT(self,y,X,sigma_theta,x_dim):
                    
            to_return = [y[i]-np.dot(X[i][0:x_dim],sigma_theta) for i in range(len(X))]
            return np.array([i[0] for i in to_return])
        
        def get_RT_o(self,y,X,sigma_theta,x_dim):
            
            to_return = [y[i]-np.dot(X[i][0:x_dim],sigma_theta) for i in range(len(X))]
            return np.array([i for i in to_return])
        
        
        def standardize(self,data):
            
            key_lookup = [i for i in sorted(data.keys())]
            X = [data[i] for i in key_lookup]
            
            new_x = preprocessing.scale(np.array(X))
            return {i:new_x[i] for i in key_lookup}

        def get_phi_from_history_lookups(self,history_lookups):
            
            
            all_data = []
            all_users = []
            all_steps = []
            
            for h_i in history_lookups['base']:
                vec = history_lookups['base'][h_i]
                rvec = history_lookups['resp'][h_i]
                prob = history_lookups['probs'][h_i]
                action = history_lookups['actions'][h_i]
                user = history_lookups['users'][h_i]
                steps = history_lookups['steps'][h_i]
                
                v = [1]
                v.extend(vec)
                v.append(prob*1)
                v.extend([prob*r for r in rvec])
                v.append((action-prob))
                v.extend([(action-prob)*r for r in rvec])
                all_users.append(user)
                all_steps.append(steps)
                all_data.append(v)
            return np.array(all_data),np.array(all_users),np.array([[s] for s in all_steps])

        def get_form_TS(self,history_lookups):
            keys = history_lookups['base'].keys()
            context = np.array([history_lookups['base'][k] for k in keys])
            steps = np.array([history_lookups['steps'][k] for k in keys])
            actions = np.array([history_lookups['actions'][k] for k in keys])
            probs = np.array([history_lookups['probs'][k] for k in keys])
            return context,steps,probs,actions

        def continuous_temperature(self,weather_key):
            dist = self.weather_label_to_val[weather_key]
                
            x = np.random.normal(loc=dist[0],scale=dist[1])

                
              
            return x

        def get_location_prior(self,group_id,day_of_week,time_of_day,seed=None):
            #random.seed(seed)
            loc_lookup=self.loc_lookup_prior
            key = '{}-{}-{}'.format(group_id,day_of_week,time_of_day)

                            ##make a bit smoother while loop instead
            if key in loc_lookup:
                ps = loc_lookup[key]
            else:
                print('missing key')
                                            
            val = np.argmax(seed.multinomial(1,ps))
            return val


        def get_next_location(self,gid,dow,tod,loc,seed=None):
     
            loc_dists = self.loc_lookup_transition
        
            relevant_context = [gid,dow,tod,loc]
        
            context_key = '-'.join([str(c) for c in relevant_context])
            
            dist = loc_dists[context_key]
            
            val = np.argmax(seed.multinomial(1,dist))
            
            return val
        

        def get_weather_prior(self,time_of_day,month,seed=None):
    
            loc_lookup = self.temperature_lookup_prior
            key = '{}-{}'.format(time_of_day,month)

            if key in loc_lookup:
                ps = loc_lookup[key]
            else:
                key =  '{}'.format(time_of_day)
                ps = loc_lookup[key]


            val = np.argmax(seed.multinomial(1,ps))
       
            return val

        def get_next_weather(self,tod,month,weather,seed=None):
            
            loc_dists = self.temperature_lookup_transition
                            
            relevant_context = [tod,month,weather]

            context_key = '-'.join([str(c) for c in relevant_context])
          
            dist = loc_dists[context_key]

            val = np.argmax(seed.multinomial(1,dist))
            
      
            
            return val


        def get_steps_no_action(self,gid,tod,dow,loc,pre,wea,seed=None):
            #random.seed(seed)
            keys = ['gid',str(gid),'tod',str(tod),'dow',str(dow),'wea',str(wea),'pre',str(pre),'loc',str(loc)]
    
            new_key = '-'.join(keys)
    
    
    
            dist = self.dists[new_key]
            scale=dist[1]
        
            if scale==0:
                scale=scale+.001
            x=seed.normal(loc=dist[0],scale=scale)
  
            return x

        def get_steps_action(self,context,seed=None):
            
            ids = ['aint','gid','tod','dow','wea','pre','loc']
            context = [str(c) for c in context]
            new_key = []
            message_type=context[0]
            context[0]='1'
    
            for i in range(len(ids)):
                new_key.append(ids[i])
                new_key.append(context[i])
            new_key = '-'.join(new_key)



            dist = self.dists_intervention_anti_sedentary[new_key]


            scale=dist[1]
            if scale==0:
                scale=scale+.001
            x = seed.normal(loc=dist[0],scale=scale)

            return x
