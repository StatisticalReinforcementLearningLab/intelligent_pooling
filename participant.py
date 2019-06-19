import numpy as np
import pandas as pd
import math

class participant:
    
    
    
    '''
    A population is an object which tracks participants. 
    It also tracks the total number of active days. 
    Also which participants are involved at which times. 
    '''
    
    def __init__(self, pid=None,times=None,decision_times=None,gid=None,Z=None,rg=None,beta=None):
        self.root = None
       
        self.pid = pid
        
        self.times =  times
        
        self.decision_times = decision_times
        
        self.gid = gid 
        
        self.current_day_counter = 0
        self.current_day=times[0]
        
        self.first_week = None
        
        self.history = {}
        
        self.tod = None
        self.dow = None
        self.weather = None
        self.location = None
        
        ##intervention related states
        self.ltps = None
        self.pds = None
        self.variation = None
        self.dosage = None
        self.duration = None
        self.inaction_dur = None
        self.action_dur = None
        
        self.available = None
        self.steps_last_time_period = 0
        self.steps = 0
        self.last_update_day = times[0]
        self.Z=Z
        self.rando_gen = rg
        self.beta = beta
    
    def set_current_day(self,time):
        if time.date()!=self.current_day.date():
            self.current_day=time
            self.current_day_counter=self.current_day_counter+1
    
    def get_current_day(self):
        pass

    def set_inaction_duration(self,duration):
        self.inaction_dur=duration
        
    def set_action_duration(self,duration):
        self.action_dur=duration
        
    def set_duration(self,duration):
        self.duration=duration
        
    def set_available(self,available):
        self.available=available
        
    def update_inaction_duration(self):
        self.inaction_dur=self.inaction_dur+1
        
    def update_action_duration(self):
        self.action_dur=self.action_dur+1  
        
        
    def update_duration(self,steps):
        if steps>0:
            self.update_action_duration()
            self.set_inaction_duration(0)
        else:
            self.update_inaction_duration()
            self.set_action_duration(0)
        duration = self.action_dur
        if self.action_dur==0:
            duration = self.inaction_dur
            
        duration = int(duration>5)
        self.duration = duration
        
    def set_dosage(self,dosage):
        self.dosage = dosage
    
    def update_dosage(self,action):
        current_dosage = self.dosage
        if action==1:
            current_dosage = current_dosage+2
        else:
            current_dosage=current_dosage-1
        if current_dosage>100:
            current_dosage=100
        if current_dosage<1:
            current_dosage=1 
        self.set_dosage(current_dosage)
        

        
    def set_tod(self,tod):
        self.tod=tod
    
    def get_tod(self):
        return self.tod
        
            
    def set_dow(self,dow):
        self.dow = dow
    
    def get_dow(self):
        return self.dow
        
    def set_wea(self,wea):
        self.weather = wea
    
    def get_wea(self):
        return self.weather
            
    def set_loc(self,location):
        self.location = location
    
    def get_loc(self):
        return self.location
    
    def set_last_time_period_steps(self,steps):
        self.ltps = steps
        
    def set_yesterday_steps(self,steps):
        self.pds = steps
        
    def set_variation(self,var):
        self.variation = var
    
    def get_prekey(self,duration,pre):
        if pre==1:
            if duration == 0:
                prekey = 0
            else:
                prekey = 1
        else:
            if duration == 1:
                prekey= 3
            else:
                prekey = 2
        return prekey
    
    def get_pretreatment(self,steps):
        steps = math.log(steps+.5)
        return int(steps>math.log(.5))
        
    def find_last_time_period_steps(self,timestamp):
       
        return self.get_pretreatment(self.steps_last_time_period)
        
    def find_yesterday_steps(self,time):
        yesterday = time-pd.DateOffset(days=1)
        
        y_steps = [self.history[t]['steps'] for t in self.history.keys() if t.date()==\
                  yesterday]
        
        y_steps = sum(y_steps)**.5
        return y_steps
    
    
    def get_day_steps_raw(self,day):
        
        
        y_steps = [self.history[t]['steps'] for t in self.history.keys() if t.date()==\
                  day.date()]
        return y_steps
    
    def steps_range(self,time_one,time_two):
        
        y_steps = [self.history[t]['steps'] for t in self.history.keys() if t>=time_one and t<time_two]
        return y_steps
    
    def get_day_slices(self,start,end):
        days = pd.date_range(start = start,end =end,freq='D')
        return days
    
    
    def find_variation(self,time):
         
        if time<self.times[0].date()+pd.DateOffset(days=8):
            start = self.times[0]
        else:
            start = time-pd.DateOffset(days=8)

        end = time-pd.DateOffset(days=2)        
        days = self.get_day_slices(start,end)
        stds = [np.array(self.get_day_steps_raw(d)).std() for d in days]
        median = np.median(np.array(stds))
        
        yesterday = time-pd.DateOffset(days=1)
        yesterday_steps = np.array(self.get_day_steps_raw(yesterday)).std()
        
        
        
        return int(yesterday_steps>median)
        
        
