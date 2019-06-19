import pickle
import participant
import random
import numpy as np

class TS_global_params:
    
    
    
    '''
    Keeps track of hyper-parameters for any TS procedure. 
    '''
    
    def __init__(self,xi=10,baseline_features=None,psi_features=None,responsivity_keys=None,uparams = None):
        self.nums = set([np.float64,int,float])
        self.pi_max = 0.8
        self.pi_min = 0.1
        self.sigma =1.15            
        self.baseline_features=baseline_features
        

        self.responsivity_keys = responsivity_keys
        self.num_baseline_features = len(baseline_features)
        self.psi_features = psi_features
        self.num_responsivity_features = len(responsivity_keys)
       
        self.psi_indices = psi_features

        
        self.xi  = xi
        
        self.update_period=7
        self.gamma_mdp = .9
        self.lambda_knot = .9 
        self.prob_sedentary = .9 
        self.weight = .5
        
        self.inv_term = None
        self.to_save_params = {}
        
      
        self.theta_dim =1+self.num_baseline_features + 2*(1+self.num_responsivity_features)
        self.baseline_indices =  [i for i in range(self.theta_dim)]
        #print(self.theta_dim)
        self.mu_theta =np.zeros(self.theta_dim)
        self.mu_theta[0]=4.6
        self.sigma_theta =self.get_theta(self.theta_dim)
        self.lr = 0.0001
      
        self.sigma_u =np.array([[0.06449696, 0.01502549 ],[ 0.01502549,    0.00896479]])
        
        self.rho_term =1.6248689729968946
        
        self.u1 =0.06449696
        
        self.u2 =0.00896479
        
        self.noise_term =1.3305
        
        self.o_noise_term =1.3305
   
        self.cov=np.array([1])

        self.decision_times = 1
        self.kdim = self.theta_dim+2

        self.last_global_update_time = None
        
        self.standardize=False
        
        self.user_id_index=None
        self.user_day_index = None
        self.write_directory ='../temp'

        self.updated_cov = False
        self.history = None
        self.mus0 = None
        self.sigmas0 =None
        
        self.mus1 = None
        self.sigmas1 =None
        
        self.mus2 = None
        self.sigmas2 = None
        if uparams is not None:
            self.init_u_params(uparams)
    
    def init_u_params(self,uparams):
        self.u1 = uparams[0]
        self.u2 = uparams[1]
        self.u3 = uparams[2]
        self.u4 = uparams[3]
    
        self.r12 = uparams[4]
        self.r13 = uparams[5]
        self.r14 = uparams[6]
        self.r23 = uparams[7]
        self.r24 = uparams[8]
        self.r34 = uparams[9]
        
        self.init_u1 = uparams[0]
        self.init_u2 = uparams[1]
        self.init_u3 = uparams[2]
        self.init_u4 = uparams[3]
        
        self.init_r12 = uparams[4]
        self.init_r13 = uparams[5]
        self.init_r14 = uparams[6]
        self.init_r23 = uparams[7]
        self.init_r24 = uparams[8]
        self.init_r34 = uparams[9]
    
    
    
    def feat0_function(self,z,x):
        
        
        temp =  [1]
        temp.extend(z)

        if type(x) in self.nums:
        
            temp.append(x)
        else:
            temp.extend(x)
        return temp

    def feat1_function(self,z,x):
        temp =  [1]
        temp.extend(z)
        if type(x) in self.nums:
        
            temp.append(x)
        else:
            temp.extend(x)
        return temp    
        
        
    def feat2_function(self,z,x):
        temp = [1,z[0]]
        if type(x) in self.nums:
        
            temp.append(x)
        else:
            temp.extend(x)
        
        return temp
            
    def get_mu0(self,z_init):
        return [0 for i in range(len(self.feat0_function(z_init,0)))]
    
    def get_mu1(self,num_baseline_features):
        return [0 for i in range(num_baseline_features+1)]
    
    def get_mu2(self,num_responsivity_features):
        return [0 for i in range(num_responsivity_features+1)]
    
    def get_asigma(self,adim):
        return np.diag([1 for i in range(adim)])
    
    
    def comput_rho(self,sigma_u):
        t =sigma_u[0][0]**.5 * sigma_u[1][1]**.5
        r = (sigma_u[0][1]+t)/t
        return r
    
    
    def update_params(self,pdict):
        self.noise_term=pdict['noise']
        self.sigma_u = pdict['sigma_u']
      
        self.rho_term = self.comput_rho(pdict['sigma_u'])
        self.cov = pdict['cov']
        self.updated_cov=True
    
    def update_params_more(self,pdict):
        self.noise_term=pdict['noise']
        
      
        
        self.cov = pdict['cov']
        self.update_uparams(pdict['uparams'])
        self.updated_cov=True
    

    def get_theta(self,dim_baseline):
        m = 1*np.eye(dim_baseline)

        return m

    def update_cov(self,current_dts):
        cov = np.eye(current_dts)

        self.cov=cov


    def update_mus(self,pid,mu_value,which_mu):
        if which_mu==0:
            self.mus0=mu_value
        
        if which_mu==1:
            self.mus1=mu_value
        
        if which_mu==2:
            self.mus2=mu_value

    def update_sigmas(self,pid,sigma_value,which_sigma):
        if which_sigma==0:
            self.sigmas0=sigma_value
        
        if which_sigma==1:
            self.sigmas1=sigma_value

        if which_sigma==2:
            self.sigmas2=sigma_value
