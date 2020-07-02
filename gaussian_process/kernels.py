######### change path according to where you store library files #########
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.special import gamma
from lib.rprop import rprop
import math

class kernel:
    def __init__(self,alpha=0.5,beta=0.5,v=np.pi/np.sqrt(12.),optimizer="slsqp", n_restarts_optimizer=10,epsilon=1.e-5):
        self.alpha=alpha
        self.beta=beta
        self.v=v
        self.optimizer=optimizer
        self.n_restarts_optimizer=n_restarts_optimizer
        self.epsilon=epsilon
    
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y.T
         
        bounds_ = ((1.e-10,1.-1.e-5),(1.e-10,1.-1.e-5),(1.e-10,1000.))
        arg_results=[]
        val_results=[]
        
        for i in range(self.n_restarts_optimizer):
            self.Ve=np.random.beta(self.alpha,self.beta)
            self.tau2=np.random.beta(self.alpha,self.beta)
            sqrtv=np.sqrt(self.v)
            if len(self.X_train.shape)==1:
                self.phi=np.ptp(self.X_train,axis=0)*np.absolute(np.random.normal(0.,sqrtv))
            elif len(self.X_train.shape)==2:
                self.phi=np.ptp(self.X_train,axis=0)*np.absolute(np.random.normal(0.,sqrtv,self.X_train.shape[1]))
            sqrt_theta_initial=self.transform_theta(np.hstack([self.Ve,self.tau2,self.phi]))
            print("Restart number: ",i)            
            if self.optimizer=="l_bfgs_b":
                result=minimize(self.objective_function,sqrt_theta_initial,method='L-BFGS-B',jac=True)#,bounds=bounds_)
                #print("hyperparameters: ",result.x,"objective functionn: ",self.objective_function(result.x)[0])
                arg_results.append(result.x)
                val_results.append(self.objective_function(result.x)[0])
                
            elif self.optimizer=="slsqp":
                result=minimize(self.objective_function,sqrt_theta_initial,method='SLSQP',jac=False)#,bounds=bounds_)
                #print("hyperparameters: ",result.x,"objective function: ",self.objective_function(result.x)[0])
                arg_results.append(result.x)
                val_results.append(self.objective_function(result.x)[0])
                
            elif self.optimizer=="rprop":
                result=rprop(self.objective_function,sqrt_theta_initial,bounds=bounds_)
                #print("hyperparameters: ",result,"objective function: ",self.objective_function(result)[0])
                arg_results.append(result)
                val_results.append(self.objective_function(result)[0])
        
        self.Ve=self.inverse_transform_theta(arg_results[np.nanargmin(np.asarray(val_results))][0])
        self.tau2=self.inverse_transform_theta(arg_results[np.nanargmin(np.asarray(val_results))][1])
        self.phi=self.inverse_transform_theta(arg_results[np.nanargmin(np.asarray(val_results))][2])
        print(self.Ve,self.tau2,self.phi,np.nanmin(np.asarray(val_results)))
        return arg_results,val_results
    
    def distances(self,X1,X2):
        if len(X1.shape)==1:
            return (np.tile(X2,(X1.shape[0],1))-np.tile(X1,(X2.shape[0],1)).T)
        elif len(X1.shape)==2:
            z1=np.tile(X2,(X1.shape[0],1,1))
            z2=np.tile(X1,(X2.shape[0],1,1))
        return z1-np.transpose(z2,(1,0,2))
        
    def k(self,X1,X2,tau2_,phi_):
        if len(X1.shape)==1:
            return tau2_*np.exp(-np.power(self.distances(X1,X2),2)*np.power(phi_,2))
        elif len(X1.shape)==2:
            return tau2_*np.exp(-np.linalg.norm(np.power(self.distances(X1,X2),2)*np.power(phi_,2),axis=2))
    
    def log_marginal_likelihood(self,K,Kinv,y):
        determinant_K_epsilon=np.linalg.det(K)
        if determinant_K_epsilon==0.:
            determinant_K_epsilon=np.linalg.det(K+self.epsilon*np.eye(K.shape[0]))
        return -0.5*np.log(determinant_K_epsilon)-0.5*(y.T @ Kinv @ y)
    
    def prior_Ve_distribution(self,Ve_):
        normalization = gamma(self.alpha+self.beta)/(gamma(self.alpha)*gamma(self.beta))
        return normalization*np.power(Ve_,self.alpha-1.)*np.power(1.-Ve_,self.beta-1.)
    
    def prior_tau2_distribution_gamma(self,tau2_):
        normalization = np.power(self.beta,self.alpha)/gamma(self.alpha)
        return normalization*np.power(tau2_,self.alpha-1.)*np.exp(-self.beta*tau2_)
    """ 
    def prior_tau2_distribution_beta(self,tau2_):
        normalization = gamma(self.alpha+self.beta)/(gamma(self.alpha)*gamma(self.beta))
        return normalization*np.power(tau2_,self.alpha-1.)*np.power(1.-tau2_,self.beta-1.)
    """
    def prior_phi_distribution(self,phi_):
        if isinstance(phi_,(np.ndarray)):
            normalization = 2./np.power(2.*np.pi*self.v,phi_.shape[0]/2.)
        else:
            normalization = 2./np.sqrt(2.*np.pi*self.v)
        return normalization*np.exp(-np.power(np.linalg.norm(phi_),2)/(2.*self.v))
    
    def log_marginal_posterior(self,K,Kinv,y,Ve_,tau2_,phi_):
        return self.log_marginal_likelihood(K,Kinv,y)+np.log(self.prior_Ve_distribution(Ve_))+np.log(self.prior_tau2_distribution_gamma(tau2_))+np.log(self.prior_phi_distribution(phi_))
    
    def gradient_log_marginal_posterior(self,Sigma,K,Kinv,y,Ve_,tau2_,phi_):
        alpha = Kinv @ y
        grad_Ve = 0.5*np.trace(alpha @ alpha.T-Kinv)+(self.alpha-1.)/Ve_-(self.beta-1.)/(1.-Ve_)
        grad_tau2 = 0.5*np.trace(alpha @ alpha.T-Kinv @ Sigma)/tau2_+(self.alpha-1.)/tau2_-self.beta
        Xi_Xj = self.distances(self.X_train,self.X_train)
        grad_phi = []
        for i in range(phi_.shape[0]):
            grad_phi.append(0.5*np.trace(alpha @ alpha.T-Kinv @ (np.power(Xi_Xj[:,:,i],2)*Sigma))+phi_[i]/self.v) #check sign in this line
        return np.hstack([grad_Ve,grad_tau2,grad_phi])
    
    def objective_function(self,sqrt_theta_):
        #theta_[0]=Ve,theta_[1]=tau2,theta_[2]=phi
        theta_=self.inverse_transform_theta(sqrt_theta_)
        Sigma=self.k(self.X_train,self.X_train,theta_[1],theta_[2:])
        K=Sigma+theta_[0]*np.eye(Sigma.shape[0])
        Kinv=np.linalg.pinv(K)
        value=-self.log_marginal_posterior(K,Kinv,self.y_train,theta_[0],theta_[1],theta_[2:])
        gradient=-self.gradient_log_marginal_posterior(Sigma,K,Kinv,self.y_train,theta_[0],theta_[1],theta_[2:])
        return value#,gradient
    
    def transform_theta(self,theta):
        return np.sqrt(theta)
    
    def inverse_transform_theta(self,theta_transform):
        return np.power(theta_transform,2)

    def log_marginal_likelihood_check(self,sqrt_theta_):
        #theta_[0]=Ve,theta_[1]=tau2,theta_[2]=phi
        theta_=self.inverse_transform_theta(sqrt_theta_)
        Sigma=self.k(self.X_train,self.X_train,theta_[1],theta_[2:])
        K=Sigma+theta_[0]*np.eye(Sigma.shape[0])
        Kinv=np.linalg.pinv(K)
        log_marginal_likelihood_=-0.5*np.log(np.linalg.det(K))-0.5*(self.y_train.T @ Kinv @ self.y_train)
        determinant_K=np.linalg.det(K)
        return log_marginal_likelihood_,determinant_K


        

        
