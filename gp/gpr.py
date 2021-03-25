"""
This package is adapted from Sklearn Gaussian Process Regression by Jan Hendrik Metzen.

Pending modification to support ARD priors. Needs to have internal normalization before using ARD priors
"""

import numpy as np
from .kernels import *
from .priors import *
from .optimizers import rprop
from scipy.linalg import cholesky, cho_solve, solve_triangular
from operator import itemgetter
from scipy.optimize import fmin_l_bfgs_b
import warnings
from sklearn.exceptions import ConvergenceWarning

class gaussian_process_regressor:
    """
    optimizers = 'fmin_l_bfgs_b', rprop
    """
    #=====================================================================
    # rprop optimizer DOESN'T WORK CURRENTLY! NEEDS TO BE DEBUGGED.
    #=====================================================================
    def __init__(self,kernel='sq_exp',optimizer='fmin_l_bfgs_b',prior='ard',normalize_x=False,normalize_y=False,n_restarts_optimizer = 0,sqrt_transform_theta = False):
        self.kernel=kernel
        self.optimizer=optimizer
        if self.optimizer=='rprop':
            self.optimizer=rprop
        self.prior=prior
        self.normalize_x=normalize_x
        self.normalize_y=normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.sqrt_transform_theta = sqrt_transform_theta
        if self.prior=='ard' and (self.normalize_x==False or self.normalize_y==False):
            self.normalize_x=True
            self.normalize_y=True
            print("Setting internal normalization of x_train and y_train to True to be consistent with ARD prior specs")
        
        if self.prior=='ard' and self.kernel=='sq_exp': #generalize this later on
            self.prior_phi = half_normal(var=np.pi/np.sqrt(12.))
            self.prior_tau2 = gamma(k=1.,theta=1.)
            self.prior_noise2 = gamma(k=1.,theta=1.) #beta(a=1.1,b=1.1) causes log_prior = -inf at initial_noise2=1.
        
        if self.optimizer==rprop and self.sqrt_transform_theta==False:
            self.sqrt_transform_theta=True
            print("Setting sqrt_transform_theta to True for proper functioning of rprop.")
    
    def fit(self,x_train,y_train,initial_noise2=1.,initial_phi=1.,initial_tau2=1.):
        if len(x_train.shape)!=2 or len(y_train.shape)!=2:
            raise ValueError("x_train and y_train must be 2d numpy arrays (n_samples,m_features)")
        # Normalize x and y (zero mean, unit variance per column)
        if self.normalize_x:
            self.x_train_mean = np.mean(x_train, axis=0)
            self.x_train_std = np.std(x_train, axis=0)
            x_train = (x_train - self.x_train_mean)/self.x_train_std
        else:
            self.x_train_mean = np.zeros(x_train.shape[1])
            self.x_train_std = np.ones(x_train.shape[1])
        if self.normalize_y:
            self.y_train_mean = np.mean(y_train, axis=0)
            self.y_train_std = np.std(y_train, axis=0)
            y_train = (y_train - self.y_train_mean)/self.y_train_std            
        else:
            self.y_train_mean = np.zeros(y_train.shape[1])
            self.y_train_std = np.ones(y_train.shape[1])
        
        self.x_train = np.copy(x_train)
        self.y_train = np.copy(y_train)
        self.noise2 = initial_noise2
        if np.iterable(initial_phi):
            self.phi = initial_phi
        else:
            self.phi = initial_phi*np.ones(x_train.shape[1])
        self.tau2 = initial_tau2
        

        #=================================================================================
        # The rest of this function is copied from Sklearn GP and modified
        #=================================================================================
        # Choose hyperparameters based on maximizing the log-marginal
        # likelihood (potentially starting from several initial values)
        # First optimize starting from parameters specified in kernel
        parameters_initial = np.concatenate(([self.noise2],[self.tau2],self.phi))
        if self.sqrt_transform_theta==True:
            parameters_initial = self.transform_theta(parameters_initial)
        bounds = []
        for i in parameters_initial:
            bounds.append((1.e-5,1.e5))
        optima = [(self._constrained_optimization(self.obj_func, parameters_initial, bounds))]
        
        # Additional runs are performed from log-uniform chosen initial
        # parameters
        if self.n_restarts_optimizer > 0:
            if not np.isfinite(bounds).all():
                raise ValueError("Multiple optimizer restarts (n_restarts_optimizer>0) requires that all bounds are finite.")
            for iteration in range(self.n_restarts_optimizer):
                if self.prior==None:
                    parameters_initial = np.exp(np.array([np.random.uniform(np.log(bounds[i][0]),np.log(bounds[i][1])) for i in range(len(bounds))]))
                elif self.prior=='ard' and self.kernel=='sq_exp':
                    parameters_initial[0] = self.prior_noise2.sample()
                    parameters_initial[1] = self.prior_noise2.sample()
                    for i in range(2,len(parameters_initial)):
                        parameters_initial[i] = self.prior_phi.sample()
                        
                print(parameters_initial)
                if self.sqrt_transform_theta==True:
                    parameters_initial = self.transform_theta(parameters_initial)
                optima.append(self._constrained_optimization(self.obj_func, parameters_initial,bounds))
        
        # Select result from run with minimal (negative) log-marginal likelihood
        obj_func_values = list(map(itemgetter(1), optima))
        optimum_parameters = optima[np.argmin(obj_func_values)][0]
        #===========================================================================================================
        # This unpacking needs to be standardized (separate function?) (search for any use of the word 'parameter')
        #===========================================================================================================
        if self.sqrt_transform_theta:
            optimum_parameters = self.inverse_transform_theta(optimum_parameters)
        self.noise2 = optimum_parameters[0]
        self.tau2 = optimum_parameters[1]
        self.phi = optimum_parameters[2:]
        
        self.objective_function_value_ = -np.min(obj_func_values)
        
        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = covariance(self.x_train, self.phi, self.tau2, eval_gradient=False)
        K[np.diag_indices_from(K)] += self.noise2
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError:
            print ("The kernel is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the noise parameter of your "
                        "GaussianProcessRegressor estimator.")
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train)  # Line 3
    
    def log_marginal_likelihood(self,parameters=None,eval_gradient=False):
        if not np.iterable(parameters):
            parameters = np.concatenate(([self.noise2],[self.tau2],self.phi))
        
        phi = parameters[2:]
        tau2 = parameters[1]
        noise2 = parameters[0]
        
        #=============================================================================
        # The rest of this function is copied from Sklearn GP Regression and modified 
        #=============================================================================
        if eval_gradient:
            K, K_gradient_phi, K_gradient_tau2 = covariance(self.x_train, phi, tau2, eval_gradient=True)
        else:
            K = covariance(self.x_train, phi, tau2)
        
        K_noisy = np.copy(K)
        K_noisy[np.diag_indices_from(K_noisy)] += noise2
        try:
            L = cholesky(K_noisy, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(parameters)) if eval_gradient else -np.inf
        
        alpha = cho_solve((L, True), self.y_train)  # Line 3
        
        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self.y_train, alpha) - np.log(np.diag(L)).sum() - K_noisy.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions
        
        if eval_gradient:  # compare Equation 5.9 from GPML
            # This sklearn GP implementation to efficiently calculate only required terms (only diagonal terms required for trace) speeds up by a factor 2-3 (in the range len(x_train) = 1000-2000 over naive implementation
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_phi = (0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient_phi)).sum(-1)
            log_likelihood_gradient_tau2 = (0.5 * np.einsum("ijl,ij->l", tmp, K_gradient_tau2)).sum(-1)
            log_likelihood_gradient_noise2 = (0.5 * np.einsum("iil,ii->l", tmp, np.eye(K.shape[0]))).sum(-1)
            log_likelihood_gradient = np.concatenate(([log_likelihood_gradient_noise2,log_likelihood_gradient_tau2],log_likelihood_gradient_phi))
                    
        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood
    
    #=======================================================================
    # obj_func should be modifed to accept priors
    #=======================================================================
    def obj_func(self, parameters, eval_gradient=True):
        if self.sqrt_transform_theta==True:
            parameters = self.inverse_transform_theta(parameters)
        if eval_gradient:
            lml, grad_lml = self.log_marginal_likelihood(parameters, eval_gradient=True)
            if self.prior==None:
                return -lml, -grad_lml
            elif self.prior=='ard':
                log_posterior = lml + self.prior_noise2.logPDF(parameters[0]) + self.prior_tau2.logPDF(parameters[1]) 
                grad_log_posterior = grad_lml + self.prior_noise2.grad_logPDF(parameters[0]) + self.prior_tau2.grad_logPDF(parameters[1]) 
                for i in range(2,len(parameters)):
                    log_posterior += self.prior_phi.logPDF(parameters[i])
                    grad_log_posterior += self.prior_phi.grad_logPDF(parameters[i])
                return -log_posterior, -grad_log_posterior
        else:
            if self.prior==None:
                return -self.log_marginal_likelihood(parameters)
            elif self.prior=='ard':
                log_posterior = self.log_marginal_likelihood(parameters) + self.prior_noise2.logPDF(parameters[0]) + self.prior_tau2.logPDF(parameters[1])
                for i in range(2,len(parameters)):
                    log_posterior += self.prior_phi.logPDF(parameters[i])
                return -log_posterior
    
    #============================================================================================
    # The function below is copied from sklearn GP and not modified yet. Works perfectly though
    #============================================================================================
    def _constrained_optimization(self, obj_func, parameters_initial, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            parameters_opt, func_min, convergence_dict = fmin_l_bfgs_b(obj_func, parameters_initial, bounds=bounds)
            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict,
                              ConvergenceWarning)
        elif callable(self.optimizer):
            parameters_opt, func_min = \
                self.optimizer(obj_func, parameters_initial, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return parameters_opt, func_min
        
    def transform_theta(self,theta):
        return np.sqrt(theta)
    
    def inverse_transform_theta(self,theta_transform):
        return theta_transform**2

        
    #=======================================================================
    # The function below is copied from sklearn GP and modified
    #=======================================================================
    def predict(self, X, return_std=True, return_cov=False):
        """Predict using the Gaussian process regression model

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated

        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean

        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points

        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        # Transform X to normalized x
        if self.normalize_x:
            x = (X-self.x_train_mean)/self.x_train_std
        else:
            x = X
        K_trans = covariance((x,self.x_train), self.phi, self.tau2, eval_gradient=False)
        y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        # Undo y normalization
        if self.normalize_y:
            y_mean = self.y_train_std*y_mean + self.y_train_mean
        
        if return_cov or return_std:
            # The diagonal is tested to be same as (y_std**2) in sklearn GP
            v = cho_solve((self.L_, True), K_trans.T)  # Line 5
            y_cov = covariance(x, self.phi, self.tau2) - K_trans.dot(v)  # Line 6
            # Undo y normalization
            if self.normalize_y:
                if len(self.y_train_std)==1:
                    y_cov = y_cov*(self.y_train_std**2)
                else:
                    print("Couldn't calculate y_cov as normalization support hasn't been coded yet")
                    return y_mean
            if return_cov and not return_std:
                return y_mean, y_cov
            elif return_std and not return_cov:
                return y_mean, np.sqrt(np.diag(y_cov)).reshape(-1,1)
            else:
                return y_mean, np.sqrt(np.diag(y_cov)).reshape(-1,1), y_cov
        else:
            return y_mean
       
