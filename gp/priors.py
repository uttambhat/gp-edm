import numpy as np
from scipy.special import gamma as gamma_func, beta as beta_func

"""
Distribution class can have
1. sampler: priors.beta.sample()
2. PDF: priors.beta.PDF(x) #need for objective function
3. Grad PDF wrt variable: priors.beta.grad_PDF(x) #need for optimizing obj func
"""

class beta:
    """
    Class for beta probability distribution,
    f(x; a,b) = (1/B(a,b)) * x^{a-1} * (1-x)^{b-1}
    (Ref: https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html)
    """
    def __init__(self,a,b):
        self.a = a
        self.b = b
        self.normalization = (1./beta_func(self.a,self.b))
        self.log_normalization = np.log(self.normalization)
    
    def sample(self,size=None):
        return np.random.beta(self.a,self.b,size)
    
    def PDF(self,x):
        return self.normalization*(x**(self.a-1.))*((1.-x)**(self.b-1))
    
    def grad_PDF(self,x):
        return self.normalization*((self.a!=1.)*(self.a-1.)*(x**(self.a-2.))*((1.-x)**(self.b-1.)) - (self.b!=1.)*(self.b-1.)*(x**(self.a-1.))*((1.-x)**(self.b-2.)))
    
    def logPDF(self,x):
        return self.log_normalization + (self.a-1.)*np.log(x) + (self.b-1.)*np.log(1.-x)
    
    def grad_logPDF(self,x):
        return (self.a-1.)/x - (self.b-1.)/(1.-x)
        

class gamma:
    """
    Class for gamma probability distribution,
    p(x) = x^{k-1} \exp{-x/theta} / (theta^k * Gamma(k))
    (Ref: https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html)

    """
    def __init__(self,k,theta):
        self.k = k
        self.theta = theta
        self.normalization = 1./((self.theta**self.k)*gamma_func(self.k))
        self.log_normalization = np.log(self.normalization)
    
    def sample(self,size=None):
        return np.random.gamma(self.k,self.theta,size)
    
    def PDF(self,x):
        return self.normalization*(x**(self.k-1.))*np.exp(-x/self.theta)
    
    def grad_PDF(self,x):
        return self.normalization*((self.k!=1.)*(x**(self.k-2))*np.exp(-x/self.theta) - (1./self.theta)*(x**(self.k-1.))*np.exp(-x/self.theta))
    
    def logPDF(self,x):
        return self.log_normalization + (self.k-1.)*np.log(x) - x/self.theta
    
    def grad_logPDF(self,x):
        return (self.k-1.)/x - 1./self.theta

class half_normal:
    """
    Class for half-normal prior distribution (zero mean Gaussian for x>0 (normalized accordingly))
    """
    def __init__(self,var):
        self.var = var
        self.sigma = np.sqrt(var)
        self.normalization = np.sqrt(2./(np.pi*var))
        self.log_normalization = np.log(self.normalization)
    
    def sample(self,size=None):
        return np.absolute(np.random.normal(0.,self.sigma,size))
    
    def PDF(self,x):
        return (x>=0.)*self.normalization*np.exp(-(x**2)/(2.*self.var))
    
    def grad_PDF(self,x):
        return -(x>=0.)*self.normalization*(x/self.var)*np.exp(-(x**2)/(2.*self.var))
    
    def logPDF(self,x):
        return self.log_normalization - (x**2)/(2.*self.var)
    
    def grad_logPDF(self,x):
        return -x/self.var

class centered_normal:
    """
    Class for Normal (Gaussian) prior distribution
    """
    def __init__(self,var):
        self.var = var
        self.sigma = np.sqrt(var)
        self.normalization = np.sqrt(1./(np.pi*var))
        self.log_normalization = np.log(self.normalization)
    
    def sample(self,size=None):
        return np.random.normal(0.,self.sigma,size)
    
    def PDF(self,x):
        return self.normalization*np.exp(-(x**2)/(2.*self.var))
    
    def grad_PDF(self,x):
        return -self.normalization*(x/self.var)*np.exp(-(x**2)/(2.*self.var))
    
    def logPDF(self,x):
        return self.log_normalization - (x**2)/(2.*self.var)
    
    def grad_logPDF(self,x):
        return -x/self.var

