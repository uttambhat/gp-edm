"""
Kernels for Gaussian Process Regression
"""
import numpy as np
from scipy.spatial.distance import pdist,squareform,cdist


def covariance(X,phi,tau2=1.,eval_gradient=False):
    """
    Description
    -----------
    Calculates the covariance matrix.
    tau2 exp(-0.5*phi^2 (x1-x2)^2)
    
    Parameters
    ----------
    X : (n x m) numpy array or a tuple of (n1 x m) and (n2 x m) numpy arrays
        The n rows are the different datapoints and the m columns represent
        the different features
    phi : scalar or (m,) shaped numpy array of inverse length-scales 
    tau2 : Amplitude of the squared exponential kernel
    
    Returns
    -------
    cov : (n x n) numpy array
    cov_grad : (n x n x m) numpy array with gradient
    
    """
    nax = np.newaxis
    phi = phi[nax,:]
    if type(X)!=tuple:
        lnC = (pdist(X*phi))**2
        cov = squareform((tau2)*np.exp(-0.5*lnC))
        np.fill_diagonal(cov,tau2)
        
    else:
        lnC=(cdist(X[0]*phi,X[1]*phi))**2
        cov = (tau2)*np.exp(-0.5*lnC)
    
    if(eval_gradient):
        if type(X)==tuple:
            raise ValueError("Currently, gradient can only be evaluated for cov(X,X) only")
        cov_grad_phi = phi[nax,...]*((X[:, nax, :] - X[nax, :, :]) ** 2)
        cov_grad_phi *= cov[..., nax]
        cov_grad_tau2 = cov
        return cov,cov_grad_phi,cov_grad_tau2
    else:
        return cov
