import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class gaussian_process:
    def __init__(self,covariance_kernel):
        self.covariance_kernel=covariance_kernel
    
    def prior_sample(self,x_array):
        return np.random.multivariate_normal(np.zeros(x_array.shape),self.covariance_kernel.k(x_array,x_array,self.covariance_kernel.tau2,self.covariance_kernel.phi))
    
    def fit(self,training_data_x,training_data_y):
        self.X=[training_data_x]
        self.Y=[np.atleast_2d(training_data_y)]
        self.noise=self.covariance_kernel.Ve
        self.covariance_matrix_00=self.covariance_kernel.k(self.X[0],self.X[0],self.covariance_kernel.tau2,self.covariance_kernel.phi)
        self.covariance_matrix_00_noisy=self.covariance_matrix_00+self.noise*np.eye(self.covariance_matrix_00.shape[0])
        self.covariance_matrix_00_noisy_inv=np.linalg.inv(self.covariance_matrix_00_noisy)
        
    
    def predict(self,prediction_data_x):
        self.X.append(prediction_data_x)
        
        self.covariance_matrix=[]
        for i in range(len(self.X)):
            K_i=[]
            for j in range(len(self.X)):
                K_i.append(self.covariance_kernel.k(self.X[i],self.X[j],self.covariance_kernel.tau2,self.covariance_kernel.phi))
            self.covariance_matrix.append(K_i)
        
        self.mean=(self.covariance_matrix[1][0] @ self.covariance_matrix_00_noisy_inv @ self.Y[0].T)
        self.mean=self.mean.reshape(self.mean.shape[0])
        self.cov=self.covariance_matrix[1][1]-(self.covariance_matrix[1][0] @ self.covariance_matrix_00_noisy_inv @ self.covariance_matrix[0][1])
    
    def plot_prediction(self,number_of_predictions=1):
        if len(self.Y)==1:
            self.Y.append(np.random.multivariate_normal(self.mean,self.cov,number_of_predictions).T)
        elif len(self.Y)>1:
            self.Y[1]=np.random.multivariate_normal(self.mean,self.cov,number_of_predictions).T
        if len(self.X[0].shape)==1:
            for i in range(self.Y[1].shape[1]):
                plt.plot(self.X[1],self.Y[1][:,i])
            plt.scatter(self.X[0],self.Y[0][0])
            plt.show()
        elif len(self.X[0].shape)==2:
            xx=self.X[1][:,0].reshape(np.unique(self.X[1][:,0]).shape[0],np.unique(self.X[1][:,1]).shape[0])
            yy=self.X[1][:,1].reshape(np.unique(self.X[1][:,0]).shape[0],np.unique(self.X[1][:,1]).shape[0])    
            plt3d = plt.figure().gca(projection='3d')
            plt3d.plot_surface(xx,yy,self.Y[1].reshape(xx.shape),alpha=0.3)
            plt.show()




        

