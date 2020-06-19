######### change path according to where you store library files #########
import sys, os
sys.path.append(os.path.abspath("..")+"/python_homemade_commons")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gaussian_process.gaussian_process_regression as gp
import gaussian_process.kernels as kernels

import generic.grid as grid
import generic.normalize as normalize
import generic.plot_array_util as pltarray
import ml_data_prep.time_series_to_ml_edm as data_prep

########## Get data ################
time_series_data = np.loadtxt("../python_time_series_generators/time_series_data/time_series_04.txt")

########## Log and Normalize time-series values #############
#time_series_data = np.log(time_series_data)
mean = np.mean(time_series_data)
stdev = np.sqrt(np.var(time_series_data))
#time_series_data_normalized = (time_series_data-mean)/stdev
time_series_data_normalized = time_series_data[:1000]

################## Data preparation ############################

X_column_list = [0,1]
y_column_list = [0]
number_of_delays = 4
test_fraction = 0.5

X_train,y_train,X_test,y_test = data_prep.prepare(time_series_data_normalized,X_column_list,y_column_list,number_of_delays,test_fraction)

######### GP specific reshaping data ##############

X = [X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]),X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])]
Y = [y_train.T,y_test.T]

############# Fit hyper parameters for the kernel #######################

covariance_kernel=kernels.kernel()
(fit_data_args,fit_data_vals)=covariance_kernel.fit(X[0],Y[0])
regression=gp.gaussian_process(covariance_kernel)

############# Fit GP and predict #####################

regression.fit(X[0],Y[0])
regression.predict(X[1])

############## Plot and compare attractor ##################

ax=plt.axes(projection='3d')
ax.scatter(X[1][:,0],X[1][:,1],Y[1][0])
ax.scatter(X[1][:,0],X[1][:,1],regression.mean)
ax.set_zlim(-5.,5.)
plt.show()

ax=plt.axes(projection='3d')
ax.scatter(X[1][:,2],X[1][:,3],Y[1][0])
ax.scatter(X[1][:,2],X[1][:,3],regression.mean)
ax.set_zlim(-5.,5.)
plt.show()

#Check the inverse length scale parameter

############# Plot and compare time-series predictions #######################
plt.plot(Y[1][0])
plt.plot(regression.mean)
plt.show()





