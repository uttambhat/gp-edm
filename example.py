import numpy as np
import gp.gpr as gp
import edm.simple as edm
from helpers import dataprep, plotter

### Load data ###
data_original = np.loadtxt("datasets/timeseries_l96_N10_F10.txt")[:500]

### Normalize and log-transform if required ###
data,mean,stdev = dataprep.normalize(data_original)

### Fit parameters ###
x_columns = [0,1,2] #independent variables
y_columns = [0]     #dependent variables
number_of_lags = [2,3,3] #number of time-lags for each independent variable
lag = [1,1,1] #lag for each variable (or scalar for same lag across all variables)
forecast_steps_ahead = 1 #number of time-steps ahead for forecasting
test_fraction = 0.2 #fraction of data to be used to calculate out-of-sample error (can be set to zero to use all data for training)
x_train,y_train,x_test,y_test = edm.construct(data,x_columns,y_columns,number_of_lags,lag,forecast_steps_ahead,test_fraction)

### Fit GP ###
model = gp.gaussian_process_regressor(kernel='sq_exp',optimizer='fmin_l_bfgs_b',prior='ard',n_restarts_optimizer=10) #reduce n_restarts_optimizer to reduce runtime at the risk of a bad optimum
model.fit(x_train,y_train)

### Predict ###
y_predict,y_error = model.predict(x_test,return_std=True)

### Plot test vs. predictions ###
plotter.compare_output_timeseries(y_test,y_predict,y_error=y_error,display_rmse=True)

