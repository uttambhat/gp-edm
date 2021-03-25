import numpy as np
import gp.gpr as gp
from helpers import dataprep
import matplotlib.pyplot as plt
import time
from gp.optimizers import rprop

x = np.arange(0.,10.,0.1).reshape(-1,1)
y = np.sin(x) + np.random.normal(0.,0.1,x.shape)
x = x

### Fit GP ###
model = gp.gaussian_process_regressor(normalize_x=True,normalize_y=True,optimizer='fmin_l_bfgs_b',prior='ard',n_restarts_optimizer=10) #rprop doesn't work (please debug) (alt = 'fmin_l_bfgs_b')
start = time.time()
model.fit(x,y)
end = time.time()
print(end-start)

### Predict ###
x_test = x
y_pred,y_pred_cov = model.predict(x_test,return_cov=True)
y_error = np.sqrt(np.diag(y_pred_cov))

### Plot test vs. predictions ###
plt.scatter(x,y)
plt.plot(x_test,y_pred)
plt.fill_between(x_test.flatten(),y_pred.flatten()-y_error,y_pred.flatten()+y_error)
plt.show()
#plotter.compare(x_test,y_test,y_predict)


# GPY
import GPy as gpy

kernel = gpy.kern.RBF(x.shape[1],ARD=1) + gpy.kern.White(x.shape[1])
model_gpy = gpy.models.GPRegression(x,y,kernel)

########### Fitting kernel hyperparameters ##########
start = time.time()
model_gpy.optimize(optimizer='scg',max_iters=100)
end = time.time()
print(end-start)

y_pred_gpy,y_error_gpy = model_gpy.predict(x_test)
plt.scatter(x,y)
plt.plot(x_test,y_pred_gpy)
plt.fill_between(x_test.flatten(),y_pred_gpy.flatten()-y_error_gpy.flatten(),y_pred_gpy.flatten()+y_error_gpy.flatten())
plt.show()

