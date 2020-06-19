import numpy as np

def rprop(objective_function,theta_initial,bounds):     
    precision=1.e-5
    eta_plus=1.1
    eta_minus=0.8
    step_size_bounds=[1.e-10,1.e2]
    step_size=0.01*np.ones(theta_initial.shape)
    theta=theta_initial
    (value,gradient)=objective_function(theta)
    delta_theta=-np.sign(gradient)*step_size
    theta=theta+delta_theta
    gradient_previous=np.copy(gradient)
    count=0
    
    while(np.linalg.norm(delta_theta)>precision and count<10000):
        (value,gradient)=objective_function(theta)
        
        for i in range(theta.shape[0]):
            step_size[i] = eta_plus*step_size[i] if np.sign(gradient[i])==np.sign(gradient_previous[i]) else eta_minus*step_size[i] if np.sign(gradient[i])==-np.sign(gradient_previous[i]) else step_size[i]
        gradient_previous=gradient
        for i in range(step_size.shape[0]):
            step_size[i]=np.sign(step_size[i])*np.clip(np.absolute(step_size[i]),step_size_bounds[0],step_size_bounds[1])
        delta_theta=-np.sign(gradient)*step_size
        theta=theta+delta_theta
        #print(theta,gradient,step_size)
        count += 1
        if count%100==0:
            print("Rprop iteration: ",count)
    return theta
