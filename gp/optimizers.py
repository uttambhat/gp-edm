import numpy as np

def rprop(obj_func, parameters_initial, bounds):
    """
    Rprop algorithm ported from Steve Munch's MATLAB implementation 
    (/home/uttam/Dropbox/c/delay_embedding/arxiv_and_unused/matlab/GP_simple/fmingrad_Rprop.m),
    particularly all the algorithmic constants (stopping condition, increase-decrease factors etc)
    
    """
    #==================================================
    # Fails to fit sine curves. To be debugged.
    #==================================================
    m = len(parameters_initial)
    step_size = 0.1*np.ones(m)
    step_size_min = 1.e-6
    step_size_max = 50.
    step_increase_factor = 1.2
    step_decrease_factor = 0.5
    max_iterations = 200
    
    parameters = parameters_initial
    f,grad = obj_func(parameters)
    grad_norm = np.linalg.norm(grad)
    
    relative_change = 10.
    iteration = 0
    while (grad_norm > 1.e-8 and iteration<max_iterations and relative_change>1.e-8):
        # update parameters
        parameters_new = parameters - np.sign(grad)*step_size
        f_new,grad_new = obj_func(parameters_new)
        grad_norm = np.linalg.norm(grad_new)
        relative_change = np.abs(f_new/f - 1.)
        
        # update step_size
        grad_product = grad*grad_new
        step_size = np.minimum(step_size_max, np.maximum(step_size_min, step_size*(1.+(step_increase_factor-1.)*(grad_product>0.)+(step_decrease_factor-1.)*(grad_product<0.))))

        # prepare for next step
        parameters = parameters_new
        grad = grad_new
        f = f_new
        iteration += 1
    
    f_opt,grad_opt = obj_func(parameters)
    return parameters,f_opt

    
    
