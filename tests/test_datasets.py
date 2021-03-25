import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.,10.,0.1).reshape(-1,1)
y = np.sin(x) + np.random.normal(0.,0.1,x.shape)
x = x/100.

# Try various different x_scales. Apparently GP is sensitive to this scale. Normalizing x-scale is one way to rectify this
