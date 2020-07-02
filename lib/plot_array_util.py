import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def plot_1d(x_array):
    plt.plot(x_array[:,0],x_array[:,1])
    plt.show()

def plot_2d(x_array):
    plt.plot(x_array[:,1],x_array[:,2])
    plt.show()

def plot_3d(x_array):
    ax=plt.axes(projection='3d')
    ax.plot3D(x_array[:,0],x_array[:,1],x_array[:,2])
    plt.show()
    
def scatterplot_1d(x_array):
    plt.scatter(x_array[:,0],x_array[:,1])
    plt.show()

def scatterplot_2d(x_array):
    plt.scatter(x_array[:,1],x_array[:,2])
    plt.show()

def scatterplot_3d(x_array):
    ax=plt.axes(projection='3d')
    ax.scatter(x_array[:,0],x_array[:,1],x_array[:,2])
    plt.show()

def scatterplot_3d_compare(x_array,y_array):
    ax=plt.axes(projection='3d')
    ax.scatter(x_array[:,0],x_array[:,1],x_array[:,2])
    ax.scatter(y_array[:,0],y_array[:,1],y_array[:,2])
    plt.show()

def surfaceplot_3d(x_array):
    ax=plt.axes(projection='3d')
    ax.plot_surface(x_array[:,0],x_array[:,1],x_array[:,2])
    plt.show()

def scatterplot_surfaceplot_3d(x_array,y_array):
    ax=plt.axes(projection='3d')
    ax.scatter(x_array[:,0],x_array[:,1],x_array[:,2])
    ax.plot_surface(y_array[:,:,0],y_array[:,:,1],y_array[:,:,2],cmap='viridis', edgecolor='none')
    plt.show()


