import numpy as np

def construct(end_points):
    if len(end_points.shape)==1:
        return np.arange(end_points[0],end_points[1],end_points[2])
    elif len(end_points.shape)==2:
        if end_points.shape[0]==1:
            return np.arange(end_points[0,0],end_points[0,1],end_points[0,2])
        elif end_points.shape[0]==2:
            xx,yy=np.mgrid[end_points[0,0]:end_points[0,1]:end_points[0,2],end_points[1,0]:end_points[1,1]:end_points[1,2]]
            return(np.vstack((xx.flatten(), yy.flatten())).T)
        elif end_points.shape[0]==3:
            xx,yy,zz=np.mgrid[end_points[0,0]:end_points[0,1]:end_points[0,2],end_points[1,0]:end_points[1,1]:end_points[1,2],end_points[2,0]:end_points[2,1]:end_points[2,2]]
            return(np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T)
        

        
