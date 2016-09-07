import numpy as n
def Ax(x,y,z):
    """ 
    Vector potential Ax(x,y,z)
    """
    p = n.zeros_like(x)
    return p

def Ay(x,y,z):
    """
    Vector potentail Ay(x,y,z)
    """
    p = n.zeros_like(x)
    return p
    
def Az(x,y,z,Lx=25.6,Ly=12.8):
    """
    Vector potential Az(x,y,z)
    Peturbation function for GEM reconnection probelm
    Default values for Lx = 25.6 and Ly=12.8 are utilized
    """
    p = -0.1*n.cos(2*n.pi*x/Lx)*n.cos(n.pi*y/Ly);
    return p
