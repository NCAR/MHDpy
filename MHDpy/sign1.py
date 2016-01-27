"""
Compute return sign of number with zero replaced as 1
"""
def sign1(x):
    """
    Emulates fortran sign function by returning 1 for zero locations 
    Requries:
        x - array to check for signs
    Returns:
        a - array same size x with sign values
    """  
    import numpy as n
    
    a = n.sign(x)
    a[a == 0] = 1
    
    return a
    