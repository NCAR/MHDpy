"""
Compute left and right TVD states
"""
def TVD2(fm1,f0,f1,f2,f=None):
    """
    Function compoutes the left and right PDM limited states 
    Requries:
        fm1 - ?
        f1 - ?
        f2 - ?
        f - optional
    Returns:
        (f_left,f_right) - Limited left/right state values
    """  
    import numpy as n

    if (f == None):
        f = 0.5*(f0+f1)
 
    # left value
    eps=1.0e-10
    r_i = (f0-fm1)/(f1-f0+eps)
    phi_i = (r_i + abs(r_i))/(1+abs(r_i)) # Van Leer slope limiter
    # phi_i = max(0,max(min(2*r_i,1),min(r_i,2))) # Superbee slope limiter
    f_left = f0 - phi_i*(f0-f)
    
    # right value
    r_i = (f1-f2)/(f0-f1+eps)
    phi_i = (r_i + abs(r_i))/(1+abs(r_i))
    # phi_i = max(0,max(min(2*r_i,1),min(r_i,2)))
    f_right = f1 - phi_i*(f1-f)
    
    return (f_left,f_right)
