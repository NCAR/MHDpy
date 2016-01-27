"""
Compute left and right PDM states
"""
def PDM2(f0,f1,f2,f3,f,PDMB):
    """
    Function compoutes the left and right PDM limited states 
    Requries:
        f0 - ?
        f1 - ?
        f2 - ?
        f3 - ?
        f - ?
        PDMB - value of the B parameter in the limiter 
    Returns:
        (f_left,f_right) - Limited left/right state values
    """  
    import numpy as n
    import MHDpy
    ##
    # f0=1
    # f1=1
    # f2=1
    # f3=0
    
    # second order interp
    # f=0.5*(f1+f2) 
    
    # fourth order interp
    # f = (-f0+9*f1+9*f2-f3)/16
    # f = (-f0+7*f1+7*f2-f3)/12
    
    #NB - the maximum and minimum functions do an element by element comparison
    #     of the maxtrix elements and return the corresponding max/min
    maxf = n.maximum(f1,f2)
    minf = n.minimum(f1,f2)
    
    f = n.maximum(minf,n.minimum(f,maxf))
    
    df0 = PDMB*(f1-f0)
    df1 = PDMB*(f2-f1)
    df2 = PDMB*(f3-f2)
    
    s0 = MHDpy.sign1(df0)
    s1 = MHDpy.sign1(df1)
    s2 = MHDpy.sign1(df2)
    
    df0 = n.abs(df0)
    df1 = n.abs(df1)
    df2 = n.abs(df2)
    
    q0 = n.abs(s0+s1)
    q1 = n.abs(s1+s2)
    
    df_left = f - f1
    df_righ = f2 - f
    
    f_left = f - s1*n.maximum(0,n.abs(df_left) - q0*df0)
    f_right= f + s1*n.maximum(0,n.abs(df_righ) - q1*df2) 
    
    return (f_left,f_right)
