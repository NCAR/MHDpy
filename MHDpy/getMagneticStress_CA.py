"""
Compute the conserved variables and return them in a tuple
"""
def getMagneticStress_CA(rho,Vx,Vy,Vz,p,bx,by,bz,CA,direction):
    """
    Function computes the conserved form of the plasma parameters 
    Requries:
        rho,vx,vy,vz,p - plasma variables
        bx,by,bz - magnetic field
        direction - sweep dir 1 = x, 2 = y, 3 = z
    Returns:
        (Frho_p,FrhoVx_p,FrhoVy_p,FrhoVz_p,Feng_p,
         Frho_n,FrhoVx_n,FrhoVy_n,FrhoVz_n,Feng_n) - Tuple of fluxes
    """
    import numpy as n
    from scipy.special import erfc
    
    # This function calculates the positive/negative moments of a Gaussian
    # distribution for the magnetic stress calculation. The width of the Gaussian
    # distribution function is the fast speed. This is also adapted from:
    #
    # Kun Xu, Gas-Kinetic Theory-Based Flux Splitting Method for Ideal 
    # Magnetohydrodynamics, 
    # Journal of Computational Physics, Volume 153, Issue 2, 10 August 1999, 
    # Pages 334-352, ISSN 0021-9991, 
    # http://dx.doi.org/10.1006/jcph.1999.6280.
    
    # magnetic pressure    
    b = n.sqrt(bx**2+by**2+bz**2)
    va2 = b**2/rho
    valf = va2*CA**2/(va2+CA**2)
    # total pressure 
    ptot = 2*p/rho+valf
    # "temperature" of the magnetic distribution function         
    lamda = 1/ptot       
    
    Vx0_p = 0.5*erfc(-n.sqrt(lamda)*Vx) # zeroth velocity moment - positive x
    Vx0_n = 0.5*erfc(+n.sqrt(lamda)*Vx) # zeroth velocity moment - negative x
    
    Vy0_p = 0.5*erfc(-n.sqrt(lamda)*Vy) # zeroth velocity moment - positive y
    Vy0_n = 0.5*erfc(+n.sqrt(lamda)*Vy) # zeroth velocity moment - negative y
    
    Vz0_p = 0.5*erfc(-n.sqrt(lamda)*Vz) # zeroth velocity moment - positive z
    Vz0_n = 0.5*erfc(+n.sqrt(lamda)*Vz) # zeroth velocity moment - negative z
    
    if (direction ==1):
        n1=1
        n2=0
        n3=0
        Vn0_p = Vx0_p     
        Vn0_n = Vx0_n
        bn = bx
    elif(direction==2):
        n1=0
        n2=1
        n3=0
        Vn0_p = Vy0_p     
        Vn0_n = Vy0_n
        bn = by
    elif(direction==3):
        n1=0
        n2=0
        n3=1
        Vn0_p = Vz0_p     
        Vn0_n = Vz0_n
        bn = bz
    
    # Stresses in the positive x direction
    Bstress_x_p = (0.5*b**2*n1 - bx*bn) * Vn0_p
    Bstress_y_p = (0.5*b**2*n2 - by*bn) * Vn0_p
    Bstress_z_p = (0.5*b**2*n3 - bz*bn) * Vn0_p
    
    # Stresses in the negative x direction
    Bstress_x_n = (0.5*b**2*n1 - bx*bn) * Vn0_n
    Bstress_y_n = (0.5*b**2*n2 - by*bn) * Vn0_n
    Bstress_z_n = (0.5*b**2*n3 - bz*bn) * Vn0_n
    
    return (Bstress_x_p,Bstress_y_p,Bstress_z_p,
            Bstress_x_n,Bstress_y_n,Bstress_z_n)
