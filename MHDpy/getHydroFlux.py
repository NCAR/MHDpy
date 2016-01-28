"""
Compute the conserved variables and return them in a tuple
"""
def getHydroFlux(rho,Vx,Vy,Vz,p,gamma,direction):
    """
    Function computes the conserved form of the plasma parameters 
    Requries:
        rho,vx,vy,vz,p - plasma variables
        gamma - ratio of specific heats
        direction - sweep dir 1 = x, 2 = y, 3 = z
    Returns:
        (Frho_p,FrhoVx_p,FrhoVy_p,FrhoVz_p,Feng_p,
         Frho_n,FrhoVx_n,FrhoVy_n,FrhoVz_n,Feng_n) - Tuple of fluxes
    """
    import numpy as n
    from scipy.special import erfc

    # This function calculates the positive/negative moments of a Gaussian
    # distribution for the face flux calculation. The width of the Gaussian
    # distribution function is the fluid temperature. Details can be found in
    #
    # Kun Xu, Gas-Kinetic Theory-Based Flux Splitting Method for Ideal 
    # Magnetohydrodynamics, 
    # Journal of Computational Physics, Volume 153, Issue 2, 10 August 1999, 
    # Pages 334-352, ISSN 0021-9991, 
    # http://dx.doi.org/10.1006/jcph.1999.6280.
        
    eng= 0.5*rho*(Vx**2+Vy**2+Vz**2)+p/(gamma-1)
    ptot = p 
    lamda = rho/(2*ptot) # temperature of the fluid distribution function 
    
    # zeroth velocity moment - positive x
    Vx0_p = 0.5*erfc(-n.sqrt(lamda)*Vx)
    # zeroth velocity moment - negative x                           
    Vx0_n = 0.5*erfc(+n.sqrt(lamda)*Vx)
    # first  velocity moment - positive x                           
    Vx1_p = Vx*Vx0_p + 0.5*n.exp(-lamda*Vx**2)/n.sqrt(n.pi*lamda)
    # first  velocity moment - negative x 
    Vx1_n = Vx*Vx0_n - 0.5*n.exp(-lamda*Vx**2)/n.sqrt(n.pi*lamda) 
    
    # zeroth velocity moment - positive y
    Vy0_p = 0.5*erfc(-n.sqrt(lamda)*Vy)
    # zeroth velocity moment - negative y                           
    Vy0_n = 0.5*erfc(+n.sqrt(lamda)*Vy)
    # first  velocity moment - positive y                           
    Vy1_p = Vy*Vy0_p + 0.5*n.exp(-lamda*Vy**2)/n.sqrt(n.pi*lamda)
    # first  velocity moment - negative y 
    Vy1_n = Vy*Vy0_n - 0.5*n.exp(-lamda*Vy**2)/n.sqrt(n.pi*lamda) 
    
    # zeroth velocity moment - positive z
    Vz0_p = 0.5*erfc(-n.sqrt(lamda)*Vz)
    # zeroth velocity moment - negative z                           
    Vz0_n = 0.5*erfc(+n.sqrt(lamda)*Vz)
    # first  velocity moment - positive z                           
    Vz1_p = Vz*Vz0_p + 0.5*n.exp(-lamda*Vz**2)/n.sqrt(n.pi*lamda)
    # first  velocity moment - negative z 
    Vz1_n = Vz*Vz0_n - 0.5*n.exp(-lamda*Vz**2)/n.sqrt(n.pi*lamda) 
    
    if (direction ==1): # normal direction is x
        n1=1
        n2=0
        n3=0
        Vn0_p = Vx0_p     
        Vn0_n = Vx0_n     
        Vn1_p = Vx1_p     
        Vn1_n = Vx1_n     
        Vn = Vx       
    elif(direction==2): # normal direction is y
        n1=0
        n2=1
        n3=0
        Vn0_p = Vy0_p     
        Vn0_n = Vy0_n
        Vn1_p = Vy1_p     
        Vn1_n = Vy1_n
        Vn = Vy      
    elif(direction==3): # normal direction is z
        n1=0
        n2=0
        n3=1
        Vn0_p = Vz0_p     
        Vn0_n = Vz0_n
        Vn1_p = Vz1_p     
        Vn1_n = Vz1_n
        Vn = Vz
        
    # Fluxes in the positive direction
    Frho_p   = rho  * Vn1_p
    FrhoVx_p = rho*Vx * Vn1_p + (ptot*n1 ) * Vn0_p
    FrhoVy_p = rho*Vy * Vn1_p + (ptot*n2 ) * Vn0_p
    FrhoVz_p = rho*Vz * Vn1_p + (ptot*n3 ) * Vn0_p
    Feng_p   = (eng + 0.5*p) * Vn1_p + (0.5*p*Vn)*Vn0_p
    
    # Fluxes in the negative direction
    Frho_n   = rho  * Vn1_n
    FrhoVx_n = rho*Vx * Vn1_n + (ptot*n1) * Vn0_n
    FrhoVy_n = rho*Vy * Vn1_n + (ptot*n2) * Vn0_n
    FrhoVz_n = rho*Vz * Vn1_n + (ptot*n3) * Vn0_n
    Feng_n   = (eng + 0.5*p) * Vn1_n + (0.5*p*Vn)*Vn0_n
            
    return (Frho_p,FrhoVx_p,FrhoVy_p,FrhoVz_p,Feng_p,
            Frho_n,FrhoVx_n,FrhoVy_n,FrhoVz_n,Feng_n) 
