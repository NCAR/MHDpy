"""
Compute the conserved variables and return them in a tuple
"""
def getConservedVariables(rho,vx,vy,vz,p,gamma):
    """
    Function computes the conserved form of the plasma parameters 
    Requries:
        rho,vx,vy,vz,p - plasma variables
        gamma - ratio of specific heats
    Returns:
        rho,rhovx,rhovy,rhovz,eng - plasma variables in conserved form
    """               
    import numpy as n
    rhovx = rho*vx
    rhovy = rho*vy
    rhovz = rho*vz
    eng= 0.5*rho*(vx**2+vy**2+vz**2)+p/(gamma-1)
    
    return (rho,rhovx,rhovy,rhovz,eng)
      
    