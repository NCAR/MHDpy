"""
Compute the timestep based upon CFL condition
"""
def getDT(rho,vx,vy,vz,bx,by,bz,p,gamma,dx,dy,dz,CFL):
    """
    Function computes the timestep based upon computed wave speeds and the 
    CFL condition 
    Requries:
        rho,vx,vy,vz,bx,by,bz,p - plasma variables and magnetic field vars
        gamma - ratio of specific heats
        dx,dy,dz - size of edges
        CFL - CFL condition parameter
    Returns:
        dt - smallest allowed timestep
    """               
    import numpy as n
    
    Vfluid = n.sqrt(vx**2+vy**2+vz**2)
    Btotal = n.sqrt(bx**2+by**2+bz**2)
    Valfvn = Btotal/n.sqrt(rho)
    Vsound = n.sqrt(gamma*p/rho)
    
    VCFL = Vfluid+n.sqrt(Valfvn**2+Vsound**2)
    dtCFL = CFL /(VCFL/dx+VCFL/dy+VCFL/dz)
    dt = dtCFL.min()
    
    return (dt)
      
 