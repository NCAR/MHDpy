"""
Function for computing metric parameters from grid
"""
def Metrics(x,y,z,NO):
    """
    Function for generating a Uniform 3D Cartesian grid from -1 to 1 in all 
    directions.  
    Requries:
        x - locations of cell corners in the x direction
        y - locations of cell corners in the y direction
        z - locations of cell corners in the z direction
        NO - order of numerical scheme to be applied
    Returns:
        xc,yc,zc - arrays of cell center locations 
        xi,yi,zi - i-face centers where bi is defined
        xj,yj,zj - j-face centers where bj is defined
        xk,yk,zk - k-face centers where bk is defined
        dx,dy,dz - lengths of each cell edge
        All are returned in a massive tuple   
    """

    # locations for cell centers
    xc = 0.125*( x[:-1,:-1,:-1] + x[:-1,:-1,1:] + 
                x[1:,:-1,1:] + x[1:,:-1,:-1] + 
                x[:-1,1:,:-1] + x[:-1,1:,1:] + 
                x[1:,1:,1:] + x[1:,1:,:-1] )
    yc = 0.125*( y[:-1,:-1,:-1] + y[:-1,:-1,1:] + 
                y[1:,:-1,1:] + y[1:,:-1,:-1] + 
                y[:-1,1:,:-1] + y[:-1,1:,1:] + 
                y[1:,1:,1:] + y[1:,1:,:-1] )
    zc = 0.125*( z[:-1,:-1,:-1] + z[:-1,:-1,1:] + 
                z[1:,:-1,1:] + z[1:,:-1,:-1] + 
                z[:-1,1:,:-1] + z[:-1,1:,1:] + 
                z[1:,1:,1:] + z[1:,1:,:-1] )

    # locations for i-face centers -  where bi is defined 
    xi = x[:,:-1,:-1]
    yi = 0.25*( y[:,:-1,:-1] + y[:,1:,:-1] + y[:,:-1,1:] + y[:,1:,1:] )
    zi = 0.25*( z[:,:-1,:-1] + z[:,1:,:-1] + z[:,:-1,1:] + z[:,1:,1:] )
    
    # locations for j-face centers - where bj is defined 
    xj = 0.25*( x[:-1,:,:-1] + x[1:,:,:-1] + x[:-1,:,1:] + x[1:,:,1:])
    yj = y[:-1,:,:-1];
    zj = 0.25*( z[:-1,:,:-1] + z[1:,:,:-1] + z[:-1,:,1:] + z[1:,:,1:])
    
    # locations for k-face centers - where bk is defined 
    xk = 0.25*( x[:-1,:-1,:] + x[1:,:-1,:] + x[:-1,1:,:] + x[1:,1:,:])
    yk = 0.25*( y[:-1,:-1,:] + y[1:,:-1,:] + y[:-1,1:,:] + y[1:,1:,:])
    zk = z[:-1,:-1,:]
    
    # lengths of each cell edges
    dx = x[1:,:-1,:-1] - x[:-1,:-1,:-1]
    dy = y[:-1,1:,:-1] - y[:-1,:-1,:-1]
    dz = z[:-1,:-1,1:] - z[:-1,:-1,:-1]
        
    return (xc,yc,zc,xi,yi,zi,xj,yj,zj,xk,yk,zk,dx,dy,dz)