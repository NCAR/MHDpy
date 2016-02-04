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
        I,J,K - cell center indices including ghost cells
        if_act,jf_act,kf_act - indices for active cell centers 
        if_act,jf_act,kf_act - indices for active face centers
        ic_lb,jc_lb,kc_lb,if_lb,jf_lb,kf_lb - left boundary indices
        ic_rb,jc_rb,kc_rb,if_rb,jf_rb,kf_rb - right boundary indices
        All are returned in a massive tuple   
    """
    import numpy as n 
    
    print 'Python Index Version'
    (nx_total,ny_total,nz_total)=x.shape;
    NO2 = NO/2;
    
    nx = nx_total-1-NO
    ny = ny_total-1-NO
    nz = nz_total-1-NO
    
    # I,J,K are the indices for cell centers
    # As index arrays they must be broadcastable to the same shape so the use
    # None as the dummy indices makes this possible 
    I = n.arange(nx_total-1)[:,None,None]
    J = n.arange(ny_total-1)[None,:,None]
    K = n.arange(nz_total-1)[None,None,:]
    
    # Ip1,Jp1,Kp1 are the indices for cell corners
    Ip1 = n.arange(nx_total)[:,None,None]
    Jp1 = n.arange(ny_total)[None,:,None]
    Kp1 = n.arange(nz_total)[None,None,:]
    
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
    
    # index of active cell centers
    ic_act = n.arange(NO2,NO2+nx)[:,None,None]
    jc_act = n.arange(NO2,NO2+ny)[None,:,None]
    kc_act = n.arange(NO2,NO2+nz)[None,None,:]
    
    # index of active face centers
    if_act = n.arange(NO2,NO2+nx+1)[:,None,None]
    jf_act = n.arange(NO2,NO2+ny+1)[None,:,None]
    kf_act = n.arange(NO2,NO2+nz+1)[None,None,:]
    
    # index of left-boudary for cell centers
    ic_lb = n.arange(NO2)[:,None,None]
    jc_lb = n.arange(NO2)[None,:,None]
    kc_lb = n.arange(NO2)[None,None,:]
    
    # index of right-boundary for cell centers
    ic_rb = n.arange(nx+NO2, nx+NO2+NO2)[:,None,None]
    jc_rb = n.arange(ny+NO2, ny+NO2+NO2)[None,:,None]
    kc_rb = n.arange(nz+NO2, nz+NO2+NO2)[None,None,:]
    
    # index of left-boundary for face centers
    if_lb = n.arange(NO2)[:,None,None]
    jf_lb = n.arange(NO2)[None,:,None]
    kf_lb = n.arange(NO2)[None,None,:]
    
    # index of right-boundary for face centers
    if_rb = n.arange(nx+NO2+1, nx+NO2+NO2+1)[:,None,None]
    jf_rb = n.arange(ny+NO2+1, ny+NO2+NO2+1)[None,:,None]
    kf_rb = n.arange(nz+NO2+1, nz+NO2+NO2+1)[None,None,:]
    
    return (xc,yc,zc,xi,yi,zi,xj,yj,zj,xk,yk,zk,dx,dy,dz,I,J,K,
            ic_act,jc_act,kc_act,if_act,jf_act,kf_act,
            ic_lb,jc_lb,kc_lb,if_lb,jf_lb,kf_lb,
            ic_rb,jc_rb,kc_rb,if_rb,jf_rb,kf_rb)