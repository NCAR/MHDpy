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
    xc = 0.125*( x[I,J,K] + x[I,J,K+1] + 
                x[I+1,J,K+1] + x[I+1,J,K] + 
                x[I,J+1,K] + x[I,J+1,K+1] + 
                x[I+1,J+1,K+1] + x[I+1,J+1,K] )
    yc = 0.125*( y[I,J,K] + y[I,J,K+1] + 
                y[I+1,J,K+1] + y[I+1,J,K] + 
                y[I,J+1,K] + y[I,J+1,K+1] + 
                y[I+1,J+1,K+1] + y[I+1,J+1,K] )
    zc = 0.125*( z[I,J,K] + z[I,J,K+1] + 
                z[I+1,J,K+1] + z[I+1,J,K] + 
                z[I,J+1,K] + z[I,J+1,K+1] + 
                z[I+1,J+1,K+1] + z[I+1,J+1,K] )
    
    # locations for i-face centers -  where bi is defined 
    xi = x[Ip1,J,K]
    yi = 0.25*( y[Ip1,J,K] + y[Ip1,J+1,K] + y[Ip1,J,K+1] + y[Ip1,J+1,K+1] )
    zi = 0.25*( z[Ip1,J,K] + z[Ip1,J+1,K] + z[Ip1,J,K+1] + z[Ip1,J+1,K+1] )
    
    # locations for j-face centers - where bj is defined 
    xj = 0.25*( x[I,Jp1,K] + x[I+1,Jp1,K] + x[I,Jp1,K+1] + x[I+1,Jp1,K+1])
    yj = y[I,Jp1,K];
    zj = 0.25*( z[I,Jp1,K] + z[I+1,Jp1,K] + z[I,Jp1,K+1] + z[I+1,Jp1,K+1])
    
    # locations for k-face centers - where bk is defined 
    xk = 0.25*( x[I,J,Kp1] + x[I+1,J,Kp1] + x[I,J+1,Kp1] + x[I+1,J+1,Kp1])
    yk = 0.25*( y[I,J,Kp1] + y[I+1,J,Kp1] + y[I,J+1,Kp1] + y[I+1,J+1,Kp1])
    zk = z[I,J,Kp1]
    
    # lengths of each cell edges
    dx = x[I+1,J,K] - x[I,J,K]
    dy = y[I,J+1,K] - y[I,J,K]
    dz = z[I,J,K+1] - z[I,J,K]
    
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