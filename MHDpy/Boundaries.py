"""
Function for implementing boundary conditions
"""
def Boundaries(rho,p,vx,vy,vz,bi,bj,bk,NO,
               ic_act,jc_act,kc_act,if_act,jf_act,kf_act,
               ic_lb,ic_rb,jc_lb,jc_rb,kc_lb,kc_rb,
               if_lb,if_rb,jf_lb,jf_rb,kf_lb,kf_rb):
    """
    Function for apply peroidic, outflow, or extrapoloation BC.  At this time 
    choice is done by uncommenting sections, need to be made a function option 
    Requries:
        rho,p,vx,vy,vz,bi,bj,bk - plasma and magentic field variables
        NO - order of numerical scheme to be applied
        ic_act,jc_act,kc_act - indices for active cell centers 
        if_act,jf_act,kf_act - indices for active face centers
        ic_lb,jc_lb,kc_lb,if_lb,jf_lb,kf_lb - left boundary indices
        ic_rb,jc_rb,kc_rb,if_rb,jf_rb,kf_rb - right boundary indices
    Returns:
        rho,p,vx,vy,vz,bi,bj,bk - plasma and magentic field variables with BCs
    """               
    import numpy as n                                      
    NO2 = NO/2
    [nx,ny,nz]=rho.shape
    nx = nx - NO
    ny = ny - NO
    nz = nz - NO
    # 
    # # --outflow boundary - x direction
    # rho[ic_lb,:,:] =rho[NO-ic_lb-1,:,:]
    # vx[ic_lb,:,:]  = vx[NO-ic_lb-1,:,:]
    # vy[ic_lb,:,:]  = vy[NO-ic_lb-1,:,:]
    # vz[ic_lb,:,:]  = vz[NO-ic_lb-1,:,:]
    # p[ic_lb,:,:]   =  p[NO-ic_lb-1,:,:]
    # bi[if_lb,:,:]  = bi[NO-if_lb-1,:,:] 
    # bj[ic_lb,:,:]  = bj[NO-ic_lb-1,:,:]
    # bk[ic_lb,:,:]  = bk[NO-ic_lb-1,:,:]
    # 
    # # --outflow boundary - x direction
    # rho[ic_rb,:,:] = rho[2*nx+NO-ic_rb-1,:,:]
    # p[ic_rb,:,:] = p[2*nx+NO-ic_rb-1,:,:]
    # vx[ic_rb,:,:] = vx[2*nx+NO-ic_rb-1,:,:]
    # vy[ic_rb,:,:] = vy[2*nx+NO-ic_rb-1,:,:]
    # vz[ic_rb,:,:] = vz[2*nx+NO-ic_rb-1,:,:]
    #  'bi' has dimension [Ip1,J,K], so the I boundary use i-face active index
    # bi[if_rb,:,:] = bi[2*nx+NO-if_rb+1,:,:] 
    #  'bj' has dimension [I,Jp1,K], so the I boundary use i-center active index
    # bj[ic_rb,:,:] = bj[2*nx+NO-ic_rb-1,:,:]
    #  'bk' has dimension [I,J,Kp1], so the I boundary use i-center active index   
    # bk[ic_rb,:,:] = bk[2*nx+NO-ic_rb-1,:,:]   
    
    # 
    # --periodic boundary - x direction
    rho[ic_rb,:,:] = rho[ic_act[0:NO2],:,:]
    p[ic_rb,:,:]   = p[ic_act[0:NO2],:,:]
    vx[ic_rb,:,:]  = vx[ic_act[0:NO2],:,:]
    vy[ic_rb,:,:]  = vy[ic_act[0:NO2],:,:]
    vz[ic_rb,:,:]  = vz[ic_act[0:NO2],:,:]
    # 'bi' has dimension [Ip1,J,K], so the I boundary use i-face active index
    bi[if_rb,:,:]  = bi[if_act[0:NO2],:,:]
    # 'bj' has dimension [I,Jp1,K], so the I boundary use i-center active index 
    bj[ic_rb,:,:]  = bj[ic_act[0:NO2],:,:] 
    # 'bk' has dimension [I,J,Kp1], so the I boundary use i-center active index  
    bk[ic_rb,:,:]  = bk[ic_act[0:NO2],:,:]   
    
    # --periodic boundary - x direction
    rho[ic_lb,:,:] = rho[ic_act[-NO2:],:,:]
    vx[ic_lb,:,:]  = vx[ic_act[-NO2:],:,:]
    vy[ic_lb,:,:]  = vy[ic_act[-NO2:],:,:]
    vz[ic_lb,:,:]  = vz[ic_act[-NO2:],:,:]
    p[ic_lb,:,:]   =  p[ic_act[-NO2:],:,:]
    bi[if_lb,:,:]  = bi[if_act[-NO2:],:,:] 
    bj[ic_lb,:,:]  = bj[ic_act[-NO2:],:,:]
    bk[ic_lb,:,:]  = bk[ic_act[-NO2:],:,:]
    
    # --periodic boundary - y direction
    # rho[:,jc_lb,:] =rho[:,jc_act[:,-NO2:,:],:]
    # vx[:,jc_lb,:]  = vx[:,jc_act[:,-NO2:,:],:]
    # vy[:,jc_lb,:]  = vy[:,jc_act[:,-NO2:,:],:]
    # vz[:,jc_lb,:]  = vz[:,jc_act[:,-NO2:,:],:]
    # p[:,jc_lb,:]   =  p[:,jc_act[:,-NO2:,:],:]
    #  'bi' has dimension [Ip1,J,K], so the J boundary use j-center active index
    # bi[:,jc_lb,:]  = bi[:,jc_act[:,-NO2:,:],:]
    #  'bj' has dimension [I,Jp1,K], so the J boundary use j-face active index   
    # bj[:,jf_lb,:]  = bj[:,jf_act[:,-NO2,:],:]
    #  'bk' has dimension [I,J,Kp1], so the J boundary use j-center active index   
    # bk[:,kc_lb,:]  = bk[:,jc_act[:,-NO2:,:],:]   
    # 
    # # --periodic boundary - y direction
    # rho[:,jc_rb,:] = rho[:,jc_act[:,:NO2,:],:]
    # vx[:,jc_rb,:]  = vx[:,jc_act[:,:NO2,:],:]
    # vy[:,jc_rb,:]  = vy[:,jc_act[:,:NO2,:],:]
    # vz[:,jc_rb,:]  = vz[:,jc_act[:,:NO2,:],:]
    # p[:,jc_rb,:]   =  p[:,jc_act[:,:NO2,:],:]
    # bi[:,jc_rb,:]  = bi[:,jc_act[:,:NO2,:],:]
    # bj[:,jf_rb,:]  = bj[:,jf_act[:,:NO2,:],:]
    # bk[:,jc_rb,:]  = bk[:,jc_act[:,:NO2,:],:]
    
    # 
    # # --routflow boundary - y direction
    rho[:,jc_lb,:] =rho[:,NO-jc_lb-1,:]
    vx[:,jc_lb,:]  = vx[:,NO-jc_lb-1,:]
    vy[:,jc_lb,:]  = vy[:,NO-jc_lb-1,:]
    vz[:,jc_lb,:]  = vz[:,NO-jc_lb-1,:]
    p[:,jc_lb,:]   =  p[:,NO-jc_lb-1,:]
    #  'bi' has dimension [Ip1,J,K], so the J boundary use j-center active index
    bi[:,jc_lb,:]  = bi[:,NO-jc_lb-1,:]
    #  'bj' has dimension [I,Jp1,K], so the J boundary use j-face active index   
    bj[:,jf_lb,:]  = bj[:,NO-jf_lb-1,:]
    #  'bk' has dimension [I,J,Kp1], so the J boundary use j-center active index   
    bk[:,jc_lb,:]  = bk[:,NO-jc_lb-1,:]   
    
    # --routflow boundary - y direction
    rho[:,jc_rb,:] =rho[:,2*ny+NO-jc_rb-1,:]
    vx[:,jc_rb,:]  = vx[:,2*ny+NO-jc_rb-1,:]
    vy[:,jc_rb,:]  = vy[:,2*ny+NO-jc_rb-1,:]
    vz[:,jc_rb,:]  = vz[:,2*ny+NO-jc_rb-1,:]
    p[:,jc_rb,:]   =  p[:,2*ny+NO-jc_rb-1,:]
    bi[:,jc_rb,:]  = bi[:,2*ny+NO-jc_rb-1,:]
    bj[:,jf_rb,:]  = bj[:,2*ny+NO-jf_rb+1,:]
    bk[:,jc_rb,:]  = bk[:,2*ny+NO-jc_rb-1,:]
    
    # # --routflow boundary - z direction
    # rho[:,:,kc_lb] =rho[:,:,NO-kc_lb-1]
    # vx[:,:,kc_lb]  = vx[:,:,NO-kc_lb-1]
    # vy[:,:,kc_lb]  = vy[:,:,NO-kc_lb-1]
    # vz[:,:,kc_lb]  = vz[:,:,NO-kc_lb-1]
    # p[:,:,kc_lb]   =  p[:,:,NO-kc_lb-1]
    #  'bi' has dimension [Ip1,J,K], so the J boundary use j-center active index
    # bi[:,:,kc_lb]  = bi[:,:,NO-kc_lb-1]
    #  'bj' has dimension [I,Jp1,K], so the J boundary use j-face active index   
    # bj[:,:,kc_lb]  = bj[:,:,NO-kc_lb-1]
    #  'bk' has dimension [I,J,Kp1], so the J boundary use j-center active index   
    # bk[:,:,kf_lb]  = bk[:,:,NO-kf_lb-1]   
    # # --routflow boundary - z direction
    # rho[:,:,kc_rb] =rho[:,:,2*nz+NO-kc_rb-1]
    # vx[:,:,kc_rb]  = vx[:,:,2*nz+NO-kc_rb-1]
    # vy[:,:,kc_rb]  = vy[:,:,2*nz+NO-kc_rb-1]
    # vz[:,:,kc_rb]  = vz[:,:,2*nz+NO-kc_rb-1]
    # p[:,:,kc_rb]   =  p[:,:,2*nz+NO-kc_rb-1]
    # bi[:,:,kc_rb]  = bi[:,:,2*nz+NO-kc_rb-1]
    # bj[:,:,kc_rb]  = bj[:,:,2*nz+NO-kc_rb-1]
    # bk[:,:,kf_rb]  = bk[:,:,2*nz+NO-kf_rb+1]
    
    # 
    # --zeroth order extraplation boundary - z direction
    kc_b1 = n.array([kc_act[0,0,0], kc_act[0,0,0], kc_act[0,0,0], 
                    kc_act[0,0,0]])[None,None,:]
    kf_b1 = n.array([kf_act[0,0,0], kf_act[0,0,0], kf_act[0,0,0], 
                    kc_act[0,0,0]])[None,None,:]
    
    rho[:,:,kc_lb] =rho[:,:,kc_b1]
    vx[:,:,kc_lb]  = vx[:,:,kc_b1]
    vy[:,:,kc_lb]  = vy[:,:,kc_b1]
    vz[:,:,kc_lb]  = vz[:,:,kc_b1]
    p[:,:,kc_lb]   =  p[:,:,kc_b1]
    bi[:,:,kc_lb]  = bi[:,:,kc_b1]
    bj[:,:,kc_lb]  = bj[:,:,kc_b1]
    bk[:,:,kf_lb]  = bk[:,:,kf_b1]
    
    kc_b2 = n.array([kc_act[0,0,-1], kc_act[0,0,-1], 
                    kc_act[0,0,-1], kc_act[0,0,-1]])[None,None,:]
    kf_b2 = n.array([kf_act[0,0,-1], kf_act[0,0,-1], 
                    kf_act[0,0,-1], kf_act[0,0,-1]])[None,None,:]
    # --zeroth order extrapolation boundary - z direction
    rho[:,:,kc_rb] =rho[:,:,kc_b2]
    vx[:,:,kc_rb]  = vx[:,:,kc_b2]
    vy[:,:,kc_rb]  = vy[:,:,kc_b2]
    vz[:,:,kc_rb]  = vz[:,:,kc_b2]
    p[:,:,kc_rb]   =  p[:,:,kc_b2]
    #  'bi' has dimension [Ip1,J,K], so the K boundary use k-center active index
    bi[:,:,kc_rb]  = bi[:,:,kc_b2]
    #  'bj' has dimension [I,Jp1,K], so the K boundary use k-center active index   
    bj[:,:,kc_rb]  = bj[:,:,kc_b2]
    #  'bk' has dimension [I,J,Kp1], so the K boundary use k-face active index   
    bk[:,:,kf_rb]  = bk[:,:,kf_b2]   