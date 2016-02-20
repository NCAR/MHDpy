"""
Function for implementing boundary conditions
"""
def Boundaries(rho,p,vx,vy,vz,bi,bj,bk,NO,
               xtype='PER',ytype='PER',ztype='PER'):
    """
    Function for apply peroidic, outflow, or extrapoloation BC.  At this time 
    choice is done by uncommenting sections, need to be made a function option 
    Requries:
        rho,p,vx,vy,vz,bi,bj,bk - plasma and magentic field variables
        NO - order of numerical scheme to be applied
        xtype,ytype,ztype - Type of BC PER - peroidic, OUT - outflow, 
                            EXP - extrapolation
    Returns:
        rho,p,vx,vy,vz,bi,bj,bk - plasma and magentic field variables with BCs
    """               
    import numpy as n                                      
    NO2 = NO/2
    [nx,ny,nz]=rho.shape
    nx = nx - NO
    ny = ny - NO
    nz = nz - NO
 
    if (xtype == 'OUT'):
        ## --outflow boundary - x direction
        #:NO2 gives the left boundary cells
        # NO-1:NO2-1:-1 gives the first NO2 active cells in reverse order      
        rho[:NO2,:,:] =rho[NO-1:NO2-1:-1,:,:]
        vx[:NO2,:,:]  = vx[NO-1:NO2-1:-1,:,:]
        vy[:NO2,:,:]  = vy[NO-1:NO2-1:-1,:,:]
        vz[:NO2,:,:]  = vz[NO-1:NO2-1:-1,:,:]
        p[:NO2,:,:]   =  p[NO-1:NO2-1:-1,:,:]
        bi[:NO2,:,:]  = bi[NO-1:NO2-1:-1,:,:] 
        bj[:NO2,:,:]  = bj[NO-1:NO2-1:-1,:,:]
        bk[:NO2,:,:]  = bk[NO-1:NO2-1:-1,:,:]
        
        #-NO2: gives the right boundary cells
        #-NO2-1:-NO-1:-1 gives last NO2 active cells in reverse order
        rho[-NO2:,:,:] = rho[-NO2-1:-NO-1:-1,:,:]
        p[-NO2:,:,:] = p[-NO2-1:-NO-1:-1,:,:]
        vx[-NO2:,:,:] = vx[-NO2-1:-NO-1:-1,:,:]
        vy[-NO2:,:,:] = vy[-NO2-1:-NO-1:-1,:,:]
        vz[-NO2:,:,:] = vz[-NO2-1:-NO-1:-1,:,:]
        bi[-NO2:,:,:] = bi[-NO2-1:-NO-1:-1,:,:] 
        bj[-NO2:,:,:] = bj[-NO2-1:-NO-1:-1,:,:]  
        bk[-NO2:,:,:] = bk[-NO2-1:-NO-1:-1,:,:]   
    elif (xtype == 'PER'):
        # --periodic boundary - x direction
        #-NO2: gives the right boundary cells
        #NO2:NO gives first NO2 active cells 
        rho[-NO2:,:,:] = rho[NO2:NO,:,:]
        p[-NO2:,:,:]   = p[NO2:NO,:,:]
        vx[-NO2:,:,:]  = vx[NO2:NO,:,:]
        vy[-NO2:,:,:]  = vy[NO2:NO,:,:]
        vz[-NO2:,:,:]  = vz[NO2:NO,:,:]
        bi[-NO2:,:,:]  = bi[NO2:NO,:,:]
        bj[-NO2:,:,:]  = bj[NO2:NO,:,:] 
        bk[-NO2:,:,:]  = bk[NO2:NO,:,:]   
        
        #:NO2 give left boudnary cells
        #-NO:-NO2 gives last NO2 active cells
        rho[:NO2,:,:] = rho[-NO:-NO2,:,:]
        vx[:NO2,:,:]  = vx[-NO:-NO2,:,:]
        vy[:NO2,:,:]  = vy[-NO:-NO2,:,:]
        vz[:NO2,:,:]  = vz[-NO:-NO2,:,:]
        p[:NO2,:,:]   =  p[-NO:-NO2,:,:]
        bi[:NO2,:,:]  = bi[-NO:-NO2,:,:] 
        bj[:NO2,:,:]  = bj[-NO:-NO2,:,:]
        bk[:NO2,:,:]  = bk[-NO:-NO2,:,:]
    elif (xtype == 'EXP'):
        # --zeroth order extraplation boundary - x direction
        #for left side want to replicate the first active cell
        exp = n.zeros(NO2,dtype=n.integer)
        exp[:] = NO2
        #:NO2 give left boudnary cells
        rho[:NO2,:,:] =rho[exp,:,:]
        vx[:NO2,:,:]  = vx[exp,:,:]
        vy[:NO2,:,:]  = vy[exp,:,:]
        vz[:NO2,:,:]  = vz[exp,:,:]
        p[:NO2,:,:]   =  p[exp,:,:]
        bi[:NO2,:,:]  = bi[exp,:,:]
        bj[:NO2,:,:]  = bj[exp,:,:]
        bk[:NO2,:,:]  = bk[exp,:,:]
        
        #for the right side replicate the last active cell
        exp[:] = nx + NO2 -1
        #-NO2: gives the right boundary cells
        rho[-NO2:,:,:] =rho[exp,:,:]
        vx[-NO2:,:,:]  = vx[exp,:,:]
        vy[-NO2:,:,:]  = vy[exp,:,:]
        vz[-NO2:,:,:]  = vz[exp,:,:]
        p[-NO2:,:,:]   =  p[exp,:,:]
        bj[-NO2:,:,:]  = bj[exp,:,:]
        bk[-NO2:,:,:]  = bk[exp,:,:]   
    
    if (ytype == 'PER'):    
        #--periodic boundary - y direction
        #:NO2 give left boudnary cells
        #-NO:-NO2 gives last NO2 active cells
        rho[:,:NO2,:] =rho[:,-NO:-NO2,:]
        vx[:,:NO2,:]  = vx[:,-NO:-NO2,:]
        vy[:,:NO2,:]  = vy[:,-NO:-NO2,:]
        vz[:,:NO2,:]  = vz[:,-NO:-NO2,:]
        p[:,:NO2,:]   =  p[:,-NO:-NO2,:]
        bi[:,:NO2,:]  = bi[:,-NO:-NO2,:]
        bj[:,:NO2,:]  = bj[:,-NO:-NO2,:]
        bk[:,:NO2,:]  = bk[:,-NO:-NO2,:]   
        
        #-NO2: gives the right boundary cells
        #NO2:NO gives first NO2 active cells 
        rho[:,-NO2:,:] = rho[:,NO2:NO,:]
        vx[:,-NO2:,:]  = vx[:,NO2:NO,:]
        vy[:,-NO2:,:]  = vy[:,NO2:NO,:]
        vz[:,-NO2:,:]  = vz[:,NO2:NO,:]
        p[:,-NO2:,:]   =  p[:,NO2:NO,:]
        bi[:,-NO2:,:]  = bi[:,NO2:NO,:]
        bj[:,-NO2:,:]  = bj[:,NO2:NO,:]
        bk[:,-NO2:,:]  = bk[:,NO2:NO,:]
    elif(ytype == 'OUT'):
        # --routflow boundary - y direction 
        #:NO2 gives the left boundary cells
        # NO-1:NO2-1:-1 gives the first NO2 active cells in reverse order 
        rho[:,:NO2,:] =rho[:,NO-1:NO2-1:-1,:]
        vx[:,:NO2,:]  = vx[:,NO-1:NO2-1:-1,:]
        vy[:,:NO2,:]  = vy[:,NO-1:NO2-1:-1,:]
        vz[:,:NO2,:]  = vz[:,NO-1:NO2-1:-1,:]
        p[:,:NO2,:]   =  p[:,NO-1:NO2-1:-1,:]
        bi[:,:NO2,:]  = bi[:,NO-1:NO2-1:-1,:]
        bj[:,:NO2,:]  = bj[:,NO-1:NO2-1:-1,:]
        bk[:,:NO2,:]  = bk[:,NO-1:NO2-1:-1,:]   
        
        #-NO2: gives the right boundary cells
        #-NO2-1:-NO-1:-1 gives last NO2 active cells in reverse order
        rho[:,-NO2:,:] =rho[:,-NO2-1:-NO-1:-1,:]
        vx[:,-NO2:,:]  = vx[:,-NO2-1:-NO-1:-1,:]
        vy[:,-NO2:,:]  = vy[:,-NO2-1:-NO-1:-1,:]
        vz[:,-NO2:,:]  = vz[:,-NO2-1:-NO-1:-1,:]
        p[:,-NO2:,:]   =  p[:,-NO2-1:-NO-1:-1,:]
        bi[:,-NO2:,:]  = bi[:,-NO2-1:-NO-1:-1,:]
        bj[:,-NO2:,:]  = bj[:,-NO2-1:-NO-1:-1,:]
        bk[:,-NO2:,:]  = bk[:,-NO2-1:-NO-1:-1,:]
    
    elif(ytype == 'EXP'):
        # --zeroth order extraplation boundary - y direction
        #for left side want to replicate the first active cell
        exp = n.zeros(NO2,dtype=n.integer)
        exp[:] = NO2
        #:NO2 give left boudnary cells
        rho[:,:NO2,:] =rho[:,exp,:]
        vx[:,:NO2,:]  = vx[:,exp,:]
        vy[:,:NO2,:]  = vy[:,exp,:]
        vz[:,:NO2,:]  = vz[:,exp,:]
        p[:,:NO2,:]   =  p[:,exp,:]
        bi[:,:NO2,:]  = bi[:,exp,:]
        bj[:,:NO2,:]  = bj[:,exp,:]
        bk[:,:NO2,:]  = bk[:,exp,:]
        
        #for the right side replicate the last active cell
        exp[:] = ny + NO2 -1
        #-NO2: gives the right boundary cells
        rho[:,-NO2:,:] =rho[exp]
        vx[:,-NO2:,:]  = vx[exp]
        vy[:,-NO2:,:]  = vy[exp]
        vz[:,-NO2:,:]  = vz[exp]
        p[:,-NO2:,:]   =  p[exp]
        bi[:,-NO2:,:]  = bi[exp]
        bj[:,-NO2:,:]  = bj[exp]
        bk[:,-NO2:,:]  = bk[exp]
        
    if (ztype == 'OUT'):
         # --routflow boundary - z direction
         #:NO2 gives the left boundary cells
         # NO-1:NO2-1:-1 gives the first NO2 active cells in reverse order 
         rho[:,:,:NO2] =rho[:,:,NO-1:NO2-1:-1]
         vx[:,:,:NO2]  = vx[:,:,NO-1:NO2-1:-1]
         vy[:,:,:NO2]  = vy[:,:,NO-1:NO2-1:-1]
         vz[:,:,:NO2]  = vz[:,:,NO-1:NO2-1:-1]
         p[:,:,:NO2]   =  p[:,:,NO-1:NO2-1:-1]
         bi[:,:,:NO2]  = bi[:,:,NO-1:NO2-1:-1]
         bj[:,:,:NO2]  = bj[:,:,NO-1:NO2-1:-1]
         bk[:,:,:NO2]  = bk[:,:,NO-1:NO2-1:-1]   
         #-NO2: gives the right boundary cells
         #-NO2-1:-NO-1:-1 gives last NO2 active cells in reverse order
         rho[:,:,-NO2:] =rho[:,:,-NO2-1:-NO-1:-1]
         vx[:,:,-NO2:]  = vx[:,:,-NO2-1:-NO-1:-1]
         vy[:,:,-NO2:]  = vy[:,:,-NO2-1:-NO-1:-1]
         vz[:,:,-NO2:]  = vz[:,:,-NO2-1:-NO-1:-1]
         p[:,:,-NO2:]   =  p[:,:,-NO2-1:-NO-1:-1]
         bi[:,:,-NO2:]  = bi[:,:,-NO2-1:-NO-1:-1]
         bj[:,:,-NO2:]  = bj[:,:,-NO2-1:-NO-1:-1]
         bk[:,:,-NO2:]  = bk[:,:,-NO2-1:-NO-1:-1]
    elif(ztype == 'PER'):
        #--periodic boundary - z direction
        #:NO2 give left boudnary cells
        #-NO:-NO2 gives last NO2 active cells
        rho[:,:,:NO2] =rho[:,:,-NO:-NO2]
        vx[:,:,:NO2]  = vx[:,:,-NO:-NO2]
        vy[:,:,:NO2]  = vy[:,:,-NO:-NO2]
        vz[:,:,:NO2]  = vz[:,:,-NO:-NO2]
        p[:,:,:NO2]   =  p[:,:,-NO:-NO2]
        bi[:,:,:NO2]  = bi[:,:,-NO:-NO2]   
        bj[:,:,:NO2]  = bj[:,:,-NO:-NO2]
        bk[:,:,:NO2]  = bk[:,:,-NO:-NO2]   
        
        #-NO2: gives the right boundary cells
        #NO2:NO gives first NO2 active cells 
        rho[:,:,-NO2:] = rho[:,:,NO2:NO]
        vx[:,:,-NO2:]  = vx[:,:,NO2:NO]
        vy[:,:,-NO2:]  = vy[:,:,NO2:NO]
        vz[:,:,-NO2:]  = vz[:,:,NO2:NO]
        p[:,:,-NO2:]   =  p[:,:,NO2:NO]
        bi[:,:,-NO2:]  = bi[:,:,NO2:NO]
        bj[:,:,-NO2:]  = bj[:,:,NO2:NO]
        bk[:,:,-NO2:]  = bk[:,:,NO2:NO]
        
    elif(ztype == 'EXP'):
        # --zeroth order extraplation boundary - z direction
        #for left side want to replicate the first active cell
        exp = n.zeros(NO2,dtype=n.integer)
        exp[:] = NO2
        #:NO2 give left boudnary cells
        rho[:,:,:NO2] =rho[:,:,exp]
        vx[:,:,:NO2]  = vx[:,:,exp]
        vy[:,:,:NO2]  = vy[:,:,exp]
        vz[:,:,:NO2]  = vz[:,:,exp]
        p[:,:,:NO2]   =  p[:,:,exp]
        bi[:,:,:NO2]  = bi[:,:,exp]
        bj[:,:,:NO2]  = bj[:,:,exp]
        bk[:,:,:NO2]  = bk[:,:,exp]
        
        #for the right side replicate the last active cell
        exp[:] = nz + NO2 -1
        #-NO2: gives the right boundary cells
        rho[:,:,-NO2:] =rho[:,:,exp]
        vx[:,:,-NO2:]  = vx[:,:,exp]
        vy[:,:,-NO2:]  = vy[:,:,exp]
        vz[:,:,-NO2:]  = vz[:,:,exp]
        p[:,:,-NO2:]   =  p[:,:,exp]
        bi[:,:,-NO2:]  = bi[:,:,exp]
        bj[:,:,-NO2:]  = bj[:,:,exp]
        bk[:,:,-NO2:]  = bk[:,:,exp]   