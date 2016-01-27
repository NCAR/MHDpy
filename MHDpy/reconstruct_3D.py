"""
Compute the left and right states
"""
def reconstruct_3D(rho_h,if_act,jf_act,kf_act,
          PDMB,direction,limiter_type='PDM'):
    """
    Function compoutes the left and right states at the 
    Requries:
        rho_h - parameter to reconstruct
        if_act,jf_act,kf_act - location of active faces
        PDMB - B parameter for PDM method
        direction - 1 for x, 2 for y, and 3 for z
        limiter_type - limiter_type of limiter to use in 
    Returns:
        rho_left,rho_right - left and right states
    """               
    import numpy as n
    import MHDpy
    
    sizetuple = (if_act[-1,0,0]+1,jf_act[0,-1,0]+1,kf_act[0,0,-1]+1)
    rho_left = n.zeros(sizetuple)
    rho_right = n.zeros(sizetuple)
    rho_interp = n.zeros(sizetuple)
    if (direction==1):
        #8-th order reconstruction
      
        if ( limiter_type == '8th'):
            rho_left[if_act,jf_act,kf_act] = (
                                            -3*rho_h[if_act-4,jf_act,kf_act]+
                                            29*rho_h[if_act-3,jf_act,kf_act]-
                                            139*rho_h[if_act-2,jf_act,kf_act]+
                                            533*rho_h[if_act-1,jf_act,kf_act]+
                                            533*rho_h[if_act,jf_act,kf_act]-
                                            139*rho_h[if_act+1,jf_act,kf_act]+
                                            29*rho_h[if_act+2,jf_act,kf_act]-
                                            3*rho_h[if_act+3,jf_act,kf_act])/840
            rho_right[if_act,jf_act,kf_act] = rho_left[if_act,jf_act,kf_act]
        elif(limiter_type == 'PDM'):
            rho_interp[if_act,jf_act,kf_act] = (
                                            -3*rho_h[if_act-4,jf_act,kf_act]+
                                            29*rho_h[if_act-3,jf_act,kf_act]-
                                            139*rho_h[if_act-2,jf_act,kf_act]+
                                            533*rho_h[if_act-1,jf_act,kf_act]+
                                            533*rho_h[if_act,jf_act,kf_act]-
                                            139*rho_h[if_act+1,jf_act,kf_act]+
                                            29*rho_h[if_act+2,jf_act,kf_act]-
                                            3*rho_h[if_act+3,jf_act,kf_act])/840
            
            # rho_interp[if_act,jf_act,kf_act] = (
            #                                -1*rho_h[if_act-2,jf_act,kf_act]+
            #                                7*rho_h[if_act-1,jf_act,kf_act]+
            #                                7*rho_h[if_act,jf_act,kf_act]-
            #                                rho_h[if_act+1,jf_act,kf_act])/12
            # limiting
            (rho_left[if_act,jf_act,kf_act],
            rho_right[if_act,jf_act,kf_act])= MHDpy.PDM2(
                                                rho_h[if_act-2,jf_act,kf_act],
                                                rho_h[if_act-1,jf_act,kf_act],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act+1,jf_act,kf_act],
                                                rho_interp[if_act,jf_act,kf_act],
                                                PDMB)
        elif(limiter_type == 'TVD'):
            (rho_left[if_act,jf_act,kf_act],
            rho_right[if_act,jf_act,kf_act])= MHDpy.TVD(
                                                rho_h[if_act-2,jf_act,kf_act],
                                                rho_h[if_act-1,jf_act,kf_act],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act+1,jf_act,kf_act])
        elif(limiter_type == 'WENO'):
            rho_left[if_act,jf_act,kf_act] = MHDpy.WENO5(
                                                rho_h[if_act-3,jf_act,kf_act],
                                                rho_h[if_act-2,jf_act,kf_act],
                                                rho_h[if_act-1,jf_act,kf_act],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act+1,jf_act,kf_act])
            rho_right[if_act,jf_act,kf_act]= MHDpy.WENO5(
                                                rho_h[if_act+2,jf_act,kf_act],
                                                rho_h[if_act+1,jf_act,kf_act],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act-1,jf_act,kf_act],
                                                rho_h[if_act-2,jf_act,kf_act])

        
    elif(direction==2):
   
        if ( limiter_type == '8th'):
            rho_left[if_act,jf_act,kf_act] = (
                                            -3*rho_h[if_act,jf_act-4,kf_act]+
                                            29*rho_h[if_act,jf_act-3,kf_act]-
                                            139*rho_h[if_act,jf_act-2,kf_act]+
                                            533*rho_h[if_act,jf_act-1,kf_act]+
                                            533*rho_h[if_act,jf_act,kf_act]-
                                            139*rho_h[if_act,jf_act+1,kf_act]+
                                            29*rho_h[if_act,jf_act+2,kf_act]-
                                            3*rho_h[if_act,jf_act+3,kf_act])/840
            rho_right[if_act,jf_act,kf_act] = rho_left[if_act,jf_act,kf_act]
        elif(limiter_type == 'PDM'):
            rho_interp[if_act,jf_act,kf_act] = (
                                            -3*rho_h[if_act,jf_act-4,kf_act]+
                                            29*rho_h[if_act,jf_act-3,kf_act]-
                                            139*rho_h[if_act,jf_act-2,kf_act]+
                                            533*rho_h[if_act,jf_act-1,kf_act]+
                                            533*rho_h[if_act,jf_act,kf_act]-
                                            139*rho_h[if_act,jf_act+1,kf_act]+
                                            29*rho_h[if_act,jf_act+2,kf_act]-
                                            3*rho_h[if_act,jf_act+3,kf_act])/840
            
            #rho_interp[if_act,jf_act,kf_act] = (
            #                               -1*rho_h[if_act,jf_act-2,kf_act]+
            #                              7*rho_h[if_act,jf_act-1,kf_act]+
            #                             7*rho_h[if_act,jf_act,kf_act]-
            #                            rho_h[if_act,jf_act+1,kf_act])/12
    
            (rho_left[if_act,jf_act,kf_act],
            rho_right[if_act,jf_act,kf_act])= MHDpy.PDM2(
                                                rho_h[if_act,jf_act-2,kf_act],
                                                rho_h[if_act,jf_act-1,kf_act],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act,jf_act+1,kf_act],
                                                rho_interp[if_act,jf_act,kf_act],
                                                PDMB)

        elif (limiter_type== 'TVD'):
            (rho_left[if_act,jf_act,kf_act],
            rho_right[if_act,jf_act,kf_act])= MHDpy.TVD(
                                                rho_h[if_act,jf_act-2,kf_act],
                                                rho_h[if_act,jf_act-1,kf_act],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act,jf_act+1,kf_act])
        elif(limiter_type == 'WENO'):
            rho_left[if_act,jf_act,kf_act] = MHDpy.WENO5(
                                                rho_h[if_act,jf_act-3,kf_act],
                                                rho_h[if_act,jf_act-2,kf_act],
                                                rho_h[if_act,jf_act-1,kf_act],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act,jf_act+1,kf_act])
            rho_right[if_act,jf_act,kf_act]= MHDpy.WENO5(
                                                rho_h[if_act,jf_act+2,kf_act],
                                                rho_h[if_act,jf_act+1,kf_act],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act,jf_act-1,kf_act],
                                                rho_h[if_act,jf_act-2,kf_act])

    elif(direction==3):
        if( limiter_type == '8th'):
            rho_left[if_act,jf_act,kf_act] = (
                                            -3*rho_h[if_act,jf_act,kf_act-4]+
                                            29*rho_h[if_act,jf_act,kf_act-3]-
                                            139*rho_h[if_act,jf_act,kf_act-2]+
                                            533*rho_h[if_act,jf_act,kf_act-1]+
                                            533*rho_h[if_act,jf_act,kf_act]-
                                            139*rho_h[if_act,jf_act,kf_act+1]+
                                            29*rho_h[if_act,jf_act,kf_act+2]-
                                            3*rho_h[if_act,jf_act,kf_act+3])/840
            rho_right[if_act,jf_act,kf_act] = rho_left[if_act,jf_act,kf_act]
        elif(limiter_type == 'PDM'):
            rho_interp[if_act,jf_act,kf_act] = (
                                            -3*rho_h[if_act,jf_act,kf_act-4]+
                                            29*rho_h[if_act,jf_act,kf_act-3]-
                                            139*rho_h[if_act,jf_act,kf_act-2]+
                                            533*rho_h[if_act,jf_act,kf_act-1]+
                                            533*rho_h[if_act,jf_act,kf_act]-
                                            139*rho_h[if_act,jf_act,kf_act+1]+
                                            29*rho_h[if_act,jf_act,kf_act+2]-
                                            3*rho_h[if_act,jf_act,kf_act+3])/840
            
            #rho_interp[if_act,jf_act,kf_act] = (
            #                                -1*rho_h[if_act,jf_act,kf_act-2]+
            #                                7*rho_h[if_act,jf_act,kf_act-1]+
            #                                7*rho_h[if_act,jf_act,kf_act]-
            #                                rho_h[if_act,jf_act,kf_act+1])/12
            
            (rho_left[if_act,jf_act,kf_act],
            rho_right[if_act,jf_act,kf_act])= MHDpy.PDM2(
                                                rho_h[if_act,jf_act,kf_act-2],
                                                rho_h[if_act,jf_act,kf_act-1],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act,jf_act,kf_act+1],
                                                rho_interp[if_act,jf_act,kf_act],
                                                PDMB)
        elif(limiter_type == 'TVD'):
            (rho_left[if_act,jf_act,kf_act],
            rho_right[if_act,jf_act,kf_act])= MHDpy.TVD(
                                                rho_h[if_act,jf_act,kf_act-2],
                                                rho_h[if_act,jf_act,kf_act-1],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act,jf_act,kf_act+1])
        elif(limiter_type == 'WENO'):
            rho_left[if_act,jf_act,kf_act] = MHDpy.WENO5(
                                                rho_h[if_act,jf_act,kf_act-3],
                                                rho_h[if_act,jf_act,kf_act-2],
                                                rho_h[if_act,jf_act,kf_act-1],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act,jf_act,kf_act+1])
            rho_right[if_act,jf_act,kf_act]= MHDpy.WENO5(
                                                rho_h[if_act,jf_act,kf_act+2],
                                                rho_h[if_act,jf_act,kf_act+1],
                                                rho_h[if_act,jf_act,kf_act],
                                                rho_h[if_act,jf_act,kf_act-1],
                                                rho_h[if_act,jf_act,kf_act-2])
    
    return (rho_left,rho_right)
