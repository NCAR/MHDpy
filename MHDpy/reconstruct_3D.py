"""
Compute the left and right states
"""
def reconstruct_3D(rho_h,if_act,jf_act,kf_act,NO2,
          PDMB,direction,limiter_type='PDM'):
    """
    Function compoutes the left and right states at the 
    Requries:
        rho_h - parameter to reconstruct
        if_act,jf_act,kf_act - location of active faces
        NO2 - Half numerical order
        PDMB - B parameter for PDM method
        direction - 1 for x, 2 for y, and 3 for z
        limiter_type - limiter_type of limiter to use in 
    Returns:
        rho_left,rho_right - left and right states
    """               
    import numpy as n
    import MHDpy
    
    if (direction==1):
        #8-th order reconstruction
      
        if ( limiter_type == '8th'):
            rho_left = (
                        -3*rho_h[NO2-4:-NO2-4+1,NO2:-NO2+1,NO2:-NO2+1]+
                        29*rho_h[NO2-3:-NO2-3+1,NO2:-NO2+1,NO2:-NO2+1]-
                        139*rho_h[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2+1]+
                        533*rho_h[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2+1]+
                        533*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1]-
                        139*rho_h[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2+1]+
                        29*rho_h[NO2+2:-NO2+2+1,NO2:-NO2+1,NO2:-NO2+1]-
                        3*rho_h[NO2+3:,NO2:-NO2+1,NO2:-NO2+1])/840
            rho_right = rho_left
        elif(limiter_type == 'PDM'):
            rho_interp = (
                        -3*rho_h[NO2-4:-NO2-4+1,NO2:-NO2+1,NO2:-NO2+1]+
                        29*rho_h[NO2-3:-NO2-3+1,NO2:-NO2+1,NO2:-NO2+1]-
                        139*rho_h[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2+1]+
                        533*rho_h[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2+1]+
                        533*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1]-
                        139*rho_h[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2+1]+
                        29*rho_h[NO2+2:-NO2+2+1,NO2:-NO2+1,NO2:-NO2+1]-
                        3*rho_h[NO2+3:,NO2:-NO2+1,NO2:-NO2+1])/840
            
            # rho_interp = (
            #                -1*rho_h[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2+1]+
            #                7*rho_h[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2+1]+
            #                7*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1]-
            #                rho_h[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2+1])/12
            # limiting
            (rho_left,rho_right)= MHDpy.PDM2(
                                    rho_h[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1    ,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_interp,PDMB)
        elif(limiter_type == 'TVD'):
            (rho_left,rho_right)= MHDpy.TVD(
                                    rho_h[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2+1])
        elif(limiter_type == 'WENO'):
            rho_left = MHDpy.WENO5(
                                    rho_h[NO2-3:-NO2-3+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2+1])
            rho_right = MHDpy.WENO5(
                                    rho_h[NO2+2:-NO2+2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2+1])

        
    elif(direction==2):
   
        if ( limiter_type == '8th'):
            rho_left = (
                        -3*rho_h[NO2:-NO2+1,NO2-4:-NO2-4+1,NO2:-NO2+1]+
                        29*rho_h[NO2:-NO2+1,NO2-3:-NO2-3+1,NO2:-NO2+1]-
                        139*rho_h[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2+1]+
                        533*rho_h[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2+1]+
                        533*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1]-
                        139*rho_h[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2+1]+
                        29*rho_h[NO2:-NO2+1,NO2+2:-NO2+2+1,NO2:-NO2+1]-
                        3*rho_h[NO2:-NO2+1,NO2+3:,NO2:-NO2+1])/840
            rho_right = rho_left
        elif(limiter_type == 'PDM'):
            rho_interp = (
                        -3*rho_h[NO2:-NO2+1,NO2-4:-NO2-4+1,NO2:-NO2+1]+
                        29*rho_h[NO2:-NO2+1,NO2-3:-NO2-3+1,NO2:-NO2+1]-
                        139*rho_h[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2+1]+
                        533*rho_h[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2+1]+
                        533*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1]-
                        139*rho_h[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2+1]+
                        29*rho_h[NO2:-NO2+1,NO2+2:-NO2+2+1,NO2:-NO2+1]-
                        3*rho_h[NO2:-NO2+1,NO2+3:,NO2:-NO2+1])/840
            
            #rho_interp = (
            #          -1*rho_h[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2+1]+
            #          7*rho_h[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2+1]+
            #          7*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1]-
            #          rho_h[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2+1])/12
    
            (rho_left,rho_right)= MHDpy.PDM2(
                                    rho_h[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2+1],
                                    rho_interp,PDMB)

        elif (limiter_type== 'TVD'):
            (rho_left,rho_right)= MHDpy.TVD(
                                    rho_h[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2+1])
        elif(limiter_type == 'WENO'):
            rho_left = MHDpy.WENO5(
                                    rho_h[NO2:-NO2+1,NO2-3:-NO2-3+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2+1])
            rho_right = MHDpy.WENO5(
                                    rho_h[NO2:-NO2+1,NO2+2:-NO2+2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2+1])

    elif(direction==3):
        if( limiter_type == '8th'):
            rho_left = (
                        -3*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-4:-NO2-4+1]+
                        29*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-3:-NO2-3+1]-
                        139*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-2:-NO2-2+1]+
                        533*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-1:-NO2-1+1]+
                        533*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1]-
                        139*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+1:-NO2+1+1]+
                        29*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+2:-NO2+2+1]-
                        3*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+3:])/840
            rho_right = rho_left
        elif(limiter_type == 'PDM'):
            rho_interp = (
                        -3*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-4:-NO2-4+1]+
                        29*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-3:-NO2-3+1]-
                        139*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-2:-NO2-2+1]+
                        533*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-1:-NO2-1+1]+
                        533*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1]-
                        139*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+1:-NO2+1+1]+
                        29*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+2:-NO2+2+1]-
                        3*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+3:])/840
            
            #rho_interp = (
            #            -1*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-2:-NO2-2+1]+
            #            7*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-1:-NO2-1+1]+
            #            7*rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1]-
            #            rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+1:-NO2+1+1])/12
            
            (rho_left,rho_right)= MHDpy.PDM2(
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-2:-NO2-2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-1:-NO2-1+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+1:-NO2+1+1],
                                    rho_interp,PDMB)
        elif(limiter_type == 'TVD'):
            (rho_left,rho_right)= MHDpy.TVD(
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-2:-NO2-2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-1:-NO2-1+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+1:-NO2+1+1])
        elif(limiter_type == 'WENO'):
            rho_left = MHDpy.WENO5(
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-3:-NO2-3+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-2:-NO2-2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-1:-NO2-1+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+1:-NO2+1+1])
            rho_right= MHDpy.WENO5(
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+2:-NO2+2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2+1:-NO2+1+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-1:-NO2-1+1],
                                    rho_h[NO2:-NO2+1,NO2:-NO2+1,NO2-2:-NO2-2+1])
    
    return (rho_left,rho_right)
