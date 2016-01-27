"""
Compute average velocity at cell corner from cell centers
"""
def center2corner(vx,I,J,K,ic_act,jc_act,kc_act,if_act,jf_act,kf_act,
          PDMB,direction,limiter_type='PDM'):
    """
    Function compoutes average values at cell corners from cell centers.
    Requries:
        vx - velocity
        I,J,K - locations of all cells
        ic_act,jc_act,kc_act - location of active cell centers
        if_act,jf_act,kf_act - location of active faces
        PDMB - B parameter for PDM method
        direction - type of plane must be xy,yz,zx
        limiter_type - type of limiter to use in 
    Returns:
        Ei,Ej,Ek - Elecrtic fields along cell edges
    """  
    import numpy as n
    import MHDpy

    ###########################################################################
    # calculate the averaged values at cell corners, from cell centers.
    # Detailed description see Lyon et al., [2004]. Note that the dimensions of
    # arrays in this function varies depending on the location of the values,
    # which can be tricky.
    #
    # MAIN ALGORITHM (in the x-y plane): 
    #   STEP 1: interpolate cell-centered values in the x-direction to cell
    #           faces
    #   STEP 2: interpolate the cell-faced values in the y-direction to cell
    #           cornor
    #   STEP 3: using limiters to split the interpolated cell-corner values to
    #           8 states in the four quadrants
    #   STEP 4: pick one state in each quadrants based on the larger relative
    #           differences in each direction
    #   STEP 5: average the 4 states in each quadrant to get an estimate of the
    #           average value at cell corner
    ###########################################################################

    
    if (direction == 'xy'):
        # step 1 & 2: interpolate velocity from cell center to corner
        #         first interpolate in the x-direction, so the index for
        #         x-dimension changes from ic_act to if_act,for the y-direction,
        #         interpolation is done for the whole y-domain, i.e., J
        #         then interpolate in the y-direction, so the index for 
        #         y-dimension changes from J to jf_act
        
        # The Matlab version of this code uses syntax X[if_act,jf_act,kc_act]
        # This resuls in arrays of the size of last element of each index array
        # The elements before the index arrays are set to zero.  In order to 
        # emulate that syntax here we need to use zeros to declare
        # the arrays first then we can just use Bin's syntax 
        # This should be refactored to just use the arrays the active size, but
        # since this mechanism is used throughout the code we are just going to 
        # use this work around for now.
        
        vx_interp_x = n.zeros((if_act[-1,0,0]+1,J[0,-1,0]+1,kc_act[0,0,-1]+1))
        sizetuple = (if_act[-1,0,0]+1,jf_act[0,-1,0]+1,kc_act[0,0,-1]+1)
        vx_interp = n.zeros(sizetuple)
        vx_00_x = n.zeros(sizetuple)
        vx_10_x = n.zeros(sizetuple)
        vx_00_y = n.zeros(sizetuple)
        vx_01_y = n.zeros(sizetuple)
        vx_10_y = n.zeros(sizetuple)
        vx_11_y = n.zeros(sizetuple)
        vx_01_x = n.zeros(sizetuple)
        vx_11_x = n.zeros(sizetuple)

        if ( limiter_type == '8th'):   
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (if_act, jf_act, kc_act]
            vx_interp_x[if_act,J,kc_act] = (
                -3*vx[if_act-4,J,kc_act]+
                29*vx[if_act-3,J,kc_act]-
                139*vx[if_act-2,J,kc_act]+
                533*vx[if_act-1,J,kc_act]+
                533*vx[if_act,J,kc_act]-
                139*vx[if_act+1,J,kc_act]+
                29*vx[if_act+2,J,kc_act]-
                3*vx[if_act+3,J,kc_act])/840
            vx_interp[if_act,jf_act,kc_act] = (
                -3*vx_interp_x[if_act,jf_act-4,kc_act]+
                29*vx_interp_x[if_act,jf_act-3,kc_act]-
                139*vx_interp_x[if_act,jf_act-2,kc_act]+
                533*vx_interp_x[if_act,jf_act-1,kc_act]+
                533*vx_interp_x[if_act,jf_act,kc_act]-
                139*vx_interp_x[if_act,jf_act+1,kc_act]+
                29*vx_interp_x[if_act,jf_act+2,kc_act]-
                3*vx_interp_x[if_act,jf_act+3,kc_act])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            vx_00_x[if_act,jf_act,kc_act] = vx_interp[if_act,jf_act,kc_act]
            vx_10_x[if_act,jf_act,kc_act] = vx_interp[if_act,jf_act,kc_act]
            vx_00_y[if_act,jf_act,kc_act] = vx_interp[if_act,jf_act,kc_act]
            vx_01_y[if_act,jf_act,kc_act] = vx_interp[if_act,jf_act,kc_act]
            vx_10_y[if_act,jf_act,kc_act] = vx_interp[if_act,jf_act,kc_act]
            vx_11_y[if_act,jf_act,kc_act] = vx_interp[if_act,jf_act,kc_act]
            vx_01_x[if_act,jf_act,kc_act] = vx_interp[if_act,jf_act,kc_act]
            vx_11_x[if_act,jf_act,kc_act] = vx_interp[if_act,jf_act,kc_act]
        elif (limiter_type == 'PDM'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (if_act, jf_act, kc_act]
            vx_interp_x[if_act,J,kc_act] = (
                -3*vx[if_act-4,J,kc_act]+
                29*vx[if_act-3,J,kc_act]-
                139*vx[if_act-2,J,kc_act]+
                533*vx[if_act-1,J,kc_act]+
                533*vx[if_act,J,kc_act]-
                139*vx[if_act+1,J,kc_act]+
                29*vx[if_act+2,J,kc_act]-
                3*vx[if_act+3,J,kc_act])/840
            vx_interp[if_act,jf_act,kc_act] = (
                -3*vx_interp_x[if_act,jf_act-4,kc_act]+
                29*vx_interp_x[if_act,jf_act-3,kc_act]-
                139*vx_interp_x[if_act,jf_act-2,kc_act]+
                533*vx_interp_x[if_act,jf_act-1,kc_act]+
                533*vx_interp_x[if_act,jf_act,kc_act]-
                139*vx_interp_x[if_act,jf_act+1,kc_act]+
                29*vx_interp_x[if_act,jf_act+2,kc_act]-
                3*vx_interp_x[if_act,jf_act+3,kc_act])/840
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            (vx_00_x[if_act,jf_act,kc_act],
            vx_10_x[if_act,jf_act,kc_act])=MHDpy.PDM2(
                                                vx[if_act-2,jf_act-1,kc_act],
                                                vx[if_act-1,jf_act-1,kc_act],
                                                vx[if_act,jf_act-1,kc_act],
                                                vx[if_act+1,jf_act-1,kc_act],
                                                vx_interp[if_act,jf_act,kc_act],
                                                PDMB)
            (vx_00_y[if_act,jf_act,kc_act],
            vx_01_y[if_act,jf_act,kc_act])= MHDpy.PDM2(
                                                vx[if_act-1,jf_act-2,kc_act],
                                                vx[if_act-1,jf_act-1,kc_act],
                                                vx[if_act-1,jf_act,kc_act],
                                                vx[if_act-1,jf_act+1,kc_act],
                                                vx_interp[if_act,jf_act,kc_act],
                                                PDMB)
            (vx_10_y[if_act,jf_act,kc_act],
            vx_11_y[if_act,jf_act,kc_act])= MHDpy.PDM2(
                                                vx[if_act,jf_act-2,kc_act],
                                                vx[if_act,jf_act-1,kc_act],
                                                vx[if_act,jf_act,kc_act],
                                                vx[if_act,jf_act+1,kc_act],
                                                vx_interp[if_act,jf_act,kc_act],
                                                PDMB)
            (vx_01_x[if_act,jf_act,kc_act],
            vx_11_x[if_act,jf_act,kc_act])= MHDpy.PDM2(
                                                vx[if_act-2,jf_act,kc_act],
                                                vx[if_act-1,jf_act,kc_act],
                                                vx[if_act,jf_act,kc_act],
                                                vx[if_act+1,jf_act,kc_act],
                                                vx_interp[if_act,jf_act,kc_act],
                                                PDMB)
        elif(limiter_type == 'TVD'):
            # put in TVD limiter here
            vx_interp[if_act,jf_act,kc_act] = 0.25*(
                                            vx[if_act-1,jf_act-1,kc_act] + 
                                            vx[if_act,jf_act-1,kc_act] +
                                            vx[if_act-1,jf_act,kc_act] + 
                                            vx[if_act,jf_act,kc_act])
                                                
            (vx_00_x[if_act,jf_act,kc_act],
            vx_10_x[if_act,jf_act,kc_act])= MHDpy.TVD2(
                                    vx[if_act-2,jf_act-1,kc_act],
                                    vx[if_act-1,jf_act-1,kc_act],
                                    vx[if_act,jf_act-1,kc_act],
                                    vx[if_act+1,jf_act-1,kc_act],
                                    vx_interp[if_act,jf_act,kc_act])
            (vx_00_y[if_act,jf_act,kc_act],
            vx_01_y[if_act,jf_act,kc_act])=MHDpy.TVD2(
                                    vx[if_act-1,jf_act-2,kc_act],
                                    vx[if_act-1,jf_act-1,kc_act],
                                    vx[if_act-1,jf_act,kc_act],
                                    vx[if_act-1,jf_act+1,kc_act],
                                    vx_interp[if_act,jf_act,kc_act])
            (vx_10_y[if_act,jf_act,kc_act],
            vx_11_y[if_act,jf_act,kc_act])=MHDpy.TVD2(
                                    vx[if_act,jf_act-2,kc_act],
                                    vx[if_act,jf_act-1,kc_act],
                                    vx[if_act,jf_act,kc_act],
                                    vx[if_act,jf_act+1,kc_act],
                                    vx_interp[if_act,jf_act,kc_act])
            (vx_01_x[if_act,jf_act,kc_act],
            vx_11_x[if_act,jf_act,kc_act])=MHDpy.TVD2(
                                vx[if_act-2,jf_act,kc_act],
                                vx[if_act-1,jf_act,kc_act],
                                vx[if_act,jf_act,kc_act],
                                vx[if_act+1,jf_act,kc_act],
                                vx_interp[if_act,jf_act,kc_act])
                

        # need to declare a few more worker arrays
        diff = n.zeros(sizetuple)
        vx_00 = n.zeros(sizetuple)
        vx_10 = n.zeros(sizetuple)
        vx_01 = n.zeros(sizetuple)
        vx_11 = n.zeros(sizetuple)
        vx_avg = n.zeros(sizetuple)
        # quadrant 00 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff[if_act,jf_act,kc_act] = n.sign(n.abs(vx[if_act-2,jf_act-1,kc_act]-
                                            vx[if_act-1,jf_act-1,kc_act]) - 
                                            n.abs(vx[if_act-1,jf_act-2,kc_act]-
                                            vx[if_act-1,jf_act-1,kc_act]))
        # pick the state in the direction that has a bigger difference 
        # (if no difference then average the two states)
        vx_00[if_act,jf_act,kc_act] = ((1+diff[if_act,jf_act,kc_act])/2*
                                        vx_00_x[if_act,jf_act,kc_act] +
                                        (1-diff[if_act,jf_act,kc_act])/2*
                                        vx_00_y[if_act,jf_act,kc_act])
        
        # quadrant 10 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff[if_act,jf_act,kc_act] = n.sign(n.abs(vx[if_act+1,jf_act-1,kc_act]-
                                            vx[if_act,jf_act-1,kc_act]) - 
                                            n.abs(vx[if_act,jf_act-2,kc_act]-
                                            vx[if_act,jf_act-1,kc_act]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_10[if_act,jf_act,kc_act] = ((1+diff[if_act,jf_act,kc_act])/2*
                                        vx_10_x[if_act,jf_act,kc_act] +
                                        (1-diff[if_act,jf_act,kc_act])/2*
                                        vx_10_y[if_act,jf_act,kc_act])
        
        # quadrant 01 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff[if_act,jf_act,kc_act] = n.sign(n.abs(vx[if_act-2,jf_act,kc_act]-
                                            vx[if_act-1,jf_act,kc_act]) - 
                                            n.abs(vx[if_act-1,jf_act+1,kc_act]-
                                            vx[if_act-1,jf_act,kc_act]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_01[if_act,jf_act,kc_act] = ((1+diff[if_act,jf_act,kc_act])/2*
                                        vx_01_x[if_act,jf_act,kc_act] +
                                        (1-diff[if_act,jf_act,kc_act])/2
                                        *vx_01_y[if_act,jf_act,kc_act])
        
        # quadrant 11 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff[if_act,jf_act,kc_act] = n.sign(n.abs(vx[if_act+1,jf_act,kc_act]-
                                            vx[if_act,jf_act,kc_act]) - 
                                            n.abs(vx[if_act,jf_act+1,kc_act]-
                                            vx[if_act,jf_act,kc_act]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_11[if_act,jf_act,kc_act] = ((1+diff[if_act,jf_act,kc_act])/2*
                                        vx_11_x[if_act,jf_act,kc_act] + 
                                        (1-diff[if_act,jf_act,kc_act])/2*
                                        vx_11_y[if_act,jf_act,kc_act])
        
        # calculate average vx at the cell corner by averaging the four states
        vx_avg[if_act,jf_act,kc_act] = 0.25*(vx_00[if_act,jf_act,kc_act] + 
                                            vx_01[if_act,jf_act,kc_act] + 
                                            vx_10[if_act,jf_act,kc_act] + 
                                            vx_11[if_act,jf_act,kc_act])
        
        return vx_avg
        
    elif (direction == 'yz'): #
        # step 1 & 2: interpolate velocity from cell center to corner
        #         first interpolate in the y-direction, so the index for
        #         y-dimension changes from jc_act to jf_act,for the y-direction,
        #         interpolation is done for the whole z-domain, i.e., K
        #         then interpolate in the z-direction, so the index for 
        #         z-dimension changes from K to kf_act
        
        vx_interp_x = n.zeros((ic_act[-1,0,0]+1,jf_act[0,-1,0]+1,K[0,0,-1]+1))
        sizetuple = (ic_act[-1,0,0]+1,jf_act[0,-1,0]+1,kf_act[0,0,-1]+1)
        vx_interp = n.zeros(sizetuple)
        vx_00_x = n.zeros(sizetuple)
        vx_10_x = n.zeros(sizetuple)
        vx_00_y = n.zeros(sizetuple)
        vx_01_y = n.zeros(sizetuple)
        vx_10_y = n.zeros(sizetuple)
        vx_11_y = n.zeros(sizetuple)
        vx_01_x = n.zeros(sizetuple)
        vx_11_x = n.zeros(sizetuple)

        if ( limiter_type == '8th'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (ic_act, jf_act, kf_act]
            vx_interp_x[ic_act,jf_act,K] = (
                                            -3*vx[ic_act,jf_act-4,K]+
                                            29*vx[ic_act,jf_act-3,K]-
                                            139*vx[ic_act,jf_act-2,K]+
                                            533*vx[ic_act,jf_act-1,K]+
                                            533*vx[ic_act,jf_act,K]-
                                            139*vx[ic_act,jf_act+1,K]+
                                            29*vx[ic_act,jf_act+2,K]-
                                            3*vx[ic_act,jf_act+3,K])/840
            vx_interp[ic_act,jf_act,kf_act] = (
                                    -3*vx_interp_x[ic_act,jf_act,kf_act-4]+
                                    29*vx_interp_x[ic_act,jf_act,kf_act-3]-
                                    139*vx_interp_x[ic_act,jf_act,kf_act-2]+
                                    533*vx_interp_x[ic_act,jf_act,kf_act-1]+
                                    533*vx_interp_x[ic_act,jf_act,kf_act]-
                                    139*vx_interp_x[ic_act,jf_act,kf_act+1]+
                                    29*vx_interp_x[ic_act,jf_act,kf_act+2]-
                                    3*vx_interp_x[ic_act,jf_act,kf_act+3])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            vx_00_x[ic_act,jf_act,kf_act] = vx_interp[ic_act,jf_act,kf_act]
            vx_10_x[ic_act,jf_act,kf_act] = vx_interp[ic_act,jf_act,kf_act]
            vx_00_y[ic_act,jf_act,kf_act] = vx_interp[ic_act,jf_act,kf_act]
            vx_01_y[ic_act,jf_act,kf_act] = vx_interp[ic_act,jf_act,kf_act]
            vx_10_y[ic_act,jf_act,kf_act] = vx_interp[ic_act,jf_act,kf_act]
            vx_11_y[ic_act,jf_act,kf_act] = vx_interp[ic_act,jf_act,kf_act]
            vx_01_x[ic_act,jf_act,kf_act] = vx_interp[ic_act,jf_act,kf_act]
            vx_11_x[ic_act,jf_act,kf_act] = vx_interp[ic_act,jf_act,kf_act]
            
        elif( limiter_type == 'PDM'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (ic_act, jf_act, kf_act]
            vx_interp_x[ic_act,jf_act,K] = (
                                            -3*vx[ic_act,jf_act-4,K]+
                                            29*vx[ic_act,jf_act-3,K]-
                                            139*vx[ic_act,jf_act-2,K]+
                                            533*vx[ic_act,jf_act-1,K]+
                                            533*vx[ic_act,jf_act,K]-
                                            139*vx[ic_act,jf_act+1,K]+
                                            29*vx[ic_act,jf_act+2,K]-
                                            3*vx[ic_act,jf_act+3,K])/840
            vx_interp[ic_act,jf_act,kf_act] = (
                                    -3*vx_interp_x[ic_act,jf_act,kf_act-4]+
                                    29*vx_interp_x[ic_act,jf_act,kf_act-3]-
                                    139*vx_interp_x[ic_act,jf_act,kf_act-2]+
                                    533*vx_interp_x[ic_act,jf_act,kf_act-1]+
                                    533*vx_interp_x[ic_act,jf_act,kf_act]-
                                    139*vx_interp_x[ic_act,jf_act,kf_act+1]+
                                    29*vx_interp_x[ic_act,jf_act,kf_act+2]-
                                    3*vx_interp_x[ic_act,jf_act,kf_act+3])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            (vx_00_x[ic_act,jf_act,kf_act],
            vx_10_x[ic_act,jf_act,kf_act])= MHDpy.PDM2(
                                                vx[ic_act,jf_act-2,kf_act-1],
                                                vx[ic_act,jf_act-1,kf_act-1],
                                                vx[ic_act,jf_act,kf_act-1],
                                                vx[ic_act,jf_act+1,kf_act-1],
                                                vx_interp[ic_act,jf_act,kf_act],
                                                PDMB)
            (vx_00_y[ic_act,jf_act,kf_act],
            vx_01_y[ic_act,jf_act,kf_act])= MHDpy.PDM2(
                                                vx[ic_act,jf_act-1,kf_act-2],
                                                vx[ic_act,jf_act-1,kf_act-1],
                                                vx[ic_act,jf_act-1,kf_act],
                                                vx[ic_act,jf_act-1,kf_act+1],
                                                vx_interp[ic_act,jf_act,kf_act],
                                                PDMB)
            (vx_10_y[ic_act,jf_act,kf_act],
            vx_11_y[ic_act,jf_act,kf_act])= MHDpy.PDM2(
                                                vx[ic_act,jf_act,kf_act-2],
                                                vx[ic_act,jf_act,kf_act-1],
                                                vx[ic_act,jf_act,kf_act],
                                                vx[ic_act,jf_act,kf_act+1],
                                                vx_interp[ic_act,jf_act,kf_act],
                                                PDMB)
            (vx_01_x[ic_act,jf_act,kf_act],
            vx_11_x[ic_act,jf_act,kf_act])= MHDpy.PDM2(
                                                vx[ic_act,jf_act-2,kf_act],
                                                vx[ic_act,jf_act-1,kf_act],
                                                vx[ic_act,jf_act,kf_act],
                                                vx[ic_act,jf_act+1,kf_act],
                                                vx_interp[ic_act,jf_act,kf_act],
                                                PDMB)
        elif(limiter_type == 'TVD'):
            # put in TVD limiter here
            vx_interp[ic_act,jf_act,kf_act] = 0.25*(
                                                vx[ic_act,jf_act-1,kf_act-1] + 
                                                vx[ic_act,jf_act,kf_act-1] + 
                                                vx[ic_act,jf_act-1,kf_act] + 
                                                vx[ic_act,jf_act,kf_act])
                                                
            (vx_00_x[ic_act,jf_act,kf_act],
            vx_10_x[ic_act,jf_act,kf_act])= MHDpy.TVD2(
                                                vx[ic_act,jf_act-2,kf_act-1],
                                                vx[ic_act,jf_act-1,kf_act-1],
                                                vx[ic_act,jf_act,kf_act-1],
                                                vx[ic_act,jf_act+1,kf_act-1],
                                                vx_interp[ic_act,jf_act,kf_act])
            (vx_00_y[ic_act,jf_act,kf_act],
            vx_01_y[ic_act,jf_act,kf_act])= MHDpy.TVD2(
                                                vx[ic_act,jf_act-1,kf_act-2],
                                                vx[ic_act,jf_act-1,kf_act-1],
                                                vx[ic_act,jf_act-1,kf_act],
                                                vx[ic_act,jf_act-1,kf_act+1],
                                                vx_interp[ic_act,jf_act,kf_act])
            (vx_10_y[ic_act,jf_act,kf_act],
            vx_11_y[ic_act,jf_act,kf_act])= MHDpy.TVD2(
                                                vx[ic_act,jf_act,kf_act-2],
                                                vx[ic_act,jf_act,kf_act-1],
                                                vx[ic_act,jf_act,kf_act],
                                                vx[ic_act,jf_act,kf_act+1],
                                                vx_interp[ic_act,jf_act,kf_act])
            (vx_01_x[ic_act,jf_act,kf_act],
            vx_11_x[ic_act,jf_act,kf_act])= MHDpy.TVD2(
                                                vx[ic_act,jf_act-2,kf_act],
                                                vx[ic_act,jf_act-1,kf_act],
                                                vx[ic_act,jf_act,kf_act],
                                                vx[ic_act,jf_act+1,kf_act],
                                                vx_interp[ic_act,jf_act,kf_act])
            

        
        # quadrant 00 - get the left states in both y and z direction
        # calculate the difference between y and z directions
        # need to declare a few more worker arrays
        diff = n.zeros(sizetuple)
        vx_00 = n.zeros(sizetuple)
        vx_10 = n.zeros(sizetuple)
        vx_01 = n.zeros(sizetuple)
        vx_11 = n.zeros(sizetuple)
        vx_avg = n.zeros(sizetuple)
        
        diff[ic_act,jf_act,kf_act] = n.sign(n.abs(vx[ic_act,jf_act-2,kf_act-1]-
                                            vx[ic_act,jf_act-1,kf_act-1]) - 
                                            n.abs(vx[ic_act,jf_act-1,kf_act-2]-
                                            vx[ic_act,jf_act-1,kf_act-1]))
        # pick the state in the direction that has a bigger difference 
        # (if no difference then average the two states)
        vx_00[ic_act,jf_act,kf_act] = ((1+diff[ic_act,jf_act,kf_act])/2*
                                        vx_00_x[ic_act,jf_act,kf_act] + 
                                        (1-diff[ic_act,jf_act,kf_act])/2*
                                        vx_00_y[ic_act,jf_act,kf_act])
        
        # quadrant 10 - get the left states in both y and z direction
        # calculate the difference between y and z directions
        diff[ic_act,jf_act,kf_act] = n.sign(n.abs(vx[ic_act,jf_act+1,kf_act-1]-
                                            vx[ic_act,jf_act,kf_act-1]) - 
                                            n.abs(vx[ic_act,jf_act,kf_act-2]-
                                            vx[ic_act,jf_act,kf_act-1]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_10[ic_act,jf_act,kf_act] = ((1+diff[ic_act,jf_act,kf_act])/2*
                                        vx_10_x[ic_act,jf_act,kf_act] + 
                                        (1-diff[ic_act,jf_act,kf_act])/2*
                                        vx_10_y[ic_act,jf_act,kf_act])
        
        # quadrant 01 - get the left states in both y and z direction
        # calculate the difference between y and z directions
        diff[ic_act,jf_act,kf_act] = n.sign(n.abs(vx[ic_act,jf_act-2,kf_act]-
                                            vx[ic_act,jf_act-1,kf_act]) - 
                                            n.abs(vx[ic_act,jf_act-1,kf_act+1]-
                                            vx[ic_act,jf_act-1,kf_act]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_01[ic_act,jf_act,kf_act] = ((1+diff[ic_act,jf_act,kf_act])/2*
                                        vx_01_x[ic_act,jf_act,kf_act] + 
                                        (1-diff[ic_act,jf_act,kf_act])/2*
                                        vx_01_y[ic_act,jf_act,kf_act])
        
        # quadrant 11 - get the left states in both y and z direction
        # calculate the difference between y and z directions
        diff[ic_act,jf_act,kf_act] = n.sign(n.abs(vx[ic_act,jf_act+1,kf_act]-
                                            vx[ic_act,jf_act,kf_act]) - 
                                            n.abs(vx[ic_act,jf_act,kf_act+1]-
                                            vx[ic_act,jf_act,kf_act]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_11[ic_act,jf_act,kf_act] = ((1+diff[ic_act,jf_act,kf_act])/2*
                                        vx_11_x[ic_act,jf_act,kf_act] + 
                                        (1-diff[ic_act,jf_act,kf_act])/2*
                                        vx_11_y[ic_act,jf_act,kf_act])
        
        # calculate average vx at the cell corner by averaging the four states
        vx_avg[ic_act,jf_act,kf_act] = 0.25*(vx_00[ic_act,jf_act,kf_act] + 
                                            vx_01[ic_act,jf_act,kf_act] + 
                                            vx_10[ic_act,jf_act,kf_act] + 
                                            vx_11[ic_act,jf_act,kf_act])
        return vx_avg
                                        
    elif (direction == 'zx'): # FIXME!!!!!! NOT tested yet..
        # step 1 & 2: interpolate velocity from cell center to corner
        #         first interpolate in the z-direction, so the index for
        #         z-dimension changes from kc_act to kf_act, for the x-direction,
        #         interpolation is done for the whole x-domain, i.e., J
        #         then interpolate in the x-direction, so the index for x-dimension
        #         changes from J to jf_act
        vx_interp_x = n.zeros((I[-1,0,0]+1,jc_act[0,-1,0]+1,kf_act[0,0,-1]+1))
        sizetuple = (if_act[-1,0,0]+1,jc_act[0,-1,0]+1,kf_act[0,0,-1]+1)
        vx_interp = n.zeros(sizetuple)
        vx_00_x = n.zeros(sizetuple)
        vx_10_x = n.zeros(sizetuple)
        vx_00_y = n.zeros(sizetuple)
        vx_01_y = n.zeros(sizetuple)
        vx_10_y = n.zeros(sizetuple)
        vx_11_y = n.zeros(sizetuple)
        vx_01_x = n.zeros(sizetuple)
        vx_11_x = n.zeros(sizetuple)
  
        if(limiter_type == '8th'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (if_act, jc_act, kf_act]
            
          
            vx_interp_x[I,jc_act,kf_act] = (
                                            -3*vx[I,jc_act,kf_act-4]+
                                            29*vx[I,jc_act,kf_act-3]-
                                            139*vx[I,jc_act,kf_act-2]+
                                            533*vx[I,jc_act,kf_act-1]+
                                            533*vx[I,jc_act,kf_act]-
                                            139*vx[I,jc_act,kf_act+1]+
                                            29*vx[I,jc_act,kf_act+2]-
                                            3*vx[I,jc_act,kf_act+3])/840
            vx_interp[if_act,jc_act,kf_act] = (
                                    -3*vx_interp_x[if_act-4,jc_act,kf_act]+
                                    29*vx_interp_x[if_act-3,jc_act,kf_act]-
                                    139*vx_interp_x[if_act-2,jc_act,kf_act]+
                                    533*vx_interp_x[if_act-1,jc_act,kf_act]+
                                    533*vx_interp_x[if_act,jc_act,kf_act]-
                                    139*vx_interp_x[if_act+1,jc_act,kf_act]+
                                    29*vx_interp_x[if_act+2,jc_act,kf_act]-
                                    3*vx_interp_x[if_act+3,jc_act,kf_act])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            vx_00_x[if_act,jc_act,kf_act] = vx_interp[if_act,jc_act,kf_act]
            vx_10_x[if_act,jc_act,kf_act] = vx_interp[if_act,jc_act,kf_act]
            vx_00_y[if_act,jc_act,kf_act] = vx_interp[if_act,jc_act,kf_act]
            vx_01_y[if_act,jc_act,kf_act] = vx_interp[if_act,jc_act,kf_act]
            vx_10_y[if_act,jc_act,kf_act] = vx_interp[if_act,jc_act,kf_act]
            vx_11_y[if_act,jc_act,kf_act] = vx_interp[if_act,jc_act,kf_act]
            vx_01_x[if_act,jc_act,kf_act] = vx_interp[if_act,jc_act,kf_act]
            vx_11_x[if_act,jc_act,kf_act] = vx_interp[if_act,jc_act,kf_act]

        elif(limiter_type == 'PDM'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (if_act, jc_act, kf_act]
            vx_interp_x[I,jc_act,kf_act] = (
                                            -3*vx[I,jc_act,kf_act-4]+
                                            29*vx[I,jc_act,kf_act-3]-
                                            139*vx[I,jc_act,kf_act-2]+
                                            533*vx[I,jc_act,kf_act-1]+
                                            533*vx[I,jc_act,kf_act]-
                                            139*vx[I,jc_act,kf_act+1]+
                                            29*vx[I,jc_act,kf_act+2]-
                                            3*vx[I,jc_act,kf_act+3])/840
            vx_interp[if_act,jc_act,kf_act] = (
                                    -3*vx_interp_x[if_act-4,jc_act,kf_act]+
                                    29*vx_interp_x[if_act-3,jc_act,kf_act]-
                                    139*vx_interp_x[if_act-2,jc_act,kf_act]+
                                    533*vx_interp_x[if_act-1,jc_act,kf_act]+
                                    533*vx_interp_x[if_act,jc_act,kf_act]-
                                    139*vx_interp_x[if_act+1,jc_act,kf_act]+
                                    29*vx_interp_x[if_act+2,jc_act,kf_act]-
                                    3*vx_interp_x[if_act+3,jc_act,kf_act])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            (vx_00_x[if_act,jc_act,kf_act],
            vx_10_x[if_act,jc_act,kf_act]) = MHDpy.PDM2(
                                                vx[if_act-1,jc_act,kf_act-2],
                                                vx[if_act-1,jc_act,kf_act-1],
                                                vx[if_act-1,jc_act,kf_act],
                                                vx[if_act-1,jc_act,kf_act+1],
                                                vx_interp[if_act,jc_act,kf_act],
                                                PDMB)
            (vx_00_y[if_act,jc_act,kf_act],
            vx_01_y[if_act,jc_act,kf_act])= MHDpy.PDM2(
                                                vx[if_act-2,jc_act,kf_act-1],
                                                vx[if_act-1,jc_act,kf_act-1],
                                                vx[if_act,jc_act,kf_act-1],
                                                vx[if_act+1,jc_act,kf_act-1],
                                                vx_interp[if_act,jc_act,kf_act],
                                                PDMB)
            (vx_10_y[if_act,jc_act,kf_act],
            vx_11_y[if_act,jc_act,kf_act])= MHDpy.PDM2(
                                                vx[if_act-2,jc_act,kf_act],
                                                vx[if_act-1,jc_act,kf_act],
                                                vx[if_act,jc_act,kf_act],
                                                vx[if_act+1,jc_act,kf_act],
                                                vx_interp[if_act,jc_act,kf_act],
                                                PDMB)
            (vx_01_x[if_act,jc_act,kf_act],
            vx_11_x[if_act,jc_act,kf_act])= MHDpy.PDM2(
                                                vx[if_act,jc_act,kf_act-2],
                                                vx[if_act,jc_act,kf_act-1],
                                                vx[if_act,jc_act,kf_act],
                                                vx[if_act,jc_act,kf_act+1],
                                                vx_interp[if_act,jc_act,kf_act],
                                                PDMB)
        elif(limiter_type == 'TVD'):
            # put in TVD limiter here
            vx_interp[if_act,jc_act,kf_act] = 0.25*(
                                                vx[if_act-1,jc_act,kf_act-1] + 
                                                vx[if_act-1,jc_act,kf_act] + 
                                                vx[if_act,jc_act,kf_act-1] + 
                                                vx[if_act,jc_act,kf_act])
                                                
            (vx_00_x[if_act,jc_act,kf_act],
            vx_10_x[if_act,jc_act,kf_act])= MHDpy.TVD2(
                                                vx[if_act-1,jc_act,kf_act-2],
                                                vx[if_act-1,jc_act,kf_act-1],
                                                vx[if_act-1,jc_act,kf_act],
                                                vx[if_act-1,jc_act,kf_act+1],
                                                vx_interp[if_act,jc_act,kf_act])
            (vx_00_y[if_act,jc_act,kf_act],
            vx_01_y[if_act,jc_act,kf_act])= MHDpy.TVD2(
                                                vx[if_act-2,jc_act,kf_act-1],
                                                vx[if_act-1,jc_act,kf_act-1],
                                                vx[if_act,jc_act,kf_act-1],
                                                vx[if_act+1,jc_act,kf_act-1],
                                                vx_interp[if_act,jc_act,kf_act])
            (vx_10_y[if_act,jc_act,kf_act],
            vx_11_y[if_act,jc_act,kf_act])= MHDpy.TVD2(
                                                vx[if_act-2,jc_act,kf_act],
                                                vx[if_act-1,jc_act,kf_act],
                                                vx[if_act,jc_act,kf_act],
                                                vx[if_act+1,jc_act,kf_act],
                                                vx_interp[if_act,jc_act,kf_act])
            (vx_01_x[if_act,jc_act,kf_act],
            vx_11_x[if_act,jc_act,kf_act])= MHDpy.TVD2(
                                                vx[if_act,jc_act,kf_act-2],
                                                vx[if_act,jc_act,kf_act-1],
                                                vx[if_act,jc_act,kf_act],
                                                vx[if_act,jc_act,kf_act+1],
                                                vx_interp[if_act,jc_act,kf_act])

    
        # quadrant 00 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        # need to declare a few more worker arrays
        diff = n.zeros(sizetuple)
        vx_00 = n.zeros(sizetuple)
        vx_10 = n.zeros(sizetuple)
        vx_01 = n.zeros(sizetuple)
        vx_11 = n.zeros(sizetuple)
        vx_avg = n.zeros(sizetuple)
        diff[if_act,jc_act,kf_act] = n.sign(n.abs(vx[if_act-1,jc_act,kf_act-2]-
                                            vx[if_act-1,jc_act,kf_act-1]) - 
                                            n.abs(vx[if_act-2,jc_act,kf_act-1]-
                                            vx[if_act-1,jc_act,kf_act-1]))
        # pick the state in the direction that has a bigger difference 
        # (if no difference then average the two states)
        vx_00[if_act,jc_act,kf_act] = ((1+diff[if_act,jc_act,kf_act])/2*
                                        vx_00_x[if_act,jc_act,kf_act] + 
                                        (1-diff[if_act,jc_act,kf_act])/2*
                                        vx_00_y[if_act,jc_act,kf_act])
        
        # quadrant 10 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff[if_act,jc_act,kf_act] = n.sign(n.abs(vx[if_act-1,jc_act,kf_act+1]-
                                            vx[if_act-1,jc_act,kf_act]) - 
                                            n.abs(vx[if_act-2,jc_act,kf_act]-
                                            vx[if_act-1,jc_act,kf_act]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_10[if_act,jc_act,kf_act] = ((1+diff[if_act,jc_act,kf_act])/2*
                                        vx_10_x[if_act,jc_act,kf_act] + 
                                        (1-diff[if_act,jc_act,kf_act])/2*
                                        vx_10_y[if_act,jc_act,kf_act])
        
        # quadrant 01 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff[if_act,jc_act,kf_act] = n.sign(n.abs(vx[if_act,jc_act,kf_act-2]-
                                            vx[if_act,jc_act,kf_act-1]) - 
                                            n.abs(vx[if_act+1,jc_act,kf_act-1]-
                                            vx[if_act,jc_act,kf_act-1]))
        # pick the state that has a bigger difference (if no difference then average the two states)
        vx_01[if_act,jc_act,kf_act] = ((1+diff[if_act,jc_act,kf_act])/2*
                                        vx_01_x[if_act,jc_act,kf_act] + 
                                        (1-diff[if_act,jc_act,kf_act])/2*
                                        vx_01_y[if_act,jc_act,kf_act])
        
        # quadrant 11 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff[if_act,jc_act,kf_act] = n.sign(n.abs(vx[if_act,jc_act,kf_act+1]-
                                            vx[if_act,jc_act,kf_act]) - 
                                            n.abs(vx[if_act+1,jc_act,kf_act]-
                                            vx[if_act,jc_act,kf_act]))
        # pick the state that has a bigger difference (if no difference then average the two states)
        vx_11[if_act,jc_act,kf_act] = ((1+diff[if_act,jc_act,kf_act])/2*
                                        vx_11_x[if_act,jc_act,kf_act] + 
                                        (1-diff[if_act,jc_act,kf_act])/2*
                                        vx_11_y[if_act,jc_act,kf_act])
        
        # calculate average vx at the cell corner by averaging the four states
        vx_avg[if_act,jc_act,kf_act] = 0.25*(
                                            vx_00[if_act,jc_act,kf_act] + 
                                            vx_01[if_act,jc_act,kf_act] + 
                                            vx_10[if_act,jc_act,kf_act] + 
                                            vx_11[if_act,jc_act,kf_act])
        return vx_avg

    
    
