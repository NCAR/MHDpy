"""
Compute average velocity at cell corner from cell centers
"""
def center2corner(vx,NO2,PDMB,direction,limiter_type='PDM'):
    """
    Function compoutes average values at cell corners from cell centers.
    Requries:
        vx - velocity
        NO2 - Half numerical order
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
        #     first interpolate in the x-direction, so the index for
        #     x-dimension changes from ic_act to NO2:-NO2+1,for the y-direction,
        #     interpolation is done for the whole y-domain, i.e., J
        #     then interpolate in the y-direction, so the index for 
        #     y-dimension changes from J to jf_act
    

        if ( limiter_type == '8th'):   
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (NO2:-NO2+1, jf_act, kc_act]
            vx_interp_x = (
                -3*vx[NO2-4:-NO2-4+1,:,NO2:-NO2]+
                29*vx[NO2-3:-NO2-3+1,:,NO2:-NO2]-
                139*vx[NO2-2:-NO2-2+1,:,NO2:-NO2]+
                533*vx[NO2-1:-NO2-1+1,:,NO2:-NO2]+
                533*vx[NO2:-NO2+1,:,NO2:-NO2]-
                139*vx[NO2+1:-NO2+1+1,:,NO2:-NO2]+
                29*vx[NO2+2:-NO2+2+1,:,NO2:-NO2]-
                3*vx[NO2+3:,:,NO2:-NO2])/840
            vx_interp = (
                -3*vx_interp_x[:,NO2-4:-NO2-4+1,:]+
                29*vx_interp_x[:,NO2-3:-NO2-3+1,:]-
                139*vx_interp_x[:,NO2-2:-NO2-2+1,:]+
                533*vx_interp_x[:,NO2-1:-NO2-1+1,:]+
                533*vx_interp_x[:,NO2:-NO2+1,:]-
                139*vx_interp_x[:,NO2+1:-NO2+1+1,:]+
                29*vx_interp_x[:,NO2+2:-NO2+2+1,:]-
                3*vx_interp_x[:,NO2+3:,:])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            vx_00_x = vx_interp
            vx_10_x = vx_interp
            vx_00_y = vx_interp
            vx_01_y = vx_interp
            vx_10_y = vx_interp
            vx_11_y = vx_interp
            vx_01_x = vx_interp
            vx_11_x = vx_interp
        elif (limiter_type == 'PDM'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (NO2:-NO2+1, jf_act, kc_act]
            vx_interp_x = (
                -3*vx[NO2-4:-NO2-4+1,:,NO2:-NO2]+
                29*vx[NO2-3:-NO2-3+1,:,NO2:-NO2]-
                139*vx[NO2-2:-NO2-2+1,:,NO2:-NO2]+
                533*vx[NO2-1:-NO2-1+1,:,NO2:-NO2]+
                533*vx[NO2:-NO2+1,:,NO2:-NO2]-
                139*vx[NO2+1:-NO2+1+1,:,NO2:-NO2]+
                29*vx[NO2+2:-NO2+2+1,:,NO2:-NO2]-
                3*vx[NO2+3:,:,NO2:-NO2])/840
            vx_interp = (
                -3*vx_interp_x[:,NO2-4:-NO2-4+1,:]+
                29*vx_interp_x[:,NO2-3:-NO2-3+1,:]-
                139*vx_interp_x[:,NO2-2:-NO2-2+1,:]+
                533*vx_interp_x[:,NO2-1:-NO2-1+1,:]+
                533*vx_interp_x[:,NO2:-NO2+1,:]-
                139*vx_interp_x[:,NO2+1:-NO2+1+1,:]+
                29*vx_interp_x[:,NO2+2:-NO2+2+1,:]-
                3*vx_interp_x[:,NO2+3:,:])/840
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            (vx_00_x,vx_10_x)=MHDpy.PDM2(
                                    vx[NO2-2:-NO2-2+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                    vx[NO2-1:-NO2-1+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                    vx[NO2:-NO2+1    ,NO2-1:-NO2-1+1,NO2:-NO2],
                                    vx[NO2+1:-NO2+1+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                    vx_interp,PDMB)
            (vx_00_y,vx_01_y)= MHDpy.PDM2(
                                    vx[NO2-1:-NO2-1+1,NO2-2:-NO2-2+1,NO2:-NO2],
                                    vx[NO2-1:-NO2-1+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2+1    ,NO2:-NO2],
                                    vx[NO2-1:-NO2-1+1,NO2+1:-NO2+1+1,NO2:-NO2],
                                    vx_interp,PDMB)
            (vx_10_y,vx_11_y)= MHDpy.PDM2(
                                    vx[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2],
                                    vx[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                    vx[NO2:-NO2+1,NO2:-NO2+1    ,NO2:-NO2],
                                    vx[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2],
                                    vx_interp,PDMB)
            (vx_01_x,vx_11_x)= MHDpy.PDM2(
                                    vx[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2],
                                    vx[NO2:-NO2+1    ,NO2:-NO2+1,NO2:-NO2],
                                    vx[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2],
                                    vx_interp,PDMB)
        elif(limiter_type == 'TVD'):
            # put in TVD limiter here
            vx_interp = 0.25*(
                                vx[NO2-1:-NO2-1+1,NO2-1:-NO2-1+1,NO2:-NO2] + 
                                vx[NO2:-NO2+1    ,NO2-1:-NO2-1+1,NO2:-NO2] +
                                vx[NO2-1:-NO2-1+1,NO2:-NO2+1    ,NO2:-NO2] + 
                                vx[NO2:-NO2+1    ,NO2:-NO2+1    ,NO2:-NO2])
                                                
            (vx_00_x,vx_10_x)= MHDpy.TVD2(
                                vx[NO2-2:-NO2-2+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                vx[NO2-1:-NO2-1+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                vx[NO2:-NO2+1    ,NO2-1:-NO2-1+1,NO2:-NO2],
                                vx[NO2+1:-NO2+1+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                vx_interp)
            (vx_00_y,vx_01_y)=MHDpy.TVD2(
                                vx[NO2-1:-NO2-1+1,NO2-2:-NO2-2+1,NO2:-NO2],
                                vx[NO2-1:-NO2-1+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                vx[NO2-1:-NO2-1+1,NO2:-NO2+1    ,NO2:-NO2],
                                vx[NO2-1:-NO2-1+1,NO2+1:-NO2+1+1,NO2:-NO2],
                                vx_interp)
            (vx_10_y,vx_11_y)=MHDpy.TVD2(
                                vx[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2],
                                vx[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2],
                                vx[NO2:-NO2+1,NO2:-NO2+1    ,NO2:-NO2],
                                vx[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2],
                                vx_interp)
            (vx_01_x,vx_11_x)=MHDpy.TVD2(
                                vx[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2],
                                vx[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2],
                                vx[NO2:-NO2+1    ,NO2:-NO2+1,NO2:-NO2],
                                vx[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2],
                                vx_interp)
                
        # quadrant 00 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff = n.sign(n.abs(vx[NO2-2:-NO2-2+1,NO2-1:-NO2-1+1,NO2:-NO2]-
                            vx[NO2-1:-NO2-1+1,NO2-1:-NO2-1+1,NO2:-NO2]) - 
                            n.abs(vx[NO2-1:-NO2-1+1,NO2-2:-NO2-2+1,NO2:-NO2]-
                            vx[NO2-1:-NO2-1+1,NO2-1:-NO2-1+1,NO2:-NO2]))
        # pick the state in the direction that has a bigger difference 
        # (if no difference then average the two states)
        vx_00 = ((1+diff)/2*vx_00_x +(1-diff)/2*vx_00_y)
        
        # quadrant 10 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff = n.sign(n.abs(vx[NO2+1:-NO2+1+1,NO2-1:-NO2-1+1,NO2:-NO2]-
                            vx[NO2:-NO2+1    ,NO2-1:-NO2-1+1,NO2:-NO2]) - 
                            n.abs(vx[NO2:-NO2+1,NO2-2:-NO2-2+1,NO2:-NO2]-
                            vx[NO2:-NO2+1,NO2-1:-NO2-1+1,NO2:-NO2]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_10 = ((1+diff)/2*vx_10_x +(1-diff)/2*vx_10_y)
        
        # quadrant 01 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff = n.sign(n.abs(vx[NO2-2:-NO2-2+1,NO2:-NO2+1,NO2:-NO2]-
                            vx[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2]) - 
                            n.abs(vx[NO2-1:-NO2-1+1,NO2+1:-NO2+1+1,NO2:-NO2]-
                            vx[NO2-1:-NO2-1+1,NO2:-NO2+1,NO2:-NO2]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_01 = ((1+diff)/2*vx_01_x +(1-diff)/2*vx_01_y)
        
        # quadrant 11 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff = n.sign(n.abs(vx[NO2+1:-NO2+1+1,NO2:-NO2+1,NO2:-NO2]-
                            vx[NO2:-NO2+1    ,NO2:-NO2+1,NO2:-NO2]) - 
                            n.abs(vx[NO2:-NO2+1,NO2+1:-NO2+1+1,NO2:-NO2]-
                            vx[NO2:-NO2+1,NO2:-NO2+1,NO2:-NO2]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_11 = ((1+diff)/2*vx_11_x + (1-diff)/2*vx_11_y)
        
        # calculate average vx at the cell corner by averaging the four states
        vx_avg = 0.25*(vx_00 + vx_01 + vx_10 + vx_11)
        
        return vx_avg
        
    elif (direction == 'yz'): #
        # step 1 & 2: interpolate velocity from cell center to corner
        #         first interpolate in the y-direction, so the index for
        #         y-dimension changes from jc_act to jf_act,for the y-direction,
        #         interpolation is done for the whole z-domain, i.e., K
        #         then interpolate in the z-direction, so the index for 
        #         z-dimension changes from K to kf_act

        if ( limiter_type == '8th'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (ic_act, jf_act, kf_act]
            vx_interp_x = (
                -3*vx[NO2:-NO2,NO2-4:-NO2-4+1,:]+
                29*vx[NO2:-NO2,NO2-3:-NO2-3+1,:]-
                139*vx[NO2:-NO2,NO2-2:-NO2-2+1,:]+
                533*vx[NO2:-NO2,NO2-1:-NO2-1+1,:]+
                533*vx[NO2:-NO2,NO2:-NO2+1,:]-
                139*vx[NO2:-NO2,NO2+1:-NO2+1+1,:]+
                29*vx[NO2:-NO2,NO2+2:-NO2+2+1,:]-
                3*vx[NO2:-NO2,NO2+3:,:])/840
            vx_interp = (
                -3*vx_interp_x[:,:,NO2-4:-NO2-4+1]+
                29*vx_interp_x[:,:,NO2-3:-NO2-3+1]-
                139*vx_interp_x[:,:,NO2-2:-NO2-2+1]+
                533*vx_interp_x[:,:,NO2-1:-NO2-1+1]+
                533*vx_interp_x[:,:,NO2:-NO2+1]-
                139*vx_interp_x[:,:,NO2+1:-NO2+1+1]+
                29*vx_interp_x[:,:,NO2+2:-NO2+2+1]-
                3*vx_interp_x[:,:,NO2+3:])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            vx_00_x = vx_interp
            vx_10_x = vx_interp
            vx_00_y = vx_interp
            vx_01_y = vx_interp
            vx_10_y = vx_interp
            vx_11_y = vx_interp
            vx_01_x = vx_interp
            vx_11_x = vx_interp
            
        elif( limiter_type == 'PDM'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (ic_act, jf_act, kf_act]
            vx_interp_x = (
                -3*vx[NO2:-NO2,NO2-4:-NO2-4+1,:]+
                29*vx[NO2:-NO2,NO2-3:-NO2-3+1,:]-
                139*vx[NO2:-NO2,NO2-2:-NO2-2+1,:]+
                533*vx[NO2:-NO2,NO2-1:-NO2-1+1,:]+
                533*vx[NO2:-NO2,NO2:-NO2+1,:]-
                139*vx[NO2:-NO2,NO2+1:-NO2+1+1,:]+
                29*vx[NO2:-NO2,NO2+2:-NO2+2+1,:]-
                3*vx[NO2:-NO2,NO2+3:,:])/840
            vx_interp = (
                -3*vx_interp_x[:,:,NO2-4:-NO2-4+1]+
                29*vx_interp_x[:,:,NO2-3:-NO2-3+1]-
                139*vx_interp_x[:,:,NO2-2:-NO2-2+1]+
                533*vx_interp_x[:,:,NO2-1:-NO2-1+1]+
                533*vx_interp_x[:,:,NO2:-NO2+1]-
                139*vx_interp_x[:,:,NO2+1:-NO2+1+1]+
                29*vx_interp_x[:,:,NO2+2:-NO2+2+1]-
                3*vx_interp_x[:,:,NO2+3:])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            (vx_00_x,vx_10_x)= MHDpy.PDM2(
                                vx[NO2:-NO2,NO2-2:-NO2-2+1,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2:-NO2+1    ,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2+1:-NO2+1+1,NO2-1:-NO2-1+1],
                                vx_interp,PDMB)
            (vx_00_y,vx_01_y)= MHDpy.PDM2(
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-2:-NO2-2+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2+1:-NO2+1+1],
                                vx_interp,PDMB)
            (vx_10_y,vx_11_y)= MHDpy.PDM2(
                                vx[NO2:-NO2,NO2:-NO2+1,NO2-2:-NO2-2+1],
                                vx[NO2:-NO2,NO2:-NO2+1,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2:-NO2+1,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2:-NO2+1,NO2+1:-NO2+1+1],
                                vx_interp,PDMB)
            (vx_01_x,vx_11_x)= MHDpy.PDM2(
                                vx[NO2:-NO2,NO2-2:-NO2-2+1,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2:-NO2+1    ,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2+1:-NO2+1+1,NO2:-NO2+1],
                                vx_interp,PDMB)
        elif(limiter_type == 'TVD'):
            # put in TVD limiter here
            vx_interp = 0.25*(
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-1:-NO2-1+1] + 
                                vx[NO2:-NO2,NO2:-NO2+1    ,NO2-1:-NO2-1+1] + 
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2:-NO2+1] + 
                                vx[NO2:-NO2,NO2:-NO2+1    ,NO2:-NO2+1])
                                                
            (vx_00_x,vx_10_x)= MHDpy.TVD2(
                                vx[NO2:-NO2,NO2-2:-NO2-2+1,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2:-NO2+1    ,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2+1:-NO2+1+1,NO2-1:-NO2-1+1],
                                vx_interp)
            (vx_00_y,vx_01_y)= MHDpy.TVD2(
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-2:-NO2-2+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2+1:-NO2+1+1],
                                vx_interp)
            (vx_10_y,vx_11_y)= MHDpy.TVD2(
                                vx[NO2:-NO2,NO2:-NO2+1,NO2-2:-NO2-2+1],
                                vx[NO2:-NO2,NO2:-NO2+1,NO2-1:-NO2-1+1],
                                vx[NO2:-NO2,NO2:-NO2+1,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2:-NO2+1,NO2+1:-NO2+1+1],
                                vx_interp)
            (vx_01_x,vx_11_x)= MHDpy.TVD2(
                                vx[NO2:-NO2,NO2-2:-NO2-2+1,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2:-NO2+1    ,NO2:-NO2+1],
                                vx[NO2:-NO2,NO2+1:-NO2+1+1,NO2:-NO2+1],
                                vx_interp)
            

        
        # quadrant 00 - get the left states in both y and z direction
        # calculate the difference between y and z directions

        diff = n.sign(n.abs(vx[NO2:-NO2,NO2-2:-NO2-2+1,NO2-1:-NO2-1+1]-
                            vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-1:-NO2-1+1]) - 
                            n.abs(vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-2:-NO2-2+1]-
                            vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2-1:-NO2-1+1]))
        # pick the state in the direction that has a bigger difference 
        # (if no difference then average the two states)
        vx_00 = ((1+diff)/2*vx_00_x + (1-diff)/2*vx_00_y)
        
        # quadrant 10 - get the left states in both y and z direction
        # calculate the difference between y and z directions
        diff = n.sign(n.abs(vx[NO2:-NO2,NO2+1:-NO2+1+1,NO2-1:-NO2-1+1]-
                            vx[NO2:-NO2,NO2:-NO2+1    ,NO2-1:-NO2-1+1]) - 
                            n.abs(vx[NO2:-NO2,NO2:-NO2+1,NO2-2:-NO2-2+1]-
                            vx[NO2:-NO2,NO2:-NO2+1,NO2-1:-NO2-1+1]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_10 = ((1+diff)/2*vx_10_x + (1-diff)/2*vx_10_y)
        
        # quadrant 01 - get the left states in both y and z direction
        # calculate the difference between y and z directions
        diff = n.sign(n.abs(vx[NO2:-NO2,NO2-2:-NO2-2+1,NO2:-NO2+1]-
                            vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2:-NO2+1]) - 
                            n.abs(vx[NO2:-NO2,NO2-1:-NO2-1+1,NO2+1:-NO2+1+1]-
                            vx[NO2:-NO2      ,NO2-1:-NO2-1+1,NO2:-NO2+1]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_01 = ((1+diff)/2*vx_01_x + (1-diff)/2*vx_01_y)
        
        # quadrant 11 - get the left states in both y and z direction
        # calculate the difference between y and z directions
        diff = n.sign(n.abs(vx[NO2:-NO2,NO2+1:-NO2+1+1,NO2:-NO2+1]-
                            vx[NO2:-NO2,NO2:-NO2+1    ,NO2:-NO2+1]) - 
                            n.abs(vx[NO2:-NO2,NO2:-NO2+1,NO2+1:-NO2+1+1]-
                            vx[NO2:-NO2      ,NO2:-NO2+1,NO2:-NO2+1]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_11 = ((1+diff)/2*vx_11_x + (1-diff)/2*vx_11_y)
        
        # calculate average vx at the cell corner by averaging the four states
        vx_avg = 0.25*(vx_00 + vx_01 + vx_10 + vx_11)
        return vx_avg
                                        
    elif (direction == 'zx'): # FIXME!!!!!! NOT tested yet..
        # step 1 & 2: interpolate velocity from cell center to corner
        #         first interpolate in the z-direction, so the index for
        #         z-dimension changes from kc_act to kf_act, for the x-direction,
        #         interpolation is done for the whole x-domain, i.e., J
        #         then interpolate in the x-direction, so the index for x-dimension
        #         changes from J to jf_act

        if(limiter_type == '8th'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (NO2:-NO2+1, jc_act, kf_act]
            
          
            vx_interp_x = (
                    -3*vx[:,NO2:-NO2,NO2-4:-NO2-4+1]+
                    29*vx[:,NO2:-NO2,NO2-3:-NO2-3+1]-
                    139*vx[:,NO2:-NO2,NO2-2:-NO2-2+1]+
                    533*vx[:,NO2:-NO2,NO2-1:-NO2-1+1]+
                    533*vx[:,NO2:-NO2,NO2:-NO2+1]-
                    139*vx[:,NO2:-NO2,NO2+1:-NO2+1+1]+
                    29*vx[:,NO2:-NO2,NO2+2:-NO2+2+1]-
                    3*vx[:,NO2:-NO2,NO2+3:])/840
            vx_interp = (
                    -3*vx_interp_x[NO2-4:-NO2-4+1,:,:]+
                    29*vx_interp_x[NO2-3:-NO2-3+1,:,:]-
                    139*vx_interp_x[NO2-2:-NO2-2+1,:,:]+
                    533*vx_interp_x[NO2-1:-NO2-1+1,:,:]+
                    533*vx_interp_x[NO2:-NO2+1,:,:]-
                    139*vx_interp_x[NO2+1:-NO2+1+1,:,:]+
                    29*vx_interp_x[NO2+2:-NO2+2+1,:,:]-
                    3*vx_interp_x[NO2+3:,:,:])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            vx_00_x = vx_interp
            vx_10_x = vx_interp
            vx_00_y = vx_interp
            vx_01_y = vx_interp
            vx_10_y = vx_interp
            vx_11_y = vx_interp
            vx_01_x = vx_interp
            vx_11_x = vx_interp

        elif(limiter_type == 'PDM'):
            # interpolate vx from cell center to cell corner, 
            # vx_interp dimension is (NO2:-NO2+1, jc_act, kf_act]
            vx_interp_x = (
                    -3*vx[:,NO2:-NO2,NO2-4:-NO2-4+1]+
                    29*vx[:,NO2:-NO2,NO2-3:-NO2-3+1]-
                    139*vx[:,NO2:-NO2,NO2-2:-NO2-2+1]+
                    533*vx[:,NO2:-NO2,NO2-1:-NO2-1+1]+
                    533*vx[:,NO2:-NO2,NO2:-NO2+1]-
                    139*vx[:,NO2:-NO2,NO2+1:-NO2+1+1]+
                    29*vx[:,NO2:-NO2,NO2+2:-NO2+2+1]-
                    3*vx[:,NO2:-NO2,NO2+3:])/840
            vx_interp = (
                    -3*vx_interp_x[NO2-4:-NO2-4+1,:,:]+
                    29*vx_interp_x[NO2-3:-NO2-3+1,:,:]-
                    139*vx_interp_x[NO2-2:-NO2-2+1,:,:]+
                    533*vx_interp_x[NO2-1:-NO2-1+1,:,:]+
                    533*vx_interp_x[NO2:-NO2+1,:,:]-
                    139*vx_interp_x[NO2+1:-NO2+1+1,:,:]+
                    29*vx_interp_x[NO2+2:-NO2+2+1,:,:]-
                    3*vx_interp_x[NO2+3:,:,:])/840
            
            # split the interpolated velocity to 8-states, 
            # 2-states in each quadrant:
            # 00_x, 00_y, 10_x, 10_y, 01_x, 01_y, 11_x, 11_y
            (vx_00_x,vx_10_x) = MHDpy.PDM2(
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2-2:-NO2-2+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2+1:-NO2+1+1],
                                    vx_interp,PDMB)
            (vx_00_y,vx_01_y)= MHDpy.PDM2(
                                    vx[NO2-2:-NO2-2+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2:-NO2+1    ,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2+1:-NO2+1+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx_interp,PDMB)
            (vx_10_y,vx_11_y)= MHDpy.PDM2(
                                    vx[NO2-2:-NO2-2+1,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2:-NO2+1    ,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2+1:-NO2+1+1,NO2:-NO2,NO2:-NO2+1],
                                    vx_interp,PDMB)
            (vx_01_x,vx_11_x)= MHDpy.PDM2(
                                    vx[NO2:-NO2+1,NO2:-NO2,NO2-2:-NO2-2+1],
                                    vx[NO2:-NO2+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2:-NO2+1,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2:-NO2+1,NO2:-NO2,NO2+1:-NO2+1+1],
                                    vx_interp,PDMB)
        elif(limiter_type == 'TVD'):
            # put in TVD limiter here
            vx_interp = 0.25*(
                            vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2-1:-NO2-1+1] + 
                            vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2:-NO2+1] + 
                            vx[NO2:-NO2+1    ,NO2:-NO2,NO2-1:-NO2-1+1] + 
                            vx[NO2:-NO2+1    ,NO2:-NO2,NO2:-NO2+1])
                                                
            (vx_00_x,vx_10_x)= MHDpy.TVD2(
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2-2:-NO2-2+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2+1:-NO2+1+1],
                                    vx_interp)
            (vx_00_y,vx_01_y)= MHDpy.TVD2(
                                    vx[NO2-2:-NO2-2+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2:-NO2+1    ,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2+1:-NO2+1+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx_interp)
            (vx_10_y,vx_11_y)= MHDpy.TVD2(
                                    vx[NO2-2:-NO2-2+1,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2:-NO2+1    ,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2+1:-NO2+1+1,NO2:-NO2,NO2:-NO2+1],
                                    vx_interp)
            (vx_01_x,vx_11_x)= MHDpy.TVD2(
                                    vx[NO2:-NO2+1,NO2:-NO2,NO2-2:-NO2-2+1],
                                    vx[NO2:-NO2+1,NO2:-NO2,NO2-1:-NO2-1+1],
                                    vx[NO2:-NO2+1,NO2:-NO2,NO2:-NO2+1],
                                    vx[NO2:-NO2+1,NO2:-NO2,NO2+1:-NO2+1+1],
                                    vx_interp)

    
        # quadrant 00 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        # need to declare a few more worker arrays
        diff = n.sign(n.abs(vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2-2:-NO2-2+1]-
                            vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2-1:-NO2-1+1]) - 
                            n.abs(vx[NO2-2:-NO2-2+1,NO2:-NO2,NO2-1:-NO2-1+1]-
                            vx[NO2-1:-NO2-1+1      ,NO2:-NO2,NO2-1:-NO2-1+1]))
        # pick the state in the direction that has a bigger difference 
        # (if no difference then average the two states)
        vx_00 = ((1+diff)/2*vx_00_x + (1-diff)/2*vx_00_y)
        
        # quadrant 10 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff = n.sign(n.abs(vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2+1:-NO2+1+1]-
                            vx[NO2-1:-NO2-1+1,NO2:-NO2,NO2:-NO2+1]) - 
                            n.abs(vx[NO2-2:-NO2-2+1,NO2:-NO2,NO2:-NO2+1]-
                            vx[NO2-1:-NO2-1+1      ,NO2:-NO2,NO2:-NO2+1]))
        # pick the state that has a bigger difference 
        # (if no difference then average the two states)
        vx_10 = ((1+diff)/2*vx_10_x + (1-diff)/2*vx_10_y)
        
        # quadrant 01 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff = n.sign(n.abs(vx[NO2:-NO2+1,NO2:-NO2,NO2-2:-NO2-2+1]-
                            vx[NO2:-NO2+1,NO2:-NO2,NO2-1:-NO2-1+1]) - 
                            n.abs(vx[NO2+1:-NO2+1+1,NO2:-NO2,NO2-1:-NO2-1+1]-
                            vx[NO2:-NO2+1          ,NO2:-NO2,NO2-1:-NO2-1+1]))
        # pick the state that has a bigger difference 
        #(if no difference then average the two states)
        vx_01 = ((1+diff)/2*vx_01_x + (1-diff)/2*vx_01_y)
        
        # quadrant 11 - get the left states in both x and y direction
        # calculate the difference between x and y directions
        diff = n.sign(n.abs(vx[NO2:-NO2+1,NO2:-NO2,NO2+1:-NO2+1+1]-
                            vx[NO2:-NO2+1,NO2:-NO2,NO2:-NO2+1]) - 
                            n.abs(vx[NO2+1:-NO2+1+1,NO2:-NO2,NO2:-NO2+1]-
                            vx[NO2:-NO2+1          ,NO2:-NO2,NO2:-NO2+1]))
        # pick the state that has a bigger difference 
        #(if no difference then average the two states)
        vx_11 = ((1+diff)/2*vx_11_x + (1-diff)/2*vx_11_y)
        
        # calculate average vx at the cell corner by averaging the four states
        vx_avg = 0.25*(vx_00 + vx_01 + vx_10 + vx_11)
        return vx_avg

    
    
