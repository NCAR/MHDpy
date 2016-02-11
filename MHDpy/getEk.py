"""
Compute the Electric field components on the cell edges
"""
def getEk(vx,vy,vz,rho,p,gamma,bi,bj,bk,bx,by,bz,
          I,J,K,ic_act,jc_act,kc_act,if_act,jf_act,kf_act,NO2,
          PDMB,limiter_type='PDM'):
    """
    Function compoutes the electric field components along the cell edges using
    a high order constrained transport method on a Yee type grid.
    Requries:
        rho,vx,vy,vz,p - plasma variables at cell centers
        gamma - ratio of specific heats
        bi,bj,bk - magnetic fluxes on cell faces
        bx,by,bz - magnetic field at cell centers
        I,J,K - locations of all cells
        ic_act,jc_act,kc_act - location of active cell centers
        if_act,jf_act,kf_act - location of active faces
        NO2 - half numerical order
        PDMB - B parameter for PDM method
        limiter_type - type of limiter to use in 
    Returns:
        Ei,Ej,Ek - Elecrtic fields along cell edges
    """               
    import numpy as n
    import MHDpy 

###########################################################################
# This code calculates the Electric field components on cell edges - Ei, Ej
# and Ek using high order constrained transport (Yee grid) method, details
# described in Lyon et al., [2004]. The electric field is estimated as:
#                      E = - v_avg x B_upwind + eta*J
# where v_avg is an average velocity at cell edges and B_upwind is the
# upwinded magnetic field at cell edges chosen based on the average velocity.
# Note that the eta*J term here is not usual resistive MHD term rather it's 
# only turned on when the limiter detects a discontinuity in the B fields. 
# The eta is set to be the local fast speed averaged around the cell edges. 
# This term is important in the regions where Alfven waves are present. 
#
# ALGORITHM (Use Ek as an example):
# STEP 1: interpolate cell-centered vx, vy to cell-edge (vx_avg, vy_avg)
# STEP 2: reconstruct Bi in y-dir to get bi_left and bi_right at cell edges
#         reconstruct Bj in x-dir to get bj_left and bj_right at cell edges
# STEP 3: pick the upwinded B components based on vx_avg and vy_avg, that
#         is, if vx_avg>0, choose bj_left, otherwise choose bj_right
#                vy_avg>0, choose bj_right ,otherwise choose bj_left
# STEP 4: compute the electric field as Ek = - v_avg x B_upwind
# STEP 5: the reconstructed Bi and Bi form a current at cell edges:
#         J = bi_left + bj_right - bi_right - bj_left the diffusion term is
#         added as v_diffusive * J, v_diffusive is chosen as the average
#         local fast mode speed. Note that this diffusive term contains
#         both numerical and Alflven resistivity.
#
# INDICES:
# For Ei, the involved velocity components are (vy, vz), and the magnetic
# components are (bj, bk) - the i-indices are ic_act
#                         - the edge-indices are [ic_act, jf_act, kf_act]
# For Ej, the involved velocity components are (vz, vx), and the magnetic
# components are (bk, bi) - the j-indices are jc_act
#                         - the edge-indices are [if_act, jc_act, kf_act]
# For Ek, the involved velocity components are (vx, vy), and the magnetic
# components are (bi, bj) - the k-indices are kc_act
#                         - the edge-indices are [if_act, jf_act, kc_act]
###########################################################################
    
    if (limiter_type == 'WENO'):
        limiter_type = 'PDM'
    
    ###########################################################################
    #                              calculate Ek
    ###########################################################################
    # step 1 - get an avg velocity at cell edges (vx_avg,vy_avg), plane = 'xy'
    # vx_avg - dimension [if_act,jf_act,kc_act]
    vx_avg = MHDpy.center2corner(vx,I,J,K,ic_act,jc_act,kc_act,
                                if_act,jf_act,kf_act,NO2,PDMB,'xy',limiter_type)
    # dimension [if_act,jf_act,kc_act] 
    vy_avg = MHDpy.center2corner(vy,I,J,K,ic_act,jc_act,kc_act,
                                if_act,jf_act,kf_act,NO2,PDMB,'xy',limiter_type) 
    
    # step 2 - reconstruct face-centered bi in the y-direction (direction=2) to
    #          reconstruct face-centered bj in the x-direction (direction=1) to 
    #          cell edges
    # dimension [if_act,jf_act,kc_act]

    [bi_left, bi_right] = MHDpy.reconstruct_3D(bi,if_act,jf_act,kc_act,NO2,PDMB,
                                        2,limiter_type)
    [bj_left, bj_right] = MHDpy.reconstruct_3D(bj,if_act,jf_act,kc_act,NO2,PDMB, 
                                        1,limiter_type) 
    
    # step 3 - pick the upwinded bi,bj based on the direction of vx_avg, vy_avg
    #          if vx_avg > 0, then bj_upwind = bj_left else bj_upwind = bj_right
    #          if vy_avg > 0, then bi_upwind = bi_left else bi_upwind = bi_right
    #          this makes the B fields are advected in the upwind direction,
    #          however, this does not account the Alfven wave information which
    #          requires step 5.
    # recontruct_3D returns if_act,jf_act,kf_act so we need
    # bi[:-1,:,:-1] and bj[:,:-1,:-1]
    bi_upwind = (
                    (1+n.sign(vy_avg))/2*bi_left[:-1,:,:-1] + 
                    (1-n.sign(vy_avg))/2*bi_right[:-1,:,:-1])
    bj_upwind = (
                    (1+n.sign(vx_avg))/2*bj_left[:,:-1,:-1] + 
                    (1-n.sign(vx_avg))/2*bj_right[:,:-1,:-1])
    
    # step - 4 compute v_avg cross b_upwind 
    Ek = -(vx_avg*bj_upwind - vy_avg*bi_upwind) # v_avg cross b_upwind
    
    # step 5:  with Alfven resistivity, the eta*j
    #          term is useful in the regions where Alfven waves/fast waves are 
    #          important, which is not the resistivity for reconnection
    #          calculate average fast speed at cell center
    
    VF = n.sqrt(vx**2+vy**2+vz**2) + n.sqrt((bx**2+by**2+bz**2)/rho)
    etak = 0.25*(
                VF[NO2-1:-NO2-1+1,NO2-1:-NO2-1+1,NO2:-NO2] + 
                VF[NO2:-NO2+1    ,NO2-1:-NO2-1+1,NO2:-NO2] + 
                VF[NO2-1:-NO2-1+1,NO2:-NO2+1    ,NO2:-NO2] + 
                VF[NO2:-NO2+1    ,NO2:-NO2+1    ,NO2:-NO2]) 
                                    
    Ek = (Ek + etak*(
        bi_left[:-1,:,:-1] + bj_right[:,:-1,:-1] - 
        bi_right[:-1,:,:-1] -bj_left[:,:-1,:-1]))
    
    ###########################################################################
    #                              calculate Ei
    ###########################################################################
    # step 1 - get an average velocity at cell corners (vy_avg,vz_avg)
    # dimension [ic_act,jf_act,kf_act]
    vy_avg = MHDpy.center2corner(vy,I,J,K,ic_act,jc_act,kc_act,
                                if_act,jf_act,kf_act,NO2,PDMB,'yz',limiter_type)
    # dimension [ic_act,jf_act,kf_act] 
    vz_avg = MHDpy.center2corner(vz,I,J,K,ic_act,jc_act,kc_act,
                                if_act,jf_act,kf_act,NO2,PDMB,'yz',limiter_type) 
    
    # step 2 - reconstruct face-centered bi, bj to cell corners
    #          bi is defined on xi[if_act,jc_act,kc_act], only need
    #          reconstruction once in the j-direction

    # dimension [ic_act,jf_act,kf_act]
    # recontruct_3D returns if_act,jf_act,kf_act so to get ic_act need :-1
    [bj_left, bj_right] = MHDpy.reconstruct_3D(bj,ic_act,jf_act,kf_act,NO2,PDMB, 
                                        3,limiter_type)
    # dimension [ic_act,jf_act,kf_act] 
    [bk_left, bk_right] = MHDpy.reconstruct_3D(bk,ic_act,jf_act,kf_act,NO2,PDMB, 
                                        2,limiter_type) 
    
    # step 3 - pick the upwinded bi,bj based on the direction of vx_avg, vy_avg
    # recontruct_3D returns if_act,jf_act,kf_act so we need
    # bk[:-1,:,:-1] and bj[:-1,:-1,:]
    bk_upwind = (
                (1+n.sign(vy_avg))/2*bk_left[:-1,:,:-1] + 
                (1-n.sign(vy_avg))/2*bk_right[:-1,:,:-1])
    bj_upwind = (
                (1+n.sign(vz_avg))/2*bj_left[:-1,:-1,:] + 
                (1-n.sign(vz_avg))/2*bj_right[:-1,:-1,:])
    
    # step - 4 compute v_avg cross b_upwind with Alfven resistivity, the eta*j
    #          term is useful in the regions where Alfven waves/fast waves are 
    #          important, which is not the resistivity for reconnection
    Ei = -(vy_avg*bk_upwind - vz_avg*bj_upwind)
    
    # calculate average fast speed at cell center
    # uses VF from Ek calc
    etai = 0.25*(VF[NO2:-NO2,NO2-1:-NO2-1+1,NO2-1:-NO2-1+1] + 
                 VF[NO2:-NO2,NO2-1:-NO2-1+1,NO2:-NO2+1] + 
                 VF[NO2:-NO2,NO2:-NO2+1,NO2-1:-NO2-1+1] + 
                 VF[NO2:-NO2,NO2:-NO2+1,NO2:-NO2+1])
    
    Ei = (Ei + etai*( 
            bj_left[:-1,:-1,:] + bk_right[:-1,:,:-1]-
            bj_right[:-1,:-1,:]- bk_left[:-1,:,:-1]))
                                                                                
    ###########################################################################                                                                               
    # calculate Ej using 2D splitting - Lyon et al., [2004]
    # step 1 - get an average velocity at cell corners (vz_avg,vx_avg)
    # dimension [if_act,jc_act,kf_act]
    vz_avg = MHDpy.center2corner(vz,I,J,K,ic_act,jc_act,kc_act,
                                if_act,jf_act,kf_act,NO2,PDMB,'zx',limiter_type)
    # dimension [if_act,jc_act,kf_act] 
    vx_avg = MHDpy.center2corner(vx,I,J,K,ic_act,jc_act,kc_act,
                                if_act,jf_act,kf_act,NO2,PDMB,'zx',limiter_type) 
    
    # step 2 - reconstruct face-centered bi, bj to cell corners
    #          bi is defined on xi[if_act,jc_act,kc_act], only need
    #          reconstruction once in the j-direction
    # dimension [if_act,jc_act,kf_act]
    # recontruct_3D returns if_act,jf_act,kf_act so to get jc_act need :-1
    [bi_left, bi_right] = MHDpy.reconstruct_3D(bi,if_act,jc_act,kf_act,NO2,PDMB, 
                                        3,limiter_type)
    # dimension [if_act,jc_act,kf_act] 
    [bk_left, bk_right] = MHDpy.reconstruct_3D(bk,if_act,jc_act,kf_act,NO2,PDMB, 
                                        1,limiter_type) 
    
    # step 3 - pick the upwinded bi,bj based on the direction of vx_avg, vy_avg
    # recontruct_3D returns if_act,jf_act,kf_act so we need
    # bk[:,:-1,:-1] and bi[:-1,:-1,:]
    bk_upwind = (
                    (1+n.sign(vx_avg))/2*bk_left[:,:-1,:-1] + 
                    (1-n.sign(vx_avg))/2*bk_right[:,:-1,:-1])
    bi_upwind = (
                    (1+n.sign(vz_avg))/2*bi_left[:-1,:-1,:] + 
                    (1-n.sign(vz_avg))/2*bi_right[:-1,:-1,:])
    
    # step - 4 compute v_avg cross b_upwind with Alfven resistivity, the eta*j
    #          term is useful in the regions where Alfven waves/fast waves are 
    #          important, which is not the resistivity for reconnection
    # v_avg cross b_upwind
    Ej = -(vz_avg*bi_upwind-vx_avg*bk_upwind)

    # calculate average fast speed at cell center
    # once again we reuse the VF calculated in Ek
    etaj = 0.25*(
                VF[NO2-1:-NO2-1+1,NO2:-NO2,NO2-1:-NO2-1+1] + 
                VF[NO2:-NO2+1    ,NO2:-NO2,NO2-1:-NO2-1+1] + 
                VF[NO2-1:-NO2-1+1,NO2:-NO2,NO2:-NO2+1] + 
                VF[NO2:-NO2+1    ,NO2:-NO2,NO2:-NO2+1])
    Ej = (Ej + etaj* (
            bk_left[:,:-1,:-1] + bi_right[:-1,:-1,:]-
            bk_right[:,:-1,:-1]- bi_left[:-1,:-1,:]))
    
    return (Ei,Ej,Ek)
