###############################################################################
#
#             3-D ideal MHD Solver on a Stretched Cartesian Grid
#    
#    Numerical methods are described in Lyon et al., [2004]. Most of the
#    LFM schemes are used in this 3-D MHD code except:
#    1. No curvilinear grid
#    2. No Ring-average needed
#    3. No Boris-correction for high Alfven speed (very low plasma beta)
#    4. No Background B field subtraction (high order Gaussin integral)
#
#    MAIN FEATURES:
#    1. Finite Volume (actually finite difference in Cartesian)
#    2. Semi-conservative - use plasma energy equation
#    3. 2nd-8th order reconstruction with TVD/WENO/PDM, default PDM
#    4. 2nd order Adams-Bashforth time-stepping scheme
#    5. Operator-splitting for the Lorentz force terms
#    6. Gas-kinetic flux functions for fluid and magnetic stresses
#    7. High order Constraint transport (Yee-Grd) to conserve Div B = 0
#    8. Resistive MHD - to be implemented, relatively simple
#    9. Hall MHD - to be implemented, can use the getEk function
#
#    ALGORITHM:
#    STEP 1: Adams-Bashforth predictor step (half time step update)
#    STEP 2: Calculate E fields at cell edges
#    STEP 3: Calculate fluid flux/stresses and magnetic stresses
#            a) x-direction 
#            b) y-direction
#            c) z-direction
#    STEP 4: Update the hydrodynamic equation without magnetic stress
#    STEP 5: Apply magnetic stress to the updated momentum
#    STEP 6: Evolve B field 
#    STEP 7: Apply boundary conditions
#
#    NOTE:
#    The cell length should be included in the reconstruction process in
#    order to obtain formal accuracy, which is probably more important in
#    the curvilinear version of the solver. For uniform cartesian, doesn't
#    matter.
###############################################################################

import numpy as n
import pylab as pl
import time,os
import MHDpy

# Model Parameters
NO = 8                # - default 8th order, hard coded for PDM
NO2 = NO/2            # num of ghost cells on each end
gamma=5/3.             # ratio of the specific heat, 5/3 for ideal gas
CFL = 0.3             # Courant number
PDMB= 4.0             # PDM beta parameter for controlling numerical diffusion
limiter_type = 'PDM'  # 'PDM' - 8th order with PDM limiter
                      # 'TVD' - 2nd order with Van-Leer limiter
                      # '8th' - 8th order without limiter
                      # 'WENO'- 5th order with WENO reconstruction 
                      #         (not tested in the getEk algorithm yet)
                      # 'PPM' - 3rd order PPM method (not tested yet)

# Grid information- nx,ny,nz are # of ACTIVE cells (no ghost cell included)
# The generated grid are cell corners, the metric function will calculate
# cell centers, faces and other grid information
nx = 128
ny = 128
nz = 1
(x,y,z)=MHDpy.Generate_Grid_3D_uniform(nx,ny,nz,NO) # This function generate a 
                                               # uniformly distributed active 
                                               # grid between -1 and 1 with 
                                               # nx, ny nz active cells in each 
                                               # direction
(nx_total,ny_total,nz_total)=x.shape           # with NO/2 ghost cells, 
                                               # nx_total, ny_total, nz_total 
                                               # are total num of cell corners
x = (x+1.0)/2.0 # map the grid from [-1 1] to [-0.5 -.5]
y = (y+1.0)/2.0 # map the grid from [-1 1] to [-0.5 0.5]
z = (z+0.0)/1.0 # z doesn't matter...

# Calculate grids and indices
# xc,yc,zc: cell centers
# xi,yi,zi: i-face cetners where bi is defined
# xj,yj,zj: j-face cetners where bj is defined
# xk,yk,zk: k-face cetners where bk is defined
# I,J,K: cell center indices including ghost cells
# ic_act,jc_act,kc_act: indices for active cell centers
# if_act,jf_act,kf_act: indices for active face centers
# "lb"s are left boundary indices, "rb"s are right boundary indices,
(xc,yc,zc,xi,yi,zi,xj,yj,zj,xk,yk,zk,dx,dy,dz,I,J,K,
ic_act,jc_act,kc_act,if_act,jf_act,kf_act,
ic_lb,jc_lb,kc_lb,if_lb,jf_lb,kf_lb,
ic_rb,jc_rb,kc_rb,if_rb,jf_rb,kf_rb)=MHDpy.Metrics(x,y,z,NO)

# Define premitive Hydrodynamic variables at cell center
rho = n.zeros(xc.shape)
vx = n.zeros(xc.shape)
vy = n.zeros(xc.shape)
vz = n.zeros(xc.shape)
p = n.zeros(xc.shape)

# Define Magnetic fields at cell faces
bi = n.zeros(xi.shape)
bj = n.zeros(xj.shape)
bk = n.zeros(xk.shape)

# Define Electric fields at cell edges
Ei = n.zeros((nx_total-1,ny_total,nz_total))
Ej = n.zeros((nx_total,ny_total-1,nz_total))
Ek = n.zeros((nx_total,ny_total,nz_total-1))
  
# 2-D Orzag-Tang vortex
rho = rho+25.0/(n.pi*36.0)
p   = p+5.0/(n.pi*12.0)
vx  = -n.sin(2.0*n.pi*yc)
vy  =  n.sin(2.0*n.pi*xc)
bi  = -1.0/n.sqrt(4.0*n.pi)*n.sin(2*n.pi*yi)
bj  = 1.0/n.sqrt(4.0*n.pi)*n.sin(2*n.pi*2*xj)

VecPot = 0# can use vector potential and Stokes theorm to calculate bi,bj,bk
# Define Electric fields at cell edges
#LAi = n.zeros((nx_total-1,ny_total,nz_total))
#LAj = n.zeros((nx_total,ny_total-1,nz_total))
#LAk = n.zeros((nx_total,ny_total,nz_total-1))
#if (VecPot==1):
#    # integrate Ax along x edge
#    for i in ic_act:
#        for j in jf_act:
#            for k in kf_act:
#                LAi[i,j,k] =  (x[i+1,j,k]-x[i,j,k])*(
#                    GaussianLineIntegral(Ax,x[i+1,j,k],y[i+1,j,k],z[i+1,j,k],
#                                    x[i,j,k],y[i,j,k],z[i,j,k]))
#
#    # integrate Ay along y edge
#    for i in if_act:
#        for j in jc_act:
#            for k in kf_act:
#                LAj[i,j,k] =  (y[i,j+1,k]-y[i,j,k])*(
#                    GaussianLineIntegral(Ay,x[i,j+1,k],y[i,j+1,k],z[i,j+1,k],
#                    x[i,j,k],y[i,j,k],z[i,j,k]))
#
#    # integrate Az along z edge
#    for i in if_act:
#        for j in jf_act:
#            for k in kc_act:
#                LAk[i,j,k] = (z[i,j,k+1]-z[i,j,k])*(
#                    GaussianLineIntegral(Az,x[i,j,k+1],y[i,j,k+1],z[i,j,k+1],
#                    x[i,j,k],y[i,j,k],z[i,j,k]))
#
#    # Stokes theorm for bi, bj, bk - face-integrated flux diveded by area
#    i=if_actj=jc_actk=kc_act
#    bi[i,j,k] = LAj[i,j,] - LAj[i,j,k+1] + LAk[i,j+1,k] - LAk[i,j,k]
#    bi[i,j,k] = bi[i,j,k]/dy[i,j,k]/dz[i,j,k]
#    
#    i=ic_actj=jf_actk=kc_act
#    bj[i,j,k] = -( LAi[i,j,k] - LAi[i,j,k+1] + LAk[i+1,j,k] - LAk[i,j,k] )
#    bj[i,j,k] = bj[i,j,k]/dz[i,j,k]/dx[i,j,k]
#    
#    i=ic_actj=jc_actk=kf_act
#    bk[i,j,k] = LAi[i,j,k] - LAi[i,j+1,k] + LAj[i+1,j,k] - LAj[i,j,k]
#    bk[i,j,k] = bk[i,j,k]/dx[i,j,k]/dy[i,j,k]

# set boundary conditions 
# For Orszag-Tang vortex simulation, use periopdic for both x and y
MHDpy.Boundaries(rho,p,vx,vy,vz,bi,bj,bk,NO,
                ic_act,jc_act,kc_act,
                if_act,jf_act,kf_act,
                ic_lb,ic_rb,jc_lb,jc_rb,kc_lb,kc_rb,
                if_lb,if_rb,jf_lb,jf_rb,kf_lb,kf_rb)
                
# calculate bx, by, bz at cell center, 2nd order accurate, equation (36) in
# Lyon et al., [2004] (the 1/8 in Lyon et al., [2004] is actually a typo)
bx = (bi[I+1,J,K] + bi[I,J,K])/2.
by = (bj[I,J+1,K] + bj[I,J,K])/2.
bz = (bk[I,J,K+1] + bk[I,J,K])/2.

# check if the electric field calculation is correct (and boundary cond)
# calculate edge-centered electric field 
(Ei,Ej,Ek) = MHDpy.getEk(vx,vy,vz,rho,p,gamma,bi,bj,bk,bx,by,bz,
                        I,J,K,ic_act,jc_act,kc_act,if_act,jf_act,kf_act,
                        PDMB,limiter_type)
# calculate cell-centered electric field: E = -vxB
Ez = -(vx*by - vy*bx)
# plot the distributions of Ek and Ez
fig,ax = pl.subplots(ncols=2,figsize=(16,8))
ax[0].pcolor(n.squeeze(x[ic_act,jc_act,kc_act[0]]),
            n.squeeze(y[ic_act,jc_act,kc_act[0]]),
            n.squeeze(Ez[ic_act,jc_act,kc_act[0]]))
ax[0].set_title('Ez')
ax[1].pcolor(n.squeeze(x[if_act,jf_act,kc_act[0]]),
            n.squeeze(y[if_act,jf_act,kc_act[0]]),
            n.squeeze(Ek[if_act,jf_act,kc_act[0]]))
ax[1].set_title('Ek')
pl.draw()



# Get conserved hydrodynamic variables
(rho,rhovx,rhovy,rhovz,eng) = MHDpy.getConservedVariables(rho,vx,vy,vz,p,gamma)

# get the first dt for A-B time stepping
dt0 = MHDpy.getDT(rho,vx,vy,vz,bx,by,bz,p,gamma,dx,dy,dz,CFL)

# Save the initial states for the first Adam-Bashforth time stepping
rho_p = rho
vx_p = vx
vy_p = vy
vz_p = vz
p_p= p
bx_p = bx
by_p = by
bz_p = bz

bi_p = bi
bj_p = bj
bk_p = bk

# MAIN LOOP
# simulation Time information
count = 0
Nstep=1000
Time = 0
RealT=0
step=0
imageNum=0
while (Time < 5.0):
#for step in n.arange(Nstep):
    Tstart=time.time()
    dt = MHDpy.getDT(rho,vx,vy,vz,bx,by,bz,p,gamma,dx,dy,dz,CFL)
    Time = Time + dt
    [rho,rhovx,rhovy,rhovz,eng] = MHDpy.getConservedVariables(rho,vx,vy,vz,
                                                                p,gamma)
    
    # Step 1: get the half time step values
    rho_h = rho + dt/dt0/2*(rho-rho_p)
    vx_h = vx + dt/dt0/2*(vx-vx_p)
    vy_h = vy + dt/dt0/2*(vy-vy_p)
    vz_h = vz + dt/dt0/2*(vz-vz_p)
    p_h = p + dt/dt0/2*(p-p_p)
    bx_h = bx + dt/dt0/2*(bx-bx_p)
    by_h = by + dt/dt0/2*(by-by_p)
    bz_h = bz + dt/dt0/2*(bz-bz_p)
    bi_h = bi + dt/dt0/2*(bi-bi_p)
    bj_h = bj + dt/dt0/2*(bj-bj_p)
    bk_h = bk + dt/dt0/2*(bk-bk_p)  
        
    rho0 = rho
    rhovx0 = rhovx
    rhovy0 = rhovy
    rhovz0 = rhovz
    eng0 = eng
    
    # save the current state vector for next AB time stepping
    rho_p = rho
    vx_p = vx
    vy_p = vy
    vz_p = vz
    bx_p = bx
    by_p = by
    bz_p = bz
    bi_p = bi
    bj_p = bj
    bk_p = bk    
    p_p= p
    dt0 = dt    
    
    #Step 2 Calculat the electric field, no resistivity term    
    (Ei,Ej,Ek) = MHDpy.getEk(vx,vy,vz,rho,p,gamma,bi,bj,bk,bx,by,bz,
                            I,J,K,ic_act,jc_act,kc_act,if_act,jf_act,kf_act,
                            PDMB,limiter_type)
    
    # Step 3 Calculate fluid and magnetic flux/stresses
    # a) Calculate fluid flux/stress in the x direction
    # # reconstruct cell centered primitive variables to cell faces
    (rho_left, rho_right) = MHDpy.reconstruct_3D(rho_h,if_act,jf_act,kf_act,
                                        PDMB,1,limiter_type)    
    (vx_left, vx_right) = MHDpy.reconstruct_3D(vx_h,if_act,jf_act,kf_act,
                                        PDMB,1,limiter_type)   
    (vy_left, vy_right) = MHDpy.reconstruct_3D(vy_h,if_act,jf_act,kf_act,
                                        PDMB,1,limiter_type)       
    (vz_left, vz_right) = MHDpy.reconstruct_3D(vz_h,if_act,jf_act,kf_act,
                                        PDMB,1,limiter_type)        
    (p_left, p_right) = MHDpy.reconstruct_3D(p_h,if_act,jf_act,kf_act,
                                        PDMB,1,limiter_type)   
    # use Gas-hydro flux flunction to calculate the net flux at cell faces
    # Here we use a Gaussian distribution, the temperature is fluid.
    # Can use waterbag in Lyon et al., [2004]. Results are very similar.
    (Frho_p,FrhoVx_p,FrhoVy_p,FrhoVz_p,Feng_p,_,_,_,_,_) = MHDpy.getHydroFlux(
                            rho_left,vx_left,vy_left,vz_left,p_left,gamma,1)
    (_,_,_,_,_,Frho_n,FrhoVx_n,FrhoVy_n,FrhoVz_n,Feng_n) = MHDpy.getHydroFlux(
                            rho_right,vx_right,vy_right,vz_right,p_right,gamma,1) 
    
    rho_flux_x = Frho_p + Frho_n
    vx_flux_x = FrhoVx_p + FrhoVx_n
    vy_flux_x = FrhoVy_p + FrhoVy_n
    vz_flux_x = FrhoVz_p + FrhoVz_n
    eng_flux_x = Feng_p + Feng_n
    
    # calculate magnetic stress in the x direction
    # Reconstruct the cell centered magnetic fields to cell faces
    (bx_left, bx_right) = MHDpy.reconstruct_3D(bx_h,if_act,jf_act,kf_act,
                                        PDMB,1,limiter_type)   
    (by_left, by_right) = MHDpy.reconstruct_3D(by_h,if_act,jf_act,kf_act,
                                        PDMB,1,limiter_type)
    (bz_left, bz_right) = MHDpy.reconstruct_3D(bz_h,if_act,jf_act,kf_act,
                                        PDMB,1,limiter_type)
    # Magnetic distribution function is also Gaussian, the "temperature" is
    # fluid+magnetic
    (Bstress_x_p, Bstress_y_p, Bstress_z_p,_, _,_) = MHDpy.getMagneticStress(
        rho_left,vx_left,vy_left,vz_left,p_left,bx_left,by_left,bz_left,1)
    (_,_,_,Bstress_x_n, Bstress_y_n, Bstress_z_n) = MHDpy.getMagneticStress(
        rho_right,vx_right,vy_right,vz_right,p_right,bx_right,by_right,bz_right,1)
    
    BstressX_x = Bstress_x_p + Bstress_x_n
    BstressY_x = Bstress_y_p + Bstress_y_n
    BstressZ_x = Bstress_z_p + Bstress_z_n    
        
    #  b) Calculate Flux in the y direction    
    (rho_left, rho_right) = MHDpy.reconstruct_3D(rho_h,if_act,jf_act,kf_act,
                                            PDMB,2,limiter_type)    
    (vx_left, vx_right) = MHDpy.reconstruct_3D(vx_h,if_act,jf_act,kf_act,
                                            PDMB,2,limiter_type)   
    (vy_left, vy_right) = MHDpy.reconstruct_3D(vy_h,if_act,jf_act,kf_act,
                                            PDMB,2,limiter_type)       
    (vz_left, vz_right) = MHDpy.reconstruct_3D(vz_h,if_act,jf_act,kf_act,
                                            PDMB,2,limiter_type)        
    (p_left, p_right) = MHDpy.reconstruct_3D(p_h,if_act,jf_act,kf_act,
                                            PDMB,2,limiter_type)   
    
    (Frho_py,FrhoVx_py,FrhoVy_py,FrhoVz_py,Feng_py,_,_,_,_,_) = MHDpy.getHydroFlux(
                        rho_left,vx_left,vy_left,vz_left,p_left,gamma,2)
    (_,_,_,_,_,Frho_ny,FrhoVx_ny,FrhoVy_ny,FrhoVz_ny,Feng_ny) = MHDpy.getHydroFlux(
                        rho_right,vx_right,vy_right,vz_right,p_right,gamma,2) 
    
    rho_flux_y = Frho_py + Frho_ny
    vx_flux_y = FrhoVx_py + FrhoVx_ny
    vy_flux_y = FrhoVy_py + FrhoVy_ny
    vz_flux_y = FrhoVz_py + FrhoVz_ny
    eng_flux_y = Feng_py + Feng_ny    
    
    # calculate magnetic stress in the Y direction
    #[bx_left, bx_right] = reconstruct_3D(bx_h,if_act,jf_act,kf_act,
    #                                    PDMB,2,limiter_type)   
    (by_left, by_right) = MHDpy.reconstruct_3D(by_h,if_act,jf_act,kf_act,
                                            PDMB,2,limiter_type)       
    (bz_left, bz_right) = MHDpy.reconstruct_3D(bz_h,if_act,jf_act,kf_act,
                                            PDMB,2,limiter_type)      
    
    (Bstress_x_p, Bstress_y_p, Bstress_z_p,_,_,_) = MHDpy.getMagneticStress(
        rho_left,vx_left,vy_left,vz_left,p_left,bx_left,by_left,bz_left,2)
    (_,_,_,Bstress_x_n, Bstress_y_n, Bstress_z_n) = MHDpy.getMagneticStress(
        rho_right,vx_right,vy_right,vz_right,p_right,bx_right,by_right,bz_right,2)
    
    BstressX_y = Bstress_x_p + Bstress_x_n
    BstressY_y = Bstress_y_p + Bstress_y_n
    BstressZ_y = Bstress_z_p + Bstress_z_n    
    
    # c) Calculate Flux in the z direction
    (rho_left, rho_right) = MHDpy.reconstruct_3D(rho_h,if_act,jf_act,kf_act,
                                            PDMB,3,limiter_type)    
    (vx_left, vx_right) = MHDpy.reconstruct_3D(vx_h,if_act,jf_act,kf_act,
                                            PDMB,3,limiter_type)   
    (vy_left, vy_right) = MHDpy.reconstruct_3D(vy_h,if_act,jf_act,kf_act,
                                            PDMB,3,limiter_type)       
    (vz_left, vz_right) = MHDpy.reconstruct_3D(vz_h,if_act,jf_act,kf_act,
                                            PDMB,3,limiter_type)        
    (p_left, p_right) = MHDpy.reconstruct_3D(p_h,if_act,jf_act,kf_act,
                                            PDMB,3,limiter_type)   
    
    (Frho_px,FrhoVx_px,FrhoVy_px,FrhoVz_px,Feng_px,_,_,_,_,_) = MHDpy.getHydroFlux(
                    rho_left,vx_left,vy_left,vz_left,p_left,gamma,3)
    (_,_,_,_,_,Frho_nx,FrhoVx_nx,FrhoVy_nx,FrhoVz_nx,Feng_nx) = MHDpy.getHydroFlux(
                    rho_right,vx_right,vy_right,vz_right,p_right,gamma,3) 
    
    rho_flux_z = Frho_px + Frho_nx
    vx_flux_z = FrhoVx_px + FrhoVx_nx
    vy_flux_z = FrhoVy_px + FrhoVy_nx
    vz_flux_z = FrhoVz_px + FrhoVz_nx
    eng_flux_z = Feng_px + Feng_nx        
    
    # calculate magnetic stress in the Z direction
    (bx_left, bx_right) = MHDpy.reconstruct_3D(bx_h,if_act,jf_act,kf_act,
                                            PDMB,3,limiter_type)   
    (by_left, by_right) = MHDpy.reconstruct_3D(by_h,if_act,jf_act,kf_act,
                                            PDMB,3,limiter_type)       
    (bz_left, bz_right) = MHDpy.reconstruct_3D(bz_h,if_act,jf_act,kf_act,
                                            PDMB,3,limiter_type)      
    
    (Bstress_x_p, Bstress_y_p, Bstress_z_p,_,_,_) = MHDpy.getMagneticStress(
        rho_left,vx_left,vy_left,vz_left,p_left,bx_left,by_left,bz_left,3)
    (_,_,_,Bstress_x_n, Bstress_y_n, Bstress_z_n) = MHDpy.getMagneticStress(
        rho_right,vx_right,vy_right,vz_right,p_right,bx_right,by_right,bz_right,3)
    
    BstressX_z = Bstress_x_p + Bstress_x_n
    BstressY_z = Bstress_y_p + Bstress_y_n
    BstressZ_z = Bstress_z_p + Bstress_z_n    
    
    # Step 4 update Hydro variables without magnetic stress - operator
    # splitting step withouth JxB force
    rho[ic_act,jc_act,kc_act] = (rho0[ic_act,jc_act,kc_act] - 
        dt/dx[ic_act,jc_act,kc_act]*(rho_flux_x[ic_act+1,jc_act,kc_act]-
        rho_flux_x[ic_act,jc_act,kc_act])- 
        dt/dy[ic_act,jc_act,kc_act]*(rho_flux_y[ic_act,jc_act+1,kc_act]-
        rho_flux_y[ic_act,jc_act,kc_act]) -
        dt/dz[ic_act,jc_act,kc_act]*(rho_flux_z[ic_act,jc_act,kc_act+1]-
        rho_flux_z[ic_act,jc_act,kc_act]) )
    rhovx[ic_act,jc_act,kc_act] = (rhovx0[ic_act,jc_act,kc_act] - 
        dt/dx[ic_act,jc_act,kc_act]*(vx_flux_x[ic_act+1,jc_act,kc_act]-
        vx_flux_x[ic_act,jc_act,kc_act]) - 
        dt/dy[ic_act,jc_act,kc_act]*(vx_flux_y[ic_act,jc_act+1,kc_act]-
        vx_flux_y[ic_act,jc_act,kc_act]) - 
        dt/dz[ic_act,jc_act,kc_act]*(vx_flux_z[ic_act,jc_act,kc_act+1]-
        vx_flux_z[ic_act,jc_act,kc_act]) )
    rhovy[ic_act,jc_act,kc_act] = (rhovy0[ic_act,jc_act,kc_act] - 
        dt/dx[ic_act,jc_act,kc_act]*(vy_flux_x[ic_act+1,jc_act,kc_act]-
        vy_flux_x[ic_act,jc_act,kc_act]) - 
        dt/dy[ic_act,jc_act,kc_act]*(vy_flux_y[ic_act,jc_act+1,kc_act]-
        vy_flux_y[ic_act,jc_act,kc_act]) - 
        dt/dz[ic_act,jc_act,kc_act]*(vy_flux_z[ic_act,jc_act,kc_act+1]-
        vy_flux_z[ic_act,jc_act,kc_act]) )
    rhovz[ic_act,jc_act,kc_act] = (rhovz0[ic_act,jc_act,kc_act] - 
        dt/dx[ic_act,jc_act,kc_act]*(vz_flux_x[ic_act+1,jc_act,kc_act]-
        vz_flux_x[ic_act,jc_act,kc_act]) - 
        dt/dy[ic_act,jc_act,kc_act]*(vz_flux_y[ic_act,jc_act+1,kc_act]-
        vz_flux_y[ic_act,jc_act,kc_act]) - 
        dt/dz[ic_act,jc_act,kc_act]*(vz_flux_z[ic_act,jc_act,kc_act+1]-
        vz_flux_z[ic_act,jc_act,kc_act]) )
    eng[ic_act,jc_act,kc_act] = (eng0[ic_act,jc_act,kc_act] - 
        dt/dx[ic_act,jc_act,kc_act]*(eng_flux_x[ic_act+1,jc_act,kc_act]-
        eng_flux_x[ic_act,jc_act,kc_act]) - 
        dt/dy[ic_act,jc_act,kc_act]*(eng_flux_y[ic_act,jc_act+1,kc_act]-
        eng_flux_y[ic_act,jc_act,kc_act]) - 
        dt/dz[ic_act,jc_act,kc_act]*(eng_flux_z[ic_act,jc_act,kc_act+1]-
        eng_flux_z[ic_act,jc_act,kc_act]) )
    
    # get plasma pressure - now rho and p are solved (to O(dt)?)
    vx = rhovx/rho
    vy = rhovy/rho
    vz = rhovz/rho
    p = (eng - 0.5*rho*(vx**2+vy**2+vz**2))*(gamma-1)
    
    # Step 5 apply the magnetic stress to the momentums - operator 
    # splitting step dealing with JxB force Since JxB doesn't heat the
    # plasma, pressure is not changed between the Step 4 and 5
    rhovx[ic_act,jc_act,kc_act] = (rhovx[ic_act,jc_act,kc_act] - 
        dt/dx[ic_act,jc_act,kc_act]*(BstressX_x[ic_act+1,jc_act,kc_act]-
        BstressX_x[ic_act,jc_act,kc_act]) - 
        dt/dy[ic_act,jc_act,kc_act]*(BstressX_y[ic_act,jc_act+1,kc_act]-
        BstressX_y[ic_act,jc_act,kc_act]) - 
        dt/dz[ic_act,jc_act,kc_act]*(BstressX_z[ic_act,jc_act,kc_act+1]-
        BstressX_z[ic_act,jc_act,kc_act]) )
    rhovy[ic_act,jc_act,kc_act] = (rhovy[ic_act,jc_act,kc_act] - 
        dt/dx[ic_act,jc_act,kc_act]*(BstressY_x[ic_act+1,jc_act,kc_act]-
        BstressY_x[ic_act,jc_act,kc_act]) - 
        dt/dy[ic_act,jc_act,kc_act]*(BstressY_y[ic_act,jc_act+1,kc_act]-
        BstressY_y[ic_act,jc_act,kc_act]) - 
        dt/dz[ic_act,jc_act,kc_act]*(BstressY_z[ic_act,jc_act,kc_act+1]-
        BstressY_z[ic_act,jc_act,kc_act]) )
    rhovz[ic_act,jc_act,kc_act] = (rhovz[ic_act,jc_act,kc_act] - 
        dt/dx[ic_act,jc_act,kc_act]*(BstressZ_x[ic_act+1,jc_act,kc_act]-
        BstressZ_x[ic_act,jc_act,kc_act]) - 
        dt/dy[ic_act,jc_act,kc_act]*(BstressZ_y[ic_act,jc_act+1,kc_act]-
        BstressZ_y[ic_act,jc_act,kc_act]) - 
        dt/dz[ic_act,jc_act,kc_act]*(BstressZ_z[ic_act,jc_act,kc_act+1]-
        BstressZ_z[ic_act,jc_act,kc_act]) )
    
    # get velocities with magnetic stress - now vx, vy, vz are solved
    vx = rhovx/rho
    vy = rhovy/rho
    vz = rhovz/rho
        
    # Step 6 update face-center magnetic fields (bi,bj,bk) through Faraday's law
    bi[if_act,jc_act,kc_act] = bi[if_act,jc_act,kc_act] - dt*( 
        1/dy[if_act,jc_act,kc_act]*(Ek[if_act,jc_act+1,kc_act]-
        Ek[if_act,jc_act,kc_act]) -
        1/dz[if_act,jc_act,kc_act]*(Ej[if_act,jc_act,kc_act+1]-
        Ej[if_act,jc_act,kc_act]) )
    bj[ic_act,jf_act,kc_act] = bj[ic_act,jf_act,kc_act] - dt*( 
        1/dz[ic_act,jf_act,kc_act]*(Ei[ic_act,jf_act,kc_act+1]-
        Ei[ic_act,jf_act,kc_act])-
        1/dx[ic_act,jf_act,kc_act]*(Ek[ic_act+1,jf_act,kc_act]-
        Ek[ic_act,jf_act,kc_act]) )
    bk[ic_act,jc_act,kf_act] = bk[ic_act,jc_act,kf_act] - dt*( 
        1/dx[ic_act,jc_act,kf_act]*(Ej[ic_act+1,jc_act,kf_act]-
        Ej[ic_act,jc_act,kf_act])-
        1/dy[ic_act,jc_act,kf_act]*(Ei[ic_act,jc_act+1,kf_act]-
        Ei[ic_act,jc_act,kf_act]) )    
                                                                
    # Step 7 Apply Boundary Conditions
    MHDpy.Boundaries(rho,p,vx,vy,vz,bi,bj,bk,NO,
                    ic_act,jc_act,kc_act,
                    if_act,jf_act,kf_act,
                    ic_lb,ic_rb,jc_lb,jc_rb,kc_lb,kc_rb,
                    if_lb,if_rb,jf_lb,jf_rb,kf_lb,kf_rb)  
                                            
    # calculate bx, by, bz at cell center, 2nd order accurate, since bi,
    # bj, bk are already modified use boundary conditions, no need bc here
    # for bx, by, bz
    bx = (bi[I+1,J,K] + bi[I,J,K])/2
    by = (bj[I,J+1,K] + bj[I,J,K])/2
    bz = (bk[I,J,K+1] + bk[I,J,K])/2
    
    RealDT = time.time() - Tstart
    RealT = RealT + RealDT
    step = step +1
    # plot results
    if((step %50)==1):
        # check divB
        divB = n.zeros((ic_act[-1,0,0]+1,jc_act[0,-1,0]+1,kc_act[0,0,-1]+1))
        divB[ic_act,jc_act,kc_act] =( ( (bi[ic_act+1,jc_act,kc_act] - 
            bi[ic_act,jc_act,kc_act])*
            dy[ic_act,jc_act,kc_act]*dz[ic_act,jc_act,kc_act] +
            (bj[ic_act,jc_act+1,kc_act] - 
            bj[ic_act,jc_act,kc_act])*
            dz[ic_act,jc_act,kc_act]*dx[ic_act,jc_act,kc_act] + 
            (bk[ic_act,jc_act,kc_act+1] - 
            bk[ic_act,jc_act,kc_act])*
            dx[ic_act,jc_act,kc_act]*dy[ic_act,jc_act,kc_act] ) /
            dx[ic_act,jc_act,kc_act]/
            dy[ic_act,jc_act,kc_act]/
            dz[ic_act,jc_act,kc_act])
    
        pl.figure()
        pl.pcolor(n.squeeze(xc[ic_act,jc_act,kc_act[0]]),
            n.squeeze(yc[ic_act,jc_act,kc_act[0]]),
            n.squeeze(rho[ic_act,jc_act,kc_act[0]]))
        pl.title('Simulation Time = %f' % Time)
        saveFigName = os.path.join('/Users/wiltbemj/Downloads/figs',
                                    'ot-%06d.png'%imageNum)
        pl.savefig(saveFigName,dpi=100)
        pl.close()
        imageNum = imageNum + 1
        print 'Loop %d Sim Time = %f Real Time = %f' % (step,Time,RealT)
        print 'Sim DT = %f Real DT = %f ' % (dt,RealDT)
        print ' Max(divB) = %f ' % n.max(n.abs(divB))


                                                                                                                                                                                                