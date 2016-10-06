"""
Compute single fluid stresses and return them as a tuple
"""
def Gas_Hydro1(rho_h,vx_h,vy_h,vz_h,p_h,dx,dy,dz,
            NO2,PDMB,limiter_type,gamma,dt,CA):
    """
    Function single fluid stresses 
    Requries:
        rho,vx,vy,vz,p - plasma variables
        bx,by,bz - cell centered magnetic field
        dx,dy,dz - grid lengths
        PDMB,limter_type - numeric type information
        gamma - ratio of specific heats
        dt - timestep
        CA - Alfven correction
    Returns:
        (drho,dpxF,dpyF,dpzF,deng) - Tuple of stresses
    """
    import MHDpy
    
    # Step 3 Calculate fluid and magnetic flux/stresses
    # a) Calculate fluid flux/stress in the x direction
    # # reconstruct cell centered primitive variables to cell faces
    (rho_left, rho_right) = MHDpy.reconstruct_3D(rho_h,NO2,PDMB,1,limiter_type)    
    (vx_left, vx_right) = MHDpy.reconstruct_3D(vx_h,NO2,PDMB,1,limiter_type)   
    (vy_left, vy_right) = MHDpy.reconstruct_3D(vy_h,NO2,PDMB,1,limiter_type)       
    (vz_left, vz_right) = MHDpy.reconstruct_3D(vz_h,NO2,PDMB,1,limiter_type)        
    (p_left, p_right) = MHDpy.reconstruct_3D(p_h,NO2,PDMB,1,limiter_type)   
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
        
    #  b) Calculate Flux in the y direction    
    (rho_left, rho_right) = MHDpy.reconstruct_3D(rho_h,NO2,PDMB,2,limiter_type)    
    (vx_left, vx_right) = MHDpy.reconstruct_3D(vx_h,NO2,PDMB,2,limiter_type)   
    (vy_left, vy_right) = MHDpy.reconstruct_3D(vy_h,NO2,PDMB,2,limiter_type)       
    (vz_left, vz_right) = MHDpy.reconstruct_3D(vz_h,NO2,PDMB,2,limiter_type)        
    (p_left, p_right) = MHDpy.reconstruct_3D(p_h,NO2,PDMB,2,limiter_type)   
    
    (Frho_py,FrhoVx_py,FrhoVy_py,FrhoVz_py,Feng_py,_,_,_,_,_) = MHDpy.getHydroFlux(
                        rho_left,vx_left,vy_left,vz_left,p_left,gamma,2)
    (_,_,_,_,_,Frho_ny,FrhoVx_ny,FrhoVy_ny,FrhoVz_ny,Feng_ny) = MHDpy.getHydroFlux(
                        rho_right,vx_right,vy_right,vz_right,p_right,gamma,2) 
    
    rho_flux_y = Frho_py + Frho_ny
    vx_flux_y = FrhoVx_py + FrhoVx_ny
    vy_flux_y = FrhoVy_py + FrhoVy_ny
    vz_flux_y = FrhoVz_py + FrhoVz_ny
    eng_flux_y = Feng_py + Feng_ny       
    
    # c) Calculate Flux in the z direction
    (rho_left, rho_right) = MHDpy.reconstruct_3D(rho_h,NO2,PDMB,3,limiter_type)    
    (vx_left, vx_right) = MHDpy.reconstruct_3D(vx_h,NO2,PDMB,3,limiter_type)   
    (vy_left, vy_right) = MHDpy.reconstruct_3D(vy_h,NO2,PDMB,3,limiter_type)       
    (vz_left, vz_right) = MHDpy.reconstruct_3D(vz_h,NO2,PDMB,3,limiter_type)        
    (p_left, p_right) = MHDpy.reconstruct_3D(p_h,NO2,PDMB,3,limiter_type)   
    
    (Frho_px,FrhoVx_px,FrhoVy_px,FrhoVz_px,Feng_px,_,_,_,_,_) = MHDpy.getHydroFlux(
                    rho_left,vx_left,vy_left,vz_left,p_left,gamma,3)
    (_,_,_,_,_,Frho_nx,FrhoVx_nx,FrhoVy_nx,FrhoVz_nx,Feng_nx) = MHDpy.getHydroFlux(
                    rho_right,vx_right,vy_right,vz_right,p_right,gamma,3) 
    
    rho_flux_z = Frho_px + Frho_nx
    vx_flux_z = FrhoVx_px + FrhoVx_nx
    vy_flux_z = FrhoVy_px + FrhoVy_nx
    vz_flux_z = FrhoVz_px + FrhoVz_nx
    eng_flux_z = Feng_px + Feng_nx          
        
    # initialze stress return variables
    drho = rho_h*0.0
    dpxF = rho_h*0.0
    dpyF = rho_h*0.0
    dpzF = rho_h*0.0
    deng = rho_h*0.0
    
    # volume integrated fluid stress
    drho[NO2:-NO2,NO2:-NO2,NO2:-NO2] = ( - 
        dt/dx[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(rho_flux_x[1:,:-1,:-1]-
        rho_flux_x[:-1,:-1,:-1])- 
        dt/dy[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(rho_flux_y[:-1,1:,:-1]-
        rho_flux_y[:-1,:-1,:-1]) -
        dt/dz[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(rho_flux_z[:-1,:-1,1:]-
        rho_flux_z[:-1,:-1,:-1]) )
    dpxF[NO2:-NO2,NO2:-NO2,NO2:-NO2] = ( - 
        dt/dx[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(vx_flux_x[1:,:-1,:-1]-
        vx_flux_x[:-1,:-1,:-1]) - 
        dt/dy[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(vx_flux_y[:-1,1:,:-1]-
        vx_flux_y[:-1,:-1,:-1]) - 
        dt/dz[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(vx_flux_z[:-1,:-1,1:]-
        vx_flux_z[:-1,:-1,:-1]) )
    dpyF[NO2:-NO2,NO2:-NO2,NO2:-NO2] = (- 
        dt/dx[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(vy_flux_x[1:,:-1,:-1]-
        vy_flux_x[:-1,:-1,:-1]) - 
        dt/dy[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(vy_flux_y[:-1,1:,:-1]-
        vy_flux_y[:-1,:-1,:-1]) - 
        dt/dz[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(vy_flux_z[:-1,:-1,1:]-
        vy_flux_z[:-1,:-1,:-1]) )
    dpzF[NO2:-NO2,NO2:-NO2,NO2:-NO2] = ( - 
        dt/dx[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(vz_flux_x[1:,:-1,:-1]-
        vz_flux_x[:-1,:-1,:-1]) - 
        dt/dy[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(vz_flux_y[:-1,1:,:-1]-
        vz_flux_y[:-1,:-1,:-1]) - 
        dt/dz[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(vz_flux_z[:-1,:-1,1:]-
        vz_flux_z[:-1,:-1,:-1]) )
    deng[NO2:-NO2,NO2:-NO2,NO2:-NO2] = ( - 
        dt/dx[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(eng_flux_x[1:,:-1,:-1]-
        eng_flux_x[:-1,:-1,:-1]) - 
        dt/dy[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(eng_flux_y[:-1,1:,:-1]-
        eng_flux_y[:-1,:-1,:-1]) - 
        dt/dz[NO2:-NO2,NO2:-NO2,NO2:-NO2]*(eng_flux_z[:-1,:-1,1:]-
        eng_flux_z[:-1,:-1,:-1]) )
        

        
    return (drho,dpxF,dpyF,dpzF,deng)
        