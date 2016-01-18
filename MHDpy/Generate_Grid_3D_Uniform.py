"""
Function for generating a Uniform 3D Cartesian grid from -1 to 1 in all 
directions
"""
def Generate_Grid_3D_uniform(nx,ny,nz,NO):
    """
    Function for generating a Uniform 3D Cartesian grid from -1 to 1 in all 
    directions.  
    Requries:
        nx - number of cell corners in the x direction
        ny - number of cell corners in the y direction
        nz - number of cell corners in the z direction
        NO - order of numerical scheme to be applied
    Returns:
        (x,y,z) - arrays of 
            
    """
    import numpy as n 
    
    # Note it is important to specify these as floats to avoind int math errors  
    xmin=-1.
    xmax=1.
    ymin=-1.
    ymax = 1.
    zmin =-1.
    zmax = 1.
    
    NO2=NO/2
    
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny
    dz = (zmax-zmin)/nz
    
    x_grid_all = n.arange((xmin-NO2*dx),(xmax+NO2*dx),dx) # x loc of cell faces
    y_grid_all = n.arange((ymin-NO2*dy),(ymax+NO2*dy),dy) # x loc of cell faces
    z_grid_all = n.arange((zmin-NO2*dz),(zmax+NO2*dz),dz) # x loc of cell faces
    
    X=n.zeros((n.size(x_grid_all),n.size(y_grid_all),n.size(z_grid_all)))
    Y=n.zeros((n.size(x_grid_all),n.size(y_grid_all),n.size(z_grid_all)))
    Z=n.zeros((n.size(x_grid_all),n.size(y_grid_all),n.size(z_grid_all)))
    
    for i in range(n.size(x_grid_all)):
        for j in range(n.size(y_grid_all)):
            for k in range(n.size(z_grid_all)):
                X[i,j,k] = x_grid_all[i]
                Y[i,j,k] = y_grid_all[j]
                Z[i,j,k] = z_grid_all[k]
    
    return (X,Y,Z)