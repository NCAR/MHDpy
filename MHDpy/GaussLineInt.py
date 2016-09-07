def gaussLineInt(func,xa,ya,za,xb,yb,zb):
    """
    Integrate FX(x,y,z) over the line (xa,ya,za) to (xb,yb,zb).  This
    subroutine does Gaussian integration with the first twelve Legendre
    polynomials as the basis fuctions.  Abromowitz and Stegun page 916.
    """
    
    # Positive zeros of 12th order Legendre polynomial
    a = [0.1252334085,  0.3678314989,  0.5873179542,
        0.7699026741,  0.9041172563,  0.9815606342]
    # Gaussian Integration coefficients for a 12th order polynomial
    wt = [0.2491470458,  0.2334925365,  0.2031674267,
        0.1600783285,  0.1069393259,  0.0471753363]
        
    dx = (xb-xa)/2.0
    dy = (yb-ya)/2.0
    dz = (zb-za)/2.0
    xbar = (xb+xa)/2.0
    ybar = (yb+ya)/2.0
    zbar = (zb+za)/2.0
    
    sum = 0.0
    for i in range(len(a)):
        sum = sum + wt[i] * (
                            func(xbar+a[i]*dx,ybar+a[i]*dy,zbar+a[i]*dz)+
                            func(xbar-a[i]*dx,ybar-a[i]*dy,zbar-a[i]*dz)
                            )

    return 0.5*sum    