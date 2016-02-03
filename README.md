# MHDpy
##            3-D ideal MHD Solver on a Stretched Cartesian Grid
Numerical methods are described in Lyon et al., [2004]. Most of the LFM schemes are used in this 3-D MHD code except:
   1. No curvilinear grid
   2. No Ring-average needed
   3. No Boris-correction for high Alfven speed (very low plasma beta)
   4. No Background B field subtraction (high order Gaussin integral)

##  MAIN FEATURES:

  1. Finite Volume (actually finite difference in Cartesian)
  2. Semi-conservative - use plasma energy equation
  3. 2nd-8th order reconstruction with TVD/WENO/PDM, default PDM
  4. 2nd order Adams-Bashforth time-stepping scheme
  5. Operator-splitting for the Lorentz force terms
  6. Gas-kinetic flux functions for fluid and magnetic stresses
  7. High order Constraint transport (Yee-Grd) to conserve Div B = 0
  8. Resistive MHD - to be implemented, relatively simple
  9. Hall MHD - to be implemented, can use the getEk function

##   ALGORITHM
The basic steps in the algorith are
  1. Adams-Bashforth predictor step (half time step update)
  2. Calculate E fields at cell edges
  3. Calculate fluid flux/stresses and magnetic stresses
    +  x-direction 
    +  y-direction
    + z-direction
  4. Update the hydrodynamic equation without magnetic stress
  5. Apply magnetic stress to the updated momentum
  6. Evolve B field 
  7. Apply boundary conditions

NOTE: The cell length should be included in the reconstruction process in order to obtain formal accuracy, which is probably more important in the curvilinear version of the solver. For uniform cartesian, doesn't matter.

NOTE: This branch is for testing the changes to python type indexing

