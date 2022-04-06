# Bubble collisions

*N.B.: This code was written in 2013 and, even though it wasn't released until 2022, it hasn't had any substantial updates in the intervening years and likely won't receive updates in the future. Further [documentation can be found here](http://clwainwright.net/bubble-collisions/).*

2013/06/20

This project will simulate the collision between two bubble universes in a background de Sitter space. It is made up of essentially two parts:

- The CosmoTransitions package, modified from its last public release to include gravitation. This set of python modules will calculate instantons involving single or multiple scalar fields. It also includes functions for finding finite-temperature potentials and the nucleation of bubbles at finite temperature, but these aren't needed for the current project.
- The adaptiveGrid code, written in C with python integration.

One generally calls the adaptiveGrid code from a python prompt, probably using one of the routines in collisionRunner.py. These routines operate in a few steps:

1. Set the model (i.e., the potential). The models are defined in collisionModel.c.
2. Calculate the instanton solution.
3. Set the simulation parameters. This includes the necessary rescaling of the potential, the simulation end point, grid density (or parameters that adaptively determine it), time step size, and output density.
4. Calculate the initial field configuration from the instantons.
5. Run the simulation. It can be useful to stop the simulation at a certain point, truncate it so that any bubble walls lie outside of its bounds, and then continue to run it.

The final output is binary data. At each time, the following is written to file:

- `nx (int)`: the number of grid points at this time
- `ny (int)`: (the number of scalar fields)*2 + 2. Each field has both its value and time derivative recorded, and the two metric functions are recorded as well.
- `t (int)`: the time value (a.k.a. "N")
- `x (double[nx])`: the grid points
- `y (double[nx][ny])`: the fields and their derivatives at each grid point. First come all of the fields, then their temporal derivatives (Pi), then the metric functions alpha and a.
- `dy (double[nx][ny])`: the spatial derivatives of everything in y.

The actual simulation happens in the files adaptiveGrid.c and bubbleEvolution.c. The former contains more general code that will run a 1+1 dimensional PDE. The latter contains the code specific to the problem at hand.

To run the code, first make sure that you have python, numpy and scipy installed. To build, run

	python setup.py build --build-lib ./

from this directory. Then load up a python prompt, import bubbleCollision, and run one of the models.

Translation between the variables in potentials.nb and the parameters in collisionModel.c:

    p1=-phi01
    p2=-phiF2
    pT=phiT2out
    pj=phimatch2
    pm=phimin2
    a1=a1
    a2=a2
    M=epsilon*(3/8*pi)^{1/2}
    M4C1=(M^4)*CF1
    M4C2=(M^4)*CF2
    VT2=VT2
    m21=-mc21
    m22=mc22
