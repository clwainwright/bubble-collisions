"""
This module provides functions for calculating geodesics on top of a
previously run simulation, as well as functions for calculating the 
comoving curvature perturbation from the geodesic information.
"""

import sys
from scipy import integrate, interpolate
import numpy as np
from numpy import sqrt, sinh, cosh, tanh, arctanh, arcsinh
from time import time
from gc import collect as collect_garbage

from .derivsAndSmoothing import deriv14
from . import simulation
        
    
def _georhs_fromData(tau, Y, chris_data, Ndata,cubic=True):
    N, dN, x, dx = Y
    G, dGdN, dGdx, dGdNx = simulation.valsOnGrid(N,x, chris_data, Ndata, cubic)
    
    return [dN, -G[0]*dN*dN - 2*G[1]*dx*dN - G[2]*dx*dx, 
        dx, -G[3]*dN*dN - 2*G[4]*dx*dN - G[5]*dx*dx]
    
def _geojac_fromData(tau, Y, chris_data, Ndata,cubic=True):
    N, dN, x, dx = Y
    G, dGdN, dGdx, dGdNx = simulation.valsOnGrid(N,x, chris_data, Ndata, cubic)
    
    jac = [[0,1,0,0], 
           [-dGdN[0]*dN*dN - 2*dGdN[1]*dx*dN - dGdN[2]*dx*dx,
            -2*G[0]*dN - 2*G[1]*dx,
            -dGdx[0]*dN*dN - 2*dGdx[1]*dx*dN - dGdx[2]*dx*dx,
            -2*G[1]*dN - 2*G[2]*dx],
           [0,0,0,1],
           [-dGdN[3]*dN*dN - 2*dGdN[4]*dx*dN - dGdN[5]*dx*dx,
            -2*G[3]*dN - 2*G[4]*dx,
            -dGdx[3]*dN*dN - 2*dGdx[4]*dx*dN - dGdx[5]*dx*dx,
            -2*G[4]*dN - 2*G[5]*dx]]
    return jac
    

def runSingleGeo(xi, tau, data, Ndata=None, x0=0.0, cubic=True, 
        integratorName='vode', **intargs):
    """
    Integrates a geodesic for a given set of simulation data.

    Parameters
    ----------
    xi : float
        The initial geodesic trajectory
    tau : array_like
        The proper times along the geodesic at which the integration should
        be saved.
    data : list
        Christoffel data output by :func:`simulation.readFromFile`.
    Ndata : array_like, optional
        If present, equal to ``array([d[0] for d in data])``.
        This is a slight optimization when running many geodesics.
    x0 : float, optional
        Initial starting placement for the geodesic.
    cubic : bool, optional
        If True (default), use cubic interpolation along simulation grid.
        If False, use linear interpolation.
    integratorName : string, optional
        Integrator to use with :class:`scipy.integrate.ode`.
    intargs
        Extra arguments to pass to the integrator.

    Returns
    -------
    :
        An array of shape ``(len(tau), 4)``, with each element corresponding to 
        ``[ N(tau), dN/dtau, x(tau), dx/dtau ]``.
    """
    if Ndata is None: 
        Ndata = np.array([d[0] for d in data])
    Y0 = [0.0, cosh(xi), x0, sinh(xi)]
    Y = np.zeros((len(tau), 4))
    r = integrate.ode(_georhs_fromData, _geojac_fromData)
    if (intargs or integratorName is not 'vode'):
        r.set_integrator(integratorName, **intargs)
    else:
        r.set_integrator(
            'vode', method='adams', with_jacobian=True, 
            nsteps=10**5, atol=1e-9, rtol=1e-9)
    r.set_initial_value(Y0, 0.0)
    r.set_f_params(data,Ndata,cubic).set_jac_params(data,Ndata,cubic)
    if tau[0] == 0:
        Y[0] = Y0
        istart = 1
    else:
        istart = 0
    for i in xrange(istart,len(tau)):
        Y[i] = r.integrate(tau[i])
        if r.successful() is not True: 
            print "Fail at tau =", tau[i]
            break
    return Y

def findGeodesics(xi, tau, chris_fname, fields_fname, xi0=0.0, cubic=True, 
        integratorName='vode', min_simulated_N=0.0, **intargs):
    """
    Integrate a grid of geodesics, returning the coordinates, fields, metric
    functions, and Christoffel symbols along the grid.

    Parameters
    ----------
    xi : list
        The different geodesic trajectories which should be integrated.
    tau : list
        Proper times at which to save the geodesics
    chris_fname : string
        File name for the christoffel symbols.
    fields_fname : string
        File name for the fields and metrics functions.
    cubic : bool, optional
        If True (default), use cubic interpolation along simulation grid.
        If False, use linear interpolation.
    integratorName : string, optional
        Integrator to use with :class:`scipy.integrate.ode`.
    min_simulated_N : float, optional
        Raise an error if the simulation didn't reach at least this value.
        Has no other affect.
    intargs
        Extra arguments to pass to the integrator.

    Returns
    -------
    :
        A dictionary containing the coordinates, fields, metric functions, and 
        Christoffel symbols. The keys *xi* and *tau* are the same as the 
        input values. Each other key corresponds to a multi-dimensional array
        with the first index corresponding to *tau* and the second index
        corresponding to *xi*.
    """
    print "\nLoading file '%s'..." % chris_fname
    data = simulation.readFromFile(chris_fname)
    Ndata = np.array([d[0] for d in data])
    if min_simulated_N > Ndata[-1]:
        raise RuntimeError(
            "File '%s' does not simulate up to 'min_simulated_N.\n'"
            "Ndata[-1] = %s; min_simulated_N = %s" %
            (chris_fname, Ndata[-1], min_simulated_N))
    Y = np.empty((len(tau), len(xi), 4)) # The geodesics
    print "Running geodesics... "
    di_print = max(len(xi) // 100, 1)
    for i in xrange(len(xi)):
        if i % di_print == 0:
            sys.stdout.write("%i "%i)
            sys.stdout.flush()
        Y[:,i] = runSingleGeo(xi[i], tau, data, Ndata, xi0, 
            cubic, integratorName, **intargs)
    print "\nLoading Christoffel symbols..."
    sys.stdout.flush()
    N,x = Y[:,:,0], Y[:,:,2]
    G = simulation.valsOnGrid(N,x,data,Ndata)
    # make sure to free up the memory before loading the next array:
    del data, Ndata; collect_garbage() 
    print "Loading file '%s'..." % fields_fname
    sys.stdout.flush()
    data = simulation.readFromFile(fields_fname)
    Ndata = np.array([d[0] for d in data])
    print "Loading field values..."
    sys.stdout.flush()
    F = simulation.valsOnGrid(N,x,data,Ndata)
    # make sure to free up the memory before loading the next array
    del data, Ndata; collect_garbage() 
    print "Assembling output dictionary..."
    sys.stdout.flush()
    rdict = {}
    ny = F.shape[-1]
    nfields = (ny-2)//2
    rdict.update(
        xi =     xi,  # xi and tau define the coordinates of the geodesics
        tau =    tau,
        N =      Y[:,:,0], 
        dNdt =   Y[:,:,1], 
        x =      Y[:,:,2],
        dxdt =   Y[:,:,3], 
        # For the following, changing the third index gets us 
        # the derivatives d/dN, d/dx and d2/dNdx
        alpha =  F[:,:,0,-2],
        a =      F[:,:,0,-1],
        phi =    F[:,:,0, 0:nfields],
        Pi =     F[:,:,0, nfields:2*nfields],
        # just load up all the Christoffels in one. Ignore derivatives.
        Gamma =  G[:,:,0,:] 
        )
    rdict["dNdxi"] = deriv14(rdict["N"], xi)
    rdict["dxdxi"] = deriv14(rdict["x"], xi)
    rdict["dphidt"] = deriv14(rdict["phi"].T, tau).T
    print "Done."
    return rdict
    
    
def interpGeos(xiP, tauP, data, key):
    """
    Does a simple interpolation along the xi-tau grid.
    Outputs the value on the grid.
    """
    nxi = len(data["xi"])
    assert(nxi >= 2)
    ixi = np.searchsorted(data["xi"], xiP) - 1
    ixi[ixi > nxi-2] = nxi-2
    ixi[ixi < 0] = 0
    
    ntau = len(data["tau"])
    assert(ntau >= 2)
    itau = np.searchsorted(data["tau"], tauP) - 1
    itau[itau > ntau-2] = ntau-2
    itau[itau < 0] = 0
    
    dxi = (xiP-data["xi"][ixi])/(data["xi"][ixi+1]-data["xi"][ixi])
    dtau = (tauP-data["tau"][itau])/(data["tau"][itau+1]-data["tau"][itau])
    
    v1 = data[key][itau,ixi]
    v2 = data[key][itau,ixi+1]
    v3 = data[key][itau+1,ixi]
    v4 = data[key][itau+1,ixi+1]
    
    if len(dxi.shape) < len(v1.shape):
        dxi = dxi[...,np.newaxis]
        dtau = dtau[...,np.newaxis]
    
    return v1*(1-dxi)*(1-dtau) + v2*dxi*(1-dtau) + v3*(1-dxi)*dtau + v4*dxi*dtau
    
def observerMetric(tau, X,Y, eta, geoData):
    """
    This function returns a dictionary containing the metric at time slice tau
    and at observer Cartesian coordinates X, Y (with Z=0) in the synchronous
    gauge, and at an observer position given by xi0 = eta 
    (eta represents a boost of the simulation coordinates, with 
    (cosh(eta) = gamma)).
    It also returns the field and its time (tau) derivative at the specified 
    points.
    
    tau, X, Y, and eta should all be numpy arrays that are broadcastable to the 
    same shape. 

    geoData should be the dictionary of arrays output by :func:`findGeodesics`.
    """

    # First, get the primed (collision frame) coords in terms of the unprimed (observation) coords
    X = np.array(X)
    Y = np.array(Y)

    R24 = 0.25*(X*X+Y*Y)
    g = cosh(eta) # gamma
    gb = sinh(eta) # gamma*beta
    D = (1-R24) + g*(1+R24) + gb*X

    XP = 2 * (g*X + gb*(1+R24) ) / D
    YP = 2 * Y / D
    
    dDdX = 0.5*(g-1)*X + gb
    dDdY = 0.5*(g-1)*Y
    dXPdX = (2*g + gb*X)/D - 2*(g*X + gb*(1+R24))*dDdX/D**2
    dXPdY = (gb*Y)/D - 2*(g*X + gb*(1+R24))*dDdY/D**2
    dYPdX = -2*Y*dDdX/D**2
    dYPdY = 2/D - 2*Y*dDdY/D**2
    
    # Then get the geodesic coordinates in terms of the primed coordinates
    rho = arctanh( YP / (1 + 0.25*(XP*XP+YP*YP) ) )
    xi = arcsinh( XP / (1 - 0.25*(XP*XP+YP*YP) ) )

    cosh_rho, sinh_rho, cosh_xi, sinh_xi = cosh(rho), sinh(rho), cosh(xi), sinh(xi)

    RP24 = (XP*XP+YP*YP)*0.25
    dxidXP = ( 1/(1-RP24) + 0.5*XP*XP/(1-RP24)**2 ) / cosh_xi
    dxidYP = ( 0.5*XP*YP/(1-RP24)**2 ) / cosh_xi
    drhodYP = ( 1/(1+RP24) - 0.5*YP*YP/(1+RP24)**2 ) * cosh_rho**2
    drhodXP = (-0.5*XP*YP/(1+RP24)**2 ) * cosh_rho**2
    

    cosh_rho, sinh_rho, cosh_xi, sinh_xi = cosh(rho), sinh(rho), cosh(xi), sinh(xi)

    # Finally, get the simulation data from the geodesic coords.
    
    N = interpGeos(xi, tau, geoData, "N")
    x = interpGeos(xi, tau, geoData, "x")
    alpha = interpGeos(xi, tau, geoData, "alpha")
    a = interpGeos(xi, tau, geoData, "a")
    dNdxi = interpGeos(xi, tau, geoData, "dNdxi")
    dNdtau = interpGeos(xi, tau, geoData, "dNdt")
    dxdxi = interpGeos(xi, tau, geoData, "dxdxi")
    dxdtau = interpGeos(xi, tau, geoData, "dxdt")
    phi = interpGeos(xi, tau, geoData, "phi")
    dphidtau = interpGeos(xi, tau, geoData, "dphidt")
    dchidrho = 1
    
    dNdXP = dNdxi*dxidXP
    dNdYP = dNdxi*dxidYP
    dxdXP = dxdxi*dxidXP
    dxdYP = dxdxi*dxidYP
    
    dNdX = dNdXP*dXPdX + dNdYP*dYPdX
    dNdY = dNdXP*dXPdY + dNdYP*dYPdY
    dxdX = dxdXP*dXPdX + dxdYP*dYPdX
    dxdY = dxdXP*dXPdY + dxdYP*dYPdY
    dchidX = dchidrho * (drhodXP*dXPdX + drhodYP*dYPdX)
    dchidY = dchidrho * (drhodXP*dXPdY + drhodYP*dYPdY)
    
    # Now we're ready to calculate the metric!
    gchichi = sinh(N)**2
    gNN = -alpha**2
    gxx = (a*cosh(N))**2
    
    gXX = gNN*dNdX**2 + gxx*dxdX**2 + gchichi*dchidX**2
    gYY = gNN*dNdY**2 + gxx*dxdY**2 + gchichi*dchidY**2
    gXY = gNN*dNdX*dNdY + gxx*dxdX*dxdY + gchichi*dchidX*dchidY
    gtt = gNN*dNdtau**2 + gxx*dxdtau**2 # should be -1
    gtX = gNN*dNdtau*dNdX + gxx*dxdtau*dxdX # should be 0
    gtY = gNN*dNdtau*dNdY + gxx*dxdtau*dxdY # should be 0
    # have to do gZZ carefully due to coordinate singularity
    gZZ = ( sinh(N) * (1+cosh_xi*cosh(rho)) / (cosh(xi)*D) )**2
    
    outvars = gXX, gYY, gZZ, gXY, gtt, gtX, gtY, phi, dphidtau, X, Y
    outkeys = "gXX, gYY, gZZ, gXY, gtt, gtX, gtY, phi, phidot, X, Y".split(", ")
    outdict = {}
    for var, key in zip(outvars, outkeys):
        outdict[key] = var
    return outdict
    
def scaleFactor(geos0, tau):
    """
    Calculate the scale factor and Hubble parameter at the given proper time
    *tau* along the geodesic grid *geos0* output by :func:`findGeodesics` for
    an unperturbed (no collision) simulation.
    """
    xi = np.array([0.0])
    tau = np.array(tau)
    N = interpGeos(xi, tau.ravel(), geos0, "N").reshape(tau.shape)
    dNdt = interpGeos(xi, tau.ravel(), geos0, "dNdt").reshape(tau.shape)
    a = sinh(N)
    H = dNdt / tanh(N)
    return a, H
    
def perturbationsFromMetric(g0, g1, a, H, divideOutCurvature=True):
    """
    Calculate the comoving cuvature perturbation R from metric information
    in synchronous gauge Cartesian coordinates.

    Parameters
    ----------
    g0 : dictionary
        The unperturbed metric given by :func:`observerMetric`.
    g1 : dictionary
        The perturbed metric given by :func:`observerMetric`.
    a : float
        The scale factor.
    H : float
        The Hubble parameter.
    divideOutCurvature : bool, optional
        If True (default), the metric is divided by the overall negative
        curvature that's present in the unperturbed bubble.

    Returns
    -------
    :
        [D, E, H*deltaphi/phidot, R]
    """
    s = g0["gXX"].shape
    G0 = np.zeros((3,3)+s)
    G0[0,0], G0[1,1], G0[2,2], G0[0,1], G0[1,0] = \
        g0["gXX"], g0["gYY"], g0["gZZ"], g0["gXY"], g0["gXY"]
    G1 = np.zeros((3,3)+s)
    G1[0,0], G1[1,1], G1[2,2], G1[0,1], G1[1,0] = \
        g1["gXX"], g1["gYY"], g1["gZZ"], g1["gXY"], g1["gXY"]
    dG = (G1-G0)/np.array(a)[None,None]**2
    if (divideOutCurvature):
        dG *= (1-0.25*(g0["X"]**2+g0["Y"]**2))**2
    
    D = -(dG[0,0]+dG[1,1]+dG[2,2])/6.0
    E = 1*dG
    for i in xrange(3):
        E[i,i] += 2*D
    Exx = E[0,0]
    
    deltaphi = (g1["phi"] - g0["phi"])[...,0]
    phidot = (g0["phidot"])[...,0]
    
    R = D + 0.25*Exx + H*deltaphi/phidot
    
    return D, E, H*deltaphi/phidot, R

