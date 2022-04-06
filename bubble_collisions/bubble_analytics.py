"""
This module provides functions for finding (semi-)analytic approximations
to the late-time field values and the comoving curvature perturbation.

These used to live in :mod:`coordAndGeos` 
(now :mod:`bubble_collisions.geodesics`), but they're not really related
to anything else in that file.
"""

import numpy as np
from bubble_collisions import simulation

def analytic_delta_phi(x_out, x_in, phi_in, eta):
    """
    Evolve a wave to late times. Assumes the wave is moving to the left.

    Parameters
    ----------
    x_out : array_like
        The values of x at which the field should be calculated
    x_in : array_like
    phi_in : array_like
        The field as a function of x at the early time.
    eta : float
        Future causal boundary of the waveform. Generally equal to 
        0.5*(pi-xsep).
    """
    y_in = phi_in
    x2 = x_in[1:]
    x1 = x_in[:-1]
    delta_y = y_in[1:] - y_in[:-1]
    dydx = delta_y / (x2-x1)
    x = x_out[..., np.newaxis]
    dxA = x-x2
    dxB = x-x1
    # delta_phi = dydx * int_dxA^dxB { phi_step(u) du }
    def clamp(x, a, b):
        s = (np.sign(x-a)+1)*0.5
        x = x*s + a*(1-s)
        s = (np.sign(b-x)+1)*0.5
        return x*s + b*(1-s)
    def halfclamp(x,a):
        s = (np.sign(x-a)+1)*0.5
        return x*s + a*(1-s)
    C = halfclamp(dxA, eta)
    D = halfclamp(dxB, eta)
    B = clamp(dxB, -eta, eta)
    A = clamp(dxA, -eta, eta)
    def step_integral(y,eta):
        y2 = y/eta
        return 0.5*y*(1-np.tan(eta)**-2) + 0.5*np.sin(y-eta)/np.sin(eta)**2
    rval = step_integral(B,eta) - step_integral(A,eta) + (D-C)
    return np.sum(rval*dydx,axis=-1)        

def analyticPerturbations(xi, fname, xsep):
    """
    Find approximate perturbation from the bubble wall profile at the collision.

    This uses a number of approximations: the metric functions are 1 (test
    fields on de Sitter space), the potential is approximately linear, and xsep
    is big enough so that ``cosh(N) = (e^N)/2`` by the time the bubbles collide.

    Parameters
    ----------
    xi : array_like
    fname : string
        File name for simulation output fields
    xsep : float
        The bubble separation

    Returns
    -------
    The perturbation as a function of the input xi.
    """
    # sin(x) = tanh(N) at the boundary
    xi = np.asanyarray(xi)
    col_N = np.arctanh(np.sin(0.5*xsep))
    data = simulation.readFromFile(fname)
    all_N = [y[0] for y in data]
    col_index = np.searchsorted(all_N, col_N)
    x = data[col_index][1]
    phi = data[col_index][2][0,:,0] # goes derivs, x index, fields index
    phidot = data[col_index][2][1,:,0] # goes derivs, x index, fields index
    # We want the wall of the collision bubble, so pick x > xsep
    # But then we really want this on the left side of the bubble, so reverse
    # everything and shift the x values
    phidot = phidot[x<0][-1] # center of observation bubble
    phi = phi[x>xsep][::-1]
    x = 2*xsep - x[x>xsep][::-1]
    x_late_time = np.arctan(np.sinh(xi))
    delta_phi = analytic_delta_phi(x_late_time, x, phi, 0.5*(np.pi-xsep))
    # phidot should really be the late-time phidot, not the phidot at the 
    # collision. Luckily we have an approximate expression for evolution of 
    # phidot, so we can transform to to late time
    phidot /= np.sinh(col_N)*(2+np.cosh(col_N))/(1+np.cosh(col_N))**2 
    return delta_phi / phidot


