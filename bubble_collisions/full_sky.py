"""
This module provides a way to calculate the comoving curvature perturbation
for full-sky bubbles. Actually, it works equally well for less than
full-sky bubbles.

The basic idea is to work directly in the comoving gauge by first defining
a comoving slice, which is equivalent to a surface of constant field.
Of course, this will only work when there is only one field dimension,
otherwise there's not guaranteed to be any constant field surface.
The basic strategy could easily be adapted to a constant energy or
density surface for the case of multiple fields.
The metric is then calculated along the slice, and, through a judicious choice
of coordinates, made to match the anisotropic hyperbolic coordinates plus
a perturbation. The metric perturbation determines the Ricci three-scalar,
which can then be integrated to calculate the co-moving curvature perturbation.
"""

import sys
from scipy import optimize, integrate, interpolate
import numpy as np
from bubble_collisions import simulation
from bubble_collisions.derivsAndSmoothing import deriv14, deriv23


class FullSkyPerturbation(object):
    """
    An object-oriented wrapper for the full-sky calculation.

    There are three steps to the full-sky calculation:

    1. Calculate the (constant field) spatial slice.
    2. Determine the proper distance along the slice.
    3. Convert coordinates to the observer frame and calculate the
       perturbation.

    Because items 1 and 2 are the same for all observers, it makes sense to
    wrap the whole computation in a single object-oriented class. That way,
    the results from parts 1 and 2 can easily be reused (and examined) for
    different observers.

    If one wishes to use something other than a constant field value to define
    the spatial slice, one could just create a subclass and override
    :meth:`calcSpatialSlice`.

    Parameters
    ----------
    data : string or data tuple
        If a string, it should be the name of a simulation (fields) output
        file. If a tuple, it should be the data returned from 
        :func:`bubble_collisions.simulation.valsOnGrid`.
    phi0 : float, optional
        The field value defining the constant phi surface.
    nx : int, optional
        Approximate number of points to use on the spatial slice.
    """
    def __init__(self, data, phi0=1.0, nx=2000):
        self.phi0 = phi0
        self.setup(data, nx)

    def setup(self, data, nx):
        """
        Sets up the calculation (steps 1 and 2 above).
        Calculates the spatial slice and integrates distance along the slice.

        This is kept as a separate function from ``__init__`` so that it
        doesn't need to be rewritten for potential subclasses.
        """
        # Get the data --
        try:
            data = simulation.readFromFile(data)
        except TypeError:
            pass
        Ndata = np.array([d[0] for d in data])
        self.data, self.Ndata = data, Ndata
        # Calculate N(x) --
        xmin = self.data[-1][1][0]
        xmax = self.data[-1][1][-1]
        x = np.linspace(xmin,xmax,nx)
        N = np.empty_like(x)
        for i in xrange(len(N)):
            if i % 100 == 0:
                sys.stdout.write("%i "%i)
                sys.stdout.flush()
            N[i] = self.calcSpatialSlice(x[i])
        sys.stdout.write('\n')
        j = np.isfinite(N)
        self.N, self.x = N[j], x[j]
        self.N_spline = interpolate.UnivariateSpline(self.x, self.N, s=0)
        # Calculate distance along the slice --
        y = simulation.valsOnGrid(self.N, self.x, self.data, self.Ndata)
        alpha = y[:,0,-2]
        beta = y[:,0,-1] # usually called the metric function 'a'
      #  dNdx = self.N_spline(self.x, 1)
        dNdx = deriv14(self.N,self.x) 
            # I trust my own derivatives better. Spline is non-local.
        dudx = np.sqrt((beta*np.cosh(self.N))**2 - (alpha*dNdx)**2)
        self.dudx_spline = interpolate.UnivariateSpline(self.x, dudx, s=0)
        self.u_spline = self.dudx_spline.antiderivative()        

    def calcSpatialSlice(self, x0):
        """
        Calculate the time N along the spatial slice at x=x0.

        Can be overridden by subclasses for different slicings.
        """
        def deltaphiForN(N):
            phi = simulation.valsOnGrid(N, x0, self.data, self.Ndata)[0,0]
            return phi - self.phi0
        if np.sign(deltaphiForN(0)) == np.sign(deltaphiForN(self.Ndata[-1])):
            return np.nan
        return optimize.brentq(deltaphiForN, 0.0, self.Ndata[-1])   

    def xi0ForObserver(self, x0):
        """
        Find xi0 given observer position x0 (can be an array).
        """
        N0 = self.N_spline(x0)
        u0 = self.u_spline(x0)
        dNdu0 = self.N_spline(x0,1) / self.dudx_spline(x0)
        xi0 = np.arcsinh(np.cosh(N0)*dNdu0)
        return xi0      

    def _ricci(self, xi, B, dB, d2B):
        """
        Ricci scalar, ignoring the overall -6 constant.
        """
        c = np.cosh(xi)
        s = np.sinh(xi)
        return ( -16*B**4*s**2 - 16*B**3*s*c*dB + 8*B**3 - 8*B**2*s*c*dB 
            - 4*(B*c*dB)**2 + 16*(B*c)**2 - 4*B*B + 20*B*dB*s*c
            - 4*B*(c*dB)**2 + 4*B*c*c*d2B - 2*B + 6*s*c*dB
            + (c*dB)**2 + 2*c*c*d2B )  *  2/c**2        

    def d2RForObserver(self, x0):
        """
        Find the second derivative of the comoving curvature perturbation for
        an observer at position x0.

        Note that this does not assume anything about large N or approximate
        de Sitter space.
        """  
        N0 = self.N_spline(x0)
        u0 = self.u_spline(x0)
        dNdu0 = self.N_spline(x0,1) / self.dudx_spline(x0)
        d2Ndu0 = self.N_spline(x0,2) / self.dudx_spline(x0)**2
        d2Ndu0 -= (self.N_spline(x0,1) * self.dudx_spline(x0,1) 
            / self.dudx_spline(x0)**3)
        coshxi_sq = 1 + (np.cosh(N0)*dNdu0)**2
        d2B = (1 + dNdu0**2 - np.sinh(N0)*np.cosh(N0)*d2Ndu0) / coshxi_sq
        return d2B

    def perturbationCenteredAt(self, x0, delta_xi=1e-3, full_output=False):
        """
        Calculates the comoving curvature perturbation as a function of xi
        for an observer centered at x=x0. Note that a single spatial location
        along the slice will map to different values of xi (in the perturbed
        case) for different observers.

        Parameters
        ----------
        x0 : float
            Location of the observer along the slice in simulation coordinates.
        delta_xi : float, optional
            Spacing between sequential output points.
        full_output : bool
            If True, output xi0.

        Returns
        -------
        xi0 : float, optional
            The position of the observer in anisotropic hyperbolic coords.
        xi : array
            The spatial anisotropic hyperbolic coordinate along the slice.
        R : array
            The comoving curvature perturbation
        """
        N0 = self.N_spline(x0)
        u0 = self.u_spline(x0)
        dNdu0 = self.N_spline(x0,1) / self.dudx_spline(x0)
        xi0 = np.arcsinh(np.cosh(N0)*dNdu0)
        a0 = np.sinh(N0) / np.cosh(xi0) # scale factor
        xi = (self.u_spline(self.x)-u0)/a0 + xi0
        B = 0.5 * (1 - (np.sinh(self.N)/(a0*np.cosh(xi)))**2)
        dB = deriv14(B,xi)
        d2B = deriv23(B,xi)
        Ricci = self._ricci(xi, B, dB, d2B)
        Ricci_interp = interpolate.UnivariateSpline(xi, Ricci, s=0)
        B_interp = interpolate.UnivariateSpline(xi, B, s=0)
        def dPsidxi(Psi, xi):
            psi, dpsi = Psi
            d2psi = 0.25*Ricci_interp(xi) - 2*(
                (np.tanh(xi)-B_interp(xi,1))/(1-2*B_interp(xi)))*dpsi
            return [dpsi, d2psi]
        xi_low = np.arange(xi0, xi[0], -delta_xi)
        xi_high = np.arange(xi0, xi[-1], delta_xi)
        psi_low = integrate.odeint(dPsidxi, [0.,0.], xi_low)[:,0]
        psi_high = integrate.odeint(dPsidxi, [0.,0.], xi_high)[:,0]
        xi_out = np.append(xi_low[::-1], xi_high[1:])
        psi_out = np.append(psi_low[::-1], psi_high[1:])
        if full_output:
            return xi0, xi_out, psi_out
        return xi_out, psi_out

