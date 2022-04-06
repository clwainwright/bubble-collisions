"""
A collection of functions for calculating fitting parameters. 
Also includes some convenience functions for plotting the fits.

Note that I never got the quadratic fit to work very well with variable
collision boundary xi0.
"""


import numpy as np
from scipy import optimize
import sys, os

from .derivsAndSmoothing import deriv14, deriv23, smooth

def stepFunction(x):
    """Returns (0, 0.5, 1) for x (<, =, >) 0.0"""
    return (np.sign(x)+1)*0.5

def powerFit(xi, xi0, A, kappa, xi_scale=0.05):
    """
    Power-law function.

    R = A Theta(xi-xi0) [(xi-xi0)/xi_scale]^kappa
    """
    xi1 = (xi-xi0)/xi_scale
    xi1 *= stepFunction(xi1)
    return A*xi1**kappa

def quadFit(xi, xi0, a, b):
    """
    Quadratic function.

    R = Theta(xi-xi0) [a (xi-x0) + b (xi-x0)^2]
    """
    xi1 = xi-xi0
    xi1 *= stepFunction(xi1)
    return a*xi1 + b*xi1**2

def calcFit(xi, R, fitFunc=powerFit, Rmin=1e-4, Rmax=1e-3, 
        start_params=(.1,2), weight_small_R=True):
    """
    Calculate the best-fit parameters for the given fitting function.

    Assumes that the input data is increasing in absolute value along the
    array, so it won't work for full-sky bubbles.

    Weights are cho

    Parameters
    ----------
    xi : array-like, one-dimensional
        The independent variable. 
    R : array-like, one-dimensional
        The dependent variable to fit. Should be same length as xi.
        Assumed to have its absolute value generally increasing along the axis.
    fitFunc : callable ``f(xi, xi0, *params)``, optional
    Rmin : float, optional
        Lower cutoff on input data R which it attempts to fit.
    Rmax : float, optional
        Upper cutoff on input data R which it attempts to fit.
    start_params : tuple, optional
        Initial guess for fitting parameters.
    weight_small_R: bool, optional
        If True, weights are proportional to 1/R (capped at 1/Rmin).
        If False, weights are uniform.

    Returns
    -------
    A tuple of best-fit parameters, including xi0.
    """
    R = np.asarray(R)
    xi = np.asarray(xi)
    j1 = np.arange(len(R))[abs(R) > Rmin][0]
    j2 = np.arange(len(R))[abs(R) > Rmax][0]
    j1 = max(0, j1 - (j2-j1))
    R = R[j1:j2]
    xi = xi[j1:j2]
    if weight_small_R:
        w = 1./np.maximum(abs(R), Rmin)
    else:
        w = np.ones_like(R)
    start_params = (xi[0],) + start_params
    popt, pcov = optimize.curve_fit(fitFunc, xi, R, start_params, 
        w, maxfev=10**4)
    return popt    
        
def runAllFits(fname, Rmin=1e-4, Rmax=1e-3, verbose=False):
    """
    Calculate fits for saved data, and saves new data to file.

    Parameters
    ----------
    fname : string
        File name for data saved with np.save(). The saved data should be a
        structured array with fields 'xi' and 'R' which correspond to 
        constant-length arrays for each data 'point' (a point here being, for
        example, the entire perturbation for a single observer).
    Rmin : float, optional
        To be passed to calcFit.
    Rmax : float, optional
        To be passed to calcFit.
    verbose : bool, optional
        If True, prints the index for each problem data point.
    """
    olddata = np.load(fname)
    data = np.zeros(olddata.shape,
        dtype=olddata.dtype.descr + [
            ('fit_quad', float, (3,)), 
            ('fit_power', float, (3,))])
    for field in olddata.dtype.fields:
        data[field] = olddata[field]
    for index, val in np.ndenumerate(data):
        import traceback
        xi, R = val['xi'], val['R']
        try: 
            val['fit_power'][:] = calcFit(xi, R, powerFit, Rmin, Rmax)
        except: 
            val['fit_power'][:] = np.nan
            if verbose:
                print("error at index %s" % index)
                traceback.print_exc()
        try: 
            val['fit_quad'][:] = calcFit(xi, R, quadFit, Rmin, Rmax)
        except: 
            val['fit_quad'][:] = np.nan
            if verbose:
                print("error at index %s" % index)
                traceback.print_exc()
    np.save(fname[:-4]+"_plus_fits.npy", data)

def plotFit(data, der=0):
    import matplotlib.pyplot as plt
    xi = data['xi']
    R = data['R']
    Rpow = powerFit(xi, *data['fit_power'])
    Rquad = quadFit(xi, *data['fit_quad'])
    print "power fit params:", data['fit_power']
    print "quad fit params:", data['fit_quad']
    if der == 0:
        plt.plot(xi, R, 'k')
        plt.plot(xi, Rpow, 'c')
        plt.plot(xi, Rquad, 'r')
    elif der == 1:
        plt.plot(xi, deriv14(R, xi), 'k')
        plt.plot(xi, deriv14(Rpow, xi), 'c')
        plt.plot(xi, deriv14(Rquad, xi), 'r')
    elif der == 2:
        plt.plot(xi, deriv23(smooth(R), xi), 'k')
        plt.plot(xi, deriv23(smooth(Rpow), xi), 'c')
        plt.plot(xi, deriv23(smooth(Rquad), xi), 'r')
    fields = data.dtype.fields
    if 'beta' in fields:
        plt.title("beta=%0.4f, xsep=%0.1f" % (data['beta'], data['xsep']))
    elif 'Delta_V' in fields and 'bar_pos' in fields:
        plt.title("bar_pos=%0.2f, Delta_V=%0.3f, xsep=%0.1f" % 
            (data['bar_pos'], data['Delta_V'], data['xsep']))
        
def plotFits(data, der=0):
    import matplotlib.pyplot as plt
    for i in xrange(data.shape[0]):
        c = plt.cm.Spectral(i*1./len(data))
        xi = data['xi'][i]
        R = data['R'][i]
        Rpow = powerFit(xi, *data['fit_power'][i])
        plt.plot(xi, R, color=c)
        plt.plot(xi, Rpow, color=c, dashes=(4,2))
        plt.axis(ymin=-.1, ymax=1.)
        plt.ylabel(r"$\mathcal{R}(\xi)$")
        plt.xlabel(r"$\xi$")
        
def plotPerturbations(data, der=0, nsmooth=5):
    """
    Plots all perturbations in an array or grid of data.

    more red = lower row, or lower in the array
    smaller dashes = lower column
    """
    import matplotlib.pyplot as plt
    assert len(data.shape) == 1 or len(data.shape) == 2 
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[1]):
            c = plt.cm.Spectral(i*1./len(data))
            if j > 0:
                dash_length = j+1
                dashes = (dash_length, 0.5*dash_length)
            else:
                dashes = ()
            xi = data['xi'][i,j]
            R = data['R'][i,j]
            if nsmooth > 0:
                R = smooth(R,nsmooth)
            if der == 0:
                y = (R)
                ylabel = r"$\mathcal{R}(\xi)$"
            elif der == 1:
                y = deriv14(R,xi)
                ylabel = r"d$\mathcal{R}(\xi)$"
            elif der == 2:
                y = deriv23((R,5),xi)
                ylabel = r"$d^2\mathcal{R}(\xi)$"
            plt.plot(xi, y, color=c, dashes=dashes)
        plt.axis(ymin=-.1, ymax=1.)
        plt.ylabel(ylabel)
        plt.xlabel(r"$\xi$")            
