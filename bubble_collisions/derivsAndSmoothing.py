"""
A set of handy functions for calculating finite-difference derivatives and
smoothing noisy data.

Note that all of these functions operate on the *final* index of the input
arrays, so that the input shape for *y* should be e.g. ``(nfields, nx)``. 
This is in contrast to the convention used elsewhere in this package
where fields data is stored in arrays of shape ``(nx, nfields)``.
"""

import numpy as np

def deriv14(y,x):
	"""
	First derivative to 4th order. The input *x* does not need to be
	evenly spaced.
	"""
	n = len(x)
	j = np.arange(5)
	j[j>4/2] -= 5
	i = np.arange(n) - j[:,np.newaxis]
	i[i<0] += 5
	i[i>=n] -= 5
	
	d1 = x[i[1]]-x[i[0]]
	d2 = x[i[2]]-x[i[0]]
	d3 = x[i[3]]-x[i[0]]
	d4 = x[i[4]]-x[i[0]]
	
	w4 = ( -d4 * (-d1*d2*d3 + d4 * (d1*d2+d2*d3+d3*d1 + d4 * (+d4-d1-d2-d3))) / (d1*d2*d3) )**-1
	w3 = ( -d3 * (-d1*d2*d4 + d3 * (d1*d2+d2*d4+d4*d1 + d3 * (-d4-d1-d2+d3))) / (d1*d2*d4) )**-1
	w2 = ( -d2 * (-d1*d4*d3 + d2 * (d1*d4+d4*d3+d3*d1 + d2 * (-d4-d1+d2-d3))) / (d1*d4*d3) )**-1
	w1 = ( -d1 * (-d4*d2*d3 + d1 * (d4*d2+d2*d3+d3*d4 + d1 * (-d4+d1-d2-d3))) / (d4*d2*d3) )**-1
	w0 = -(w1+w2+w3+w4)

	dy = w0*y[...,i[0]] + w1*y[...,i[1]] + w2*y[...,i[2]] + w3*y[...,i[3]] + w4*y[...,i[4]]
	
	return dy
	
def deriv1n(y,x,n):
	"""
	First derivative to nth order. The input *x* does not need to be
	evenly spaced.
	"""
	nx = len(x)
	j = np.arange(n+1)
	j[j>n/2] -= n+1
	i = np.arange(nx) - j[:,np.newaxis]
	i[i<0] += n+1
	i[i>=nx] -= n+1
	
	d = np.empty((n,n,nx), dtype=x.dtype)*1.0
	d[0] = x[i[1:]] - x[i[0]]
	for j in xrange(1,n):
		d[j] = np.roll(d[j-1], -1, axis=0)
	d[:,0] *= -1
	w = np.zeros((n+1,nx), dtype=y.dtype)*1.
	
	# For example, when calculating w[1], we need only use
	# w[1]: d1 = d[0,0], d2 = d[0,1], d3 = d[0,2], ..., dn = d[0,n-1]
	# and for the other weights we just increment the first index:
	# w[2]: d2 = d[1,0], d3 = d[1,1], d4 = d[1,2], ..., dn = d[1,n-2], d1 = d[1,n-1]
	# So we should be able to calculate all of them at once like this.
	s = ((2**np.arange(n-1)) & np.arange(2**(n-1))[:,np.newaxis])
	s[s>0] = (np.arange(1,n) * np.ones(2**(n-1))[:,np.newaxis])[s>0]
	w[1:] = np.sum(np.product(d[:,s],axis=2), axis=1)*d[:,0] / np.product(d[:,1:], axis=1)
	w[1:] = -w[1:]**-1
	w[0] = -np.sum(w[1:],axis=0)
	
	dy = np.sum(w*y[...,i], axis=-2)
	
	return dy
		

		
def deriv23(y,x):
	"""
	Second deriv to third order. (fourth order for uniform spacing)
	"""
	d1 = x[:-4] - x[2:-2]
	d2 = x[1:-3] - x[2:-2]
	d3 = x[3:-1] - x[2:-2]
	d4 = x[4:] - x[2:-2]
		
	w4 = 2*(d1*d2+d2*d3+d3*d1) / (d4 * (-d1*d2*d3 + d4 * (d1*d2+d2*d3+d3*d1 + d4 * (+d4-d1-d2-d3) ) ) ) 
	w3 = 2*(d1*d2+d2*d4+d4*d1) / (d3 * (-d1*d2*d4 + d3 * (d1*d2+d2*d4+d4*d1 + d3 * (-d4-d1-d2+d3) ) ) ) 
	w2 = 2*(d1*d4+d4*d3+d3*d1) / (d2 * (-d1*d4*d3 + d2 * (d1*d4+d4*d3+d3*d1 + d2 * (-d4-d1+d2-d3) ) ) ) 
	w1 = 2*(d4*d2+d2*d3+d3*d4) / (d1 * (-d4*d2*d3 + d1 * (d4*d2+d2*d3+d3*d4 + d1 * (-d4+d1-d2-d3) ) ) ) 
	w0 = -(w1+w2+w3+w4)
	
	dy = np.zeros_like(y)
	dy[...,2:-2] = w1*y[...,:-4] + w2*y[...,1:-3] + w0*y[...,2:-2] + w3*y[...,3:-1] + w4*y[...,4:]
	return dy
	
	
def smooth(x, n=2):
	"""
	Simple smoothing function; averages over 2n nearest neighbors.
	Useful for cleaning up noise from derivatives.
	"""
	assert len(x) > 2*n+1
	i = np.arange(len(x))
	i[:n] = n
	i[-n:] = i[-n-1]
	i = i[np.newaxis,:] + np.arange(-n,n+1)[:,np.newaxis]
	return np.sum(x[i], axis=0) / (2*n+1.0)

def circularSmooth(x, n=2):
	"""
	Same as :func:`smooth`, but wraps smoothing at edges.
	"""
	assert len(x) > 2*n+1
	i = np.arange(len(x))
	i = i[np.newaxis,:] + np.arange(-n,n+1)[:,np.newaxis]
	i = i % len(x)
	return np.sum(x[i], axis=0) / (2*n+1.0)
