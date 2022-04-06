#include <Python.h>

#define NO_IMPORT_ARRAY

#include <arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "gridInterpolation.h"

#define PI 3.141592653589793
#define CLAMP(i, mini, maxi) i < mini ? mini : (i > maxi ? maxi : i)

#define VERBOSE 0
#if VERBOSE
	#define BLURT() printf("%s(%d) -- %s()\n", __FILE__, __LINE__, __func__)
	#define LOGMSG(...) printf("%s(%d): ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n")
#else
	#define BLURT() ;
	#define LOGMSG(...) ;
#endif

int indexInSorted(double x0 /*target*/, double *x /*list to search*/, int nx) {
	int imin = 0, imax = nx;
	while (imin < imax) {
		int i = (imin+imax)/2;
		if (x0 > x[i]) 
			imin = i+1;
		else if (x0 < x[i])
			imax = i;
		else // if (x0 == x[i])
			return i+1;
	}
	return imin;
}

void cubicInterp(double x0, 
				 double x1, double *y1, double *dy1,
				 double x2, double *y2, double *dy2,
				 double *yout, double *dyout, int ny) {
	if (yout == NULL && dyout == NULL)
		return;
	int i;
	double Dx = x2-x1;
	double t = (x0-x1)/Dx;
	double t2 = t*t;
	double t3 = t*t2;
	for (i=0; i<ny; i++) {
		if (yout)
			yout[i] = (2*t3-3*t2+1)*y1[i] + (-2*t3+3*t2)*y2[i] + ((t3-2*t2+t)*dy1[i] + (t3-t2)*dy2[i])*Dx;
		if (dyout)
			dyout[i] = (6*(t2-t)*y1[i] + 6*(t-t2)*y2[i])/Dx + ((3*t2-4*t+1)*dy1[i] + (3*t2-2*t)*dy2[i]);
	}
}	

void bicubicInterp(double t0, double x0,
				   double tA, double x1, double **y1, double x2, double **y2,
				   double tB, double x3, double **y3, double x4, double **y4,
				   double **yout, int ny) {
	// Each **y should be a list of arrays: y, dy/dt, dy/dx, d2y/dxdt
	double y_A[ny], dydx_A[ny];
	double y_B[ny], dydx_B[ny];
	double dydt_A[ny], dydxt_A[ny];
	double dydt_B[ny], dydxt_B[ny];
	// First interpolate in the x direction
	cubicInterp(x0, x1, y1[0], y1[2], x2, y2[0], y2[2], y_A, dydx_A, ny);
	cubicInterp(x0, x3, y3[0], y3[2], x4, y4[0], y4[2], y_B, dydx_B, ny);
	cubicInterp(x0, x1, y1[1], y1[3], x2, y2[1], y2[3], dydt_A, dydxt_A, ny);
	cubicInterp(x0, x3, y3[1], y3[3], x4, y4[1], y4[3], dydt_B, dydxt_B, ny);
	// Then go in the t direction
	cubicInterp(t0, tA, y_A, dydt_A, tB, y_B, dydt_B, yout[0], yout[1], ny);
	cubicInterp(t0, tA, dydx_A, dydxt_A, tB, dydx_B, dydxt_B, yout[2], yout[3], ny);
}

void linearInterp(double x0, 
				 double x1, double *y1,
				 double x2, double *y2,
				 double *yout, int ny) {
	int i;
	double Dx = x2-x1;
	double t = (x0-x1)/Dx;
	for (i=0; i<ny; i++) {
		if (yout)
			yout[i] = (1-t)*y1[i] + t*y2[i];
	}
}	

void bilinearInterp(double t0, double x0,
					double tA, double x1, double **y1, double x2, double **y2,
					double tB, double x3, double **y3, double x4, double **y4,
					double **yout, int ny) {
	// Each **y should be a list of arrays: y, dy/dt, dy/dx, d2y/dxdt
	int i;
	for (i=0; i<4; i++) {
		double yA[ny], yB[ny];
		linearInterp(x0,x1,y1[i],x2,y2[i],yA,ny); // along x direction
		linearInterp(x0,x3,y3[i],x4,y4[i],yB,ny); // along x direction
		linearInterp(t0,tA,yA,tB,yB,yout[i],ny); // along t direction
	/*	if (i==0) {
			LOGMSG("x0,t0,y0: (%0.2f,%0.2f,%0.2f)", x0, t0, yout[0][0]);
			LOGMSG("x1,t1,y1: (%0.2f,%0.2f,%0.2f)", x1, tA, y1[0][0]);
			LOGMSG("x2,t2,y2: (%0.2f,%0.2f,%0.2f)", x2, tA, y2[0][0]);
			LOGMSG("x3,t3,y3: (%0.2f,%0.2f,%0.2f)", x3, tB, y3[0][0]);
			LOGMSG("x4,t4,y4: (%0.2f,%0.2f,%0.2f)", x4, tB, y4[0][0]);
			LOGMSG("yA,yB: %0.2f, %0.2f\n", yA[0], yB[0]);
		}*/
	}
}



void valsFromData(double t0, double x0, PyObject *fulldata, double *tdata, int nt, double **yout, int ny, int cubic) {
	int i,j,k,nx;
	double *y1[4], *y2[4], *y3[4], *y4[4];
	double x1, x2, x3, x4;
	double *ydata, *xdata;
	PyObject *data, *item, *arr, *arr_[4];

	i = indexInSorted(t0, tdata, nt); // the t index we want is between i-1 and i
	i = CLAMP(i, 1, nt-1); 

	// Start with i-1
	data = PySequence_GetItem(fulldata, i-1);
	LOGMSG("i: %i", i-1);
	// Get the x array
	item = PySequence_GetItem(data,1);
	arr = arr_[0] = PyArray_FROM_OTF(item, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	nx = PyArray_Size(arr);
	xdata = (double *)PyArray_DATA((PyArrayObject *)arr);
	j = indexInSorted(x0, xdata, nx);
	j = CLAMP(j,1,nx-1);
	LOGMSG("j: %i", j-1);
	x1 = xdata[j-1];
	x2 = xdata[j];
	Py_DECREF(item);
	// Get the y array. Shape (4, nx, ny)
	item = PySequence_GetItem(data,2);
	arr = arr_[1] = PyArray_FROM_OTF(item, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	ydata = (double *)PyArray_DATA((PyArrayObject *)arr);
	for (k=0; k<4; k++) {
		y1[k] = &(ydata[k*nx*ny + (j-1)*ny]);
		y2[k] = &(ydata[k*nx*ny + j*ny]);
	}
	Py_DECREF(item);
	Py_DECREF(data);

	// And do the same thing for i, store in x3,x4,y3,y4
	data = PySequence_GetItem(fulldata, i);
	// Get the x array
	item = PySequence_GetItem(data,1);
	arr = arr_[2] = PyArray_FROM_OTF(item, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	nx = PyArray_Size(arr);
	xdata = (double *)PyArray_DATA((PyArrayObject *)arr);
	j = indexInSorted(x0, xdata, nx);
	j = CLAMP(j,1,nx-1);
	LOGMSG("j: %i", j-1);
	x3 = xdata[j-1];
	x4 = xdata[j];
	Py_DECREF(item);
	// Get the y array. Shape (4, nx, ny)
	item = PySequence_GetItem(data,2);
	arr = arr_[3] = PyArray_FROM_OTF(item, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	ydata = (double *)PyArray_DATA((PyArrayObject *)arr);
	for (k=0; k<4; k++) {
		y3[k] = &(ydata[k*nx*ny + (j-1)*ny]);
		y4[k] = &(ydata[k*nx*ny + j*ny]);
	}
	Py_DECREF(item);
	Py_DECREF(data);
	
	// And now interpolate.
	if (cubic)
		bicubicInterp(t0, x0, tdata[i-1], x1, y1, x2, y2, tdata[i], x3, y3, x4, y4, yout, ny);
	else
		bilinearInterp(t0, x0, tdata[i-1], x1, y1, x2, y2, tdata[i], x3, y3, x4, y4, yout, ny);
	LOGMSG("x0,t0,y0: (%0.2f,%0.2f,%0.2f)", x0, t0, yout[0][0]);
	LOGMSG("x1,t1,y1: (%0.2f,%0.2f,%0.2f)", x1, tdata[i-1], y1[0][0]);
	LOGMSG("x2,t2,y2: (%0.2f,%0.2f,%0.2f)", x2, tdata[i-1], y2[0][0]);
	LOGMSG("x3,t3,y3: (%0.2f,%0.2f,%0.2f)", x3, tdata[i], y3[0][0]);
	LOGMSG("x4,t4,y4: (%0.2f,%0.2f,%0.2f)\n", x4, tdata[i], y4[0][0]);
	
	// Now we can release the arrays.
	// (before I was doing this before the interpolation, assuming that fulldata kept its reference
	// count above zero. This was a bad assumption, because PyArray_FROM_OTF generally returns a
	// brand new object)
	for (k=0;k<4;k++)
		Py_DECREF(arr_[k]);
}

PyMethodDef valsOnGrid_methdef = {
    "valsOnGrid", (PyCFunction)valsOnGrid_toPy, 
    METH_VARARGS | METH_KEYWORDS, 
"valsOnGrid(Nvals, xvals, data, Ndata=None, cubic=True)\n"
"\n"
"Use interpolation to find the field/metric/christoffel values on an\n"
"output simulation grid.\n"
"\n"
"Parameters\n"
"----------\n"
"	Nvals : array\n"
"	xvals : array\n"
"		The values at which one wants to do the interpolation.\n"
"		*xvals* and *Nvals* must have the same shape.\n"
"	data : list\n"
"		Output from :func:`readFromFile`.\n"
"	Ndata : array, optional\n"
"		The time variable for each time slice along the simulation.\n"
"		If provided, should be equal to ``[d[0] for d in data]``.\n"
"	cubic: bool, optional\n"
"		True for cubic interpolation, False for linear interpolation.\n"
"\n"
"Returns\n"
"-------\n"
"	array\n"
"		The field/metric/christoffel values and their derivatives at\n"
"		each of the input points.\n"
};

PyObject *valsOnGrid_toPy(PyObject *self, PyObject *args, PyObject *keywds) {
	// Input is tvals, xvals, data, Ndata
	// tvals and xvals should have exactly the same shape.
    static char *kwlist[] = {"Nvals", "xvals", "data", "Ndata","cubic", NULL};
	PyObject *t0Obj, *x0Obj, *fulldata, *tdataObj; 
	int cubic = 1;
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOO|i", kwlist, &t0Obj, &x0Obj, &fulldata, &tdataObj, &cubic))
		return NULL;
    tdataObj = PyArray_FROM_OTF(tdataObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY); 
    t0Obj = PyArray_FROM_OTF(t0Obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY); 
	x0Obj = PyArray_FROM_OTF(x0Obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
	if ( tdataObj==NULL || t0Obj==NULL || x0Obj == NULL) {
		Py_XDECREF(tdataObj);
		Py_XDECREF(t0Obj);
		Py_XDECREF(x0Obj);
		return NULL;
	}

	int nt = PyArray_Size(tdataObj);
	int nin = PyArray_Size(t0Obj); // number of inputs
	if (nin != PyArray_Size(x0Obj)) {
		Py_XDECREF(tdataObj);
		Py_XDECREF(t0Obj);
		Py_XDECREF(x0Obj);
		return NULL;
	}
	double *tdata = (double *)PyArray_DATA((PyArrayObject *)tdataObj);
	double *t0 = (double *)PyArray_DATA((PyArrayObject *)t0Obj);
	double *x0 = (double *)PyArray_DATA((PyArrayObject *)x0Obj);
	
	// Find out what ny is
	PyObject *slice0, *yslice0;
	slice0 = PySequence_GetItem(fulldata, 0);
	yslice0 = PySequence_GetItem(slice0,2);
	npy_intp ny = ((npy_intp *)PyArray_DIMS((PyArrayObject *)yslice0))[2];
	Py_DECREF(slice0); Py_DECREF(yslice0);
	
	// Make the output array.
	int ndim = PyArray_NDIM((PyArrayObject *)t0Obj)+2;
	npy_intp dimensions[ndim];
	memcpy(dimensions, PyArray_DIMS((PyArrayObject *)t0Obj), 
		sizeof(npy_intp)*(ndim-2));
	dimensions[ndim-2] = 4;
	dimensions[ndim-1] = ny;
	PyObject *YoutObj = PyArray_SimpleNew(ndim, dimensions, NPY_DOUBLE);
	double *yout = (double *)PyArray_DATA((PyArrayObject *)YoutObj);
	
	// Now start iterating over the array.
	int i,k;
	for (i=0; i<nin; i++) {
		double *y[4];
		for (k=0; k<4; k++)
			y[k] = &(yout[i*4*ny + k*ny]);
		valsFromData(t0[i], x0[i], fulldata, tdata, nt, y, ny, cubic);
	}
	
	// And we're done.
	return YoutObj;
}






