/*
This file will contain extensions for running a 1D adaptive grid routine in python.
My plan is to have pretty much all of the numerics written here, and then pass results
back to python for plotting and inspection.
*/

/*
 Structure of the program should go
 1. Get initial conditions (in python)
 In python, loop the following:
 2. Make the grid (in python? could move to c later)
 3. and interpolate the data onto the grid (in python using scipy)
 3. Make and fill the regions (in c)
 4. Evolve the regions a few times (in c)
 5. Record the results (from python)
*/
 
#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <adaptiveGrid.h>

#define VERBOSE 1
#if VERBOSE
	#define BLURT() printf("%s(%d) -- %s()\n", __FILE__, __LINE__, __func__)
	#define LOGMSG(...) printf("%s(%d): ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n")
#else
	#define BLURT() ;
	#define LOGMSG(...) ;
#endif

#define pow2N(N) (1 << N)

// This following function is kind of ridiculous, but oh well.
// (For some reason my old interp function seems to be screwing up.)
// It uses the python c api to call the scipy interpolation functions, which
// in turn call some fortran routines.
// This whole thing could be written in two lines in python:
//		tck = interpolate.splprep(yold, u=xold, k=3, s=0)[0]
//		ynew = np.array(interpolate.splev(xnew,tck,0)).T

static PyObject *_interp_splprep = NULL; // This is initialized in the init function below.
static PyObject *_interp_splev = NULL; // This is initialized in the init function below.
double *interpGrid_scipy(double *xold, int nold, double *yold, int ny, double *xnew, int nnew) {
	PyObject *Xold, *Yold, *Xnew, *Ynew;
	PyObject *temp, *args, *kw;
	PyObject *tck;
	npy_intp dim[2];
	double *ynew, *y;
	int i,j;
	
	dim[0] = nold;
	Xold = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, xold);
	dim[1] = ny;
	temp = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, yold);
	Yold = PyArray_Transpose((PyArrayObject *)temp,NULL); Py_DECREF(temp);
	dim[0] = nnew;
	Xnew = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, xnew);
	
	args = Py_BuildValue("(O)", Yold); Py_DECREF(Yold);
	kw = Py_BuildValue("{s:O, s:i, s:i}","u",Xold,"k",3,"s",0); Py_DECREF(Xold);
	temp = PyObject_Call(_interp_splprep, args, kw); Py_DECREF(args); Py_DECREF(kw);
	tck = PySequence_GetItem(temp, 0); Py_DECREF(temp);
	args = Py_BuildValue("(O,O,i)",Xnew,tck,0); Py_DECREF(Xnew); Py_DECREF(tck);
	temp = PyObject_CallObject(_interp_splev, args); Py_DECREF(args);
	Ynew = PyArray_FromAny(temp, NULL,0,0,NPY_C_CONTIGUOUS,NULL); Py_DECREF(temp); 
	
	// Really we want to output transpose(Ynew).
	y = (double *)PyArray_DATA(Ynew);
	ynew = malloc(sizeof(double)*ny*nnew);
	for (i=0; i<ny; i++) {
		for (j=0; j<nnew; j++)
			ynew[j*ny+i] = y[j+i*nnew];
	}
	
	Py_DECREF(Ynew);
	return ynew;
}


// -------------------------------------------------------------------------------------------------------
// File I/O.
// -------------------------------------------------------------------------------------------------------

void writeToFile(FILE *fptr, int32_t nx, int32_t ny, double t, double *x, double *y, double *c1) {
	int i,k;
	double *dy;
	
	if (nx == 0) {
		fwrite(&nx, sizeof(int), 1, fptr);
		return;
	}
	
	// First calculate the first deriv.
	dy = malloc(sizeof(double)*nx*ny);	
	for (k=0; k<ny; k++) {
		dy[k] = dy[k+ny] = 0;
		dy[(nx-1)*ny+k] = dy[(nx-2)*ny+k] = 0;
	}
	for (i=0; i<nx-4; i++) {
		for (k=0; k<ny; k++) {
			dy[ny*(i+2)+k] = y[ny*i+k]*c1[i*5] + y[ny*(i+1)+k]*c1[i*5+1] + y[ny*(i+2)+k]*c1[i*5+2]
			+ y[ny*(i+3)+k]*c1[i*5+3] + y[ny*(i+4)+k]*c1[i*5+4];
		}
	}
	
	// Then write everything to file.
	fwrite(&nx, sizeof(int), 1, fptr);
	fwrite(&ny, sizeof(int), 1, fptr);
	fwrite(&t, sizeof(double), 1, fptr);
	fwrite(x, sizeof(double), nx, fptr);
	fwrite(y, sizeof(double), nx*ny, fptr);
	fwrite(dy, sizeof(double), nx*ny, fptr);
	free(dy);
}

static PyObject *readFromFile_toPy(PyObject *self, PyObject *args) {
	PyObject *X, *Y, *dY, *listOut, *dataTuple;
	npy_intp dimensions[2];
	int32_t nx, ny;
	double t;
	
	int numItems = 0, maxItems = 10000000; // don't read more than 10mil, for sanity
	
	char *fname;
	FILE *fptr;
	
	if (!PyArg_ParseTuple(args, "s", &fname) ){
		return NULL;
	}

	listOut = PyList_New(0);
	
	fptr = fopen(fname, "r");
//	fread(&nx, sizeof(int32_t), 1, fptr);
	while (fread(&nx, sizeof(int32_t), 1, fptr) && numItems < maxItems) {
		fread(&ny, sizeof(int32_t), 1, fptr);
		dimensions[0] = nx;
		dimensions[1] = ny;
		X = PyArray_SimpleNew(1, dimensions, NPY_DOUBLE);
		Y = PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
		dY = PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
		fread(&t, sizeof(double), 1, fptr);
		fread(PyArray_DATA(X), sizeof(double), nx, fptr);
		fread(PyArray_DATA(Y), sizeof(double), nx*ny, fptr);
		fread(PyArray_DATA(dY), sizeof(double), nx*ny, fptr);
		dataTuple = Py_BuildValue("(dOOO)", t,X,Y,dY);
		Py_DECREF(X); Py_DECREF(Y); Py_DECREF(dY);
		PyList_Append(listOut, dataTuple);
		Py_DECREF(dataTuple);
//		fread(&nx, sizeof(int32_t), 1, fptr);
		numItems++;
	}
	
	return listOut;
}


// -------------------------------------------------------------------------------------------------------
// Simulating a wave equation.
// -------------------------------------------------------------------------------------------------------

int dY_wave(double t, double *y, double *c1, double *c2, int nx, double *dy) {
	// dy1 = y2
	// dy2 = d^2(y1)/dx^2

	int i;
	
	dy[0]=dy[1]=dy[2]=dy[3]= 0; // assume no change at the boundaries
	dy[2*nx-4]=dy[2*nx-3]=dy[2*nx-2]=dy[2*nx-1]= 0;
	
//	for (i=0; i<nx; i++) {
//		LOGMSG("y --- %0.3e  %0.3e", y[2*i], y[2*i+1]);
//	}
	
	for (i=0; i<nx-4; i++) {
		dy[2*(i+2)] = y[2*(i+2)+1];
		dy[2*(i+2)+1] = y[2*i]*c2[i*5] + y[2*(i+1)]*c2[i*5+1] + y[2*(i+2)]*c2[i*5+2]
			+ y[2*(i+3)]*c2[i*5+3] + y[2*(i+4)]*c2[i*5+4];
//		LOGMSG("Y --- %0.3e  %0.3e", y[2*(i+2)], y[2*(i+2)+1]);
//		LOGMSG("dY --- %0.3e  %0.3e", dy[2*(i+2)], dy[2*(i+2)+1]);
//		LOGMSG("c2: %0.2e  %0.2e  %0.2e  %0.2e  %0.2e\n", c2[5*i], c2[5*i+1], c2[5*i+2], c2[5*i+3], c2[5*i+4]);
	}
	
	return 0;
}

double *wave_gridDensity(double *x, double *y, double *c1, int nx, double gain, double minm) {
	double *m, dydx, dydt;
	int i;
	
	m = malloc(sizeof(double)*nx);
	for (i=2; i<nx-2; i++) {
		dydt = y[2*i+1];
		dydx = c1[5*i-10]*y[2*(i-2)] + c1[5*i-9]*y[2*(i-1)] + c1[5*i-8]*y[2*i]
			+ c1[5*i-7]*y[2*(i+1)] + c1[5*i-6]*y[2*(i+2)];
		m[i] = sqrt(dydx*dydx+dydt*dydt)*gain + minm;
//		LOGMSG("i=%i, x=%0.3f, y=%0.3f, dydt=%0.3f, dydx=%0.3f, m=%0.3f", i, x[i],y[2*i],dydt,dydx,m[i]);
	}
	m[0] = m[1] = m[2];
	m[nx-1] = m[nx-2] = m[nx-3];
	
	return m;
}

double evolveWave(double *x0, double *y0, int nx0, double t0, double tmax,
				double **xout, double **yout, int *nxout) {
	int i, ny, nx, nxnew, Nmax, nsteps;
	double *c1, *c2, *x, *xnew, *y, *ynew, *dx, *m;
	double x0min, x0max, xmin, xmax, mindx, t, dt;
	double cfl = 0.2;
	int numEvolve = 5; // evolve 5 times before regridding.
	Region *R, *R0 = NULL;
	
	int minRegionWidth = 16;
	double gridUniformity = M_LN2 / minRegionWidth; // zero is perfectly uniform, large number means not (necessarily) uniform.
	// (This last line ensures that the grid density never drops by more than
	// a factor of two in minRegionWidth points. If this number were much larger,
	// you'd have the problem that solution in the fine regions could evolve into
	// the coarse regions before we have a chance to re-adapt the grid.)
	
	FILE *fptr = fopen("/Users/maxwain/Desktop/wavedata.dat","w");
		
	ny = 2;
	nx = nx0;
	
	*xout = *yout = NULL;
	
	// The first thing we want to do is copy the initial data.
	// This is important because we free the data from the previous iteration,
	// and we don't want to free the initial data (which this loop shouldn't control).
	x = malloc(sizeof(double)*nx);
	y = malloc(sizeof(double)*nx*ny);
	memcpy(x,x0, sizeof(double)*nx);
	memcpy(y,y0, sizeof(double)*nx*ny);
	x0min = x[0];
	x0max = x[nx-1];
	
	// Get the derivative coefficients. Necessary for calculating grid density
	// before we make the first grid.
	c1 = malloc(sizeof(double)*5*(nx-4));
	c2 = malloc(sizeof(double)*5*(nx-4));
	calcDerivCoefs(x, nx, c1, c2);
		
	t = t0;
	// begin our grand loop.
	BLURT();
	while (t < tmax) {
		// Make the new grid.
		LOGMSG("Making the grid.");
		xmin = x0min - t;
		xmax = x0max + t;
		m = wave_gridDensity(x, y, c1, nx, /*gain =*/ 20.0, /*min density =*/ 10.0);
		xnew = remakeGrid(x, m, nx, gridUniformity, xmin, xmax, &nxnew);
		ynew = interpGrid_scipy(x, nx, y, ny, xnew, nxnew);
		// Set the boundaries of the grid where we've over-extended
		for (i=0; xnew[i] < x[0]; i++) {
			ynew[2*i] = y0[0];
			ynew[2*i+1] = y0[1];
		}
		for (i=nxnew-1; xnew[i] > x[nx-1]; i--) {
			ynew[2*i] = y0[2*nx0-2];
			ynew[2*i+1] = y0[2*nx0-1];
		}
		LOGMSG("uniformity=%0.4f, xmin,xmax=%0.3f,%0.3f, nxnew=%i", gridUniformity,xmin,xmax,nxnew);
/*		for (i=0; i<nx; i++) {
			LOGMSG("%i: x=%0.4f, m=%0.4f", i, x[i], m[i]);
		}
		for (i=0; i<nxnew; i++) {
			LOGMSG("%i: xnew=%0.4f, ynew=%0.4f", i, xnew[i], ynew[2*i]);
		}
*/		// Free the old grid.
		free(x); free(y); free(m);
		x = xnew; y = ynew; nx = nxnew;
		// Recalculate the deriv coefs.
		LOGMSG("Calculating the coefs");
		free(c1); free(c2);
		c1 = malloc(sizeof(double)*5*(nx-4));
		c2 = malloc(sizeof(double)*5*(nx-4));
		calcDerivCoefs(x, nx, c1, c2);
		// Make the regions.
		LOGMSG("Making the regions");
		dx = malloc(sizeof(double)*nx);
		mindx = dx[0] = x[1]-x[0];
		for (i=1; i<nx-1; i++) {
			dx[i] = 0.5*(x[i+1]-x[i-1]);
			if (dx[i] < mindx) mindx = dx[i];
		}
		dx[nx-1] = x[nx-1]-x[nx-2];
		if (dx[nx-1] < mindx) mindx = dx[nx-1];
//		if (mindx < 0) break;
//		if (nx > 2000) break;
		LOGMSG("nx=%i, mindx=%f, minRegionWidth=%i", nx, mindx, minRegionWidth);
		R0 = makeRegions(dx, nx, mindx, 100, /*slop =*/ 0.2, minRegionWidth);
		fillRegions(R0, y, ny);
		free(dx);
		// Find the coarsest region.
		BLURT();
		Nmax = 0;
		R = R0;
		while (R) {
			if (R->N > Nmax) Nmax = R->N;
	//		LOGMSG("a region, N=%i", R->N);
			R = R->nextReg;
		}
		nsteps = pow2N(Nmax);
		dt = mindx * cfl;
		LOGMSG("About to evolve: Nmax=%i, dt=%0.5f, t=%0.3f", Nmax, dt, t);
		for (i=0; i<numEvolve; i++) {
			// Evolve the regions
		//	LOGMSG("Evolving the regions");
			if(!evolveRegions(R0, dt, t, nsteps, 2, c1, c2, &dY_wave))
				return -1;
			t += dt*nsteps;
		}
		// extract the y data from the regions.
		free(y);
		y = dataFromRegions(R0, ny, NULL);
		deallocRegions(R0);
		LOGMSG("writing data to file");
		writeToFile(fptr, nx, ny, t, x, y, c1);
		LOGMSG("end loop\n");
	}
	free(c1); free(c2);	
	fclose(fptr);
	
	// Return the data.
	*xout = x;
	*yout = y;
	*nxout = nx;
	
	LOGMSG("return nx = %i",nx);
	
	return t;
}

// -------------------------------------------------------------------------------------------------------
// Python extension functions.
// -------------------------------------------------------------------------------------------------------

// Tests the makeRegions function. Output is a 2d numpy array.
static PyObject *test_makeRegions(PyObject *self, PyObject *args)
{
	PyArrayObject *X;
	PyObject *outarr;
	int i,n, nreg;
	double *dx, dxmin;
	Region *r, *r0;
	double *out;
	npy_intp dimensions[2];
	double eps;
	int minWidth;
		
	if (!PyArg_ParseTuple(args, "O!di", &PyArray_Type, &X,&eps,&minWidth) ){
		return NULL;
	}
	
	n = 1;
	for (i = 0; i<X->nd; i++)
		n *= X->dimensions[i];
	dx = (double *)X->data;
	dxmin = dx[0];
	for (i = 1; i<n; i++)
		if (dx[i] < dxmin)
			dxmin = dx[i];
	
	LOGMSG("nx=%i, eps=%f, minWidth=%i", n, eps, minWidth);
	BLURT();
	LOGMSG("here");
	
	nreg = 0;
	r0 = r = makeRegions(dx, n, dxmin, 100, eps, minWidth);
	BLURT();
	while (r) {
		nreg++;
		r = r->nextReg;
	}
	
	BLURT();
	
	dimensions[0] = 2;
	dimensions[1] = nreg*2;
	outarr = PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
	out = (double *)PyArray_DATA(outarr);
	
	i = 0;
	r = r0;
	while (r->nextReg) {
		out[i] = r->startIndex;
		out[i+1] = r->nextReg->startIndex;
		out[i+nreg*2] = out[i+1+nreg*2] = r->N;
		r = r->nextReg;
		i += 2;
		free(r->prevReg);
	}
	out[i] = r->startIndex;
	out[i+1] = n;
	out[i+nreg*2] = out[i+1+nreg*2] = r->N;
	free(r);
	BLURT();
	
	return outarr;
}

static PyObject *test_derivCoefs(PyObject *self, PyObject *args)
{
	PyArrayObject *X;
	npy_intp nx;
	npy_intp dimensions[3];
	double *coefs;
	PyObject *outarr;
	
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X) ){
		return NULL;
	}
	
	nx = X->dimensions[0];
	dimensions[0] = 2;
	dimensions[1] = nx;
	dimensions[2] = 5;
	outarr = PyArray_SimpleNew(3, dimensions, NPY_DOUBLE);
	
	coefs = (double *)PyArray_DATA(outarr);
	calcDerivCoefs((double *)X->data, nx, coefs, coefs+5*(nx));
	
	return outarr;
}

static PyObject *test_deriv(PyObject *self, PyObject *args)
{
	PyArrayObject *X, *Y;
	npy_intp nx;
	npy_intp dimensions[2];
	double *c1, *c2, *dy, *d2y;
	PyObject *outarr;
	
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X, &PyArray_Type, &Y) ){
		return NULL;
	}
	
	nx = X->dimensions[0];
	
	c1 = malloc(sizeof(double)*nx*5);
	c2 = malloc(sizeof(double)*nx*5);
	calcDerivCoefs((double *)X->data, nx, c1, c2);
	
	dimensions[0] = 2;
	dimensions[1] = nx;
	outarr = PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
	dy = (double *)PyArray_DATA(outarr);
	d2y = dy + nx;
	
	calcDerivs((double *)Y->data, c1, nx, 1, 0, dy, 1, 0);
	calcDerivs((double *)Y->data, c2, nx, 1, 0, d2y, 1, 0);
	
	free(c1); free(c2);
	
	return outarr;
}

static PyObject *waveEvolve_py(PyObject *self, PyObject *args)
{
	PyArrayObject *X, *Y;
	PyObject *Yout;
	npy_intp dimensions[2];
	double *dx, *x, *y, dxmin, *c1, *c2, *yout, dt;
	int i,nx,Nmax=0;
	Region *r, *r0;

	if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &X, &PyArray_Type, &Y, &dt) ){
		return NULL;
	}
	
	// First make the regions.
	nx = X->dimensions[0];
	x = (double *)X->data;
	dx = malloc(sizeof(double)*nx);
	dx[0] = x[1]-x[0];
	dx[nx-1] = x[nx-1]-x[nx-2];
	dxmin = dx[0] > dx[nx-1] ? dx[nx-1] : dx[0];
	for (i=1; i<nx-1; i++) {
		dx[i] = 0.5*(x[i+1]-x[i-1]);
		if (dx[i] < dxmin) dxmin = dx[i];
	}
	r0 = makeRegions(dx, nx, dxmin, 100, .1, 16);
	y = (double *)Y->data;
	fillRegions(r0, y, 2);
		
	// Calculate the derivative coefficients.
	c1 = malloc(sizeof(double)*5*(nx-4));
	c2 = malloc(sizeof(double)*5*(nx-4));
	calcDerivCoefs(x, nx, c1, c2);
	
	printRegions(r0);
	
	// Evolve the regions
	r = r0;
	while (r) {
		if (r->N > Nmax) Nmax = r->N;
		LOGMSG("r: i0=%i, w=%i, L=%i, R=%i", r->startIndex, r->width, r->leftOverlap, r->rightOverlap);
		r = r->nextReg;
	}
	LOGMSG("Nmax: %i", Nmax);
	evolveRegions(r0, dt/pow2N(Nmax), 0.0, pow2N(Nmax), 2, c1, c2, &dY_wave);
	
	// Extract and return the data from the regions.
	yout = dataFromRegions(r0, 2, &nx);
	dimensions[0] = nx;
	dimensions[1] = 2;
	Yout = PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
	memcpy(PyArray_DATA(Yout), yout, sizeof(double)*nx*2);
	
	free(dx); free(c1); free(c2); free(yout);
	deallocRegions(r0);
	
	return Yout;
}

static PyObject *waveEvolve_py2(PyObject *self, PyObject *args) {
	PyArrayObject *X, *Y;
	PyObject *Yout, *Xout, *tupleOut;
	npy_intp dimensions[2];
	double *x, *y, *xout, *yout;
	double tmax;
	int nx, nxout;
	
	if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &X, &PyArray_Type, &Y, &tmax) ){
		return NULL;
	}
	
	nx = X->dimensions[0];
	x = (double *)X->data;
	y = (double *)Y->data;
	
	tmax = evolveWave(x, y, nx, 0.0, tmax, &xout, &yout, &nxout);
	if (xout == NULL) return NULL;
	
	dimensions[0] = nxout;
	dimensions[1] = 2;
	
	Yout = PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
	memcpy(PyArray_DATA(Yout), yout, sizeof(double)*nxout*2);
	Xout = PyArray_SimpleNew(1, dimensions, NPY_DOUBLE);
	memcpy(PyArray_DATA(Xout), xout, sizeof(double)*nxout);
	
	tupleOut = Py_BuildValue("(dOO)",tmax,Xout,Yout);
	Py_DECREF(Xout);
	Py_DECREF(Yout);
	
	return tupleOut;
}

static PyObject *remakeGrid_py(PyObject *self, PyObject *args)
{
	PyArrayObject *X, *grid_density;
	double grid_uniformity, xmin, xmax, *xout;
	PyObject *Xout;
	npy_intp dimensions[1];
	int nout;
	
	if (!PyArg_ParseTuple(args, "O!O!ddd", &PyArray_Type, &X, &PyArray_Type, &grid_density, 
						  &grid_uniformity, &xmin, &xmax) ) {
		return NULL;
	}
	
	xout = remakeGrid((double *)X->data, (double *)grid_density->data, X->dimensions[0], 
					  grid_uniformity, xmin, xmax, &nout);
	dimensions[0] = nout;
	Xout = PyArray_SimpleNew(1, dimensions, NPY_DOUBLE);
	memcpy(PyArray_DATA(Xout), xout, sizeof(double)*nout);
	free(xout);
	
	return Xout;
}
	

static PyObject *test_interpGrid(PyObject *self, PyObject *args)
{
	PyArrayObject *Xold, *Xnew, *Yold;
	int nold, nnew, ny;
	npy_intp dimensions[2];
	double *ynew;
	PyObject *outarr;
	
	if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &Xold, &PyArray_Type, &Yold, &PyArray_Type, &Xnew) ){
		return NULL;
	}
	
	nold = Xold->dimensions[0];
	nnew = Xnew->dimensions[0];
	ny = Yold->dimensions[1];
	dimensions[0] = nnew;
	dimensions[1] = ny;
	outarr = PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);


	BLURT();
	LOGMSG("nold %i, ny %i, nnew %i", nold,ny,nnew);
	ynew = interpGrid((double *)Xold->data, nold, (double *)Yold->data, ny, (double *)Xnew->data, nnew);
	BLURT();
	memcpy(PyArray_DATA(outarr), ynew, sizeof(double)*ny*nnew);
	free(ynew);
	
	return outarr;
}


static PyMethodDef gridPyMethods[] = {
    {"makeRegions",  test_makeRegions, METH_VARARGS,
		"Makes the regions from an array dx. Output is a 2d numpy array."},
	{"derivCoefs", test_derivCoefs, METH_VARARGS, ""},
	{"deriv", test_deriv, METH_VARARGS, ""},
	{"waveEvolve", waveEvolve_py, METH_VARARGS, 
		"Input is x, y (shape (nx,2)), dt."},
	{"waveEvolve2", waveEvolve_py2, METH_VARARGS, 
		"Input is x, y (shape (nx,2)), tmax."},
	{"remakeGrid", remakeGrid_py, METH_VARARGS, 
		"Input is x, grid_density (len nx), grid_uniformity, xmin, xmax."},
	{"interpGrid", test_interpGrid, METH_VARARGS, 
		"Input is Xold, Yold, Xnew."},
	{"readFromFile", readFromFile_toPy, METH_VARARGS, 
		"Input is filename, output is [(t1,x1,y1,dy1),(t2,x2,y2,dy2),...]."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyMODINIT_FUNC
initadaptiveGridTest(void)
{
	PyObject *interp_module;
	char docstring[] = "docstring...";
    (void) Py_InitModule3("adaptiveGridTest", gridPyMethods, docstring);
	import_array();

	interp_module = PyImport_ImportModule("scipy.interpolate");
	_interp_splprep = PyObject_GetAttrString(interp_module, "splprep");
	_interp_splev = PyObject_GetAttrString(interp_module, "splev");
	Py_DECREF(interp_module);	
}

