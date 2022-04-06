
#include <Python.h>
#define NO_IMPORT_ARRAY
#include <arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>

#include "bubbleEvolution.h"
#include "adaptiveGrid.h"
#include "model_objects.h"


#define VERBOSE 1
#if VERBOSE
#define BLURT() printf("%s(%d) -- %s()\n", __FILE__, __LINE__, __func__)
#define LOGMSG(...) printf("%s(%d): ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n")
#else
#define BLURT() ;
#define LOGMSG(...) ;
#endif

#define ERRMSG(...) printf("ERROR -- %s(%d): ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n")

#define pow2N(N) (1 << N)
#define PI 3.141592653589793

#pragma mark --Global variables
// ----------------------------------------------------------------------
// Global variables.
// ----------------------------------------------------------------------

// First, let's define some global (static) variables that can be set by the user.
// All of the data is stored in arrays ordered by (nx, ny). 
// The field indices go (phi1, phi2, phi3, ..., dphi1, dphi2, ..., alpha, a)

typedef struct {
    npy_intp nfields;
    ScalarFunc V;
    ScalarFunc dV;
    PyObject *obj;
} BareModelObject;
static BareModelObject the_model = {0, NULL, NULL, NULL};

static char *fname = NULL; // Output file name for fields and metric
static char *fchrisname = NULL; // Output christoffel file name
static int xres=1; // How closely spaced we want to save the data. 1 is save every point, 2 is every other, etc.
//static double tres=0.0; // Approximate spacing between save times.
static double *tout = NULL; // Output time slices, or spacing between time slices
static int ntout = 0; // Number of output time slices, or -1 for constant spacing between outputs
static int exact_tout=0; // Non-zero for integration to align exactly to desired output slices.
static int minRegionWidth = 40; // Minimum number of points per region
static int Nmax0 = 7;

static double cfl = 0.2; // The size of the timestep relative to the step-size.
static int stepsPerRegrid = 2;
static double minStepsPerPeriod = 30.0; // where a period is given by the largest frequency of oscillations about a minimum.
static double highestMass = 1e-100; // used for determining the period.

// The following are for the monitor function
static PyObject *monitor_callback;

// The following two are used to stop the simulation if it takes too long
static double max_run_time = 0.0;
static double check_run_time_before_N = 0.0;

/*
double *(*V_func)(int, int, double *) = NULL;
double *(*dV_func)(int, int, double *) = NULL;
*/

#pragma mark --Python wrappers
// ----------------------------------------------------------------------
// Wrapper and python integration functions.
// ----------------------------------------------------------------------

PyMethodDef setModel_methdef = {
    "setModel", (PyCFunction)setModel, METH_VARARGS, 
"setModel(model_obj)\n"
"\n"
"Sets the model to be used in the simulation.\n"
"\n"
"Parameters\n"
"----------\n"
"model_obj\n"
"    Either a model object defined in :mod:`models` or a python object\n"
"    (NOT IMPLEMENTED) that has methods ``V()`` and ``dV()`` as well as\n"
"    attribute ``nfields``. The methods should accept input of shape\n"
"    ``(nx, ny)`` where ``ny >= nfields`` and output arrays of shape\n"
"    ``(nx,)`` and ``(nx, nfields)`` for the potential and its derivative,\n"
"    respectively.\n"};

PyObject *setModel(PyObject *self, PyObject *args) {
	PyObject *new_model_obj;
    if (!PyArg_ParseTuple(args, "O", &new_model_obj))
    	return NULL;
    PyObject *is_model_obj = PyObject_CallMethod(
    	new_model_obj, "check_model_object", NULL);
    if (!is_model_obj || !PyObject_IsTrue(is_model_obj)) {
    	// Should try to make it handle python classes too.
    	PyErr_SetString(PyExc_TypeError, 
    		"The model must be a `ModelObject` type returned from "
    		"the `models` module.");
    	return NULL;
    }
    else {
    	the_model.nfields = ((ModelObject *)new_model_obj)->nfields;
    	the_model.V = ((ModelObject *)new_model_obj)->V;
    	the_model.dV = ((ModelObject *)new_model_obj)->dV;
    }

    // Get rid of the old model and get a reference to the new one.
    Py_INCREF(new_model_obj);
    Py_XDECREF(the_model.obj);
    the_model.obj = (PyObject *)new_model_obj;

    Py_INCREF(Py_None);
    return Py_None;
}

PyMethodDef setFileParams_methdef = {
    "setFileParams", (PyCFunction)setFileParams, METH_VARARGS | METH_KEYWORDS, 
"Sets parameters for outputting to file.\n"
"\n"
"Parameters\n"
"----------\n"
"fields_file_name : string, optional\n"
"    Name of the file which will store the fields and metric functions.\n"
"christoffel_file_name : string, optional\n"
"    Name of the file which will store the chrisoffel symbols.\n"
"xres : int, optional\n"
"    Distance between sequential outputs in the x direction. Defaults to\n"
"    1, indicating that every grid point along the time slice gets saved\n"
"    during every write. A value of 2 would save every other grid point, etc.\n"
"tout : float or array, optional\n"
"    If a float, indicates spacing between sequential output time-slices.\n"
"    If an array, specifies the times at which output should be written.\n"
"    Defaults to NULL (no output).\n"
"exact_tout: bool, optional\n"
"    If True, align the simulation time slices to match exactly with \n"
"    *tout*. Otherwise, the simulation writes to at the first time slices\n"
"    directly following the values specified in tout. Defaults to False.\n"
};

PyObject *setFileParams(PyObject *self, PyObject *args, PyObject *keywds) {
    static char *kwlist[] = {
    	"fields_file_name", "christoffel_file_name", 
    	"xres", "tout", "exact_tout", NULL};
	PyObject *fnameObj = NULL;
	PyObject *fchrisObj = NULL;
	PyObject *toutObj = NULL;
	char *s;
	
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|OOiOi", kwlist, 
    	&fnameObj, &fchrisObj, &xres, &toutObj, &exact_tout))
			return NULL;

	if (fnameObj) {
		if (fname) free(fname);
		fname = NULL;
		if (fnameObj && PyString_Check(fnameObj)) {
			s = PyString_AsString(fnameObj);
			if (strlen(s) > 0) {
				fname = malloc(sizeof(char)*(strlen(s)+1));
				strcpy(fname, s);
			}
		}
	}
	if (fchrisObj) {
		if (fchrisname) free(fchrisname);
		fchrisname = NULL;
		if (fchrisObj && PyString_Check(fchrisObj)) {
			s = PyString_AsString(fchrisObj);
			if (strlen(s) > 0) {
				fchrisname = malloc(sizeof(char)*(strlen(s)+1));
				strcpy(fchrisname, s);
			}
		}
	}
	if (toutObj) {
		if (tout) free(tout);
		toutObj = PyArray_FROM_OTF(toutObj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
		ntout = PyArray_SIZE((PyArrayObject *)toutObj);
		tout = (double *)malloc(sizeof(double)*ntout);
		memcpy(tout, PyArray_DATA((PyArrayObject *)toutObj), sizeof(double)*ntout);
		if (PyArray_NDIM((PyArrayObject *)toutObj) == 0)
			ntout = -1;
		Py_DECREF(toutObj);
	}
	
	LOGMSG("setFileParams: \n\tfname = %s; xres = %i; ntout = %i", fname ? fname : "NULL", xres, ntout);
				
	Py_INCREF(Py_None);
	return Py_None;
}

PyMethodDef setIntegrationParams_methdef = {
    "setIntegrationParams", (PyCFunction)setIntegrationParams, 
    METH_VARARGS | METH_KEYWORDS, 
"Sets parameters that govern integration behavior.\n"
"\n"
"Parameters\n"
"----------\n"
"cfl : float, optional\n"
"    The `Courant number <http://en.wikipedia.org/wiki/Courant–Friedrichs–Lewy_condition>`_\n"
"    used to calculate the temporal step size. Defaults to 0.2.\n"
"minStepsPerPeriod : float, optional\n"
"    The minimum number of time steps per period of oscillation given a\n"
"    characteristic mass scale *mass_osc*. That is, the time step will \n"
"    never be so large such that there are fewer than *minStepsPerPeriod*\n"
"    steps per oscillatory period. Defaults to 30.0.\n"
"mass_osc : float, optional\n"
"    The characteristic mass scale for oscillaitons. This should generally\n"
"    be the largest mass (given by eigenvalues of the Hessian matrix) in\n"
"    the model. Defaults to 1e-100 (effectively zero).\n"
"stepsPerRegrid : int, optional\n"
"    The number of integration steps taken in the coarsest region before\n"
"    recalculating the entire grid spacing. Defaults to 2.\n"
"minRegionWidth : int, optional\n"
"    Minimum number of grid points per integration region. Each integration\n"
"    region is integrated separately at each time step, with each region\n"
"    having some small amount of overlap with its neighbors. If this number\n"
"    is too small (or Nmax is too large), sharp features can evolve from\n"
"    higher to lower resolution regions before the simulation has a chance\n"
"    to recalculate the grid spacing. Defaults to 40; must be at least 16.\n"
"Nmax : int, optional\n"
"    The log base 2 of the ratio between the time steps in the lowest and\n"
"    highest resolution regions. Defaults to 7.\n"
};

PyObject *setIntegrationParams(PyObject *self, PyObject *args, PyObject *keywds) {
    static char *kwlist[] = {"cfl", "minStepsPerPeriod", "mass_osc", "stepsPerRegrid", "minRegionWidth", "Nmax", NULL};
	
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|dddiii", kwlist, 
									 &cfl, &minStepsPerPeriod, &highestMass, &stepsPerRegrid, &minRegionWidth, &Nmax0))
		return NULL;

	LOGMSG("setIntegrationParams: \n\t"
		   "cfl = %0.3f; minStepsPerPeriod = %0.2f; mass_osc = %f\n\t"
		   "stepsPerRegrid = %i; minRegionWidth = %i; Nmax = %i",
		   cfl, minStepsPerPeriod, highestMass, stepsPerRegrid, minRegionWidth, Nmax0);
	
	Py_INCREF(Py_None);
	return Py_None;
}

PyMethodDef setMonitorCallback_methdef = {
    "setMonitorCallback", (PyCFunction)setMonitorCallback, METH_VARARGS, 
"Sets the object that determines the grid spacing.\n"
"\n"
"The object must be callable (it can either be a function or a python\n"
"class instance). It can must have a call signature of (N, x, y) where\n"
"*N* is the simulation time variable, *x* is an array of grid points, and\n"
"*y* is an array of shape ``(nx,2*nfields+2)`` that contains the fields \n"
":math:`\\phi_i`, their conjugate momenta :math:`\\Pi_i`, and the metric \n"
"functions :math:`\\alpha` and :math:`a`. The return value should be an\n"
"array of length *nx* indicating the desired density of points for a new grid.\n"
};

PyObject *setMonitorCallback(PyObject *self, PyObject *args) {
    PyObject *result = NULL;
    PyObject *temp;
	
    if (PyArg_ParseTuple(args, "O:set_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
        }
        Py_XINCREF(temp);         /* Add a reference to new callback */
        Py_XDECREF(monitor_callback);  /* Dispose of previous callback */
        monitor_callback = temp;       /* Remember new callback */
        /* Boilerplate to return "None" */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}

PyMethodDef setTimeConstraints_methdef = {
    "setTimeConstraints", (PyCFunction)setTimeConstraints, METH_VARARGS, 
"setTimeConstraints(max_run_time, N_last_check_time)\n"
"\n"
"Set the computer time constraints for the simulation. \n"
"\n"
"If the simulation has been running for *max_run_time* (in seconds)\n"
"and has not yet reached *N=N_last_check_time*, abort the simulation.\n"
"Both values default to zero, so time constraints are not checked\n"
"by default.\n"
};

PyObject *setTimeConstraints(PyObject *self, PyObject *args) {
    if (!PyArg_ParseTuple(args, "dd", &max_run_time, &check_run_time_before_N))
		return NULL;
    	
	Py_INCREF(Py_None);
	return Py_None;
}

PyMethodDef remakeGrid_methdef = {
    "remakeGrid", (PyCFunction)remakeGrid_frompy, METH_VARARGS, 
"For debugging only."
};

PyObject *remakeGrid_frompy(PyObject *self, PyObject *args) {
    PyObject *xobj, *mobj;
    if (!PyArg_ParseTuple(args, "OO", &xobj, &mobj))
        return NULL;
        
    xobj = PyArray_FROM_OTF(xobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    mobj = PyArray_FROM_OTF(mobj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);

    int nx = PyArray_SIZE((PyArrayObject*)xobj);
    double *m = (double*)PyArray_DATA((PyArrayObject*)mobj);
    double *x = (double*)PyArray_DATA((PyArrayObject*)xobj);
    double xmin = x[0];
    double xmax = x[nx-1];
    int nx_out;
    double gridUniformity = M_LN2 / minRegionWidth;     
    double *xnew = remakeGrid(x,m,nx,gridUniformity,xmin,xmax,&nx_out);

    Py_DECREF(xobj);
    Py_DECREF(mobj);
    npy_intp dims = nx_out;
    xobj = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject *)xobj), xnew, sizeof(double)*nx_out);
    free(xnew);

    return xobj;
}

PyMethodDef runCollision_methdef = {
    "runCollision", (PyCFunction)runCollision, METH_VARARGS | METH_KEYWORDS, 
"runCollision(x, y, t0, tmax, exactTmax=True, growBounds=True, overwrite=True, alphaMax=1e5)\n"
"\n"
"Runs the simulation.\n"
"\n"
"Parameters\n"
"----------\n"
"x : array\n"
"    Initial grid values.\n"
"y : array\n"
"    Initial field values on the grid. Should be shape ``(nx, 2*nfields+2)``,\n"
"    and contain the fields :math:`\\phi_i`, their conjugate momenta\n"
"    :math:`\\Pi_i`, and the metric functions :math:`\\alpha` and :math:`a`.\n"
"t0 : float\n"
"    Time variable *N* along the initial time slice.\n"
"tmax : float\n"
"    Simulation stops when it reaches *tmax*.\n"
"exactTmax : bool, optional\n"
"    If True, the simulations stops exactly at *tmax* and the final output\n"
"    is at *tmax*. Otherwise, the final time step is determined from the \n"
"    CFL condition.\n"
"growBounds : bool, optional\n"
"    If True, expand the boundaries of the simulation to include all space\n"
"    in the future lightcone of the initial grid. If False, keep the simulation\n"
"    boundaries fixed.\n"
"overwrite : bool, optional\n"
"    If True, overwrite the output files. If False, append output.\n"
"alphaMax : float, optional\n"
"    Maximum value of the metric function :math:`\\alpha` before the\n"
"    simulation aborts (large :math:`\\alpha` usually indicates the end of\n"
"    inflation).\n"
"\n"
"Returns\n"
"-------\n"
"    t_final : double\n"
"    x_out : array\n"
"    y_out : array\n"
};

PyObject *runCollision(PyObject *self, PyObject *args, PyObject *keywds) {
	PyObject *Xin, *Yin, *Xout, *Yout, *tupleOut;
	int nx, ny, nxout, exactTmax=1, growBounds=1, overwrite=1;
	double *x, *y, *xout, *yout;
	double t0, tmax;
	npy_intp dimensions[2];
	npy_intp ndimy, ndimx, *xdims, *ydims;
	double alphaMax = 1e5;

	if (the_model.obj == NULL) {
		PyErr_SetString(PyExc_RuntimeError, 
			"Trying to run simulation without a model set.");
		return NULL;
	}
	if (monitor_callback == NULL) {
		PyErr_SetString(PyExc_RuntimeError, 
			"Trying to run simulation without a monitor function set.");
		return NULL;
	}
	BLURT();
    static char *kwlist[] = {
    	"x", "y", "t0", "tmax", 
    	"exactTmax", "growBounds", "overwrite", "alphaMax", NULL};
    BLURT();
	if (!PyArg_ParseTupleAndKeywords(
			args, keywds, "OOdd|iiid",kwlist, 
			&Xin, &Yin, &t0, &tmax, &exactTmax, &growBounds, &overwrite, &alphaMax) ){
		BLURT();
		return NULL;
	}
    BLURT();
    Xin = PyArray_FROM_OTF(Xin, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    Yin = PyArray_FROM_OTF(Yin, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (Yin == NULL || Xin == NULL) {
    	PyErr_SetString(PyExc_TypeError, 
    		"Initial conditions not convertable to arrays.");
    	Py_XDECREF(Xin); Py_XDECREF(Yin); return NULL;
    }
    BLURT();

    ndimy = PyArray_NDIM((PyArrayObject *)Yin);
    ndimx = PyArray_NDIM((PyArrayObject *)Xin);
    if (ndimx != 1 || ndimy != 2) {
    	PyErr_SetString(PyExc_ValueError, 
    		"Input arrays must have shape(x) = (nx,) and shape(y) = (nx, 2*nfields+2).");
    	Py_XDECREF(Xin); Py_XDECREF(Yin); return NULL;
    }
    xdims = PyArray_DIMS((PyArrayObject *)Xin);
    ydims = PyArray_DIMS((PyArrayObject *)Yin);
    if (xdims[0] != ydims[0] || ydims[1] != 2*the_model.nfields+2) {
    	PyErr_SetString(PyExc_ValueError, 
    		"Input arrays must have shape(x) = (nx,) and shape(y) = (nx, 2*nfields+2).");
    	Py_XDECREF(Xin); Py_XDECREF(Yin); return NULL;
    }
	nx = xdims[0];
	ny = ydims[1];
	x = (double *)PyArray_DATA((PyArrayObject *)Xin);
	y = (double *)PyArray_DATA((PyArrayObject *)Yin);
	tmax = evolveBubbles(x, y, nx, t0, tmax, 
		alphaMax, exactTmax, growBounds, overwrite, &xout, &yout, &nxout);
	Py_DECREF(Xin); Py_DECREF(Yin);
	if (xout == NULL) {
		if (!PyErr_Occurred())
			PyErr_SetString(PyExc_RuntimeError,
				"Unhandled error in simulation; xout == NULL");
		return NULL;
	}

	dimensions[0] = nxout;
	dimensions[1] = ny;
	
	Yout = PyArray_SimpleNew(2, dimensions, NPY_DOUBLE);
	memcpy(PyArray_DATA((PyArrayObject *)Yout), yout, sizeof(double)*nxout*ny);
	Xout = PyArray_SimpleNew(1, dimensions, NPY_DOUBLE);
	memcpy(PyArray_DATA((PyArrayObject *)Xout), xout, sizeof(double)*nxout);
	free(xout); free(yout);
	
	tupleOut = Py_BuildValue("(dOO)",tmax,Xout,Yout);
	Py_DECREF(Xout);
	Py_DECREF(Yout);
	
	return tupleOut;
}

#pragma mark -
#pragma mark Computation functions
// ----------------------------------------------------------------------
// Computation functions.
// ----------------------------------------------------------------------


static PyObject *_interp_splprep = NULL; // This is initialized in the init function below.
static PyObject *_interp_splev = NULL; // This is initialized in the init function below.
static PyObject *_interp_module = NULL;

double *interpGrid_scipy(double *xold, int nold, double *yold, int ny, double *xnew, int nnew) {
	PyObject *Xold, *Yold, *Xnew, *Ynew;
	PyObject *temp, *args, *kw;
	PyObject *tck;
	npy_intp dim[2];
	double *ynew, *y;
	int i,j;
	
	if (_interp_module == NULL) {
		_interp_module = PyImport_ImportModule("scipy.interpolate");
		_interp_splprep = PyObject_GetAttrString(_interp_module, "splprep");
		_interp_splev = PyObject_GetAttrString(_interp_module, "splev");
		Py_DECREF(_interp_module);	
	}
	
	dim[0] = nold;
	Xold = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, xold);
	dim[1] = ny;
	temp = PyArray_SimpleNewFromData(2, dim, NPY_DOUBLE, yold);
	Yold = PyArray_Transpose((PyArrayObject *)temp,NULL); Py_DECREF(temp);
	
	args = Py_BuildValue("(O)", Yold); Py_DECREF(Yold);
	kw = Py_BuildValue("{s:O, s:i, s:i}","u",Xold,"k",3,"s",0); Py_DECREF(Xold);
	temp = PyObject_Call(_interp_splprep, args, kw); Py_DECREF(args); Py_DECREF(kw);
	if (temp == NULL) return NULL;
	tck = PySequence_GetItem(temp, 0); Py_DECREF(temp);
	dim[0] = nnew;
	Xnew = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, xnew);
	args = Py_BuildValue("(O,O,i)",Xnew,tck,0); Py_DECREF(Xnew); Py_DECREF(tck);
	temp = PyObject_CallObject(_interp_splev, args); Py_DECREF(args);
	if (temp == NULL) return NULL;
	Ynew = PyArray_FromAny(temp, NULL,0,0,NPY_ARRAY_C_CONTIGUOUS,NULL); 
	Py_DECREF(temp); 
	
	// Really we want to output transpose(Ynew).
	y = (double *)PyArray_DATA((PyArrayObject *)Ynew);
	ynew = malloc(sizeof(double)*ny*nnew);
	for (i=0; i<ny; i++) {
		for (j=0; j<nnew; j++)
			ynew[j*ny+i] = y[j+i*nnew];
	}
	
	Py_DECREF(Ynew);
	return ynew;
}

double *monitorFunc(double t, int nx, double *x, double *y, double *c1) {
	double *m;
	double dydx[nx*the_model.nfields];
	int k, ny;

	m = malloc(sizeof(double)*nx);
	
	// Need to calculate dy.
	ny = the_model.nfields*2 + 2;
	for (k=0; k<the_model.nfields; k++) {
		calcDerivs(y, c1, nx, ny, k, dydx, the_model.nfields, k);
//		LOGMSG("vMonitorGain: %0.4e", vMonitorGain[k]);
	}
	
//	else if (monitor_type == CALL_BACK) 
	{
		// Make the input for the callback
		npy_intp ydim[2] = {nx,ny};
		PyObject *yobj = PyArray_SimpleNew(2, ydim, NPY_DOUBLE);
		PyObject *xobj = PyArray_SimpleNew(1, ydim, NPY_DOUBLE);
		double *yobj_data = (double *)PyArray_DATA((PyArrayObject *)yobj); 
		double *xobj_data = (double *)PyArray_DATA((PyArrayObject *)xobj); 
		memcpy(yobj_data, y, sizeof(double)*nx*ny);
		memcpy(xobj_data, x, sizeof(double)*nx);
		PyObject *arglist = Py_BuildValue("(dOO)", t, xobj, yobj);

		// Run the callback
		PyObject *result = PyObject_CallObject(monitor_callback, arglist);
		Py_DECREF(arglist);
		Py_DECREF(xobj);
		Py_DECREF(yobj);
		if (result == NULL) {
			free(m);
			return NULL;
		}

		// Assemble the result
		PyObject *mobj = PyArray_FROM_OTF(result, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
		assert( PyArray_SIZE((PyArrayObject *)mobj) == nx );
		memcpy(m, PyArray_DATA((PyArrayObject *)mobj), sizeof(double)*nx);
		Py_DECREF(mobj);
		Py_DECREF(result);
	}
	
	return m;
}

int dY_bubbles(double t, double *y, double *c1, double *c2, int nx, double *dY_out) {
	int i, k, ny;
	double ct, st, tt; // hyperbolic functions
	
	ny = 2*the_model.nfields+2;
	
	ct = cosh(t);
	st = sinh(t);
	tt = tanh(t);
	
	// calculate the spatial derivs.
	double *dy = malloc(sizeof(double)*nx*the_model.nfields);
	double *d2y = malloc(sizeof(double)*nx*the_model.nfields);
	double *alphaa = malloc(sizeof(double)*nx);
	double *dalphaa = malloc(sizeof(double)*nx);
	for (k=0; k<the_model.nfields; k++) {
		calcDerivs(y, c1, nx, ny, k, dy, the_model.nfields, k);
		calcDerivs(y, c2, nx, ny, k, d2y, the_model.nfields, k);
	}
	for (i=0; i<nx; i++)
		alphaa[i] = y[ny*i+(ny-2)] / y[ny*i+(ny-1)];
	calcDerivs(alphaa, c1, nx, 1, 0, dalphaa, 1, 0);

	// Calculate the potential and its gradient
	int errcode;
    // The potential and its gradient.
	double *V = malloc(sizeof(double)*nx);
    double *dV = malloc(sizeof(double)*nx*the_model.nfields);
	errcode = the_model.V(the_model.obj, nx, ny, y, V);
	if (errcode < 0) return errcode;
	errcode = the_model.dV(the_model.obj, nx, ny, y, dV);
	if (errcode < 0) return errcode;
	
	// Start calculating dY_out
	for (i=0; i<nx; i++) {
		double A, B, a, alpha;
		alpha = y[ny*i+ny-2];
		a = y[ny*i+ny-1];
		A = (tt+0.5/tt) - 0.5*alpha*alpha*(1./(ct*st) + 8*PI*tt*V[i]);
		B = 0.0;
		for (k=0; k<the_model.nfields; k++) {
			double Pi_k = y[ny*i+the_model.nfields+k];
			double Phi_k = dy[the_model.nfields*i+k]/ct;
			double dPhi_k = d2y[the_model.nfields*i+k]/ct;
			B += Phi_k*Phi_k + Pi_k*Pi_k;
			dY_out[ny*i+k] = alphaa[i]*Pi_k; // dphi/dN
			dY_out[ny*i+the_model.nfields+k] = -(tt+2.0/tt)*Pi_k
				+ (dalphaa[i]*Phi_k + alphaa[i]*dPhi_k)/ct
				- alpha*a*dV[the_model.nfields*i+k]; // dPi/dN
		}
		B *= 2*PI*alphaa[i]*alphaa[i]*tt;
		dY_out[ny*i+ny-2] = alpha*(A+B); // dalpha/dN
		dY_out[ny*i+ny-1] = a*(-A+B); // da/dN
	}

    free(dy);
    free(d2y);
    free(alphaa);
    free(dalphaa);
    free(V);
    free(dV);

	return 0;
}

/*double speedOfLight(double N) {
	return 1./cosh(N);
}*/

double evolveBubbles(double *x0, double *y0, int nx0, double t0, double tmax, double alphaMax, 
					 int exactTmax, int growBounds, int overwritefile,
					 double **xout, double **yout, int *nxout) {
	int i,k, ny, nx, nxnew, Nmax, nsteps, err, nreg, itout;
	double *c1, *c2, *x, *xnew, *y, *ynew, *dx, *m;
	double x0min, x0max, xmin, xmax, mindx, t, dt, t_write;
	Region *R, *R0 = NULL;
	FILE *fieldsFilePtr, *chrisFilePtr;
    clock_t start_clock, stop_clock;
    PyObject *stdout_obj;

    stdout_obj = PySys_GetObject("stdout");

//	int minRegionWidth = 16;
	double gridUniformity = M_LN2 / minRegionWidth; 
		// zero is perfectly uniform, large number means not (necessarily) uniform.
		// (This last line ensures that the grid density never drops by more than
		// a factor of two in minRegionWidth points. If this number were much larger,
		// you'd have the problem that solution in the fine regions could evolve into
		// the coarse regions before we have a chance to re-adapt the grid.)
	ny = 2*the_model.nfields+2;
	nx = nx0;

	fieldsFilePtr = fname ? fopen(fname, overwritefile ? "w" : "a") : NULL;
	chrisFilePtr = fchrisname ? fopen(fchrisname, overwritefile ? "w" : "a") : NULL;
	*xout = *yout = NULL;
	
	// The first thing we want to do is copy the initial data.
	// This is important because we free the data from the previous iteration,
	// and we don't want to free the initial data (which this loop shouldn't control).
	x = malloc(sizeof(double)*nx);
	y = malloc(sizeof(double)*nx*ny);
	memcpy(x,x0, sizeof(double)*nx);
	memcpy(y,y0, sizeof(double)*nx*ny);
	if (growBounds) {
		x0min = x[0] + 2*atan(tanh(t0/2.0));
		x0max = x[nx-1] - 2*atan(tanh(t0/2.0));
	}
	else {
		x0min = x[0];
		x0max = x[nx-1];
	}
		
	// Get the derivative coefficients. Necessary for calculating grid density
	// before we make the first grid.
	c1 = malloc(sizeof(double)*5*nx);
	c2 = malloc(sizeof(double)*5*nx);
	calcDerivCoefs(x, nx, c1, c2);
	
	t = t0;
	t_write = t0;
	if (ntout > 0 && tout) {
		itout = 1;
		while (itout < ntout && tout[itout] < t0) 
			itout++;
		if (itout < ntout)
			t_write = tout[itout-1]; // The first write is *before* t0, so we write immediately.
		else
			t_write = tmax*2; // never write
	}
	
	start_clock = clock();
    stop_clock = clock() + max_run_time * CLOCKS_PER_SEC;
	
	// begin our grand loop.
	while (t < tmax) {
		// Check for a keyboard interrupt.
		if (PyErr_Occurred()) {
			LOGMSG("Error occured!!!!");
			goto out;
		}
		
		// Check for alphaMax
		for (i=0; i<nx; i++) {
			if (y[i*ny+ny-2] > alphaMax) {
				LOGMSG("Reached alphaMax");
				goto out;
			}
		}
        if (t < check_run_time_before_N && clock() > stop_clock) {
            LOGMSG("Reached stop_clock. Aborting the simulation.");
            goto out;
        }
		
		// Make the new grid.
		LOGMSG("Making the grid.");
		if (growBounds) {
			// This way of extending the grid keeps the maximum space-like separation from the center of the bubble constant...
		//	xmin = -acos(cos(x0min)/cosh(t)); 
		//	xmax = x0max + (x0min - xmin)
			// ...while this way includes everything that was causally connected to the initial conditions.
			xmin = x0min - 2*atan(tanh(t/2.0));
			xmax = x0max + 2*atan(tanh(t/2.0));
		}
		else {
			xmin = x0min;
			xmax = x0max;
		}
		
		m = monitorFunc(t, nx, x, y, c1);
		if (m == NULL) {
			LOGMSG("Error in monitorFunc.");
			goto out;
		}
		xnew = remakeGrid(x, m, nx, gridUniformity, xmin, xmax, &nxnew);
		free(m);
		if (xnew == NULL) {
			LOGMSG("Error in remakeGrid.");
			goto out;
		}
      /*  for (i=1; i<nxnew; i++) {
            if (xnew[i] < xnew[i-1]) {
                LOGMSG("Error in remakeGrid.");
                goto out;
            }
        }*/
		ynew = interpGrid_scipy(x, nx, y, ny, xnew, nxnew);
		if (ynew == NULL) {
			LOGMSG("Error in interpGrid_scipy.");
			free(xnew);
			goto out;
		}
		if (growBounds) {
			// Set the boundaries of the grid where we've over-extended
			for (i=0; xnew[i] < x[0] || i < 2; i++) { 
				// For some reason we get weird boundary effects (only on the left) unless we make absolute certain
				// to zero the first two.
				for (k=0; k<ny; k++)
					ynew[ny*i+k] = y0[k];
			}
			for (i=nxnew-1; xnew[i] > x[nx-1] || i > nxnew-3; i--) {
				for (k=0; k<ny; k++)
					ynew[ny*i+k] = y0[k];
			}
		}
		LOGMSG("uniformity=%0.4f, xmin,xmax=%0.3f,%0.3f, nxnew=%i", gridUniformity,xmin,xmax,nxnew);
		// Free the old grid.
		free(x); free(y);
		x = xnew; y = ynew; nx = nxnew;
		// Recalculate the deriv coefs.
		LOGMSG("Calculating the coefs");
		free(c1); free(c2);
		c1 = malloc(sizeof(double)*5*nx);
		c2 = malloc(sizeof(double)*5*nx);
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
		LOGMSG("nx=%i, mindx=%0.2e", nx, mindx);
		R0 = makeRegions(dx, nx, mindx, Nmax0, /*slop =*/ 0.2, minRegionWidth);
		fillRegions(R0, y, ny);
		free(dx);
		// Find the coarsest region.
		Nmax = 0;
		R = R0;
		nreg = 0;
		while (R) {
			if (R->N > Nmax) Nmax = R->N;
			//		LOGMSG("a region, N=%i", R->N);
			R = R->nextReg;
			nreg++;
		}
		nsteps = pow2N(Nmax);
		//dt = mindx * cfl / speedOfLight(t);
		{
			// We need to find the maximum speed of light. This sets the minimum time step.
			double maxSpeedOfLight, speedOfLight;
			maxSpeedOfLight = speedOfLight = y[(ny-2)] / y[(ny-1)];
			for (i=1; i<nx; i++) {
				speedOfLight = y[ny*i+(ny-2)] / y[ny*i+(ny-1)];
				if (maxSpeedOfLight < speedOfLight)
					maxSpeedOfLight = speedOfLight;
			}
			LOGMSG("maxSpeedOfLight: %f", maxSpeedOfLight);
			maxSpeedOfLight /= cosh(t);
			dt = mindx * cfl / maxSpeedOfLight;
		}
		if (1) {
			// We also need to figure out the smallest period of oscillation, and make sure the time steps
			// are smaller than that.
			// Frequency of oscillations goes like omega^2 = alpha^2 d^2V
			double maxAlpha = 1.0;
			double osc_period;
			for (i=0; i<nx; i++)
				if (maxAlpha < y[ny*i+(ny-2)]) maxAlpha = y[ny*i+(ny-2)];
			osc_period = 2*PI / (maxAlpha * highestMass);
			if (dt * nsteps * minStepsPerPeriod > osc_period)
				dt = osc_period/(nsteps*minStepsPerPeriod);
			LOGMSG("period of oscillations: %f", osc_period);
			LOGMSG("dt from oscillations: %0.3e", osc_period/(nsteps*minStepsPerPeriod));
		}
		if (1) {
			// Finally, the step size shouldn't be much smaller than the characteristic change in alpha and a.
			// For now, approximate this just by the first term.
			double t_metric = 1./1.5; // really tanh N + 1/(2 tanh N), but this only matters in the large N limit anyways.
			if (dt * nsteps * minStepsPerPeriod > t_metric)
				dt = t_metric/(nsteps*minStepsPerPeriod);
		}
		LOGMSG("About to evolve %i regions: Nmax=%i, dt=%0.2e, t=%0.3f", nreg, Nmax, dt, t);
		for (i=0; i<stepsPerRegrid; i++) {
			// Evolve the regions
			//	LOGMSG("Evolving the regions");
			if (exactTmax && t + dt*nsteps >= tmax) {
				dt = (tmax-t)/nsteps;
				err = evolveRegions(R0, dt, t, nsteps, ny, c1, c2, &dY_bubbles);
				t = tmax;
			}
			else if (exact_tout && t + dt*nsteps >= t_write && t_write > t) {
				dt = (t_write-t)/nsteps;
				err = evolveRegions(R0, dt, t, nsteps, ny, c1, c2, &dY_bubbles);
				t = t_write;
			}
			else {
				err = evolveRegions(R0, dt, t, nsteps, ny, c1, c2, &dY_bubbles);
				t += dt*nsteps;
			}
			if (err == -1) return -1;

			// extract the y data from the regions.
			if (t >= t_write && (fieldsFilePtr || chrisFilePtr) && tout) {
				free(y);
				y = dataFromRegions(R0, ny, NULL);
		//		writeToFile(fieldsFilePtr, nx, ny, xres, t, x, y, c1, c2);
				writeFieldsAndChristoffelsToFile(fieldsFilePtr, chrisFilePtr, nx, xres, t, x, y, c1, c2);
				// advance t_write
				if (ntout > 0) {
					while (itout < ntout && t >= tout[itout])
						itout++;
					t_write = itout < ntout ? tout[itout] : tmax*2;
				}
				else
					t_write = t + *tout;

                { // Write to stdout
                    char s[1000];
                    PyObject *rval;
                    snprintf(
                        s, 1000,
                        "Writing simulation to file:\n "
                        "  Nmax=%i, dt=%0.2e, t=%0.3f, nx=%i\n", 
                        Nmax, dt, t, nx);
                    rval = PyObject_CallMethod(stdout_obj, "write", "(s)", s);
                    Py_XDECREF(rval);       
                }   
			}
		}
		free(y);
		y = dataFromRegions(R0, ny, NULL);
		deallocRegions(R0);
		LOGMSG("t=%f, tmax=%f", t,tmax);
		LOGMSG("end loop\n");
	}

out:
	// End of loop, return the data.
	*xout = x;
	*yout = y;
	*nxout = nx;
	
	LOGMSG("return nx = %i",nx);	
	{
		char s[1000];
		PyObject *rval;
		snprintf(
			s, 1000,
			"End simulation. Elapsed time = %0.3f. "
			"Nmax=%i, dt=%0.2e, t=%0.3f, nx=%i\n", 
			(clock()-start_clock)*1.0/CLOCKS_PER_SEC,
			Nmax, dt, t, nx);
		rval = PyObject_CallMethod(stdout_obj, "write", "(s)", s);
		Py_XDECREF(rval);		
	}	
//out:
	free(c1); free(c2);	
	if (fieldsFilePtr) fclose(fieldsFilePtr);
	if (chrisFilePtr) fclose(chrisFilePtr);
		
	return t;
}	

#pragma mark -
#pragma mark File I/O
// ----------------------------------------------------------------------
// File I/O
// ----------------------------------------------------------------------

void writeFieldsAndChristoffelsToFile(FILE *fFields, FILE *fChris, int32_t nx, int32_t skipx,
									  double t, double *x, double *y, double *c1, double *c2) {
	int32_t i, j, k, ny;
	double ct, st, tt; // hyperbolic functions
		
	ny = 2*the_model.nfields+2;
	
	ct = cosh(t);
	st = sinh(t);
	tt = tanh(t);
	
	double *dydx = malloc(sizeof(double)*nx*ny);
	double *d2ydx2 = malloc(sizeof(double)*nx*ny);
	for (k=0; k<ny; k++) {
		calcDerivs(y, c1, nx, ny, k, dydx, ny, k);
		calcDerivs(y, c2, nx, ny, k, d2ydx2, ny, k);
	}
	double *dydN = malloc(sizeof(double)*nx*ny);
	double *d2alphadN2_arr = malloc(sizeof(double)*nx);
	double *d2adN2_arr = malloc(sizeof(double)*nx);
	
/*	double alphaa_arr[nx], dalphaadx_arr[nx];
	for (i=0; i<nx; i++)
		alphaa_arr[i] = y[i*ny+ny-2]/y[i*ny+ny-1];
	calcDerivs(alphaa_arr, c1, nx, 1, 0, dalphaadx_arr, 1, 0);
*/	
    // The potential and its gradient.
	double *V = malloc(sizeof(double)*nx);
    double *dV = malloc(sizeof(double)*nx*the_model.nfields);
	the_model.V(the_model.obj, nx, ny, y, V);
	the_model.dV(the_model.obj, nx, ny, y, dV);
		
	// Start calculating dY_out
	for (i=0; i<nx; i++) {
		double alpha = y[ny*i+ny-2];
		double a = y[ny*i+ny-1];
		double A = (tt+0.5/tt) - 0.5*alpha*alpha*(1./(ct*st) + 8*PI*tt*V[i]);
		double B = 0.0;
		double dAdN = 0.0, dBdN = 0.0;
		
		double alphaa = alpha/a;
		double dalphaadx = alphaa*(dydx[ny*i+ny-2]/alpha - dydx[ny*i+ny-1]/a);
	//	double alphaa = alpha/a;
	//	double dalphaadx = dalphaadx_arr[i];
		
		for (k=0; k<the_model.nfields; k++) {
		//	double phi = y[ny*i+k];
			double Pi = y[ny*i+the_model.nfields+k];
			double dPidx = dydx[ny*i+the_model.nfields+k];
			double Phi = dydx[ny*i+k]/ct;
			double dPhidx = d2ydx2[ny*i+k]/ct;
			
			double dphidN = alphaa * Pi;
			double dPhidN = -tt*Phi + (Pi*dalphaadx + dPidx*alphaa)/ct;
			double dPidN = -(tt+2.0/tt)*Pi + (dalphaadx*Phi + alphaa*dPhidx)/ct - alpha*a*dV[the_model.nfields*i+k];

			B += Phi*Phi + Pi*Pi;
			dAdN += dV[the_model.nfields*i+k]*dphidN;
			dBdN += dPhidN*Phi + dPidN*Pi;
			
			dydN[ny*i+k] = dphidN;
			dydN[ny*i+the_model.nfields+k] = dPidN;
		}
		dAdN *= 8*PI*tt;
		dAdN += 8*PI*V[i]/(ct*ct);
		dAdN -= 1./(ct*ct) + 1./(st*st);
		dAdN *= - 0.5*alpha*alpha;
		
		double PiTanhAlphaA2 = PI*tt*alphaa*alphaa;
		dBdN *= 4*PiTanhAlphaA2;
		dBdN += 8*PiTanhAlphaA2*B*A; // where 2*A = dalphadN/alpha - dadN/a
		dBdN += (2*PI/(ct*ct))*alphaa*alphaa*B;
		B *= 2*PiTanhAlphaA2;

		double dalphadN = alpha*(A+B);
		double dadN = a*(-A+B);
		dydN[ny*i+ny-2] = dalphadN;
		dydN[ny*i+ny-1] = dadN;
		
		dAdN += (1./(ct*ct) - 0.5/(st*st)) - alpha*dalphadN*(1./(ct*st) + 8*PI*tt*V[i]);
		d2alphadN2_arr[i] = dalphadN*(A+B) + alpha*(dAdN+dBdN);
		d2adN2_arr[i] = dadN*(-A+B) + a*(-dAdN+dBdN);
	}	
	
	// Get the mixed space/time derivs.
	double *d2ydNdx = malloc(sizeof(double)*nx*ny);
	for (k=0; k<ny; k++)
		calcDerivs(dydN, c1, nx, ny, k, d2ydNdx, ny, k);
	
	double *G = malloc(sizeof(double)*nx*6);
    double *dGdN = malloc(sizeof(double)*nx*6);
	// Now calculate the Christoffel symbols.
	for (i=0; i<nx; i++) {
		int ialpha = ny*i+ny-2;
		double alpha = y[ialpha];
		double dalphadx = dydx[ialpha];
		double dalphadN = dydN[ialpha];
		double d2alphadNdx = d2ydNdx[ialpha];
		double d2alphadN2 = d2alphadN2_arr[i];
		
		int ia = ny*i+ny-1;
		double a = y[ia];
		double dadx = dydx[ia];
		double dadN = dydN[ia];
		double d2adNdx = d2ydNdx[ia];
		double d2adN2 = d2adN2_arr[i];
		
		G[i*6+0] = dalphadN/alpha; // GNNN
		G[i*6+1] = dalphadx/alpha; // GNNx
		G[i*6+2] = (a*st+dadN*ct)*a*ct/(alpha*alpha); // GNxx
		G[i*6+3] = alpha*dalphadx/(a*a*ct*ct); // GxNN
		G[i*6+4] = tt + dadN/a; // GxNx
		G[i*6+5] = dadx/a; // Gxxx
		
		dGdN[i*6+0] = d2alphadN2/alpha - G[i*6+0]*G[i*6+0];
		dGdN[i*6+1] = d2alphadNdx/alpha - dalphadx*dalphadN/(alpha*alpha);
		dGdN[i*6+2] = (a*( (a+d2adN2)*ct*ct + 4*dadN*ct*st + a*st*st ) + dadN*dadN*ct*ct) / (alpha*alpha);
		dGdN[i*6+2] -= 2*G[i*6+2]*dalphadN/alpha;
		dGdN[i*6+3] = (dalphadN*dalphadx + alpha*d2alphadNdx)/(a*a*ct*ct);
		dGdN[i*6+3] -= 2*G[i*6+3]*(tt+dadN/a);
		dGdN[i*6+4] = 1/(ct*ct) + d2adN2/a - dadN*dadN/(a*a);
		dGdN[i*6+5] = d2adNdx/a - dadN*dadx/(a*a);
	}
	
	// And calculate the spatial derivs of the Christoffels.
    double *dGdx = malloc(sizeof(double)*6*nx);
    double *d2GdNdx = malloc(sizeof(double)*6*nx);
	for (k=0; k<6; k++) {
		calcDerivs(G, c1, nx, 6, k, dGdx, 6, k);
		calcDerivs(dGdN, c1, nx, 6, k, d2GdNdx, 6, k);
	}
		
	// Now write everything to file.
	if (skipx < 1) skipx = 1;
	int32_t nxout = (nx-1)/skipx + 1;
	int32_t nderivs = 3;
	int32_t nG = 6;

	if (fFields) {
		fwrite(&nxout, sizeof(int32_t), 1, fFields);
		fwrite(&ny, sizeof(int32_t), 1, fFields);
		fwrite(&nderivs, sizeof(int32_t), 1, fFields);
		fwrite(&t, sizeof(double), 1, fFields);
	}
	if (fChris) {
		fwrite(&nxout, sizeof(int32_t), 1, fChris);
		fwrite(&nG, sizeof(int32_t), 1, fChris);
		fwrite(&nderivs, sizeof(int32_t), 1, fChris);
		fwrite(&t, sizeof(double), 1, fChris);
	}
	
	if (skipx == 1) { // don't need to shorten the arrays
		if (fFields) {
			fwrite(x, sizeof(double), nxout, fFields);
			fwrite(y, sizeof(double), nxout*ny, fFields);
			fwrite(dydN, sizeof(double), nxout*ny, fFields);
			fwrite(dydx, sizeof(double), nxout*ny, fFields);
			fwrite(d2ydNdx, sizeof(double), nxout*ny, fFields);
		}
		if (fChris) {
			fwrite(x, sizeof(double), nxout, fChris);
			fwrite(G, sizeof(double), nxout*nG, fChris);
			fwrite(dGdN, sizeof(double), nxout*nG, fChris);
			fwrite(dGdx, sizeof(double), nxout*nG, fChris);
			fwrite(d2GdNdx, sizeof(double), nxout*nG, fChris);
		}
	}
	else { // Need to shorten the arrays
		double *x_out = malloc(sizeof(double)*nxout);
		double *y_out = malloc(sizeof(double)*nxout*ny);
		double *dydN_out = malloc(sizeof(double)*nxout*ny);
		double *dydx_out = malloc(sizeof(double)*nxout*ny);
		double *d2ydNdx_out = malloc(sizeof(double)*nxout*ny);
		double *G_out = malloc(sizeof(double)*nxout*nG);
		double *dGdN_out = malloc(sizeof(double)*nxout*nG);
		double *dGdx_out = malloc(sizeof(double)*nxout*nG);
		double *d2GdNdx_out = malloc(sizeof(double)*nxout*nG);
		
		for (i=0,j=0; j<nxout; j++, i+=skipx) {
			x_out[j] = x[i];
			for (k=0; k<ny; k++) {
				y_out[j*ny+k] = y[i*ny+k];
				dydN_out[j*ny+k] = dydN[i*ny+k];
				dydx_out[j*ny+k] = dydx[i*ny+k];
				d2ydNdx_out[j*ny+k] = d2ydNdx[i*ny+k];
			}
			for (k=0; k<nG; k++) {
				G_out[j*nG+k] = G[i*nG+k];
				dGdN_out[j*nG+k] = dGdN[i*nG+k];
				dGdx_out[j*nG+k] = dGdx[i*nG+k];
				d2GdNdx_out[j*nG+k] = d2GdNdx[i*nG+k];
			}
		}
		if (fFields) {
			fwrite(x_out, sizeof(double), nxout, fFields);
			fwrite(y_out, sizeof(double), nxout*ny, fFields);
			fwrite(dydN_out, sizeof(double), nxout*ny, fFields);
			fwrite(dydx_out, sizeof(double), nxout*ny, fFields);
			fwrite(d2ydNdx_out, sizeof(double), nxout*ny, fFields);
		}
		if (fChris) {
			fwrite(x_out, sizeof(double), nxout, fChris);
			fwrite(G_out, sizeof(double), nxout*nG, fChris);
			fwrite(dGdN_out, sizeof(double), nxout*nG, fChris);
			fwrite(dGdx_out, sizeof(double), nxout*nG, fChris);
			fwrite(d2GdNdx_out, sizeof(double), nxout*nG, fChris);
		}
        free(x_out);
        free(y_out);
        free(dydN_out);
        free(dydx_out);
        free(d2ydNdx_out);
        free(G_out);
        free(dGdN_out);
        free(dGdx_out);
        free(d2GdNdx_out);
	}
	if (fFields)
		fflush(fFields);
	if (fChris)
		fflush(fChris);
    free(dydx);
    free(d2ydx2);
    free(dydN);
    free(d2alphadN2_arr);
    free(d2adN2_arr);
    free(V);
    free(dV);
    free(d2ydNdx);
    free(G);
    free(dGdN);
    free(dGdx);
    free(d2GdNdx);

	return;
}

PyMethodDef readFromFile_methdef = {
    "readFromFile", (PyCFunction)readFromFile_toPy, METH_VARARGS, 
"readFromFile(file_name)\n\n"
"Reads a simulation file.\n"
"\n"
"The input should be the name of the file. \n"
"The output will be a list of time slices. Each slice is a tuple consisting\n"
"of (*N, x, Y*), where *N* is the time value of the slice, *x* is an array\n"
"of grid points along the slice, and *Y* is an array containing the fields,\n"
"metric functions, and their derivatives along the slice:\n"
"\n"
"    Y = [y, dy/dN, dy/dx, d^2y/dNdx]\n"
"\n"
"Depending on whether a fields file or Christoffel file was loaded,\n"
"the subarray *y* will either contain the fields, their conjugate momenta, and\n"
"the metric functions, or the Christoffel symbols \n"
":math:`\\Gamma_{NN}^N`,\n"
":math:`\\Gamma_{Nx}^N`,\n"
":math:`\\Gamma_{xx}^N`,\n"
":math:`\\Gamma_{NN}^x`,\n"
":math:`\\Gamma_{Nx}^x`, and\n"
":math:`\\Gamma_{xx}^x`.\n"
};

PyObject *readFromFile_toPy(PyObject *self, PyObject *args) {
	PyObject *X, *Y, *listOut, *dataTuple;
	npy_intp dimensions[3];
	int32_t nx, ny, nderivs;
	double t, t_last=-1;
	
	int numItems = 0, maxItems = 10000000; // don't read more than 10mil, for sanity
	
	char *fname;
	FILE *fptr;
	
	if (!PyArg_ParseTuple(args, "s", &fname) ){
		return NULL;
	}

	fptr = fopen(fname, "r");
	if (fptr == NULL) {
		PyErr_Format(PyExc_ValueError, "Cannot open the file '%s'.", fname);
		return NULL;
	}

	listOut = PyList_New(0);


	while (fread(&nx, sizeof(int32_t), 1, fptr) && numItems < maxItems) {
		fread(&ny, sizeof(int32_t), 1, fptr);
		fread(&nderivs, sizeof(int32_t), 1, fptr);
		fread(&t, sizeof(double), 1, fptr);
		dimensions[0] = nx;
		X = PyArray_SimpleNew(1, dimensions, NPY_DOUBLE);
		dimensions[0] = nderivs+1;
		dimensions[1] = nx;
		dimensions[2] = ny;
		Y = PyArray_SimpleNew(3, dimensions, NPY_DOUBLE);
		fread(PyArray_DATA((PyArrayObject *)X), sizeof(double), nx, fptr);
		fread(PyArray_DATA((PyArrayObject *)Y), sizeof(double), nx*ny*(nderivs+1), fptr);
		dataTuple = Py_BuildValue("(dOO)", t,X,Y);
		Py_DECREF(X); Py_DECREF(Y);
        if (t != t_last)
		    PyList_Append(listOut, dataTuple);
		Py_DECREF(dataTuple);
		numItems++;
        t_last=t;
	}
	fclose(fptr);
	
	return listOut;
}

