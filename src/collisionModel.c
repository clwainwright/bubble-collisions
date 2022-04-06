
#include <Python.h>

#define NO_IMPORT_ARRAY

#include <arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "collisionModel.h"
#include "bubbleEvolution.h" // for setting the potential

#define PI 3.141592653589793

#define VERBOSE 1
#if VERBOSE
#define BLURT() printf("%s(%d) -- %s()\n", __FILE__, __LINE__, __func__)
#define LOGMSG(...) printf("%s(%d): ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n")
#else
#define BLURT() ;
#define LOGMSG(...) ;
#endif

#define ERRMSG(...) printf("ERROR -- %s(%d): ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n")

#pragma mark -
#pragma mark Globals
// --------------------------------------
// Globals
// --------------------------------------

int nfields = 0;
double *(*V_func)(int, int, double *) = NULL;
double *(*dV_func)(int, int, double *) = NULL;


#pragma mark -
#pragma mark Wrapper functions
// --------------------------------------
// Wrapper functions
// --------------------------------------

PyObject *V_pywrapper(PyObject *self, PyObject *args) {
	PyObject *Yin, *Vout;
	PyArrayObject *Yarr;
	double *V;
	int i, nx, ny, nd;
	npy_intp *outDim;
	
    if (!PyArg_ParseTuple(args, "O", &Yin))
		return NULL;
	
    Yarr = (PyArrayObject *) PyArray_FROM_OTF(Yin, NPY_DOUBLE, NPY_IN_ARRAY);
    if (Yarr == NULL || Yarr->nd < 1) return NULL;
	
	nx = 1;
	nd = Yarr->nd;
	outDim = malloc(sizeof(npy_intp)*(nd - 1));
	for (i=0; i < nd - 1; i++) {
		nx *= Yarr->dimensions[i];
		outDim[i] = Yarr->dimensions[i];
	}
	ny = Yarr->dimensions[nd-1];
	V = V_func(nx, ny, (double *)Yarr->data);
	
	Vout = PyArray_SimpleNew(nd-1, outDim, NPY_DOUBLE);
	memcpy(PyArray_DATA(Vout), V, nx*sizeof(double));
	
	free(V);
	Py_DECREF(Yarr);
	
	return Vout;
}

PyObject *dV_pywrapper(PyObject *self, PyObject *args) {
	PyObject *Yin, *dVout;
	PyArrayObject *Yarr;
	double *dV;
	int i, nx, ny, nd;
	npy_intp *outDim;
	
    if (!PyArg_ParseTuple(args, "O", &Yin))
		return NULL;
	
    Yarr = (PyArrayObject *) PyArray_FROM_OTF(Yin, NPY_DOUBLE, NPY_IN_ARRAY);
    if (Yarr == NULL || Yarr->nd < 1) return NULL;
	
	nx = 1;
	nd = Yarr->nd;
	outDim = malloc(sizeof(npy_intp)*(nd - 1));
	for (i=0; i < nd - 1; i++) {
		nx *= Yarr->dimensions[i];
		outDim[i] = Yarr->dimensions[i];
	}
	ny = Yarr->dimensions[nd-1];
	outDim[nd-1] = nfields;
	dV = dV_func(nx, ny, (double *)Yarr->data);
	
	dVout = PyArray_SimpleNew(nd, outDim, NPY_DOUBLE);
	memcpy(PyArray_DATA(dVout), dV, nx*nfields*sizeof(double));
	
	free(dV);
	Py_DECREF(Yarr);
	
	return dVout;
}

PyObject *V_1d_pywrapper(PyObject *self, PyObject *args) {
	PyObject *Yin, *Vout;
	double *V;
	int nx;
		
    if (!PyArg_ParseTuple(args, "O", &Yin))
		return NULL;
	
    Yin = PyArray_FROM_OTF(Yin, NPY_DOUBLE, NPY_IN_ARRAY);
    if (Yin == NULL) return NULL;
	
	nx = PyArray_SIZE(Yin);
	if (nx == 0) {
		return Yin; // If the input is an empty array, just return an empty array.
	}
	
	V = V_func(nx, 1, (double *)PyArray_DATA(Yin));
	if (PyArray_NDIM(Yin) == 0) {
		Vout = Py_BuildValue("d", *V);
	}
	else {
		Vout = PyArray_SimpleNew(PyArray_NDIM(Yin), PyArray_DIMS(Yin), NPY_DOUBLE);
		memcpy(PyArray_DATA(Vout), V, nx*sizeof(double));
	}	
	
	free(V);
	Py_DECREF(Yin);
	
	return Vout;
}

PyObject *dV_1d_pywrapper(PyObject *self, PyObject *args) {
	PyObject *Yin, *dVout;
	double *dV;
	int nx;
	
    if (!PyArg_ParseTuple(args, "O", &Yin))
		return NULL;
	
    Yin = PyArray_FROM_OTF(Yin, NPY_DOUBLE, NPY_IN_ARRAY);
    if (Yin == NULL) return NULL;
	
	nx = PyArray_SIZE(Yin);
	if (nx == 0) {
		return Yin; // If the input is an empty array, just return an empty array.
	}
	
	dV = dV_func(nx, 1, (double *)PyArray_DATA(Yin));
	if (PyArray_NDIM(Yin) == 0) {
		dVout = Py_BuildValue("d", *dV);
	}
	else {
		dVout = PyArray_SimpleNew(PyArray_NDIM(Yin), PyArray_DIMS(Yin), NPY_DOUBLE);
		memcpy(PyArray_DATA(dVout), dV, nx*sizeof(double));
	}	
	
	free(dV);
	Py_DECREF(Yin);
	
	return dVout;
}

PyObject *dV_1d_pywrapper(PyObject *self, PyObject *args);


PyObject *nfields_pywrapper(PyObject *self, PyObject *args) {
	return Py_BuildValue("i", nfields);
}


#pragma mark -
#pragma mark Model 1
// --------------------------------------
// Model 1
// V = scale*[(x^2+y^2)( (1+tilt)(x-1)^2 + (1-tilt)(y-1)^2 - c) + offset]
// where x and y have been rescaled by Mpl
// --------------------------------------

struct Model1_Params {
	double tilt;
	double c;
	double offset;
	double Mpl;
	double scale;
	
	double dxx, dxy, dyy; // Second derivs at the minimum.
};
struct Model1_Params model1_params;

double *V_model1(int nx, int ny, double *Y) {
	double x,y;
	double *V;
	int i;
	double s = model1_params.scale;
	double t = model1_params.tilt;
	double c = model1_params.c;
		
	V = malloc(nx*sizeof(double));
	for (i=0; i<nx; i++) {
		x = Y[ny*i] * model1_params.Mpl;
		y = Y[ny*i+1] * model1_params.Mpl;
		V[i] = s * ( (x*x+y*y) * ( (1+t)*(x-1)*(x-1) + (1-t)*(y-1)*(y-1) - c)  +  model1_params.offset ); 
	}
	return V;
}

double *dV_model1(int nx, int ny, double *Y) {
	double x,y;
	double *dV;
	int i;
	double s = model1_params.scale*model1_params.Mpl;
	double t = model1_params.tilt;
	double c = model1_params.c;
	
	dV = malloc(2*nx*sizeof(double));
	for (i=0; i<nx; i++) {
		x = Y[ny*i] * model1_params.Mpl;
		y = Y[ny*i+1] * model1_params.Mpl;
		dV[2*i] = 2*x * ( (1+t)*(x-1)*(x-1) + (1-t)*(y-1)*(y-1) - c);
		dV[2*i] += (x*x+y*y) * ( (1+t)*(x-1)*2 );
		dV[2*i] *= s;
		dV[2*i+1] = 2*y * ( (1+t)*(x-1)*(x-1) + (1-t)*(y-1)*(y-1) - c);
		dV[2*i+1] += (x*x+y*y) * ( (1-t)*(y-1)*2 );
		dV[2*i+1] *= s;
	}
	return dV;
}

PyObject *setModel1(PyObject *self, PyObject *args, PyObject *keywds) {
	// Input is barrier, tilt, offset, Mpl
	double barrier, tilt, offset;
	double Mpl = -1;
	
    static char *kwlist[] = {"barrier", "tilt", "offset", "Mpl", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ddd|d", kwlist, &barrier, &tilt, &offset, &Mpl))
		return NULL;
	
	model1_params.tilt = tilt;
	model1_params.c = 2*(1.0-barrier); // This definition is just to keep things somewhat consistent with testModels.py
	model1_params.offset = offset;
	if (Mpl > 0) {
		model1_params.Mpl = Mpl;
		model1_params.scale = 1.0 / ((8*PI*offset/3.)); // scale = 1/(Mpl*Hf)^2; Hf^2 = (8pi/3Mpl^2) V(0,0)
	}
	else {
		model1_params.Mpl = 1.0;
		model1_params.scale = 1.0;
	}

	// Set these to the active functions in this file
	V_func = &V_model1;
	dV_func = &dV_model1;
	
	// and also set them as the active functions in the bubbleEvolution file
	nfields = 2;
	setPotential(nfields, V_func, dV_func );

    Py_INCREF(Py_None);
    return Py_None;	
}


#pragma mark -
#pragma mark Model L2
// --------------------------------------
// Model L2
// This is the large-field inflation model used in Johnson, Peiris and Lehner.
// All parameters are fixed, but we still need to call setModelL2 to normalize it by the false vacuum
// Hubble constant.
// --------------------------------------

struct ModelL2_Params {
	double p1, p2, pT, pj, pm; // field values that define the edges of the separate pieces.
	double a1, a2;
	double M; // mass scale
	double M4C1, M4C2, VT2; // various offsets
	double m21, m22;
	double rescale; // for rescaling by the hubble constant
	int centralOnly;
};
struct ModelL2_Params modelL2_params = {
.p1 = -0.0026936953125000092,
.p2 = 0.0023907421875000072,
.pT = 0.0073689843750000114,
.pj = 0.015,
.pm = 2.75,
.a1 = 0.5,
.a2 = 0.75,
.M = 0.00345,
.M4C1 = 2.8333901249999997e-10,
.M4C2 = 2.7862010605745613e-10,
.VT2 = 1.7826771785690903e-10,
.m21 = 1.703542507086937e-08,
.m22 = 4.7531113306881428e-11,
.rescale = 1.0,
.centralOnly = 0
};

double *V_modelL2(int nx, int ny, double *y) {
	int i;
	double *V = malloc(sizeof(double)*nx);
	double M = modelL2_params.M;
	
	for (i=0; i<nx; i++) {
		double p = y[i*ny];
		if (p <= 0.0 && !modelL2_params.centralOnly) {
			p -= modelL2_params.p1;
			V[i] = -0.5*M*M*p*p + modelL2_params.a1*M*p*p*p/3.0 + 0.25 * p*p*p*p + modelL2_params.M4C1;
		}
		else if (p <= modelL2_params.pT || modelL2_params.centralOnly) {
			p -= modelL2_params.p2;
			V[i] = -0.5*M*M*p*p - modelL2_params.a2*M*p*p*p/3.0 + 0.25 * p*p*p*p + modelL2_params.M4C2;
		}
		else if (p <= modelL2_params.pj) {
			p -= modelL2_params.pT;
			V[i] = -0.5*modelL2_params.m21*p*p + modelL2_params.VT2;
		}
		else {
			p -= modelL2_params.pm;
			V[i] = 0.5*modelL2_params.m22*p*p;
		}
		V[i] *= modelL2_params.rescale;
	}
	
	return V;
}

double *dV_modelL2(int nx, int ny, double *y) {
	int i;
	double *dV = malloc(sizeof(double)*nx);
	double M = modelL2_params.M;

	for (i=0; i<nx; i++) {
		double p = y[i*ny];
		if (p <= 0.0 && !modelL2_params.centralOnly) {
			p -= modelL2_params.p1;
			dV[i] = -M*M*p + modelL2_params.a1*M*p*p +  p*p*p;
		}
		else if (p <= modelL2_params.pT || modelL2_params.centralOnly) {
			p -= modelL2_params.p2;
			dV[i] = -M*M*p - modelL2_params.a2*M*p*p +  p*p*p;
		}
		else if (p <= modelL2_params.pj) {
			p -= modelL2_params.pT;
			dV[i] = -modelL2_params.m21*p;
		}
		else {
			p -= modelL2_params.pm;
			dV[i] = modelL2_params.m22*p;
		}
		dV[i] *= modelL2_params.rescale;
	}
	
	return dV;
}

PyObject *setModelL2(PyObject *self, PyObject *args, PyObject *keywds) {
    static char *kwlist[] = {"rescale", "centralOnly", NULL};
	int rescale = 0;
	modelL2_params.centralOnly = 0;
	
	
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "|ii", kwlist, &rescale, &modelL2_params.centralOnly))
		return NULL;
	
	if (rescale) {
		double y0 = 0.0;
		modelL2_params.rescale = 1.0;
		double *V0 = V_modelL2(1, 1, &y0);
		modelL2_params.rescale = 1.0 / ((8*PI*V0[0]/3.)); // scale = 1/(Mpl*Hf)^2; Hf^2 = (8pi/3Mpl^2) V(0)
		free(V0);
		LOGMSG("L2 rescale: %f\n", modelL2_params.rescale);
	}
	else
		modelL2_params.rescale = 1.0;
	
	// Set these to the active functions in this file
	V_func = &V_modelL2;
	dV_func = &dV_modelL2;
	
	// and also set them as the active functions in the bubbleEvolution file
	nfields = 1;
	setPotential(nfields, V_func, dV_func );
	
    Py_INCREF(Py_None);
    return Py_None;	
}

#pragma mark -
#pragma mark Generic piecewise
// --------------------------------------
// This class of models implements the potentials given in the Johnson, Peiris and Lehner
// paper. The inputs are the positions of the minima and maxima and the values of the
// potential at the different minima, as well as the inflection points along the inflationary
// sections. The observation bubble is at positive field values, and the values of the potential 
// at the non-zero minima should be given in terms of the false vacuum minimum.
// --------------------------------------

struct GenericPiecewise_params {
	double pm1, pj1, pt1, pa1, pa2, pt2, pj2, pm2; // extrema and piecewise boundaries
	double m1, mt1, mt2, m2; // masses along the inflationary potentials
	double v1, vt1, v0, vt2, v2; // vacuum energies at different points
	double r1, r2; // Rescaling of quartic pieces
	double rescale; // for rescaling by the hubble constant
	int centralOnly;
	int slowRoll1;
};
struct GenericPiecewise_params gp_params = {
	.pm1 = -0.08,
	.pj1 = -.01,
	.pt1 = -.007,
	.pa1 = -.003,
	.pa2 = .002,
	.pt2 = .007,
	.pj2 = .015,
	.pm2 = 2.75,
	.m1 = 0,
	.mt1 = 0,
	.mt2 = 0,
	.m2 = 0,
	.v1 = .2,
	.vt1 = 2.2/2.7,
	.v0 = 1.0,
	.vt2 = 1.8/2.7,
	.v2 = 0.0,
	.r1 = 1.0,
	.r2 = 1.0,
	.rescale = 1.0,
	.centralOnly = 0,
	.slowRoll1 = 0
};

double *V_genericPiecewise(int nx, int ny, double *y) {
	int i;
	double *V = malloc(sizeof(double)*nx);
	
	for (i=0; i<nx; i++) {
		double p = y[i*ny];
		
		if (p < gp_params.pj1 && !gp_params.centralOnly && gp_params.slowRoll1) {
			p -= gp_params.pm1;
			V[i] = 0.5*gp_params.m1*p*p + gp_params.v1;
		}
		else if (p < gp_params.pt1 && !gp_params.centralOnly && gp_params.slowRoll1) {
			p -= gp_params.pt1;
			V[i] = 0.5*gp_params.mt1*p*p + gp_params.vt1;
		}
		else if (p < 0) {
			V[i] = gp_params.v0 + gp_params.r1*p*p*(0.5*gp_params.pt1*gp_params.pa1 
													+ p*(0.25*p - (gp_params.pt1+gp_params.pa1)/3.0));
		}
		else if (p < gp_params.pt2 || gp_params.centralOnly) {
			V[i] = gp_params.v0 + gp_params.r2*p*p*(0.5*gp_params.pt2*gp_params.pa2 
													+ p*(0.25*p - (gp_params.pt2+gp_params.pa2)/3.0));
		}
		else if (p < gp_params.pj2) {
			p -= gp_params.pt2;
			V[i] = 0.5*gp_params.mt2*p*p + gp_params.vt2;
		}
		else {
			p -= gp_params.pm2;
			V[i] = 0.5*gp_params.m2*p*p + gp_params.v2;
		}
		V[i] *= gp_params.rescale;
	}
	
	return V;
}

double *dV_genericPiecewise(int nx, int ny, double *y) {
	int i;
	double *dV = malloc(sizeof(double)*nx);
	
	for (i=0; i<nx; i++) {
		double p = y[i*ny];
		
		if (p < gp_params.pj1 && !gp_params.centralOnly && gp_params.slowRoll1) {
			p -= gp_params.pm1;
			dV[i] = gp_params.m1*p;
		}
		else if (p < gp_params.pt1 && !gp_params.centralOnly && gp_params.slowRoll1) {
			p -= gp_params.pt1;
			dV[i] = gp_params.mt1*p;
		}
		else if (p < 0) {
			dV[i] = p*(p-gp_params.pt1)*(p-gp_params.pa1)*gp_params.r1;
		}
		else if (p < gp_params.pt2 || gp_params.centralOnly) {
			dV[i] = p*(p-gp_params.pt2)*(p-gp_params.pa2)*gp_params.r2;
		}
		else if (p < gp_params.pj2) {
			p -= gp_params.pt2;
			dV[i] = gp_params.mt2*p;
		}
		else {
			p -= gp_params.pm2;
			dV[i] = gp_params.m2*p;
		}
		dV[i] *= gp_params.rescale;
	}
	
	return dV;
}

PyObject *setModel_genericPiecewise(PyObject *self, PyObject *args, PyObject *keywds) {
    static char *kwlist[] = {"pm1","pj1","pt1","pa1","pa2","pt2","pj2","pm2",
		"v0","v1","vt1","vt2","v2","rescale","centralOnly","slowRoll1", NULL};
	int rescale = -1;
	double v1 = gp_params.v1/gp_params.v0;
	double vt1 = gp_params.vt1/gp_params.v0;
	double vt2 = gp_params.vt2/gp_params.v0;
	double v2 = gp_params.v2/gp_params.v0;
	
	int success = PyArg_ParseTupleAndKeywords(args, keywds, "|dddddddddddddiii", kwlist,
											  &gp_params.pm1, &gp_params.pj1,
											  &gp_params.pt1, &gp_params.pa1,
											  &gp_params.pa2, &gp_params.pt2,
											  &gp_params.pj2, &gp_params.pm2,
											  &gp_params.v0, &v1, &vt1, &vt2, &v2,
											  &rescale, &gp_params.centralOnly, &gp_params.slowRoll1);
    if (!success)
		return NULL;
	
	gp_params.v1 = v1*gp_params.v0;
	gp_params.vt1 = vt1*gp_params.v0;
	gp_params.vt2 = vt2*gp_params.v0;
	gp_params.v2 = v2*gp_params.v0;
	
	// Set the remaining variables so that the potential is continuous.
	double dpm, dpt, v, p;
	
	// Matching at pj1
	dpm = gp_params.pj1 - gp_params.pm1;
	dpt = gp_params.pj1 - gp_params.pt1;
	// m1*dpm = mt1*dpt; v1+0.5*m1*dpm^2 = vt1+0.5*mt1*dpt^2
	// 2*(v1-vt1) + mt1*dpm*dpt = mt1*dpt^2
	gp_params.mt1 = 2*(gp_params.v1-gp_params.vt1) / (dpt * (dpt-dpm) );
	gp_params.m1 = gp_params.mt1*dpt/dpm;
	
	// matching at pj2
	dpm = gp_params.pj2 - gp_params.pm2;
	dpt = gp_params.pj2 - gp_params.pt2;
	gp_params.mt2 = 2*(gp_params.v2-gp_params.vt2) / (dpt * (dpt-dpm) );
	gp_params.m2 = gp_params.mt2*dpt/dpm;
	
	// Scaling the quartic pieces
	p = gp_params.pt1;
	v = p*p*(0.5*gp_params.pt1*gp_params.pa1 + p*(0.25*p - (gp_params.pt1+gp_params.pa1)/3.0));
	// r1*v + v0 = vt1
	gp_params.r1 = (gp_params.vt1 - gp_params.v0) / v;

	p = gp_params.pt2;
	v = p*p*(0.5*gp_params.pt2*gp_params.pa2 + p*(0.25*p - (gp_params.pt2+gp_params.pa2)/3.0));
	gp_params.r2 = (gp_params.vt2 - gp_params.v0) / v;
	
	// Rescale the whole thing if needed (this is equivalent to just scaling v0 in the input)
	if (rescale > 0) {
		gp_params.rescale = 1.0 / ((8*PI*gp_params.v0/3.)); // scale = 1/(Mpl*Hf)^2; Hf^2 = (8pi/3Mpl^2) V(0)
	}
	else if (rescale == 0)
		gp_params.rescale = 1.0;
	
	// Set these to the active functions in this file
	V_func = &V_genericPiecewise;
	dV_func = &dV_genericPiecewise;
	
	// and also set them as the active functions in the bubbleEvolution file
	nfields = 1;
	setPotential(nfields, V_func, dV_func );
	
	PyObject *rdict;
	rdict = Py_BuildValue("{s:d, s:d, s:d, s:d, s:d, s:d, s:d, s:d, "
						  "s:d, s:d, s:d, s:d, s:d, "
						  "s:d, s:d, s:d, s:d, s:d, s:i}",
						  "pm1", gp_params.pm1, "pj1", gp_params.pj1,
						  "pt1", gp_params.pt1, "pa1", gp_params.pa1,
						  "pa2", gp_params.pa2, "pt2", gp_params.pt2,
						  "pj2", gp_params.pj2, "pm2", gp_params.pm2,
						  "v0", gp_params.v0, "v1", v1, "vt1", vt1, "vt2", vt2, "v2", v2,
						  "m1", gp_params.m1, "m2", gp_params.m2, 
						  "r1", gp_params.r1, "r2", gp_params.r2, 
						  "rescale", gp_params.rescale, "centralOnly", gp_params.centralOnly);
	
	return rdict;
	
	Py_INCREF(Py_None);
	return Py_None;	
}


#pragma mark -
#pragma mark Generic piecewise, no hilltop
// --------------------------------------
// This class of models is similar to the above, except that there is no
// stationary point at the bottom of the potential barrier. Instead, the
// inflationary part of the potential is matched directly on to the quartic
// part such that the two are C1 continuous.
// 
// Inputs are the barrier position, the field value of the would-be quartic 
// minimum, the potential value at the would-be minimum (relative to the false vaccum), 
// and position of the true vacuum. If the true vacuum position is set to zero,
// (or if it otherwise cannot be matched to quartic part) the inflationary phase 
// is ignored entirely. The two sides of the potential can be set separately.
//
// The whole thing is rescaled such that the false vacuum hubble param H_F = 1.
// --------------------------------------

// a is right (positive), b is left (negative)
struct GenericPiecewise_noHilltop_params {
	double pa_bar, pb_bar; // barrier position
	double pa_qmin, pb_qmin; // would-be quartic minimum (away from phi = 0)
	double lambda_a, lambda_b; // quartic coefficient
	double pa_edge, pb_edge; // point separating the quartic from quadratic regions
	double ma, mb; // mass-squared of the inflationary phase
	double pa_vac, pb_vac; // The field at the true vacuum.
	double v0; // Value of the false vacuum (should always be 3/8pi)
};
struct GenericPiecewise_noHilltop_params gpnh_params = {
	.pa_bar = 4e-2,
	.pb_bar = -4e-2,
	.pa_qmin = 10e-2,
	.pb_qmin = -10e-2,
	.lambda_a = 1e4,
	.lambda_b = 1e4,
	.pa_edge = 0.,
	.pb_edge = 0.,
	.ma = .1,
	.mb = .1,
	.pa_vac = 0.,
	.pb_vac = 0.,
	.v0 = 3/(8*PI)
};

double *V_genericPiecewise_noHilltop(int nx, int ny, double *y) {
	int i;
	double *V = malloc(sizeof(double)*nx);
	
	for (i=0; i<nx; i++) {
		double p = y[i*ny];
		
		if (p > gpnh_params.pa_edge && gpnh_params.pa_edge > 0) {
			double delta_p = p - gpnh_params.pa_vac;
			V[i] = 0.5 * gpnh_params.ma * delta_p * delta_p;
		}
		else if (p >= 0) {
			V[i] = gpnh_params.v0 + gpnh_params.lambda_a*p*p*(0.5*gpnh_params.pa_bar*gpnh_params.pa_qmin 
															  + p*(0.25*p - (gpnh_params.pa_bar+gpnh_params.pa_qmin)/3.0));
		}
		else if (p >= gpnh_params.pb_edge || gpnh_params.pb_edge >= 0) {
			V[i] = gpnh_params.v0 + gpnh_params.lambda_b*p*p*(0.5*gpnh_params.pb_bar*gpnh_params.pb_qmin 
															  + p*(0.25*p - (gpnh_params.pb_bar+gpnh_params.pb_qmin)/3.0));
		}
		else {
			double delta_p = p - gpnh_params.pb_vac;
			V[i] = 0.5 * gpnh_params.mb * delta_p * delta_p;
		}
	}
	
	return V;
}

double *dV_genericPiecewise_noHilltop(int nx, int ny, double *y) {
	int i;
	double *dV = malloc(sizeof(double)*nx);
	
	for (i=0; i<nx; i++) {
		double p = y[i*ny];
		
		if (p > gpnh_params.pa_edge && gpnh_params.pa_edge > 0) {
			double delta_p = p - gpnh_params.pa_vac;
			dV[i] = gpnh_params.ma * delta_p;
		}
		else if (p >= 0) {
			dV[i] = gpnh_params.lambda_a * p * (p-gpnh_params.pa_bar) * (p-gpnh_params.pa_qmin);
		}
		else if (p >= gpnh_params.pb_edge || gpnh_params.pb_edge >= 0) {
			dV[i] = gpnh_params.lambda_b * p * (p-gpnh_params.pb_bar) * (p-gpnh_params.pb_qmin);
		}
		else {
			double delta_p = p - gpnh_params.pb_vac;
			dV[i] = gpnh_params.mb * delta_p;
		}
	}
	
	return dV;
}

double delta_dV_genericPiecewise_noHilltopHelper(double p_bar, double p_qmin, double lambda, double p_vac, 
													double p_edge, double *m_out) {
	double p = p_edge;
	double V = gpnh_params.v0 + lambda*p*p*(0.5*p_bar*p_qmin + p*(0.25*p - (p_bar+p_qmin)/3.0));
	double dV = lambda * p * (p-p_bar) * (p-p_qmin);
	// Want to match at p_edge: V = 0.5*m*(p-p_vac)^2
	double delta_p = p - p_vac;
	*m_out = 2*V / (delta_p*delta_p);
	
	return dV - *m_out * delta_p;
}

// Need to change this
PyObject *setModel_genericPiecewise_noHilltop(PyObject *self, PyObject *args, PyObject *keywds) {
	// First, check to see if we're setting the positive or negative side.
	int posneg = 0;
	PyObject *posneg_str_obj = PySequence_GetItem(args, 0);
	PyObject *posstr = PyUnicode_FromString("pos");
	PyObject *negstr = PyUnicode_FromString("neg");
	if (PyUnicode_Compare(posneg_str_obj, posstr) == 0)
		posneg = +1;
	else if (PyUnicode_Compare(posneg_str_obj, negstr) == 0)
		posneg = -1;
	Py_DECREF(posstr);
	Py_DECREF(negstr);
	Py_DECREF(posneg_str_obj);
	if (posneg == 0) {
		PyErr_SetString(PyExc_ValueError, "The first argument of setGenericPiecewiseModel_noHilltop() "
						"must be either 'pos' or 'neg'.");
		return NULL;
	}
	
    static char *kwlist[] = {"posneg","barrier_position","Hsq_ratio","phi_quartic_min","phi_true_vac", NULL};
	double p_bar = posneg > 0 ? gpnh_params.pa_bar : gpnh_params.pb_bar;
	double p_qmin = posneg > 0 ? gpnh_params.pa_qmin : gpnh_params.pb_qmin;
	double p_vac = posneg > 0 ? gpnh_params.pa_vac : gpnh_params.pb_vac;
	double *V_qmin_ptr = V_genericPiecewise_noHilltop(1, 1, &p_qmin);
	double Hsq_ratio = *V_qmin_ptr / gpnh_params.v0;
	free(V_qmin_ptr);
	double barrier_pos = p_bar / p_qmin;
	
	int success = PyArg_ParseTupleAndKeywords(args, keywds, "O|dddd", kwlist, posneg_str_obj, 
											  &barrier_pos, &Hsq_ratio, &p_qmin, &p_vac);
    if (!success)
		return NULL;
	
	if (posneg * p_qmin <= 0) {
		PyErr_SetString(PyExc_ValueError, "The phi_quartic_min parameter must match the sign given by 'pos' or 'neg'.");
		return NULL;
	}
	if (barrier_pos >= 0.5 || barrier_pos <= 0) {
		PyErr_SetString(PyExc_ValueError, "Must have 0 < barrier_position < 0.5.");
		return NULL;
	}
	if (Hsq_ratio >= 1) {
		PyErr_SetString(PyExc_ValueError, "Must have Hsq_ratio < 1.");
		return NULL;
	}
	
	p_bar = barrier_pos * p_qmin;
	
	// Calculate the overall quartic coefficient
	double V_temp = p_qmin*p_qmin*(0.5*p_bar*p_qmin + p_qmin*(0.25*p_qmin - (p_bar+p_qmin)/3.0));
	// v0 + lambda * V_temp = V_qmin
	// V_qmin / v0 = Hsq_ratio
	double lambda = gpnh_params.v0 * (Hsq_ratio - 1) / V_temp;
	
	// Try to match the quartic and quadratic pieces
	// First need to find the inflection point along the quartic. Start the matching search there.
	double p_inf = ( (p_bar+p_qmin) + posneg*sqrt( (p_bar-p_qmin)*(p_bar-p_qmin) + p_bar*p_qmin ) ) / 3.0;
	// Now do a binary search.
	double m_vac, p_edge; // These are the two things we need to find.
	double p1 = p_inf, p2 = p_qmin;
	double DV1 = delta_dV_genericPiecewise_noHilltopHelper(p_bar, p_qmin, lambda, p_vac, p1, &m_vac);
	double DV2 = delta_dV_genericPiecewise_noHilltopHelper(p_bar, p_qmin, lambda, p_vac, p2, &m_vac);
	double p_tol = posneg * p_qmin * 1e-10;
	if (DV1 * DV2 > 0) { // Same sign
		PyRun_SimpleString("print('Not including the slow-roll part of the potential.')");
		p_edge = 0.0;
		m_vac = 0.0;
	}
	else {
		PyRun_SimpleString("print('Including the slow-roll part of the potential.')");
		while ( posneg*(p2 - p1) > p_tol ) {
			p_edge = 0.5 * (p1+p2);
			double DVedge = delta_dV_genericPiecewise_noHilltopHelper(p_bar, p_qmin, lambda, p_vac, p_edge, &m_vac);
			if ( DVedge * DV1 > 0 ) {
				DV1 = DVedge;
				p1 = p_edge;
			}
			else {
				DV2 = DVedge;
				p2 = p_edge;
			}
		}
	}

	// We've got all the pieces. Now just put them back into the right parameters.
	if (posneg > 0) {
		gpnh_params.pa_bar = p_bar;
		gpnh_params.pa_qmin = p_qmin;
		gpnh_params.lambda_a = lambda;
		gpnh_params.pa_edge = p_edge;
		gpnh_params.ma = m_vac;
		gpnh_params.pa_vac = p_vac;
	}
	else {
		gpnh_params.pb_bar = p_bar;
		gpnh_params.pb_qmin = p_qmin;
		gpnh_params.lambda_b = lambda;
		gpnh_params.pb_edge = p_edge;
		gpnh_params.mb = m_vac;
		gpnh_params.pb_vac = p_vac;		
	}
	
	// Set these to the active functions in this file
	V_func = &V_genericPiecewise_noHilltop;
	dV_func = &dV_genericPiecewise_noHilltop;
	
	// and also set them as the active functions in the bubbleEvolution file
	nfields = 1;
	setPotential(nfields, V_func, dV_func );	
	
	Py_INCREF(Py_None);
	return Py_None;
}



#pragma mark -
#pragma mark Hilltop model
// --------------------------------------
// Two-field hilltop model
// This model uses two scalar fields: an inflaton field and a tunneling field.
// The inflaton field has a Z2 symmetry, and the tunneling happens near the origin
// so that the inflaton sits near a saddle point for awhile before rolling down.
// It is a simple quartic potential.
// --------------------------------------

struct Hilltop_params {
	double v1, v2; // vevs of inflaton and tunneling field, respectively
	double vs; // The location of the saddle point in the tunneling direction
	double lmda, beta; // quartic self-couplings
	double alpha; // coupling between the two fields (not a simple polynomial! dimension of m^2)
	double rescale; // for rescaling by the Hubble constant
};
struct Hilltop_params hilltop_params = {1,1, 1, 1,1,1, 1};

double *V_hilltop(int nx, int ny, double *y) {
	int i;
	double *V = malloc(sizeof(double)*nx);
	
	for (i=0; i<nx; i++) {
		double phi = y[i*ny]; // inflaton field
		double sig = y[i*ny+1]; // tunneling field
		
		double a; // temp value
		a = (phi*phi - hilltop_params.v1*hilltop_params.v1);
		V[i] = 0.125*hilltop_params.lmda*a*a;
		
		V[i] += hilltop_params.beta * sig*sig * 
			(0.5*hilltop_params.v2*hilltop_params.vs + sig*( 0.25*sig - (hilltop_params.v2+hilltop_params.vs)/3.));
		V[i] += 0.5*hilltop_params.alpha * sig*sig * phi*phi / (hilltop_params.v2*hilltop_params.v2 + phi*phi);
		
		V[i] *= hilltop_params.rescale;
	}
	
	return V;
}

double *dV_hilltop(int nx, int ny, double *y) {
	int i;
	double *dV = malloc(sizeof(double)*nx*2);
	
	for (i=0; i<nx; i++) {
		double phi = y[i*ny]; // inflaton field
		double sig = y[i*ny+1]; // tunneling field
		
		double a,b; // temp value
		a = (phi*phi - hilltop_params.v1*hilltop_params.v1);
		b = 1.0/(hilltop_params.v2*hilltop_params.v2 + phi*phi);
		dV[2*i+0] = 0.5*hilltop_params.lmda*a*phi;
		dV[2*i+0] += hilltop_params.alpha*sig*sig*phi*(1-phi*phi*b)*b;
		dV[2*i+1] = hilltop_params.beta * sig * 
			( hilltop_params.v2*hilltop_params.vs + sig*(sig-hilltop_params.v2-hilltop_params.vs) );
		dV[2*i+1] += hilltop_params.alpha*phi*phi*sig*b;
		dV[2*i+0] *= hilltop_params.rescale;
		dV[2*i+1] *= hilltop_params.rescale;
	}
	
	return dV;
}

PyObject *setModel_hilltop(PyObject *self, PyObject *args, PyObject *keywds) {
    static char *kwlist[] = {"v1","v2","vs","m1","m2","m1at2","rescale", NULL};
	int rescale = 0;
	double m1 = sqrt(hilltop_params.lmda)*hilltop_params.v1;
	double m2 = sqrt(hilltop_params.beta*hilltop_params.v2*hilltop_params.vs);
	double m1at2 = sqrt(hilltop_params.alpha - 0.5*m1*m1);
	
	int success = PyArg_ParseTupleAndKeywords(args, keywds, "|ddddddi", kwlist,
											  &hilltop_params.v1, &hilltop_params.v2,
											  &hilltop_params.vs,
											  &m1, &m2, &m1at2, &rescale);
    if (!success)
		return NULL;
		
	hilltop_params.lmda = (m1*m1)/(hilltop_params.v1*hilltop_params.v1);
	hilltop_params.beta = (m2*m2)/(hilltop_params.v2*hilltop_params.vs);
	hilltop_params.alpha = (m1at2*m1at2+0.5*m1*m1); // /(hilltop_params.v2*hilltop_params.v2);
	
	hilltop_params.rescale = 1.0;
	if (rescale) {
		double phiF[2] = {0.0, hilltop_params.v2};
		double *V0 = V_hilltop(1, 2, phiF);
		hilltop_params.rescale = 1.0 / ((8*PI*V0[0]/3.)); // scale = 1/(Mpl*Hf)^2; Hf^2 = (8pi/3Mpl^2) V(0)
		free(V0);
		LOGMSG("hilltop model rescale: %f\n", hilltop_params.rescale);
	}
		
	// Set these to the active functions in this file
	V_func = &V_hilltop;
	dV_func = &dV_hilltop;
	
	// and also set them as the active functions in the bubbleEvolution file
	nfields = 2;
	setPotential(nfields, V_func, dV_func );
	
	Py_INCREF(Py_None);
	return Py_None;	
}

#pragma mark -
#pragma mark Chaotic inflation
// --------------------------------------
// Chaotic inflation, comes from arXiV:1110.4773
// --------------------------------------
struct Chaotic_params {
	double sig0, phi0; // vevs of inflaton and tunneling field, respectively
	double Mv, mphi;
	double alpha, beta; // quartic self-couplings
	double rescale; // for rescaling by the Hubble constant
};
struct Chaotic_params chaotic_params = {1,1, 1,1, 1,1, 1};

double *V_chaotic(int nx, int ny, double *y) {
	int i;
	double *V = malloc(sizeof(double)*nx);
	
	for (i=0; i<nx; i++) {
		double phi = y[i*ny]; // inflaton field
		double sig = y[i*ny+1]; // tunneling field
		
		double a; // temp value
		a = sig*(phi-chaotic_params.phi0);
		V[i] = 0.5*chaotic_params.beta*a*a;
		a = chaotic_params.mphi * phi;
		V[i] += 0.5*a*a;
		a = sig - chaotic_params.sig0;
		V[i] += chaotic_params.alpha*sig*sig*(a*a+chaotic_params.Mv*chaotic_params.Mv);
	}
	
	return V;
}

double *dV_chaotic(int nx, int ny, double *y){
	int i;
	double *dV = malloc(sizeof(double)*nx*2);
	
	for (i=0; i<nx; i++) {
		double phi = y[i*ny]; // inflaton field
		double sig = y[i*ny+1]; // tunneling field
		
		double a,b; // temp value
		dV[2*i+0] = chaotic_params.mphi*chaotic_params.mphi*phi;
		dV[2*i+0] += chaotic_params.beta*sig*sig*(phi-chaotic_params.phi0);
		a = sig-chaotic_params.sig0;
		dV[2*i+1] = chaotic_params.alpha*(2*a*a+chaotic_params.sig0*a+chaotic_params.Mv*chaotic_params.Mv);
		b = phi-chaotic_params.phi0;
		dV[2*i+1] += 0.5*chaotic_params.beta*a*a;
		dV[2*i+1] *= 2*sig;
		
		dV[2*i+0] *= chaotic_params.rescale;
		dV[2*i+1] *= chaotic_params.rescale;
	}
	
	return dV;
}

PyObject *setModel_chaotic(PyObject *self, PyObject *args, PyObject *keywds) {
    static char *kwlist[] = {"sig0","phi0","Mv","mphi","alpha","beta","rescale", NULL};
	
	int success = PyArg_ParseTupleAndKeywords(args, keywds, "|ddddddd", kwlist,
											  &chaotic_params.sig0, &chaotic_params.phi0,
											  &chaotic_params.Mv, &chaotic_params.mphi,
											  &chaotic_params.alpha, &chaotic_params.beta,
											  &chaotic_params.rescale);
    if (!success)
		return NULL;
	
	
	// Set these to the active functions in this file
	V_func = &V_chaotic;
	dV_func = &dV_chaotic;
	
	// and also set them as the active functions in the bubbleEvolution file
	nfields = 2;
	setPotential(nfields, V_func, dV_func );
	
	Py_INCREF(Py_None);
	return Py_None;	
}


#pragma mark -
#pragma mark Double L2 model
// --------------------------------------
// Same as L2 model, but in two directions.
// --------------------------------------
struct DoubleL2_params {
	double xscale, yscale;
	double Vxscale, Vyscale;
	int rescaleSet;
};
struct DoubleL2_params doubleL2_params = {1,1, 1,1, 0};

double *V_doubleL2(int nx, int ny, double *y) {
	int i;
	double *y1 = malloc(sizeof(double)*nx);
	double *y2 = malloc(sizeof(double)*nx);
	for (i=0; i<nx; i++) {
		y1[i] = y[i*ny]*doubleL2_params.xscale;
		y2[i] = y[i*ny+1]*doubleL2_params.yscale;
	}
	double *V1 = V_modelL2(nx, 1, y1);
	double *V2 = V_modelL2(nx, 1, y2);
	double *V = malloc(sizeof(double)*2*nx);
	for (i=0; i<nx; i++) {
		V[i] = (V1[i]*doubleL2_params.Vxscale + V2[i]*doubleL2_params.Vyscale);
	}

	free(y1);
	free(y2);
	free(V1);
	free(V2);

	return V;
}

double *dV_doubleL2(int nx, int ny, double *y){
	int i;
	double *y1 = malloc(sizeof(double)*nx);
	double *y2 = malloc(sizeof(double)*nx);
	for (i=0; i<nx; i++) {
		y1[i] = y[i*ny]*doubleL2_params.xscale;
		y2[i] = y[i*ny+1]*doubleL2_params.yscale;
	}
	double *dV1 = dV_modelL2(nx, 1, y1);
	double *dV2 = dV_modelL2(nx, 1, y2);
	double *dV = malloc(sizeof(double)*2*nx);
	for (i=0; i<nx; i++) {
		dV[2*i] = dV1[i]*doubleL2_params.Vxscale/doubleL2_params.xscale;
		dV[2*i+1] = dV2[i]*doubleL2_params.Vxscale/doubleL2_params.xscale;
	}
	free(y1);
	free(y2);
	free(dV1);
	free(dV2);
	
	return dV;
}

PyObject *setModel_doubleL2(PyObject *self, PyObject *args, PyObject *keywds) {
    static char *kwlist[] = {"xscale","yscale","Vxscale","Vyscale","rescale","centralOnly", NULL};
	
	int success = PyArg_ParseTupleAndKeywords(args, keywds, "|ddddii", kwlist,
											  &doubleL2_params.xscale, &doubleL2_params.yscale,
											  &doubleL2_params.Vxscale, &doubleL2_params.Vyscale,
											  &doubleL2_params.rescaleSet,
											  &modelL2_params.centralOnly);
    if (!success)
		return NULL;
	
	modelL2_params.rescale = 1.0;
	if (doubleL2_params.rescaleSet) {
		double phiF[2] = {0.0, 0.0};
		double *V0 = V_doubleL2(1, 2, phiF);
		modelL2_params.rescale = 1.0 / ((8*PI*V0[0]/3.)); // scale = 1/(Mpl*Hf)^2; Hf^2 = (8pi/3Mpl^2) V(0)
		free(V0);
		LOGMSG("Rescale set: %0.4e", modelL2_params.rescale);
	}
	
	// Set these to the active functions in this file
	V_func = &V_doubleL2;
	dV_func = &dV_doubleL2;
	
	// and also set them as the active functions in the bubbleEvolution file
	nfields = 2;
	setPotential(nfields, V_func, dV_func );
	
	Py_INCREF(Py_None);
	return Py_None;	
}




#pragma mark -
#pragma mark QuadAndGaussian
// --------------------------------------
// A simple quadratic potential with a gaussian bump.
// --------------------------------------
struct QuadGauss_params {
	double m, A, sigma_sq, phi0, bump_height;
};
struct QuadGauss_params quadGauss_params = {1,.1, 0.01,-1, 1.1};

double *V_quadGauss(int nx, int ny, double *y) {
	int i;
	double *V = malloc(sizeof(double)*nx);
	for (i=0; i<nx; i++) {
		double phi = y[i*ny];
		double delta_phi = (phi-quadGauss_params.phi0);
		V[i] = 0.5*quadGauss_params.m*phi*phi;
		V[i] += quadGauss_params.A * exp(-0.5 * delta_phi*delta_phi / quadGauss_params.sigma_sq );
	}
		
	return V;
}

double *dV_quadGauss(int nx, int ny, double *y){
	int i;
	double *V = malloc(sizeof(double)*nx);
	for (i=0; i<nx; i++) {
		double phi = y[i*ny];
		double delta_phi = (phi-quadGauss_params.phi0);
		V[i] = quadGauss_params.m*phi;
		V[i] -= (quadGauss_params.A * exp(-0.5 * delta_phi*delta_phi / quadGauss_params.sigma_sq ) 
				 * delta_phi / quadGauss_params.sigma_sq);
	}
	
	return V;
}

PyObject *setModel_quadGauss(PyObject *self, PyObject *args, PyObject *keywds) {
    static char *kwlist[] = {"bumpHeight","bumpWidth","phi0", NULL};
	double sigma = sqrt(quadGauss_params.sigma_sq);
	
	int success = PyArg_ParseTupleAndKeywords(args, keywds, "|ddd", kwlist,
											  &quadGauss_params.bump_height, &sigma, &quadGauss_params.phi0);
    if (!success)
		return NULL;
	
	if (sigma <= 0) {
		PyErr_SetString(PyExc_ValueError, "Must have sigma > 0");
		return NULL;
	}
	int switched_phi = 0;
	if (quadGauss_params.phi0 > 0) {
		quadGauss_params.phi0 *= -1;
		switched_phi = 1;
	}	
	quadGauss_params.sigma_sq = sigma*sigma;
	quadGauss_params.A = -sigma*quadGauss_params.phi0*exp(.5)*quadGauss_params.bump_height;
	quadGauss_params.m = 1.0;
	
	// Find the minimum
	double phi_max = quadGauss_params.bump_height > 0 ? quadGauss_params.phi0 - sigma : quadGauss_params.phi0 + sigma;
	double phi_min = quadGauss_params.phi0 - 20*sigma;
	double phi_mid = (phi_max + phi_min) * 0.5;
	// dV should be positive at phi_max, negative at phi_min
	double *dVptr, dV;
	dVptr = dV_quadGauss(1, 1, &phi_min); dV = *dVptr; free(dVptr);
	if (dV > 0) {
		PyErr_SetString(PyExc_ValueError, "Cannot find false vacuum minimum. "
						"Potential derivative is positive far away from the bump.");
		return NULL;
	}
	dVptr = dV_quadGauss(1, 1, &phi_max); dV = *dVptr; free(dVptr);
	if (dV < 0) {
		PyErr_SetString(PyExc_ValueError, "Cannot find false vacuum minimum. "
						"Potential derivative is negative on the side of the bump.");
		return NULL;
	}
	
	while (phi_max - phi_min > -1e-9 * quadGauss_params.phi0) {
		phi_mid = (phi_max + phi_min) * 0.5;
		dVptr = dV_quadGauss(1, 1, &phi_mid); dV = *dVptr; free(dVptr);
		if (dV > 0)
			phi_max = phi_mid;
		else
			phi_min = phi_mid;
	}
	
	if (switched_phi) {
		quadGauss_params.phi0 *= -1;
		phi_mid *= -1;
	}
	
	// Rescale the potential
	double *Vptr, V;
	Vptr = V_quadGauss(1, 1, &phi_mid); V = *Vptr; free(Vptr);
	double Hf_sq = 8*PI * V / 3.0;
	quadGauss_params.A /= Hf_sq;
	quadGauss_params.m /= Hf_sq;
	
	
	// Set these to the active functions in this file
	V_func = &V_quadGauss;
	dV_func = &dV_quadGauss;
	
	// and also set them as the active functions in the bubbleEvolution file
	nfields = 1;
	setPotential(nfields, V_func, dV_func );
	
	return Py_BuildValue("d", phi_mid);	
}



