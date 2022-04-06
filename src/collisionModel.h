/*

This file specifies the model that I'll be using in my collision simulations.
Hopefully this will be faster than using the same model in python callbacks.

To set an unscaled potential (which accepts dimensionful input), set Mpl = -1.
Setting Mpl to a positive value should rescale the potentials to accept dimensionless values.

*/

//#define PY_ARRAY_UNIQUE_SYMBOL "asdffdsa"

// --------------------------------------
// These are the only true global variables in the program.
// --------------------------------------

extern double *(*V_func)(int, int, double *);
extern double *(*dV_func)(int, int, double *);
extern int nfields;

// --------------------------------------
// These first few parameters/functions are shared by all models
// --------------------------------------

PyObject *V_pywrapper(PyObject *self, PyObject *args);
PyObject *dV_pywrapper(PyObject *self, PyObject *args);
PyObject *V_1d_pywrapper(PyObject *self, PyObject *args);
PyObject *dV_1d_pywrapper(PyObject *self, PyObject *args);
PyObject *nfields_pywrapper(PyObject *self, PyObject *args);

// --------------------------------------
// Model 1
// V = scale*[(x^2+y^2)( (1+tilt)(x-1)^2 + (1-tilt)(y-1)^2 - c) + offset]
// where x and y have been rescaled by Mpl. 
// (input is unitless; above equation is unitful, despite the 1's)
// --------------------------------------

double *V_model1(int nx, int ny, double *y);
double *dV_model1(int nx, int ny, double *y);
PyObject *setModel1(PyObject *self, PyObject *args, PyObject *keywds); // barrier=1-c/2, tilt, offset, Mpl
// (use Mpl = -1 for unscaled equation)

//double *gridDensity_model1(double t, int nx, double *y, double *c1);


// --------------------------------------
// Model L2
// This is the large-field inflation model used in Johnson, Peiris and Lehner.
// All parameters are fixed, but we still need to call setModelL2 to normalize it by the false vacuum
// Hubble constant.
// --------------------------------------

double *V_modelL2(int nx, int ny, double *y);
double *dV_modelL2(int nx, int ny, double *y);
PyObject *setModelL2(PyObject *self, PyObject *args, PyObject *keywds);

// --------------------------------------
// Generic piecewise potential
// This class of models implements the potentials given in the Johnson, Peiris and Lehner
// paper. The inputs are the positions of the minima and maxima and the values of the
// potential at the different minima, as well as the inflection points along the inflationary
// sections. The observation bubble is at positive field values, and the values of the potential 
// at the non-zero minima should be given in terms of the false vacuum minimum.
// --------------------------------------

double *V_genericPiecewise(int nx, int ny, double *y);
double *dV_genericPiecewise(int nx, int ny, double *y);
PyObject *setModel_genericPiecewise(PyObject *self, PyObject *args, PyObject *keywds);

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

double *V_genericPiecewise_noHilltop(int nx, int ny, double *y);
double *dV_genericPiecewise_noHilltop(int nx, int ny, double *y);
PyObject *setModel_genericPiecewise_noHilltop(PyObject *self, PyObject *args, PyObject *keywds);


// --------------------------------------
// Two-field hilltop model
// This model uses two scalar fields: an inflaton field and a tunneling field.
// The inflaton field has a Z2 symmetry, and the tunneling happens near the origin
// so that the inflaton sits near a saddle point for awhile before rolling down.
// It is a simple quartic potential.
// --------------------------------------

double *V_hilltop(int nx, int ny, double *y);
double *dV_hilltop(int nx, int ny, double *y);
PyObject *setModel_hilltop(PyObject *self, PyObject *args, PyObject *keywds);


// --------------------------------------
// Chaotic inflation, comes from arXiV:1110.4773
// --------------------------------------

double *V_chaotic(int nx, int ny, double *y);
double *dV_chaotic(int nx, int ny, double *y);
PyObject *setModel_chaotic(PyObject *self, PyObject *args, PyObject *keywds);

// --------------------------------------
// Same as L2 model, but in two directions.
// --------------------------------------
double *V_doubleL2(int nx, int ny, double *y);
double *dV_doubleL2(int nx, int ny, double *y);
PyObject *setModel_doubleL2(PyObject *self, PyObject *args, PyObject *keywds);


// --------------------------------------
// A simple quadratic potential with a gaussian bump.
// --------------------------------------
double *V_quadGauss(int nx, int ny, double *y);
double *dV_quadGauss(int nx, int ny, double *y);
PyObject *setModel_quadGauss(PyObject *self, PyObject *args, PyObject *keywds);
