/*
...doc string etc.
*/

// ----------------------------------------------------------------------
// Wrapper and python integration functions.
// ----------------------------------------------------------------------

void setPotential( int nf, double *(*V)(int, int, double *), double *(*dV)(int, int, double *) );

extern PyMethodDef setModel_methdef;
PyObject *setModel(PyObject *self, PyObject *args);
extern PyMethodDef setFileParams_methdef;
PyObject *setFileParams(PyObject *self, PyObject *args, PyObject *keywds); // File name, xres, tres
extern PyMethodDef setIntegrationParams_methdef;
PyObject *setIntegrationParams(PyObject *self, PyObject *args, PyObject *keywds);
extern PyMethodDef setMonitorCallback_methdef;
PyObject *setMonitorCallback(PyObject *self, PyObject *args);
extern PyMethodDef setTimeConstraints_methdef;
PyObject *setTimeConstraints(PyObject *self, PyObject *args);
extern PyMethodDef runCollision_methdef;
PyObject *runCollision(PyObject *self, PyObject *args, PyObject *keywds);

extern PyMethodDef remakeGrid_methdef;
PyObject *remakeGrid_frompy(PyObject *self, PyObject *args);
	
// ----------------------------------------------------------------------
// Computation functions.
// ----------------------------------------------------------------------

double *interpGrid_scipy(double *xold, int nold, double *yold, int ny, double *xnew, int nnew);
	// This interpolates from an old grid to a new grid. It calls the scipy routines
	// interpolate.splprep and interpolate.splev. It's kind of cumbersome to do it this way,
	// but it works a little bit better (although more slowly) than my custom implementation.

double *monitorFunc(double t, int nx, double *x, double *y, double *c1);
int dY_bubbles(double t, double *y, double *c1, double *c2, int nx, double *dY_out);
//double speedOfLight(double N); // In our dS coords, c is time-dependent.
double evolveBubbles(double *x0, double *y0, int nx0, double t0, double tmax, double alphaMax, 
					 int exactTmax, int growBounds, int overwritefile,
					 double **xout, double **yout, int *nxout);
	// Main computation loop.
	// Outputs t, x, and y at the final time step. Also writes to file.

// ----------------------------------------------------------------------
// File I/O
// ----------------------------------------------------------------------

// Write the fields and the christoffel symbols to separate files.
void writeFieldsAndChristoffelsToFile(FILE *fFields, FILE *fChris, int32_t nx, int32_t skipx,
									  double t, double *x, double *y, double *c1, double *c2);
extern PyMethodDef readFromFile_methdef;
PyObject *readFromFile_toPy(PyObject *self, PyObject *args);


