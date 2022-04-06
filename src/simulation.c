/*
 *  main.c
 *  adaptiveGrid
 *
 *  Created by Max Wainwright on 1/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */


#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#include "bubbleEvolution.h"
#include "gridInterpolation.h"


// --------------------------------------
// Python initialization
// --------------------------------------


static PyMethodDef *_methods = NULL;
static char *docstring = 
"Simulate colliding bubbles.\n"
"\n"
"This module contains all of the basic code for running a simulation.\n"
"It acts as a state machine (as opposed to having an object-oriented interface),\n"
"with the simulation state being set by the various *set_()* functions.\n"
"Both the model and the monitor function will always need to be set before\n"
"running, while all other parameters have working default values. However,\n"
"one will generally also want to set the file parameters so that the\n"
"simulation data can be retrieved. The simulation can then be run using\n"
":func:`runCollision`.\n"
"\n"
"Additionally, the functions :func:`readFromFile` and :func:`valsOnGrid` \n"
"can be used to load a simulation from file and then use interpolation to\n"
"find fields, metric functions, and/or Christoffel symbols as functions\n"
"of the simulation coordinates.\n"
"\n"
"Note that this module does *not* provide any resources for calculating\n"
"either the instanton data or transforming instanton data into initial\n"
"conditions. Those jobs are handled by the *cosmoTransitions* package and the\n"
":mod:`bubble_collisions.collisionRunner` module.\n";

PyMODINIT_FUNC
initsimulation(void)
{
	free(_methods);
	_methods = calloc(10, sizeof(PyMethodDef));
	_methods[0] = setModel_methdef;
	_methods[1] = setFileParams_methdef;
	_methods[2] = setIntegrationParams_methdef;
	_methods[3] = setMonitorCallback_methdef;
	_methods[4] = setTimeConstraints_methdef;
	_methods[5] = runCollision_methdef;
	_methods[6] = readFromFile_methdef;
	_methods[7] = valsOnGrid_methdef;
	_methods[8] = remakeGrid_methdef;

    (void) Py_InitModule3("simulation", _methods, docstring);
	import_array();
}

