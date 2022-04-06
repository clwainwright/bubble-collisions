#include <Python.h>
#include "structmember.h"

#include <arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>

#include "model_objects.h"

#define VERBOSE 1
#if VERBOSE
#define BLURT() printf("%s(%d) -- %s()\n", __FILE__, __LINE__, __func__)
#define LOGMSG(...) printf("%s(%d): ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n")
#else
#define BLURT() ;
#define LOGMSG(...) ;
#endif

#define PI 3.141592653589793

#pragma mark --ModelObject base class

static PyObject *model_obj_V(ModelObject *self, PyObject *args, PyObject *keywds) {
    PyObject *Yin, *Yout;
    npy_intp num_pts, num_dims, *in_dims, ny=0, i;
    double *y_in, *y_out;

    static char *kwlist[] = {"y", "one_dim", NULL};
    int one_dim = 0;
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|i", kwlist,
        &Yin, &one_dim))
        return NULL;
    if (one_dim && self->nfields > 1) {
        PyErr_SetString(PyExc_ValueError, 
            "Field can only be one-dimensional "
            "('one_dim' parameter set to True) if nfields == 1");
        return NULL;
    }
    Yin = PyArray_FROM_OTF(Yin, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (Yin == NULL) return NULL;
    if (PyArray_Size(Yin) == 0) 
        return Yin; // Empty array. Pass through.
    
    num_dims = PyArray_NDIM((PyArrayObject *)Yin);
    in_dims = PyArray_DIMS((PyArrayObject *)Yin);
    if (one_dim) {
        ny = 1;
        num_dims++;
    }
    else if (num_dims >= 1) 
        ny = in_dims[num_dims-1];
    if (ny < self->nfields) { 
        PyErr_SetString(PyExc_ValueError, 
            "Length of final axis on input array must be at least 'nfields' long.");
        Py_DECREF(Yin); return NULL;
    }

    num_pts = 1;
    for (i=0; i < num_dims - 1; i++) 
        num_pts *= in_dims[i];

    Yout = PyArray_SimpleNew(num_dims-1, in_dims, NPY_DOUBLE);
    y_in = PyArray_DATA((PyArrayObject *)Yin);
    y_out = PyArray_DATA((PyArrayObject *)Yout);
    self->V((PyObject *)self, num_pts, ny, y_in, y_out);
    Py_DECREF(Yin);
    return Yout;   
}

static PyObject *model_obj_dV(ModelObject *self, PyObject *args, PyObject *keywds) {
    PyObject *Yin, *Yout;
    npy_intp num_pts, num_dims, *in_dims, ny=0, i;
    double *y_in, *y_out;
    
    static char *kwlist[] = {"y", "one_dim", NULL};
    int one_dim = 0;
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|i", kwlist,
        &Yin, &one_dim))
        return NULL;
    if (one_dim && self->nfields > 1) {
        PyErr_SetString(PyExc_ValueError, 
            "Field can only be one-dimensional "
            "('one_dim' parameter set to True) if nfields == 1");
        return NULL;
    }
    Yin = PyArray_FROM_OTF(Yin, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (Yin == NULL) return NULL;
    if (PyArray_Size(Yin) == 0) 
        return Yin; // Empty array. Pass through.
    
    num_dims = PyArray_NDIM((PyArrayObject *)Yin);
    in_dims = PyArray_DIMS((PyArrayObject *)Yin);
    if (one_dim) {
        ny = 1;
        num_dims++;
    }
    else if (num_dims >= 1) 
        ny = in_dims[num_dims-1];
    if (ny < self->nfields) { 
        PyErr_SetString(PyExc_ValueError, 
            "Length of final axis on input array must be at least 'nfields' long.");
        Py_DECREF(Yin); return NULL;
    }

    num_pts = 1;
    npy_intp out_dims[num_dims];
    for (i=0; i < num_dims - 1; i++) {
        num_pts *= in_dims[i];
        out_dims[i] = in_dims[i];
    }
    out_dims[num_dims-1] = self->nfields;
    if (one_dim) num_dims--;

    Yout = PyArray_SimpleNew(num_dims, out_dims, NPY_DOUBLE);
    y_in = PyArray_DATA((PyArrayObject *)Yin);
    y_out = PyArray_DATA((PyArrayObject *)Yout);
    self->dV((PyObject *)self, num_pts, ny, y_in, y_out);
    Py_DECREF(Yin);
    return Yout;
}

static PyObject * model_obj_check(ModelObject *self, PyObject *args) { 
    Py_INCREF(Py_True); 
    return Py_True; 
}

static PyMethodDef model_obj_methods[] = {
    {"V", (PyCFunction)model_obj_V, METH_VARARGS | METH_KEYWORDS, 
"V(phi, one_dim=False)\n"
"\n"
"Returns the potential as a function of the scalar field *phi*.\n"
"\n"
"If *one_dim* is False, then the final axis of *phi* should separate the\n"
"different field components and it should be at least *self.nfields* long. The\n"
"output array will have shape ``in_shape[:-1]``.\n"
"\n"
"If *one_dim* is True, then all input values are treated directly as field\n"
"values and the output array will have the same shape as the input array\n"
"(can be a single scalar).\n"
},
    {"dV", (PyCFunction)model_obj_dV, METH_VARARGS | METH_KEYWORDS, 
"dV(phi, one_dim=False)\n"
"\n"
"Returns the derivative of the potential as a function of the scalar field *phi*.\n"
"\n"
"If *one_dim* is False, then the final axis of *phi* should separate the\n"
"different field components and it should be at least *self.nfields* long. The\n"
"output array will have shape ``in_shape[:-1]+(nfields,)``.\n"
"\n"
"If *one_dim* is True, then all input values are treated directly as field\n"
"values and the output array will have the same shape as the input array\n"
"(can be a single scalar).\n"
},
    {"check_model_object", (PyCFunction)model_obj_check, METH_NOARGS, 
"check_model_object()\n\nReturns True.\n"
},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/*
static void prefill_model_obj_type(PyTypeObject *type, PyMethodDef *methods) {
    type->tp_basicsize = sizeof(ModelObject);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    type->tp_new = PyType_GenericNew;
    type->tp_dealloc = (destructor)&model_obj_dealloc;
    if (methods == NULL)
        type->tp_methods = model_obj_methods;
    else {
        int i=0;
        PyMethodDef *methods2 = methods;
        while (methods2->ml_name) {
            i++;
            methods2 = methods2+1;
        }
        PyMethodDef *methods3 = malloc(sizeof(PyMethodDef)*(4+i));
        memcpy(methods3, model_obj_methods, sizeof(PyMethodDef)*3);
        memcpy(methods3+3, methods, sizeof(PyMethodDef)*(i+1));
        type->tp_methods = methods3;
    }
}
*/
PyTypeObject model_object_type = {
    PyObject_HEAD_INIT(NULL)
};

static const char *model_obj_docstring = 
"The base class for model objects. Defines the python interface to the\n"
"potential functions and defines the structure which can be passed to other\n"
"C functions.\n";

static int init_model_obj() {
    model_object_type.tp_basicsize = sizeof(ModelObject);
    model_object_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    model_object_type.tp_new = PyType_GenericNew;

    model_object_type.tp_name = "models.ModelObject";
    model_object_type.tp_doc = model_obj_docstring;
    model_object_type.tp_methods = model_obj_methods;
    
    if (PyType_Ready(&model_object_type) < 0)
        return -1;
    Py_INCREF(&model_object_type);
    return 0;
}



#pragma mark --Test model

// A super simple test model with a two-field quadratic potential

typedef struct {
    ModelObject base;
    double m1_sq;
    double m2_sq;
    double m12_sq;
} TestModelObject;

static int test_model_V(
    TestModelObject *self, 
    npy_intp numpts, npy_intp ny, double *y_in, double *y_out)
{
    npy_intp i;
    for (i=0; i<numpts; i++) {
        double phi1 = y_in[i*ny];
        double phi2 = y_in[i*ny+1];
        y_out[i] = 0.5*self->m1_sq * phi1*phi1 
            + 0.5*self->m2_sq * phi2*phi2
            + self->m12_sq * phi1*phi2;
    }
    return 0;
}

static int test_model_dV(
    TestModelObject *self, 
    npy_intp numpts, npy_intp ny, double *y_in, double *y_out)
{
    npy_intp i;
    for (i=0; i<numpts; i++) {
        double phi1 = y_in[i*ny];
        double phi2 = y_in[i*ny+1];
        y_out[2*i] = self->m1_sq * phi1 +  self->m12_sq * phi2;
        y_out[2*i+1] = self->m2_sq * phi2 +  self->m12_sq * phi1;
    }
    return 0;
}

static int test_model_init(TestModelObject* self, PyObject *args, 
                             PyObject *keywds) {
    static char *kwlist[] = {"m1_sq","m2_sq","m12_sq",NULL};
    int success = PyArg_ParseTupleAndKeywords(
        args, keywds, "ddd", kwlist, &self->m1_sq, &self->m2_sq, &self->m12_sq);
    if (!success) return -1;
    self->base.V = (ScalarFunc)test_model_V;  
    self->base.dV = (ScalarFunc)test_model_dV;
    self->base.nfields = 2;
    return 0;
}

static PyTypeObject test_model_type = {
    PyObject_HEAD_INIT(NULL)
};

static const char *test_model_docstring = 
"TestModel(m1_sq, m2_sq, m3_sq)\n"
"\n"
"A test model using a quadratic potential and two scalar fields.\n"
"Not to be used for any sort of bubble collisions.\n";

int init_test_model_type() {
    test_model_type.tp_base = &model_object_type;
    test_model_type.tp_basicsize = sizeof(TestModelObject);
    test_model_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    test_model_type.tp_name = "models.TestModel";
    test_model_type.tp_doc = test_model_docstring;
    test_model_type.tp_init = (initproc)test_model_init;
    
    if (PyType_Ready(&test_model_type) < 0)
        return -1;
    Py_INCREF(&test_model_type);
    return 0;
}


#pragma mark --Generic piecewise, no hilltop
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
typedef struct {
    double pa_bar, pb_bar; // barrier position
    double pa_qmin, pb_qmin; // would-be quartic minimum (away from phi = 0)
    double lambda_a, lambda_b; // quartic coefficient
    double pa_edge, pb_edge; // point separating the quartic from quadratic regions
    double ma, mb; // mass-squared of the inflationary phase
    double pa_vac, pb_vac; // The field at the true vacuum.
    double v0; // Value of the false vacuum (should always be 3/8pi)
} GenericPiecewise_noHilltop_params;
static GenericPiecewise_noHilltop_params gpnh_default_params = {
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

typedef struct {
    ModelObject base;
    GenericPiecewise_noHilltop_params params;
} GenericPiecewise_NoHilltop_Model;

static int V_genericPiecewise_noHilltop(
    GenericPiecewise_NoHilltop_Model *self,
    npy_intp nx, npy_intp ny, double *y, double *V) 
{
    npy_intp i;
    for (i=0; i<nx; i++) {
        double p = y[i*ny];
        
        if (p > self->params.pa_edge && self->params.pa_edge > 0) {
            double delta_p = p - self->params.pa_vac;
            V[i] = 0.5 * self->params.ma * delta_p * delta_p;
        }
        else if (p >= 0) {
            V[i] = self->params.v0 + self->params.lambda_a*p*p*(
                0.5*self->params.pa_bar*self->params.pa_qmin 
                + p*(0.25*p - (self->params.pa_bar+self->params.pa_qmin)/3.0));
        }
        else if (p >= self->params.pb_edge || self->params.pb_edge >= 0) {
            V[i] = self->params.v0 + self->params.lambda_b*p*p*(
                0.5*self->params.pb_bar*self->params.pb_qmin 
                + p*(0.25*p - (self->params.pb_bar+self->params.pb_qmin)/3.0));
        }
        else {
            double delta_p = p - self->params.pb_vac;
            V[i] = 0.5 * self->params.mb * delta_p * delta_p;
        }
    }
    return 0;
}

static int dV_genericPiecewise_noHilltop(
    GenericPiecewise_NoHilltop_Model *self,
    npy_intp nx, npy_intp ny, double *y, double *dV)
{
    npy_intp i;    
    for (i=0; i<nx; i++) {
        double p = y[i*ny];
        
        if (p > self->params.pa_edge && self->params.pa_edge > 0) {
            double delta_p = p - self->params.pa_vac;
            dV[i] = self->params.ma * delta_p;
        }
        else if (p >= 0) {
            dV[i] = self->params.lambda_a * p * (p-self->params.pa_bar) * (p-self->params.pa_qmin);
        }
        else if (p >= self->params.pb_edge || self->params.pb_edge >= 0) {
            dV[i] = self->params.lambda_b * p * (p-self->params.pb_bar) * (p-self->params.pb_qmin);
        }
        else {
            double delta_p = p - self->params.pb_vac;
            dV[i] = self->params.mb * delta_p;
        }
    }
    return 0;
}

static double delta_dV_genericPiecewise_noHilltopHelper(
        double p_bar, double p_qmin, double lambda, double p_vac,
        double p_edge, double v0, double *m_out) {
    double p = p_edge;
    double V = v0 + lambda*p*p*(0.5*p_bar*p_qmin + p*(0.25*p - (p_bar+p_qmin)/3.0));
    double dV = lambda * p * (p-p_bar) * (p-p_qmin);
    // Want to match at p_edge: V = 0.5*m*(p-p_vac)^2
    double delta_p = p - p_vac;
    *m_out = 2*V / (delta_p*delta_p);
    
    return dV - *m_out * delta_p;
}

PyObject *genericPiecewise_noHilltop_setParams(
        GenericPiecewise_NoHilltop_Model *self, PyObject *args, PyObject *keywds) {
    // Need to parse the keywords twice. First to see if we're setting the
    // positive or negative attributes, then to actually set them.
    static char *kwlist[] = {
        "omega","mu","Delta_phi","phi0", "posneg", NULL};
    int posneg = +1;
    double barrier_pos, mu, p_qmin, p_vac;
    int success = PyArg_ParseTupleAndKeywords(args, keywds, "|ddddi", kwlist, 
        &barrier_pos, &mu, &p_qmin, &p_vac, &posneg);
    if (!success)
        return NULL;
    if (posneg != 1 && posneg != -1) {
        PyErr_SetString(PyExc_ValueError, 
            "The 'posneg' parameter should either be +1 (for setting the "
            "positive half of the potential) or -1 (for setting the negative half)");
        return NULL;
    }

    GenericPiecewise_noHilltop_params *params = &(self->params);
    p_qmin = posneg > 0 ? params->pa_qmin : params->pb_qmin;
    p_vac = posneg > 0 ? params->pa_vac : params->pb_vac;
    double p_bar = posneg > 0 ? params->pa_bar : params->pb_bar;
    double V_qmin;
    V_genericPiecewise_noHilltop(self, 1, 1, &p_qmin, &V_qmin);
    double Hsq_ratio = V_qmin / params->v0;
    mu = (1.0/Hsq_ratio) - 1;
    barrier_pos = p_bar / p_qmin;
    
    PyArg_ParseTupleAndKeywords(args, keywds, "|ddddi", kwlist, 
        &barrier_pos, &mu, &p_qmin, &p_vac, &posneg);
    if (barrier_pos >= 0.5 || barrier_pos <= 0) {
        PyErr_SetString(PyExc_ValueError, "Must have 0 < omega < 0.5.");
        return NULL;
    }
    if (mu <= 0) {
        PyErr_SetString(PyExc_ValueError, "Must have mu > 0");
        return NULL;
    }

    Hsq_ratio = 1.0 / (mu+1);
    p_qmin = posneg*fabs(p_qmin);
    p_bar = barrier_pos * p_qmin;
    
    // Calculate the overall quartic coefficient
    double V_temp = p_qmin*p_qmin * ( 
        0.5*p_bar*p_qmin + p_qmin*(0.25*p_qmin - (p_bar+p_qmin)/3.0));
    // v0 + lambda * V_temp = V_qmin
    // V_qmin / v0 = Hsq_ratio
    double lambda = params->v0 * (Hsq_ratio - 1) / V_temp;
    
    // Try to match the quartic and quadratic pieces
    // First need to find the inflection point along the quartic. 
    // Start the matching search there.
    double p_inf = ( (p_bar+p_qmin) 
        + posneg*sqrt( (p_bar-p_qmin)*(p_bar-p_qmin) + p_bar*p_qmin ) ) / 3.0;
    // Now do a binary search.
    double m_vac, p_edge; // These are the two things we need to find.
    double p1 = p_inf, p2 = p_qmin;
    double DV1 = delta_dV_genericPiecewise_noHilltopHelper(
        p_bar, p_qmin, lambda, p_vac, p1, params->v0, &m_vac);
    double DV2 = delta_dV_genericPiecewise_noHilltopHelper(
        p_bar, p_qmin, lambda, p_vac, p2, params->v0, &m_vac);
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
            double DVedge = delta_dV_genericPiecewise_noHilltopHelper(
                p_bar, p_qmin, lambda, p_vac, p_edge, params->v0, &m_vac);
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
        params->pa_bar = p_bar;
        params->pa_qmin = p_qmin;
        params->lambda_a = lambda;
        params->pa_edge = p_edge;
        params->ma = m_vac;
        params->pa_vac = p_vac;
    }
    else {
        params->pb_bar = p_bar;
        params->pb_qmin = p_qmin;
        params->lambda_b = lambda;
        params->pb_edge = p_edge;
        params->mb = m_vac;
        params->pb_vac = p_vac;     
    }
    
    Py_INCREF(Py_None);
    return Py_None;
}

static int genericPiecewise_noHilltop_init(
        GenericPiecewise_NoHilltop_Model *self, PyObject *args, PyObject *keywds) {
    self->params = gpnh_default_params;
    self->base.V = (ScalarFunc)V_genericPiecewise_noHilltop;
    self->base.dV = (ScalarFunc)dV_genericPiecewise_noHilltop;
    self->base.nfields = 1;
    PyObject *success = genericPiecewise_noHilltop_setParams(self, args, keywds);
    if (!success) return -1;
    Py_DECREF(success);
    return 0;
}

static PyTypeObject genericPiecewise_noHilltop_type = {
    PyObject_HEAD_INIT(NULL)
};

static PyMethodDef genericPiecewise_noHilltop_methods[] = {
    {"setParams", (PyCFunction)genericPiecewise_noHilltop_setParams, 
        METH_VARARGS | METH_KEYWORDS, 
"setParams(omega=None, mu=None, Delta_phi=None, phi0=None, posneg=+1)\n\n"
"Set the model parameters individually or all at once."
    },
    {NULL}  
};

static const char *genericPiecewise_noHilltop_docstring = 
"A piecewise potential with a quartic barrier and a quadratic slow-roll phase.\n"
"This is the same potential used in `arXiv:1407.2950`_, and the parameters are\n"
"described in more detail there.\n"
"\n"
".. _`arXiv:1407.2950`: http://arxiv.org/abs/arXiv:1407.2950\n"
"\n"
"Parameters\n"
"----------\n"
"omega : float, optional\n"
"    The ratio of field values at the top and bottom of the quartic barrier.\n"
"    Must satisfy *0 < omega < 0.5*.\n"
"mu : float, optional\n"
"    The difference between the metastable false-vacuum energy and the\n"
"    inflationary vacuum energy relative to the inflationary energy.\n"
"    Must satisfy *mu > 0*.\n"
"Delta_phi : float, optional\n"
"    The distance in field space between the metastable minimum and the bottom\n"
"    of the quartic barrier.\n"
"phi0 : float, optional\n"
"    The location of the absolute minimum.\n"
"posneg : int, optional\n"
"    Use +1 (default) when setting the parameters at positive field values, and\n"
"    -1 when setting the parameters at negative field values.\n";

int init_genericPiecewise_noHilltop_type() {
    genericPiecewise_noHilltop_type.tp_base = &model_object_type;
    genericPiecewise_noHilltop_type.tp_basicsize = sizeof(GenericPiecewise_NoHilltop_Model);
    genericPiecewise_noHilltop_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

    genericPiecewise_noHilltop_type.tp_name = "models.GenericPiecewise_NoHilltop_Model";
    genericPiecewise_noHilltop_type.tp_doc = genericPiecewise_noHilltop_docstring;
    genericPiecewise_noHilltop_type.tp_init = (initproc)genericPiecewise_noHilltop_init;
    genericPiecewise_noHilltop_type.tp_methods = genericPiecewise_noHilltop_methods;
    
    if (PyType_Ready(&genericPiecewise_noHilltop_type) < 0)
        return -1;
    Py_INCREF(&genericPiecewise_noHilltop_type);
    return 0;
}


#pragma mark --QuadAndGaussian
// --------------------------------------
// A simple quadratic potential with a gaussian bump.
// --------------------------------------

typedef struct {
    ModelObject base;
    double m_sq, A, sigma_sq, phi0;
    double phi_meta_min, bump_height, bump_width;
} QuadGaussModel;

static int V_quadGauss(QuadGaussModel *self, 
        npy_intp nx, npy_intp ny, double *y, double *V) 
{
    npy_intp i;
    for (i=0; i<nx; i++) {
        double phi = y[i*ny];
        V[i] = self->A * exp(-0.5 * phi*phi / self->sigma_sq );
        phi -= self->phi0;
        V[i] += 0.5*self->m_sq*phi*phi;
    }        
    return 0;
}

static int dV_quadGauss(QuadGaussModel *self, 
        npy_intp nx, npy_intp ny, double *y, double *dV) 
{
    npy_intp i;
    for (i=0; i<nx; i++) {
        double phi = y[i*ny];
        dV[i] = self->A * exp(-0.5 * phi*phi / self->sigma_sq );
        dV[i] *= -phi/self->sigma_sq;
        phi -= self->phi0;
        dV[i] += self->m_sq*phi;
    }
    return 0;
}

static int quadGauss_init(
        QuadGaussModel *self, PyObject *args, PyObject *keywds) 
{
    self->base.V = (ScalarFunc)V_quadGauss;
    self->base.dV = (ScalarFunc)dV_quadGauss;
    self->base.nfields = 1;

    static char *kwlist[] = {"bump_height","bump_width","phi0", NULL};
    double height, width, phi0;
    int success = PyArg_ParseTupleAndKeywords(
        args, keywds, "ddd", kwlist, &height, &width, &phi0);
    if (!success) return -1;
    if (fabs(height) <= 1) {
        PyErr_SetString(PyExc_ValueError, "Must have |height| > 1");
        return -1;
    }

    self->sigma_sq = width*width;
    self->m_sq = 1.0;
    self->phi0 = phi0;
    self->bump_height = height;
    self->bump_width = width;
    self->A = fabs(width*phi0)*exp(0.5)*height;

    // Find the minimum
    double phi1 = height*phi0 > 0 ? -width : +width;
    double phi2 = phi0 > 0 ? -20*width : +20*width;
    double phi_mid = 0.5 * (phi1+phi2);
    double dV1, dV2, dV_mid;
    dV_quadGauss(self, 1,1, &phi1, &dV1);
    dV_quadGauss(self, 1,1, &phi2, &dV2);
    if (dV1*dV2 >= 0) {
        PyErr_SetString(PyExc_ValueError, 
            "Cannot find the false vacuum minimum. "
            "The potential derivative has the same sign both near by and "
            "far away from the bump.");
        return -1;
    }
    while (fabs(phi1-phi2) > 1e-9*fabs(width)) {
        phi_mid = 0.5 * (phi1+phi2);
        dV_quadGauss(self, 1,1, &phi_mid, &dV_mid);
        if (dV_mid * dV1 > 0)
            phi1 = phi_mid;
        else
            phi2 = phi_mid;
    }
    self->phi_meta_min = phi_mid;

    // Rescale the potential
    double VF;
    V_quadGauss(self, 1,1, &phi_mid, &VF);
    double Hf_sq = 8*PI * VF / 3.0;
    self->A /= Hf_sq;
    self->m_sq /= Hf_sq;

    return 0;
}

static PyMemberDef quadGauss_members[] = {
    {"bump_width", T_DOUBLE, offsetof(QuadGaussModel, bump_width), READONLY, NULL},
    {"bump_eight", T_DOUBLE, offsetof(QuadGaussModel, bump_height), READONLY, NULL},
    {"phi_meta_min", T_DOUBLE, offsetof(QuadGaussModel, phi_meta_min), READONLY, NULL},
    {"phi0", T_DOUBLE, offsetof(QuadGaussModel, phi0), READONLY, NULL},
    {"m_sq", T_DOUBLE, offsetof(QuadGaussModel, m_sq), READONLY, NULL},
    {"A", T_DOUBLE, offsetof(QuadGaussModel, A), READONLY, NULL},
    {NULL,0,0,0,NULL}  
};

static PyTypeObject quadGauss_type = {
    PyObject_HEAD_INIT(NULL)
};

static const char *quadGauss_type_docstring = 
"A potential characterized by a quadratic slow-roll phase and a gaussian\n"
"bump which creates a barrier and a metastable phase.\n"
"This is the same potential used in `arXiv:1407.2950`_, and the parameters are\n"
"described in more detail there.\n"
"\n"
".. _`arXiv:1407.2950`: http://arxiv.org/abs/arXiv:1407.2950\n"
"\n"
"Parameters\n"
"----------\n"
"bump_height : float\n"
"    The height of the bump relative to the slope of the quadratic potential.\n"
"    Can be either positive or negative (for a dip instead of a bump), but must\n"
"    satisfy ``|bump_height| > 1``.\n"
"bump_width : float\n"
"    The width (standard deviation) of the gaussian bump.\n"
"phi0 : float\n"
"    The location of the inflationary minimum.\n";

int init_quadGauss_type() {
    quadGauss_type.tp_base = &model_object_type;
    quadGauss_type.tp_basicsize = sizeof(QuadGaussModel);
    quadGauss_type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;

    quadGauss_type.tp_name = "models.QuadGaussModel";
    quadGauss_type.tp_doc = quadGauss_type_docstring;
    quadGauss_type.tp_init = (initproc)quadGauss_init;
    quadGauss_type.tp_members = quadGauss_members;
    
    if (PyType_Ready(&quadGauss_type) < 0)
        return -1;
    Py_INCREF(&quadGauss_type);
    return 0;
}


#pragma mark --Initialize the module

static PyMethodDef _methods[] = {
    {NULL, NULL, 0, NULL}   // Sentinel   
};

static const char *models_docstring =
"This module contains various models (defined by their scalar potentials)\n"
"for use in the collision simulations.\n"
"\n"
"All models should inherit from :class:`ModelObject`.\n";

PyMODINIT_FUNC
initmodels(void)
{
    PyObject *module = Py_InitModule3("models", _methods, models_docstring);
    
    if(init_model_obj() >= 0)
        PyModule_AddObject(module, "ModelObject", (PyObject *)&model_object_type);
    if(init_test_model_type() >= 0)
        PyModule_AddObject(module, "TestModel", (PyObject *)&test_model_type);
    if(init_genericPiecewise_noHilltop_type() >= 0)
        PyModule_AddObject(module, "GenericPiecewise_NoHilltop_Model", 
            (PyObject *)&genericPiecewise_noHilltop_type);
    if(init_quadGauss_type() >= 0)
        PyModule_AddObject(module, "QuadGaussModel", (PyObject *)&quadGauss_type);    

    #ifndef NO_IMPORT_ARRAY
    import_array();
    #endif
}

