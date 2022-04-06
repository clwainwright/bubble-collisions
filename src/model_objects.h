#include <Python.h>

// A model_object is a python wrapper around C functions which determine
// a scalar field model. It has the following fields:
//
//    nfields : npy_intp 
//        The number of fields in the potential.
//    V : ScalarFunc
//        A scalar potential that operates over a field. Inputs are:
//            self - The model object. May contain additional parameters in
//                its struct.
//            numpts - The number of input field points
//            ny - The number of input fields per point. Can be bigger than
//                nfields. Effectively determines the spacing between adjacent
//                points in y_in.
//            y_in - Input data. Must be at least ``numpts*ny`` elements long.
//            y_out - Output data. Must be at least ``numpts`` elements long.
//        It should return -1 on error, 0 on success.
//    dV : ScalarFunc
//        Gradient of V in field space. It has the same calling signature as V,
//        but the output is ``numpts*nfields`` long. The output is ordered with
//        the gradients along different field directions grouped together for
//        individual points.
// 
// All ``model_object`` objects should also have a python function
// ``check_model_object()`` to identify themselves as such.

typedef int (*ScalarFunc)(
    PyObject *self, npy_intp numpts, npy_intp ny, double *y_in, double *y_out);

typedef struct {
    PyObject_HEAD
    npy_intp nfields;
    ScalarFunc V;
    ScalarFunc dV;
} ModelObject;


