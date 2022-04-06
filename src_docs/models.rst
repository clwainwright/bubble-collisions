models
----------------------------------------

.. automodule:: bubble_collisions.models

Creating new models
~~~~~~~~~~~~~~~~~~~
Creating a new scalar field model is a relatively straightforward procedure, and most of the coding is boilerplate. Let's walk through the pieces needed to create the :class:`TestModel` class.

First, we need to create a new structure to hold instances of the model. In the rare case where the model has no parameters, this can step can be skipped and instead one can just use the ``ModelObject`` structure. In all other cases, we'll need to define a basic structure with some parameters:

.. code-block :: c

    typedef struct {
        ModelObject base;
        double m1_sq;
        double m2_sq;
        double m12_sq;
    } TestModelObject;

Here, the various *m*-parameters represent mass-squared values that go into a 2-field quadratic potential.

Next, we need to define the potential:

.. code-block :: c

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

The call signature should be exactly the same for all different models, except that each model should use a pointer to its own model structure. This data at this pointer will contain all of the parameters that we need in the potential. The input *y_in* is going to be an array of *num_pts* points, each with dimension *ny*. This will generally be larger than the total number of field dimensions *nfield*. The output *y_out* will be pre-allocated to an array of size *numpts*. The job of this function is to fill in each output point.
The inner for loop iterates over all the points, retrieving the field values from *y_in* and storing the result in *y_out*. When finished, the function returns 0 for success or -1 if there was an error (in which case an error message should probably be set using ``PyErr_SetString()``).

The potential gradient has a very similar structure, but now the output array has length ``numpts*nfields = numpts*2``, and the output points must be filled appropriately:

.. code-block :: c

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

Next, we need to create an initialization function. For a simple model it's pretty straightforward.

.. code-block :: c

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

All models with parameters will need to set them here. See the Python docs on `parsing arguments <https://docs.python.org/2/c-api/arg.html>`_ for more info on how to do this. Then, all models will need to link to their potential and derivative functions and specify how many fields they contain. Again, return 0 for success.

We now have all of the functions set up for the model, but we still need to create a model type so that Python will know that the model exists. To do this, we create a *PyTypeObject* and a function which will initialize it:

.. code-block :: c

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

This is all essentially boilerplate and can be directly copied for other models, except for changes in variable names (and docstring). Additional properties can also be set here, such as additional class methods.

Finally, we need to add the new type in the models module initialization function:

.. code-block :: c

    PyMODINIT_FUNC initmodels(void) {
        // ...

        if(init_test_model_type() >= 0)
            PyModule_AddObject(module, "TestModel", (PyObject *)&test_model_type);

        // ...
    }

And that should be it! Lots more information found in the Python docs: `defining new types <https://docs.python.org/2/extending/newtypes.html>`_.

Base Model
~~~~~~~~~~
.. autoclass:: ModelObject
    :members:

Inflationary Models
~~~~~~~~~~~~~~~~~~~
.. autoclass:: GenericPiecewise_NoHilltop_Model
    :members:
    :show-inheritance:

.. autoclass:: QuadGaussModel
    :members:
    :show-inheritance:

.. autoclass:: TestModel
    :members:
    :show-inheritance:    
