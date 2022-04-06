
/*
 The following function will find the approximate value of the fields/metrics/Christoffel symbols
 at the desired point(s) on the simulation using bicubic interpolation. Note that the input data
 should be read in from "bubbleCollision.readFromFile2()".
*/

extern PyMethodDef valsOnGrid_methdef;
PyObject *valsOnGrid_toPy(PyObject *self, PyObject *args, PyObject *keywds);

int indexInSorted(double x0 /*target*/, double *x /*list to search*/, int nx);
// Does a simple binary search. Returns the first index which is larger than x0
// (or nx if no index is larger).