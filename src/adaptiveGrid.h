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
 
// Each separate region will handle single evolution steps separately. This is useful
// when different regions have different step sizes that need to be matched up later.
typedef struct Region {
	int startIndex;
	int width;	// width of the region, not counting overlap
	int leftOverlap;	// How much into the previous region we extend.
	int rightOverlap;	// How much into the next region we extend.
	int N;	// log2 of the region's stepsize relative to the smallest region's stepsize.
	char flag;	// stores whether a region is borderline in N.
	struct Region *nextReg;
	struct Region *prevReg;
	double *y0;	// The data to evolve.
	double *y1, *y2, *dydt; // Temporary holders for the data and its derivatives.
} Region;


Region * removeRegion(Region *r);

void printRegions(Region *r);

Region * makeRegions(double *dx, int nx, double dxmin, int Nmax, double eps, int minWidth);
	// The makeRegions() function will return a pointer to the first region.
	// It does not initialize any data (use fillRegions).
	// - dx, nx gives the input grid.
	// - dxmin is the smallest spatial step on the grid. Anything within a factor
	//   of (about) two of this will be N=0.
	// - Nmax determines the largest N for a region.
	// - eps determines how much slop there is in that factor of two.
	// - minWidth is the smallest width of the output regions. Should be
	//   at least 16 (for two RK4 steps with 4th order derivs).


void calcDerivCoefs(double *x, int nx, double *c1 /*out*/, double *c2 /*out*/);
	/* 
	calcDerivCoefs() calculates the coefficients used for finite differences with
	non-uniform step size. The arrays c1 and c2 should be allocated with size 5*nx
	each. c1 gives first-derivs, c2 gives second-derivs.
	To actually calculate the derivatives, use
		dy[i] = y[i-2]*c1[i*5] + y[i-1]*c1[i*5+1] + y[i]*c1[i*5+2] 
				+ y[i+1]*c1[i*5+3] + y[i+2]*c1[i*5+4]
	and similarly for c2.
	For the first and last two indicies, the differences are not centered, so, for example,
		dy[0] = c[0]*y[0] + c[1]*y[1] + c[2]*y[2] + c[3]*y[3] + c[4]*y[4]
		dy[1] = c[5]*y[0] + c[6]*y[1] + c[7]*y[2] + c[8]*y[3] + c[9]*y[4]
	*/

void calcDerivs(double *y, double *c, int nx, int nyin, int kin, double *dyout, int nyout, int kout);
	// Calculate the derivative of y given the derivative coefficients c.
	// kin and kout are so that we can calculate for any index of the field in y (if it is multi-dimensional).


void fillRegions(Region *r0, double *data, int ny);
	// This function allocates memory for the data for each region, and then
	// fills the allocated memory from data.

double * dataFromRegions(Region *r0, int ny, int *nx /*out*/);

void deallocRegions(Region *r0);


int evolveRegions(Region *r0, double dt, double t0, int nmax, int ny,
				   double *c1, double *c2,
				   int (*dY)(double, double *, double *, double *, int, double *));
	// r0 is start region
	// dt is the shortest time-step, t0 is starting time
	// nmax is number of (shortest) steps to evolve.
	// dY is the derivative function. dY(t, y, c1, c2, nx, dy_out) (and nc = nx-4)
	// Note that the data should be shaped like (nx, ny), such that continous
	// blocks of memory are continuous in x. Returns error code (0 is no error).

	// Each derivative needs to extend into neighboring regions. This is why we need leftOverlap and rightOverlap.
	/*
	 r.prev    r
	 - ........|... y0
	 -   ......|... k1
	 -     ....|... k2
	 -       ..|... k3
	 -         |... k4 and y1
	 */
	// Each subsequent RK step needs to extend two less into the overlap region.
	// 
	// Returns 0 for no error.

double *remakeGrid(double *x, double *grid_density, int nx, double grid_uniformity, double xmin, double xmax, int *nx_out);

double *interpGrid(double *xold, int nold, double *yold, int ny, double *xnew, int nnew);

