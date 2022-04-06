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
 
#include <math.h>
#include <stdlib.h> // for malloc(), free()
#include <string.h> // for memcpy()
#include <stdio.h> // for printf()
#include <adaptiveGrid.h>

#define VERBOSE 0
#if VERBOSE
	#define BLURT() printf("%s(%d) -- %s()\n", __FILE__, __LINE__, __func__)
	#define LOGMSG(...) printf("%s(%d): ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n")
#else
	#define BLURT() ;
	#define LOGMSG(...) ;
#endif

#define pow2N(N) (1 << N)


Region * removeRegion(Region *r) {
	// This will remove a region from the doubley linked list, and return
	// the previous region (or next region, if that's NULL). 
	// If neither the next nor the previous region is NULL, it removes
	// the next region too.
	// This is a helper function for makeRegions()
	
	Region *rtemp;
	
	if (r == NULL)
		return NULL;
	
	if (r->prevReg == NULL) {
		rtemp = r->nextReg;
		rtemp->startIndex = 0;
		if (rtemp != NULL)
			rtemp->prevReg = NULL;
		free(r);
	}
	else if (r->nextReg == NULL) {
		rtemp = r->prevReg;
		rtemp->nextReg = NULL;
		free(r);
	}
	else {
		rtemp = r->prevReg;
		rtemp->nextReg = r->nextReg->nextReg;
		rtemp->flag &= r->nextReg->flag;
		if (rtemp->nextReg != NULL)
			rtemp->nextReg->prevReg = rtemp;
		free(r->nextReg);
		free(r);
	}
	return rtemp;
}

void printRegions(Region *r) {
	while (r) {
		LOGMSG("reg -- i0:%i, w:%i, N:%i, flag:%i", r->startIndex, r->width, r->N, r->flag);
		r = r->nextReg;
	}
}

// The makeRegions() function will return a pointer to the first region.
Region * makeRegions(double *dx, int nx, double dxmin, int Nmax, double eps, int minWidth) 
{
	Region *r, *startRegion;
	double log2dx, extra, N;
	int i, width;
	
	if (nx < 2)
		return NULL;
				
/*	dx = malloc(sizeof(double)*nx);
	dx[0] = x[1]-x[0];
	dx[nx-1] = x[nx-1]-x[nx-2];
	for (i=1; i<nx-1; i++)
		dx[i] = 0.5*(x[i+1]-x[i-1]);
	
	dxmin_ = dx[0];
	for (i=1; i<nx; i++) {
		if (dx[i] < dxmin_)
			dxmin_ = dx[i];
	}
	if (dxmin)
		*dxmin = dxmin_;*/


	r = startRegion = malloc(sizeof(Region));
	log2dx = log(dx[0]/dxmin)/M_LN2;
	extra = (log2dx - floor(log2dx));
	r->N = floor(log2dx);
	r->flag = extra <= eps ? 1 : (extra > 1-eps ? 2 : 0);
		// 1 is borderline low, 2 is borderline  high, 0 is not borderline.
	r->prevReg = NULL;
	r->startIndex = 0;
	for (i=1; i<nx; i++) {
		log2dx = log(dx[i]/dxmin)/M_LN2;
		N = floor(log2dx);
		if (N > Nmax) N = Nmax;
		if (N != r->N) {
			// Make sure that we're not skipping more than one value of N at a time
			while (r->N < N-1) {
				r->nextReg = malloc(sizeof(Region));
				r->nextReg->prevReg = r;
				r = r->nextReg;
				r->startIndex = i;
				r->flag = 0;
				r->N = r->prevReg->N+1; 
			}
			while (r->N > N+1) {
				r->nextReg = malloc(sizeof(Region));
				r->nextReg->prevReg = r;
				r = r->nextReg;
				r->startIndex = i;
				r->flag = 0;
				r->N = r->prevReg->N-1; 
			}
			// Add a new region.
			r->nextReg = malloc(sizeof(Region));
			r->nextReg->prevReg = r;
			r = r->nextReg;
			r->startIndex = i;
			r->flag = 3; // = 1|2
			r->N = N;
		}
		// Check for borderline
		extra = (log2dx - floor(log2dx));
		r->flag &= ((extra <= eps) ? 1 : (extra > 1-eps ? 2 : 0));
	}
	r->nextReg = NULL;
	
	// Go through all of the regions and adjust the borderline regions.
	// If a region is borderline low (borderline=1) and surrounded by regions with a lower N,
	// drop the region down to that N.
	// Then do the reverse for borderline high regions (borderline=2).
	r = startRegion;
	while (r && (r->prevReg || r->nextReg)) {
		if (r->flag == 1 && (r->prevReg == NULL || r->prevReg->N < r->N) && (r->nextReg == NULL || r->nextReg->N < r->N)) {
			r = removeRegion(r);
			if (r->prevReg == NULL)
				startRegion = r;
		}
		else
			r = r->nextReg;
	}
	r = startRegion;
	while (r && (r->prevReg || r->nextReg)) {
		if (r->flag == 2 && (r->prevReg == NULL || r->prevReg->N > r->N) && (r->nextReg == NULL || r->nextReg->N > r->N)) {
			r = removeRegion(r);
			if (r->prevReg == NULL)
				startRegion = r;
		}
		else
			r = r->nextReg;
	}
			
	// At this point we have a doubly linked list of regions, each of which has
	// N, startIndex, flag, prevReg and nextReg filled.
	// Now, we want to make sure that no region is wider than minWidth.
	
	// Keep looping until all regions are at least minWidth, or stop when we're down to one region.
	while (startRegion && startRegion->nextReg) {
		// Remove all big spikes.
		r = startRegion;
		while (r && (r->prevReg || r->nextReg)) {
			width = r->nextReg ? r->nextReg->startIndex - r->startIndex : nx - r->startIndex;
			if (width < minWidth && (r->prevReg == NULL || r->prevReg->N < r->N) && (r->nextReg == NULL || r->nextReg->N < r->N) ) {
				r = removeRegion(r);
				if (r->prevReg == NULL)
					startRegion = r;
			}
			else
				r = r->nextReg;
		}
		
		// Big spikes are gone, but can still have quick dips or steps
		r = startRegion;
		while (r && (r->prevReg || r->nextReg)) {
			width = r->nextReg ? r->nextReg->startIndex - r->startIndex : nx - r->startIndex;
			if (width < minWidth) {
				if ((r->prevReg != NULL && r->prevReg->N > r->N) && (r->nextReg != NULL && r->nextReg->N > r->N)) {
					// small dip, expand in both directions
					r->startIndex -= (minWidth-width+1)/2;
					r->nextReg->startIndex += (minWidth-width+1)/2;
				}
				else if (r->prevReg != NULL && r->prevReg->N > r->N) {
					// expand to the left.
					r->startIndex -= (minWidth-width);
				}
				else if (r->nextReg != NULL && r->nextReg->N > r->N) {
					// expand to the right.
					r->nextReg->startIndex += (minWidth-width);
				}
			}
			r = r->nextReg;
		}
		
		// Now see if all regions are smaller than minWidth.
		width = minWidth;
		r = startRegion;
		while (r->nextReg) {
			if (r->nextReg->startIndex - r->startIndex < width) {
				width--;
				break;
			}
			r = r->nextReg;
		}
		if (width == minWidth && nx - r->startIndex >= minWidth)
			break; // we're done!
		// Otherwise, we've still got something smaller than minWidth. Loop up and remove spikes.
	}
	
	// At this point, we have the right number of regions and N, flag, startIndex, and nextReg and prevReg
	// correctly set up. Let's fill in width, leftOverlap, and rightOverlap.
	
	r = startRegion;
	while (r) {
		r->width = r->nextReg ? r->nextReg->startIndex - r->startIndex : nx - r->startIndex;
		// If we have no neighbor, overlap is 0.
		// If the neighbor is coarse (larger N), overlap is 16.
		// If the neighbor is fine (smaller N), overlap is 8.
		r->leftOverlap  = r->prevReg ? (r->prevReg->N > r->N ? 16 : 8) : 0;
		r->rightOverlap = r->nextReg ? (r->nextReg->N > r->N ? 16 : 8) : 0;
		r = r->nextReg;
	}
		
	return startRegion;
}


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
void calcDerivCoefs(double *x, int nx, double *c1 /*out*/, double *c2 /*out*/)
{
	int i,j;
	double d1, d2, d3, d4;
	int i1,i2,i3,i4,i0;
	
	for (i=0; i<nx; i++) {
		if (i > 2 && i < nx-2)
			j++;
		else if (i == 0) {
			j=0; i0=0; i1=1; i2=2; i3=3; i4=4;
		}
		else if (i == 1) {
			j=0; i0=1; i1=0; i2=2; i3=3; i4=4;
		}
		else if (i == 2) {
			j=0; i0=2; i1=0; i2=1; i3=3; i4=4;
		}
		else if (i == nx-2) {
			j=nx-5; i0=3; i1=0; i2=1; i3=2; i4=4;
		}
		else if (i == nx-1) {
			j=nx-5; i0=4; i1=0; i2=1; i3=2; i4=3;
		}
		
		d1 = x[j+i1] - x[i];
		d2 = x[j+i2] - x[i];
		d3 = x[j+i3] - x[i];
		d4 = x[j+i4] - x[i];
		
		c1[i*5+i1] = (d4*d2*d3) / ( -d1 * (-d4*d2*d3 + d1 * (d4*d2+d2*d3+d3*d4 + d1 * (-d4+d1-d2-d3))) );
		c1[i*5+i2] = (d1*d4*d3) / ( -d2 * (-d1*d4*d3 + d2 * (d1*d4+d4*d3+d3*d1 + d2 * (-d4-d1+d2-d3))) );
		c1[i*5+i3] = (d1*d2*d4) / ( -d3 * (-d1*d2*d4 + d3 * (d1*d2+d2*d4+d4*d1 + d3 * (-d4-d1-d2+d3))) );
		c1[i*5+i4] = (d1*d2*d3) / ( -d4 * (-d1*d2*d3 + d4 * (d1*d2+d2*d3+d3*d1 + d4 * (+d4-d1-d2-d3))) );
		c1[i*5+i0] = -c1[i*5+i1]-c1[i*5+i2]-c1[i*5+i3]-c1[i*5+i4];
		
		c2[i*5+i1] = 2*(d4*d2+d2*d3+d3*d4) / (d1 * (-d4*d2*d3 + d1 * (d4*d2+d2*d3+d3*d4 + d1 * (-d4+d1-d2-d3) ) ) );
		c2[i*5+i2] = 2*(d1*d4+d4*d3+d3*d1) / (d2 * (-d1*d4*d3 + d2 * (d1*d4+d4*d3+d3*d1 + d2 * (-d4-d1+d2-d3) ) ) );
		c2[i*5+i3] = 2*(d1*d2+d2*d4+d4*d1) / (d3 * (-d1*d2*d4 + d3 * (d1*d2+d2*d4+d4*d1 + d3 * (-d4-d1-d2+d3) ) ) );
		c2[i*5+i4] = 2*(d1*d2+d2*d3+d3*d1) / (d4 * (-d1*d2*d3 + d4 * (d1*d2+d2*d3+d3*d1 + d4 * (+d4-d1-d2-d3) ) ) );
		c2[i*5+i0] = -c2[i*5+i1]-c2[i*5+i2]-c2[i*5+i3]-c2[i*5+i4];
	}
}

void calcDerivs(double *y, double *c, int nx, int nyin, int kin, double *dyout, int nyout, int kout) {
	int i;
	
	// middle
	for (i=2; i<nx-2; i++) {
		dyout[nyout*i+kout] = c[5*i]*y[nyin*(i-2)+kin] + c[5*i+1]*y[nyin*(i-1)+kin] + c[5*i+2]*y[nyin*i+kin]
		+ c[5*i+3]*y[nyin*(i+1)+kin] + c[5*i+4]*y[nyin*(i+2)+kin];
	//	LOGMSG("c[%i]: %0.3e  %0.3e  %0.3e  %0.3e  %0.3e", i, c[5*i], c[5*i+1], c[5*i+2], c[5*i+3], c[5*i+4]);
	}
	if(1) {
		// left boundary
		dyout[kout] = c[0]*y[kin] + c[1]*y[nyin+kin] + c[2]*y[2*nyin+kin] + c[3]*y[3*nyin+kin] +c[4]*y[4*nyin+kin];
		dyout[nyout+kout] = c[5]*y[kin] + c[6]*y[nyin+kin] + c[7]*y[2*nyin+kin] + c[8]*y[3*nyin+kin] +c[9]*y[4*nyin+kin];
		// right boundary
		dyout[nyout*(nx-2)+kout] = c[5*nx-10]*y[nyin*(nx-5)+kin] + c[5*nx-9]*y[nyin*(nx-4)+kin] + c[5*nx-8]*y[nyin*(nx-3)+kin]
		+ c[5*nx-7]*y[nyin*(nx-2)+kin] + c[5*nx-6]*y[nyin*(nx-1)+kin]; 
		dyout[nyout*(nx-1)+kout] = c[5*nx-5]*y[nyin*(nx-5)+kin] + c[5*nx-4]*y[nyin*(nx-4)+kin] + c[5*nx-3]*y[nyin*(nx-3)+kin]
		+ c[5*nx-2]*y[nyin*(nx-2)+kin] + c[5*nx-1]*y[nyin*(nx-1)+kin]; 
	}
	else {
		dyout[kout] = dyout[nyout+kout] = dyout[nyout*(nx-2)+kout] = dyout[nyout*(nx-1)+kout] = 0.0;
	}
	
}



void fillRegions(Region *r0, double *data, int ny) {
	// This function allocates memory for the data for each region, and then
	// fills the allocated memory from data.
	
	Region *r = r0;
	int i0,w;
	
	while (r) {
		w = r->width + r->leftOverlap + r->rightOverlap;
		i0 = r->startIndex - r->leftOverlap; 
		r->y0 = malloc(sizeof(double)*w*ny);
		memcpy(r->y0, data+i0*ny, w*ny*sizeof(double));
		r->y1 = malloc(sizeof(double)*w*ny);
		r->y2 = malloc(sizeof(double)*w*ny);
		r->dydt = malloc(sizeof(double)*w*ny);		
		r = r->nextReg;
	}
}

double * dataFromRegions(Region *r0, int ny, int *nx /*out*/) {
	Region *r;
	int i0;
	double *y;
	
	r = r0;
	i0 = 0;
	while (r) {
		i0 += r->width;
		r = r->nextReg;
	}
	
	y = malloc(sizeof(double)*i0*ny);
	r = r0;
	i0 = 0;
	while (r) {
		memcpy(y+i0*ny, r->y0 + r->leftOverlap*ny, r->width*ny*sizeof(double));
		i0 += r->width;
		r = r->nextReg;
	}
	
	if (nx)
		*nx = i0;
		
	return y;
}

void deallocRegions(Region *r0) {
	Region *r, *r2;
	r = r0;
	
	while (r) {
		r2 = r;
		r = r->nextReg;
		free(r2->y0);
		free(r2->y1);
		free(r2->y2);
		free(r2->dydt);
		free(r2);
	}
}

int evolveRegions(Region *r0, double dt, double t0, int nmax, int ny,
				   double *c1, double *c2,
				   int (*dY)(double, double *, double *, double *, int, double *)) {
	// r0 is start region
	// dt is the shortest time-step, t0 is starting time
	// nmax is number of (shortest) steps to evolve.
	// dY is the derivative function. dY(t, y, c1, c2, nx, dy_out) (and nc = nx-4)
	// Note that the data should be shaped like (nx, ny), such that continous
	// blocks of memory are continuous in x.
	
	// Each derivative needs to extend into neighboring regions. This is why we need leftOverlap and rightOverlap.
	/*
	  r.prev    r
	 ........|... y0
	   ......|... k1
	     ....|... k2
	       ..|... k3
	         |... k4 and y1
	*/
	// Each subsequent RK step needs to extend two less into the overlap region.
	
	int n, i, err;
	Region *r;
	int ic, iy, w, di, dw;
	double t, Dt;
	double *y0, *y1, *y2, *dydt;
	
	for (n=1; n <= nmax; n++) {
		r = r0;
		while (r) {
			// Only evolve those regions with n % r.N = 0
			if (n % pow2N(r->N) == 0) {
				Dt = pow2N(r->N) * dt;
				// First, set up the indicies and the width
				di = r->leftOverlap > 0 ? 2 : 0;
				dw = di + (r->rightOverlap > 0 ? 2 : 0);
				iy = 0;
				w = r->width + r->leftOverlap + r->rightOverlap;
			/*	if (n % (2*N) == 0) { // second step. Don't go as far into coarse regions.
					if (r->leftOverlap > di*4) {
						iy = di*4;
						w -= di*4;
					}
					if (r->rightOverlap > di*4)
						w -= di*4;
				}*/
				ic = (r->startIndex - r->leftOverlap + iy)*5;
				iy *= ny;
				
				y0 = r->y0; // Inital values at the start of this step
				y1 = r->y1; // Value for input into dY
				y2 = r->y2; // The cumulative values from this step
				dydt = r->dydt; // the derivative, output from Y
				
				// First step.
				t = t0 + (n - pow2N(r->N))*dt;
				LOGMSG("iy %i, ic %i, w %i", iy,ic,w);
				LOGMSG("startIndex %i, leftOverlap %i, di %i", r->startIndex, r->leftOverlap, di);
				LOGMSG("dt=%0.3e, N=%i", dt, r->N);
				err = dY(t, y0+iy, c1+ic, c2+ic, w, dydt+iy);
				if (err != 0) return err;
			//	di = dw = 0; // <--- should probably remove this later...
				iy += di*ny;
				ic += di*5;
				w -= dw;
				for (i=iy; i < iy+w*ny; i++) {
					y1[i] = y0[i] + 0.5 * Dt * dydt[i];
					y2[i] = dydt[i];
				}
				
				// second step.
				t += 0.5*Dt;
				err = dY(t, y1+iy, c1+ic, c2+ic, w, dydt+iy);
				if (err != 0) return err;
				iy += di*ny;
				ic += di*5;
				w -= dw;
				for (i=iy; i < iy+w*ny; i++) {
					y1[i] = y0[i] + 0.5 * Dt * dydt[i];
					y2[i] += 2 * dydt[i];
				}

				// third step.
				err = dY(t, y1+iy, c1+ic, c2+ic, w, dydt+iy);
				if (err != 0) return err;
				iy += di*ny;
				ic += di*5;
				w -= dw;
				for (i=iy; i < iy+w*ny; i++) {
					y1[i] = y0[i] + Dt * dydt[i];
					y2[i] += 2 * dydt[i];
				}

				// fourth step.
				t += 0.5*Dt;
				err = dY(t, y1+iy, c1+ic, c2+ic, w, dydt+iy);
				if (err != 0) return err;
				iy += di*ny;
				w -= dw;
				for (i=iy; i < iy+w*ny; i++) {
					y0[i] += Dt * (y2[i] + dydt[i]) / 6.0;
				}
			}
			r = r->nextReg;
		}
		
		// Now match regions.
		r = r0;
		while (r) {
			LOGMSG("n=%i",n);
			LOGMSG("Matching region: i0=%i, w=%i, L=%i, R=%i, N=%i", r->startIndex, r->width, r->leftOverlap, r->rightOverlap, r->N);
			if (n % pow2N(r->N) == 0 && r->prevReg && n % pow2N(r->prevReg->N) == 0) {
				LOGMSG("matching left");
				memcpy(r->y0, r->prevReg->y0 + ny*(r->prevReg->leftOverlap + r->prevReg->width - r->leftOverlap), 
					   ny*r->leftOverlap*sizeof(double));
			}
			if (n % pow2N(r->N) == 0 && r->nextReg && n % pow2N(r->nextReg->N) == 0) {
				LOGMSG("matching right");
				memcpy(r->y0 + ny*(r->leftOverlap + r->width), r->nextReg->y0 + ny*r->nextReg->leftOverlap, 
					   ny*r->rightOverlap*sizeof(double));
			}
			for (i=0; i < r->width + r->leftOverlap + r->rightOverlap; i++) {
				LOGMSG("%i:  %0.3e  %0.3e",i+r->startIndex-r->leftOverlap,r->y0[2*i],r->y0[2*i+1]);
			}
			r = r->nextReg;
		}
	}
	
	return 0;
}

double *remakeGrid(double *x, double *grid_density, int nx, double grid_uniformity, double xmin, double xmax, int *nx_out) {
	double *m = grid_density;
	double a = grid_uniformity, b,c,d;
	double mmin, npts_d, *xnew, msum, msum2, dx;
	int i, j, npts;
	
	// First, trim the x array such that it lies entirely within xmin and xmax
	if (x[nx-1] < xmin || x[0] > xmax)
		return NULL;
	BLURT();
	while (x[nx-1] > xmax) nx--;
	while (x[0] < xmin) {x++; m++; nx--;}
		
	// We want to make sure that mi >= [mj^-1 + a*|xi-xj|]^-1 for every i,j in m
	// First go forwards
	for (i=1; i<nx; i++) {
		mmin = m[i-1]/( 1 + m[i-1]*a*(x[i]-x[i-1]) );
		if (mmin > m[i]) 
			m[i] = mmin;
	}
	// And now go backwards
	for (i=nx-1; i>0; i--) {
		mmin = m[i]/( 1 + m[i]*a*(x[i]-x[i-1]) );
		if (mmin > m[i-1])
			m[i-1] = mmin;
	}
	
	// Figure out how many points will actually be on the grid (npoints = 1 + \int m dx)
	npts_d = 2.0;
	for (i=1; i<nx; i++) {
		npts_d += (m[i]+m[i-1])*(x[i]-x[i-1]);
	}
	npts_d *= 0.5;
	if (xmin < x[0])
		npts_d += (x[0]-xmin) * m[0];
	if (xmax > x[nx-1])
		npts_d += (xmax-x[nx-1]) * m[nx-1];
	*nx_out = npts = (int)(ceil(npts_d));
	LOGMSG("npts: %i, %0.4e", npts, npts_d);
	if (npts <= 0)
		return NULL;
		
	
	// Make the new grid.
	xnew = malloc(sizeof(double)*npts);
	i = 1; // counter the xnew
	xnew[0] = xmin;
	msum = 0.0;
	if (xmin < x[0]) {
		msum = m[0]*(x[0]-xmin);
		dx = 1./m[0];
		for (i=1; i < msum; i++) {
			xnew[i] = xnew[i-1]+dx;
	//		LOGMSG("%i: xnew=%0.3f", i, xnew[i]);
		}
	}
	for (j=1; j<nx; j++) {
		msum2 = (x[j]-x[j-1]) * 0.5 * (m[j]+m[j-1]) + msum;
		if (msum2 > i) {
			// m = 2a (x-x0) + b
			// dN = a (x-x0)^2 + b (x-x0) = i-msum
			// a dx^2 + m[j-1] dx = i-msum
			a = 0.5*(m[j]-m[j-1])/(x[j]-x[j-1]);
			b = m[j-1];
			while (msum2 > i && i < npts) {
				c = i-msum;
				d = a*c/(b*b);
				if (abs(d) < 1e-5) // Approximate using a taylor series.
					dx = c/b * (1 + d*(-1+d*(2-5*d)));
				else
					dx = 0.5*b*(-1+sqrt(1+4*d))/a;
				xnew[i] = x[j-1]+dx;
		//		LOGMSG("%i: xnew=%0.3f", i, xnew[i]);
				i++;
			}
		}
		msum = msum2;
	}
	if (i < npts) {
		dx = (i-msum)/m[nx-1];
		xnew[i] = x[nx-1] + dx;
		i++;
	}
	dx = 1.0/m[nx-1];
	while (i < npts) {
		xnew[i] = xnew[i-1] + dx;
		i++;
	}
	
	return xnew;
}

// This next routine works ok, but the scipy routine is a little more precise.
double *interpGrid(double *xold, int nold, double *yold, int ny, double *xnew, int nnew) {
	// I'm going to try to spline interpolation.
	// Finding the coefficients is a fairly straightforward implementation from the spline definition
	// and a simple algorithm for solving banded matrices.
	// Interpolation using the coefficients uses De Boor's algorithm.
	
	double *u; // knots
	double *y0; // yold, adjusted
	double u10,u21,u32,u43,u20,u31,u42,u30,u41;
	double x0, a13,a12,a11,a23,a22,a33,d13,d12,d11,d22,d23,d00,d01,d02,d03;
	double *N1, *N2, *N3; // basis functions evaluated at knots
	double *c; // spline coefficients.
	double *y; // output
	int i,j,k, nu;
	
	// First, extend x a bit in each direction, assuming that y is constant outside of the interval.
	nu = nold+8;
	u = malloc(sizeof(double)*nu);
	memcpy(u+4, xold, sizeof(double)*nold);
	for (i=0;i<4;i++) {
		u[i] = xold[0] - (4-i)*(xold[1]-xold[0]);
		u[nu-4+i] = xold[nold-1] + (1+i)*(xold[nold-1]-xold[nold-2]);
	}
	
/*	for (i=0; i<nold; i++) {
		LOGMSG("yold: %0.3e", yold[i*ny]);
	}
	for (i=0; i<nu; i++) {
		LOGMSG("u: %0.3f",u[i]);
	}*/
	
	// Evaluate the basis functions at the knots.
	// Each function is defined over 5 knots, with the two at the ends equal to zero.
	N1 = malloc(sizeof(double)*(nu-4));
	N2 = malloc(sizeof(double)*(nu-4));
	N3 = malloc(sizeof(double)*(nu-4));
	u10 = u[1]-u[0];
	u21 = u[2]-u[1];
	u32 = u[3]-u[2];
	u20 = u[2]-u[0];
	u31 = u[3]-u[1];
	u30 = u[3]-u[0];
	
	for (i=0; i<nu-4; i++) {
		u43 = u[i+4]-u[i+3];
		u42 = u[i+4]-u[i+2];
		u41 = u[i+4]-u[i+1];
		N1[i] = (u10*u10)/(u30*u20);
		N2[i] = (u20*u32)/(u30*u31) + (u42*u21)/(u41*u31);
		N3[i] = (u43*u43)/(u41*u42);
		u10=u21; u21=u32; u32=u43;
		u20=u31; u31=u42;
		u30=u41;
	}
	
	// solve the banded matrix eqn: Yold = [N1,N2,N3] (tri-diag) x C
	y0 = malloc(sizeof(double)*(nu-4)*ny);
	memcpy(y0+(4-2)*ny, yold, sizeof(double)*ny*nold); // fill y0
	for (k=0; k<ny; k++){ // fill the beginning and ends of y0 (the extensions)
		y0[k] = y0[k+ny] = yold[k];
		y0[(nu-6)*ny+k] = y0[(nu-5)*ny+k] = yold[(nold-1)*ny+k];
	}
/*	for (i=0; i<nu-4; i++) {
		LOGMSG("y0: %0.3e,  N123: %0.3e  %0.3e  %0.3e", y0[i*ny],N1[i],N2[i],N3[i]);
	}*/
	N1[1] /= N2[0];
	for (i=2; i<nu-4; i++)
		N1[i] /= N2[i-1] - N1[i-1]*N3[i];
	for (k=0; k<ny; k++)
		y0[k] /= N2[0];
	for (i=1; i<nu-4; i++) {
		for (k=0; k<ny; k++)
			y0[i*ny+k] = (y0[i*ny+k]-y0[(i-1)*ny+k]*N3[i-1])/(N2[i]-N1[i]*N3[i-1]);
	}
/*	for (i=0; i<nu-4; i++) {
		LOGMSG("y0: %0.3e", y0[i*ny]);
	}*/
	c = y0;
	for (i=nu-6; i>=0; i--) { // backwards sweep
		for (k=0; k<ny; k++)
			c[i*ny+k] -= c[(i+1)*ny+k]*N1[i+1];
	}
	
	LOGMSG("N1,N2,N3: %p  %p  %p",N1,N2,N3);
	LOGMSG("size(N1) = %i", 8*(nu-4));
	free(N1); free(N2); free(N3);
	/*	for (i=0; i<nu-4; i++) {
		LOGMSG("y0: %0.3e", y0[i*ny]);
	}*/
	
	// And evalute the spline at the new points.
	y = malloc(sizeof(double)*nnew*ny);
	while (xnew[nnew-1] >= xold[nold-1] && nnew > 1) {
		nnew--;
		for (k=0; k<ny; k++)
			y[nnew*ny+k] = yold[(nold-1)*ny+k];
	}
	i = 0;
	while (xnew[i] <= xold[0] && i < nnew) {
		for (k=0; k<ny; k++)
			y[i*ny+k] = yold[k];
		i++;
	}
	BLURT();
	j=3;
	while (i < nnew) {
	//	LOGMSG("main eval loop. %i",i);
		if (xnew[i] >= u[j+1] && j+1 < nu) {
			j++;
			while (xnew[i] >= u[j+1] && j+1 < nu) j++;
			u30 = u[j+3]-u[j];
			u20 = u[j+2]-u[j];
			u10 = u[j+1]-u[j];
			u31 = u[j+2]-u[j-1];
			u32 = u[j+1]-u[j-2];
			u21 = u[j+1]-u[j-1];
		}
		x0 = xnew[i];
		a13 = (x0-u[j  ])/u30;
		a12 = (x0-u[j-1])/u31;
		a11 = (x0-u[j-2])/u32;
		a23 = (x0-u[j  ])/u20;
		a22 = (x0-u[j-1])/u21;
		a33 = (x0-u[j  ])/u10;
		
		for (k=0; k<ny; k++) {
			d00 = c[(j-3)*ny+k];
			d01 = c[(j-2)*ny+k];
			d02 = c[(j-1)*ny+k];
			d03 = c[(j)*ny+k];
			d13 = (1-a13)*d02 + a13*d03;
			d12 = (1-a12)*d01 + a12*d02;
			d11 = (1-a11)*d00 + a11*d01;
			d23 = (1-a23)*d12 + a23*d13;
			d22 = (1-a22)*d11 + a22*d12;
			y[i*ny+k] = (1-a33)*d22 + a33*d23;
		}
		i++;
	}
	LOGMSG("%p,  %p, %p, %p, %i",y0,c,u,y,nu);
	free(y0);
	BLURT();
	free(u);
	BLURT();
	return y;
}

