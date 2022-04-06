"""
We want to plot the simulation on sequential closed slices of dS space.
Each slice can be plotted using our favorite spherical projection, which
here I will take to be the Mollweide projection (the same one usually used
for CMB plots) centered directly between the two bubbles.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from  matplotlib.colorbar import ColorbarBase
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from bubble_collisions import simulation
from scipy import interpolate

def hsvToRgb(h,s,v):
    h = h * np.pi/180
    r = np.clip(0.5+np.cos(h),0,1)
    g = np.clip(0.5+np.cos(h-2*np.pi/3),0,1)
    b = np.clip(0.5+np.cos(h+2*np.pi/3),0,1)
    rgb = np.array([r,g,b])
    return tuple((s+rgb*(1-s))*v)

def hsv_cmap(vals, name="hsv_colormap"):
    newvals = []
    h0 = None
    for val in vals:
        h1 = val[1]
        if h0 is not None:
            if (h0 // 60 != h1 // 60):
                # add in intermediate values
                if h1 > h0:
                    h_list = np.arange(h0//60, h1//60)*60+60
                else:
                    h_list = np.arange(h0//60, h1//60,-1)*60
                if h_list[-1] == h1:
                    h_list = h_list[:-1]
                v0 = np.array(newvals[-1])
                v1 = np.array(val)
                for h2 in h_list:
                    z = (h2-h0)/(h1-h0)
                    newvals.append((1-z)*v0 + z*v1)
        newvals.append(np.array(val))
        h0 = h1
    red, green, blue = [], [], []
    for x,h,s,v in newvals:
        r,g,b = hsvToRgb(h,s,v)
        red.append((x,r,r))
        green.append((x,g,g))
        blue.append((x,b,b))
    cdict = dict(red=red,green=green,blue=blue)
    return LinearSegmentedColormap(name, cdict)


def mollweideProjection(x,y):
    theta = np.arcsin(np.clip(y,-1,1))
    lat = np.arcsin((2*theta+np.sin(2*theta))/np.pi)
    lon = 0.5*np.pi * x / np.cos(theta)
    lon[abs(lon) > np.pi] = np.nan
    return lat, lon

def mollweideCirclePlot(lon0, r):
    y = np.linspace(-1,1,500)[:,None] * np.ones((1,500))
    x = y.T * 2
    lat, lon = mollweideProjection(x, y)
    lon -= lon0
    X = np.cos(lon)*np.cos(lat)
    in_circle = (X > np.cos(r))
    plt.contour(x,y,1.0*in_circle, [-.1,.5,1.1])

def sciNotationLatex(x,i=1):
    e = -1+int(np.log10(abs(x))) if abs(x) > 0 else 0
    n = x / 10.0**e
    fmt = "$%0."+str(i)+r"f \times 10^{%i}$"
    return fmt % (n, e)

class SimulationProjector(object):
    cmap = hsv_cmap((
        (0,240,0,.6),
        (.35,177,0,1),
        (.49999,150,1,.2),
        (.5,60,1,.2),
        (.65,42,0,1),
        (1,-20,0,.6) ))

    def __init__(self, data, xsep, obs_centered=True, n_grid=500):
        self.data = data
        self.Ndata = np.array([d[0] for d in data])
        self.xsep = xsep
        x_grid = np.linspace(-2,2,2*n_grid)[None,:] * np.ones((n_grid,1))
        y_grid = np.linspace(-1,1,n_grid)[:,None] * np.ones((1,2*n_grid))
        self.lat,self.lon = mollweideProjection(x_grid, y_grid)
        if not obs_centered:
            self.lon += 0.5*xsep

    def simCoords(self, N0, show_spacelike=True):
        x = self.lon.copy()
        cosh_N = np.cos(self.lat)*np.cosh(N0)
        tanh_xi = np.sin(self.lat)/np.tanh(N0)        
        if show_spacelike:
            xsep = self.xsep
            xp = np.arccos(cosh_N / np.maximum(1,cosh_N))
            cosh_N = np.maximum(1,cosh_N)
            r0 = np.sqrt(x**2+xp**2)
            r1 = np.sqrt((x-xsep)**2+xp**2)
            i1 = x<=0
            i2 = (x>0)&(x<xsep/2)
            i3 = (x<xsep)&(x>=xsep/2)
            i4 = x>=xsep
            x[i1] = -r0[i1]
            x[i2] = np.minimum(r0[i2], xsep/2)
            x[i3] = np.maximum(xsep-r1[i3], xsep/2)
            x[i4] = xsep+r1[i4]
        return np.arccosh(cosh_N), x, np.arctanh(tanh_xi)

    def plotProjection(self, N0, index=0, mapvals=None, verbose=False,
            show_colorbar=True):
        N, x, xi = self.simCoords(N0)
        y = simulation.valsOnGrid(
            N,x, self.data, self.Ndata, True)[...,0,index]
        i = np.isfinite(y.ravel())
        ymin = np.min(y.ravel()[i])
        ymax = np.max(y.ravel()[i])
        if verbose:
            print "y min/max = %f, %f" % (ymin,ymax)
        if mapvals is None:
            z = y-ymin
            z /= ymax-ymin
        else:
            z_interp, y_interp = mapvals
            tck = interpolate.splrep(y_interp,z_interp,k=1)
            y_shape = y.shape
            z = interpolate.splev(y.ravel(), tck).reshape(y_shape)
        img = self.cmap(z)
        img[~np.isfinite(y)] = 0.0

        plt.clf()
        fig = plt.gcf()
        ax1 = fig.add_axes([0, 0, 1,1])
        ax1.imshow(img)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_axis_bgcolor('black')
        if show_colorbar:
            ax2 = fig.add_axes([.04, .05, .03,.9])
            cbar = ColorbarBase(ax2, cmap=self.cmap, orientation='vertical')
            z = np.linspace(0,1,15)
            cbar.set_ticks(z)
            if mapvals is None:
                y = z*(ymax-ymin) + ymin
            else:
                z_interp, y_interp = mapvals
                tck = interpolate.splrep(z_interp,y_interp,k=1)
                y = interpolate.splev(z, tck)
                cbar.set_ticklabels([sciNotationLatex(yi) for yi in y])
               # cbar.set_ticklabels(["%0.2e"%yi for yi in y])
                ax2.tick_params(axis='y', colors='white')
            cbar.draw_all()
            ax1.text(.01,.5,r"Inflaton field ($M_{\rm Pl}$)",
                transform=ax1.transAxes,ha = 'left', va='center',
                rotation=90, color='white')
        ax1.text(.85, .9, "$N=%0.2f$"%N0, ha='left', va='center', 
            transform=ax1.transAxes, color='white', fontsize='x-large')


    def writeMovie(self, name,framerate=30, duration=10, Nmax=4.99, 
            mapvals=None):
        FFMpegWriter = manimation.writers['ffmpeg']
        #metadata = dict(title='Movie Test', artist='Matplotlib',
        #   comment='Movie support!')        
        writer = FFMpegWriter(fps=framerate, bitrate=2000)
        fig = plt.figure(figsize=(8,4))
        ax = plt.subplot(111)
        plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
        N_list = np.linspace(0.0,Nmax,framerate*duration)
        with writer.saving(fig, name, 160):
            for N0 in N_list:
                print "N0:", N0
                self.plotProjection(N0,mapvals=mapvals)
                writer.grab_frame()        


if __name__ == "__main__":
    data = simulation.readFromFile(
    'quartic_nonident_varDV_1.00_varsig_2.25_varbp_1.00_fields_xsep=1.00.dat')
    proj = SimulationProjector(data, 1.0, False)
    mapvals = ((0,.5,.75,1),(-2e-3,0,2e-3,.111))
    proj.writeMovie("collision_movie2.mp4",30,10,4.99,mapvals)


