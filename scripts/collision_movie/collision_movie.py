import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from bubble_collisions import simulation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
        comment='Movie support!')
framerate = 30
duration = 15
writer = FFMpegWriter(fps=framerate, metadata=metadata, bitrate=2000)

data = simulation.readFromFile(
    'quartic_nonident_varDV_1.00_varsig_2.25_varbp_1.00_fields_xsep=1.00.dat')
Ndata = np.array([d[0] for d in data])
x = np.linspace(-np.pi/2, 1+np.pi/2, 5000)
N_list = np.linspace(0,4.99,framerate*duration)

fig = plt.figure()
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

with writer.saving(fig, "collision_movie.mp4", 160):
    for N in N_list:
        Y=simulation.valsOnGrid(
            N*np.ones_like(x),x, data, [d[0] for d in data], False)
        y = Y[:,0,0]
        alpha = Y[:,0,2]-1
        a = Y[:,0,3]-1
        x2 = np.cosh(N)*x
        ax1.cla()
        ax1.plot(x2,y, 'b', lw=1.5)
        ax1.set_ylabel(r"Inflaton field values ($M_{\rm Pl}$)")
        ax1.axis(xmin=x2[0],xmax=x2[-1], 
            ymin=min(-.002,min(y))*1.2, ymax=max(.002,max(y))*1.2)
        ax1.text(.85, .9, "$N=%0.2f$"%N, ha='left', va='center', 
            transform=ax1.transAxes)
        ax1.set_xticklabels([])
        ax1.set_title("Colliding bubble universes")
        ax2.cla()
        ax2.plot(x2,alpha, 'c', lw=1.5)
        ax2.plot(x2,a, color=(1,.5,0), lw=1.5)
        ax2.set_ylabel(r"Metric perturbations")
        ax2.axis(xmin=x2[0],xmax=x2[-1], 
            ymin=min(min(alpha),min(a))*1.2, 
            ymax=max(max(alpha),max(a),.005)*1.2)
        ax2.set_xlabel(r"Physical distance ($r_{\rm dS}$)")
        plt.subplots_adjust(hspace=.1, top=.95, right=.9)
        writer.grab_frame()
