import os, gzip, glob, pickle, sys, numpy as np
import healpy as hp
#import matplotlib; matplotlib.use('Agg')

make_zbins = 0
rnd_30percent_stck = 0
flog = 'logs/CONVERGENCE/sigma_1_2048_cuts.txt'#RND_30percent_stacking.txt'
frac = 0.3

# mask_sigma_cuts = hp.read_map('data/mask_sigma_cuts_1_nside2048.fits')# Remember that if mask == 1, you want to *throw* away the source

####
import os, gzip, glob, pickle, sys, numpy as np
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

def fn_radial_profile(Z, XY, bin_size = 1., minbin = 0., maxbin = 6., to_arcmins = 1):

        Z = np.asarray(Z)
        if XY == None:
                Y, X = np.indices(Z.shape)
        else:
                X, Y = XY

        #RADIUS = np.hypot(X,Y) * 60.
        RADIUS = (X**2. + Y**2.) ** 0.5
        if to_arcmins: RADIUS *= 60.
        #clf();imshow(RADIUS);colorbar();show();quit()

        #ind = np.argsort(RADIUS.flat)
        #print ind;quit()

        binarr=np.arange(minbin,maxbin,bin_size)
        radprf=np.zeros((len(binarr),3))

        hit_count=[]
        #imshow(RADIUS);colorbar()

        for b,bin in enumerate(binarr):
                ind=np.where((RADIUS>=bin) & (RADIUS<bin+bin_size))
                radprf[b,0]=(bin+bin_size/2.)
                hits = len(np.where(abs(Z[ind])>0.)[0])

                if hits>0:
                        radprf[b,1]=np.sum(Z[ind])/hits
                        radprf[b,2]=np.std(Z[ind])
                hit_count.append(hits)

        hit_count=np.asarray(hit_count)
        std_mean=np.sum(radprf[:,2]*hit_count)/np.sum(hit_count)
        errval=std_mean/(hit_count)**0.5
        radprf[:,2]=errval

        #clf();errorbar(radprf[:,0], radprf[:,1], yerr = [radprf[:,2], radprf[:,2]], color= 'g', marker = 'o')
        #show();quit()
        
        return radprf

def GetStack(files):
	STACK = None
	totfiles_bin = 0
	for f in files:
		MAP = pickle.load(gzip.open(f,'rb'))['summed']
		# print f, MAP[0,0]
		#imshow(MAP);colorbar();show();sys.exit()
		if np.isnan(np.sum(MAP)): 
			print f
			continue
		try: 
			STACK += MAP
		except: 
			STACK = MAP
		totfiles_bin += pickle.load(gzip.open(f,'rb'))['totfiles']

	return STACK/totfiles_bin, totfiles_bin


plt.figure(figsize=(14,5))

sbpl = 1

# sys.exit()

from IPython import embed
# embed()


files_1 = sorted(glob.glob('K_S_0.0_14.0/2MPZ_*'))
files_2 = sorted(glob.glob('K_S_12.0_14.0/2MPZ_*'))

z = []

stack_1, n_1 = GetStack(files_1)
stack_2, n_2 = GetStack(files_2)

#hist(z, bins =50);show();sys.exit()

# subplot(1,2,sbpl)
# if sbpl == 1:
# 	vmin, vmax = -0.006, 0.005
# elif sbpl == 2:
# 	vmin, vmax = -0.007, 0.009
# elif sbpl == 3:
# 	vmin, vmax = -0.005, 0.01
# else:
vmin, vmax = None, None

subplot(1,2,1)
im_1 = imshow(stack_1, cmap='Greys', vmax=vmax, vmin=vmin)
plt.contour(stack_1, levels=[-2*np.std(stack_1)], colors='r', ls='--')
plt.contour(stack_1, levels=[2*np.std( stack_1)], colors='r', ls='-')
plt.contour(stack_1, levels=[-3*np.std(stack_1)], colors='orange', ls='--')
plt.contour(stack_1, levels=[3*np.std( stack_1)], colors='orange', ls='-')

# if 'wclust' in zsplits:
# 	title('wcluts')
# else:
# 	title('woclust')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# colorbar(im, cax=cax)
colorbar(im_1, fraction=0.046, pad=0.04)
title(r'$K_S<14$ N = %d' %n_1)

subplot(1,2,2)
im_2 = imshow(stack_2, cmap='Greys', vmax=vmax, vmin=vmin)
plt.contour(stack_2, levels=[-2*np.std(stack_2)], colors='r', ls='--')
plt.contour(stack_2, levels=[2*np.std( stack_2)], colors='r', ls='-')
plt.contour(stack_2, levels=[-3*np.std(stack_2)], colors='orange', ls='--')
plt.contour(stack_2, levels=[3*np.std( stack_2)], colors='orange', ls='-')
colorbar(im_2, fraction=0.046, pad=0.04)
title(r'$12<K_S<14$ N = %d' %n_2)

show()

# embed()

# FULL_STACK /= totfiles
# imshow(FULL_STACK)#, vmax=0.006, vmin=0.002)
# colorbar();show() 
sys.exit()
