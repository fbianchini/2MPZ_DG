"""
This code performs xc analysis between 2MPZ catalog and PlanckDR2 lensing  (see arXiv:1607.xxxx)
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
# import seaborn as sns
import sys, pickle, gzip
import healpy as hp

from astropy.visualization import hist

from TWOmpz_utils import Load2MPZ, GetdNdzNorm, LoadPlanck

import curvspec.master as cs
import hpyroutines.utils as hpy

from cosmojo.universe import Cosmo
from cosmojo.survey import dNdzInterpolation, dNdzMagLim
from cosmojo.limber import Limber
from cosmojo.kernels import *

from IPython import embed

def SetPlotStyle():
   rc('text',usetex=True)
   rc('font',**{'family':'serif','serif':['Computer Modern']})
   plt.rcParams['axes.linewidth']  = 1.5
   plt.rcParams['axes.labelsize']  = 28
   plt.rcParams['axes.titlesize']  = 22
   plt.rcParams['xtick.labelsize'] = 20
   plt.rcParams['ytick.labelsize'] = 18
   plt.rcParams['xtick.major.size'] = 7
   plt.rcParams['ytick.major.size'] = 7
   plt.rcParams['xtick.minor.size'] = 3
   plt.rcParams['ytick.minor.size'] = 3
   plt.rcParams['legend.fontsize']  = 20
   plt.rcParams['legend.frameon']  = False

   plt.rcParams['xtick.major.width'] = 1
   plt.rcParams['ytick.major.width'] = 1
   plt.rcParams['xtick.minor.width'] = 1
   plt.rcParams['ytick.minor.width'] = 1
   # plt.clf()
   # sns.set(rc('font',**{'family':'serif','serif':['Computer Modern']}))
   # sns.set_style("ticks", {'figure.facecolor': 'grey'})

SetPlotStyle()


# Params
nside   = 128
K_S_min = 0.
K_S_max = 13.9
zbins   = [(0.,0.08), (0.08,0.24)]
# zbins     = [(0.,0.4), (0.,0.08), (0.08,0.4)]

# Load Cat 
print("...loading 2MPZ catalogue...")
twoMPZ_cat = 'fits/results16_23_12_14_73.fits'
twompz = Load2MPZ(twoMPZ_cat, K_S_min=K_S_min, K_S_max=K_S_max)
print("...done...")

plt.figure(figsize=(6,4))
plt.title(r'2MPZ - $K_S<13.9$')
# 2MPZ dN/dz
nz, z, _ = hist(twompz.ZPHOTO,'knuth', normed=1, histtype='step', label=r'Full-$z_{\rm ph}$')
# _, _, _  = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label=r'Full-$z_{\rm s}$')
# _, _, _  = hist(twompz.ZPHOTO[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label=r'Cross-$z_{\rm s}$')
z = 0.5 * (z[1:]+z[:-1])
plt.close()

nz_spec, z_spec, _ = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label=r'Full spec')
# _, _, _  = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label=r'Full-$z_{\rm s}$')
# _, _, _  = hist(twompz.ZPHOTO[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label=r'Cross-$z_{\rm s}$')
z_spec = 0.5 * (z_spec[1:]+z_spec[:-1])
plt.close()

nz_overlap, z_overlap, _ = hist(twompz.ZPHOTO[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label=r'overlap photo')
# _, _, _  = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label=r'Full-$z_{\rm s}$')
# _, _, _  = hist(twompz.ZPHOTO[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label=r'Cross-$z_{\rm s}$')
z_overlap = 0.5 * (z_overlap[1:]+z_overlap[:-1])
plt.close()

# CMB lens kernel
lcmb = LensCMB(Cosmo())

# Plot dN/dz
DNDZ = dNdzInterpolation(z, nz, bins=[0.,0.24], sigma_zph=0.015, z_min=0, z_max=1)
DNDZ_spec = dNdzInterpolation(z_spec, nz_spec, bins=[0.,0.24], sigma_zph=0.015, z_min=0, z_max=1)
# DNDZ_overlap = dNdzInterpolation(z_overlap, nz_overlap, bins=[0.,0.24], sigma_zph=0.015, z_min=0, z_max=1)
plt.plot(z, DNDZ.raw_dndz_bin(z,0), label=r'2MPZ $z \le 0.24$', color='#083D77')
nz_spec, z_spec, _ = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label=r'Spec-z', color='#8C2F39')
# plt.plot(z, DNDZ_spec.raw_dndz_bin(z,0), label=r'2MPZ $z \le 0.24$ spec')
# plt.plot(z, DNDZ_overlap.raw_dndz_bin(z,0), label=r'2MPZ $z \le 0.24$ overlap')
# plt.plot(z, DNDZ.raw_dndz_bin(z,0)/np.max(DNDZ.raw_dndz_bin(z,0)), label='Fiducial')
# for i,zbin in enumerate(zbins):
#    DNDZ = dNdzInterpolation(z, nz, bins=[zbin[0],zbin[1]], sigma_zph=0.015, z_min=0, z_max=1)
#    # plt.plot(z, DNDZ.raw_dndz_bin(z,0)/np.max(DNDZ.raw_dndz_bin(z,0)), label='Bin-%d' %(i+1))
#    plt.plot(z, DNDZ.raw_dndz_bin(z,0), label='Bin-%d' %(i+1))
   # plt.plot(z, DNDZ.raw_dndz_bin(z,0), label=r'$%.2f < z < %.2f$' %(zbin[0],zbin[1]))
plt.plot(z[z>0], 30*lcmb.W_z(z[z>0]), ls='--', label=r'$W^{\kappa}$', color='#F6AE2D')
# plt.axvline(0.08, ls='--', color='grey')
plt.xlabel(r'$z$')
# plt.xlabel(r'$z_{\rm ph}$')
plt.ylabel(r'$dN/dz$')
plt.xlim(0,0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(left=0.2)
plt.savefig('plots/dNdz_2MPZ_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'.pdf', bboxes_inches='tight')
plt.show()

embed()

exit()

# Load Planck CMB lensing map 'n' mask
fits_planck_mask = '../Data/mask_convergence_planck_2015_512.fits'
fits_planck_map  = '../Data/convergence_planck_2015_512.fits'
twoMPZ_mask = 'fits/2MPZ_mask_tot_256.fits' # N_side = 256
print("...loading Planck map/mask...")
planck_map, planck_mask = LoadPlanck(fits_planck_map, fits_planck_mask, nside, do_plots=0, filt_plank_lmin=0, pess=0)
print("...done...")

print("...reading 2MPZ mask...")
mask  = hp.read_map(twoMPZ_mask, verbose=False)
nside_mask = hp.npix2nside(mask.size)
if nside_mask != nside:
   mask = hp.ud_grade(mask, nside)
print("...done...")

# Common Planck & WISExSCOS mask  
mask *= planck_mask

# Counts 'n' delta map
counts = hpy.GetCountsMap(twompz.RA, twompz.DEC, nside, coord='G', rad=True)
delta  = hpy.Counts2Delta(counts, mask=mask)

counts = hp.ma(counts)
counts.mask = np.logical_not(mask)

delta = hp.ma(np.log(1.+delta))
delta.mask = np.logical_not(mask)

from pylab import cm

cmap = cm.rainbow
cmap.set_under('w')

plt.figure(figsize=(8,5))
hp.mollview(delta, title=r'2MPZ - $K_S<13.9$')#, cmap='bone')
hp.graticule()
plt.tight_layout()
# plt.subplots_adjust(top=0.9)
plt.savefig('plots/delta_2MPZ_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'.pdf')#, bboxes_inches='tight')

hp.mollview(mask, title=r'2MPZ - $K_S<13.9$', cmap='Greys')
hp.graticule()
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)
plt.savefig('plots/mask_2MPZ_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'.pdf')#, bboxes_inches='tight')

embed()
