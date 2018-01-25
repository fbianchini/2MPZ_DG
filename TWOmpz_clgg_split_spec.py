"""
This code performs xc analysis between WISExSCOS catalog and PlanckDR2 lensing  (see arXiv:1607.xxxx)
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import pandas as pd
from sklearn.cross_validation import train_test_split
import healpy as hp
# import seaborn as sns
import argparse, sys
from astropy.io import fits
from astropy.visualization import hist

from TWOmpz_utils import Load2MPZ, GetdNdzNorm, LoadPlanck, GetNlgg

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
   plt.rcParams['axes.linewidth']  = 2.
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

def GetAutoSpectrum(cosmo, z, dndz, bins, b=1, alpha=1., lmax=1000, sigma_zph=0., i=0):
	# DNDZ = dNdzInterpolation(z, dndz, nbins=1, z_min=z[0], z_max=z[-1])
	DNDZ = dNdzInterpolation(z, dndz, bins=bins, sigma_zph=sigma_zph, z_min=0, z_max=1)
	limb = Limber(cosmo, lmin=0, lmax=lmax)
	gals = GalsTomo(cosmo, DNDZ, b=b, alpha=alpha)
	return limb.GetCl(gals,i=i)

def GetAutoSpectrumMagLim(cosmo, a, z0, b, bias=1, alpha=1., lmax=1000, sigma_zph=0., compute_at_z0=False):
	# DNDZ = dNdzInterpolation(z, dndz, nbins=1, z_min=z[0], z_max=z[-1])
	DNDZ = dNdzMagLim(a, z0, b, sigma_zph=sigma_zph, z_min=0, z_max=1)
	limb = Limber(cosmo, lmin=0, lmax=lmax, compute_at_z0=compute_at_z0)
	gals = GalsTomo(cosmo, DNDZ, b=bias, alpha=alpha)
	return limb.GetCl(gals)

def main(args):
	SetPlotStyle()

	# Params 'n' stuff
	fits_planck_mask = '../Data/mask_convergence_planck_2015_512.fits'
	fits_planck_map  = '../Data/convergence_planck_2015_512.fits'
	twoMPZ_mask = 'fits/2MPZ_mask_tot_256.fits' # N_side = 256
	twoMPZ_cat  = 'fits/results16_23_12_14_73.fits'
	nside       = args.nside
	do_plots    = False 
	delta_ell   = args.delta_ell
	lmin        = args.lmin
	lmax        = args.lmax
	K_S_min     = args.K_S_min
	K_S_max     = args.K_S_max
	# lbmin    = 2
	# lbmax    = 10

	# Load Cat and Mask
	print("...loading 2MPZ catalogue...")
	twompz = Load2MPZ(twoMPZ_cat, K_S_min=K_S_min, K_S_max=K_S_max)
	print("...done...")

	# Load Planck CMB lensing map 'n' mask
	print("...loading Planck map/mask...")
	planck_map, planck_mask = LoadPlanck(fits_planck_map, fits_planck_mask, nside, do_plots=0, filt_plank_lmin=0)
	print("...done...")

	print("...reading 2MPZ mask...")
	mask  = hp.read_map(twoMPZ_mask, verbose=False)
	nside_mask = hp.npix2nside(mask.size)
	if nside_mask != nside:
		mask = hp.ud_grade(mask, nside)
	print("...done...")

	# Common Planck & WISExSCOS mask  
	mask *= planck_mask

	# embed()

	# Counts n overdesnity map
	cat_tmp = twompz[(twompz.ZSPEC >= args.zmin) & (twompz.ZSPEC < args.zmax)]
	a, b = train_test_split(cat_tmp, test_size=0.5)

	counts_a = hpy.GetCountsMap(a.RA, a.DEC, nside, coord='G', rad=True)
	counts_b = hpy.GetCountsMap(b.RA, b.DEC, nside, coord='G', rad=True)
	# hp.write_map('fits/counts_'+zbin+'_sum.fits', counts_sum[zbin])
	# hp.write_map('fits/counts_'+zbin+'_diff.fits', counts_diff[zbin])
	print("\tNumber of sources in bin [%.2f,%.2f) is %d" %(args.zmin, args.zmax, np.sum((counts_a+counts_b)*mask)))

	print("...converting to overdensity maps...")

	delta_a = hpy.Counts2Delta(counts_a, mask=mask)
	delta_b = hpy.Counts2Delta(counts_b, mask=mask)

	delta_sum = 0.5 * (delta_a + delta_b)
	delta_diff = 0.5 * (delta_a - delta_b)
	# hp.write_map('fits/delta_'+zbin+'_sum.fits', delta_sum[zbin])
	# hp.write_map('fits/delta_'+zbin+'_diff.fits', delta_diff[zbin])

	print("...done...")

	# embed()

	# exit()

	# XC estimator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	est = cs.Master(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell, MASTER=1, pixwin=True)
	print('...extracting gg spectra...')

	clGS, err_gg = est.get_spectra(delta_sum, analytic_errors=True)
	clGD = est.get_spectra(delta_diff)
	gg = clGS - clGD 
	counts_ = counts_a + counts_b
	delta_ = hpy.Counts2Delta(counts_, mask=mask)
	gg_, err_gg_ = est.get_spectra(delta_, nl=GetNlgg(counts_, mask=mask), pseudo=0, analytic_errors=True)
	print("\t=> Detection at rougly %.2f sigma " ) %(np.sum((gg[0:]/err_gg[0:])**2)**.5)
	print("\t=> Detection at rougly %.2f sigma "  ) %(np.sum((gg_[0:]/err_gg_[0:])**2)**.5)

	print('...done...')

	# Initializing Cosmo class
	cosmo = Cosmo()
	cosmo_nl = Cosmo(nonlinear=True)

	# embed()

	nz, z, _ = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1)
	z = 0.5 * (z[1:]+z[:-1])
	plt.close()


	fig = plt.figure()#figsize=(10, 8))

	ax = fig.add_subplot(1,1,1) 
	plt.title(r'$%.2f < z < %.2f$' %(args.zmin,args.zmax))

	# z, n_z  = GetdNdzNorm(nz=1000)
	# clgg    = GetAutoSpectrumMagLim( cosmo,    2.21, 0.053, 1.43, bias=1.24, alpha=1., lmax=1000)
	# clgg_ph = GetAutoSpectrumMagLim( cosmo_nl, 2.21, 0.053, 1.43, bias=1.24, alpha=1., lmax=1000, sigma_zph=0.03)
	# clgg_nl = GetAutoSpectrumMagLim( cosmo_nl, 2.21, 0.053, 1.43, bias=1.24, alpha=1., lmax=1000)
	clgg    = GetAutoSpectrum(cosmo,   z,  nz, [args.zmin, args.zmax], b=1., alpha=1., lmax=500, sigma_zph=0.0, i=0)
	clgg_nl = GetAutoSpectrum(cosmo_nl, z, nz, [args.zmin, args.zmax], b=1., alpha=1., lmax=500, sigma_zph=0.0, i=0)		
	# clgg    = GetAutoSpectrum(cosmo,   z,  dndz, [0., 0.08, 0.5], b=1.24, alpha=1., lmax=500, sigma_zph=0.015, i=i)
	# clgg_nl = GetAutoSpectrum(cosmo_nl, z, dndz, [0., 0.08, 0.5], b=1.24, alpha=1., lmax=500, sigma_zph=0.015, i=i)		
	# ax = fig.add_subplot(1,1,i+1) 
	lab = r'2MPZ  $%.1f < K_S < %.1f$' %(K_S_min, K_S_max)
	ax.plot(clgg, color='grey', label=r'$b=1$')
	ax.plot(clgg_nl, color='darkgrey', ls='--', label=r'NL $b=1$')
	ax.errorbar(est.lb, gg, yerr=err_gg, fmt='o', capsize=0, ms=5, label=lab)
	# ax.errorbar(est.lb+1, gg_[zbin], yerr=err_gg_[zbin], fmt='x', capsize=0, ms=5)#, label=lab)
	ax.set_ylabel(r'$C_{\ell}^{gg}$')
	ax.set_xlabel(r'Multipole $\ell$')
	ax.legend(loc='best')
	ax.axhline(ls='--', color='grey')
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax.set_xlim([2., 200])
	ax.set_yscale('log')
	ax.set_ylim([1e-5, 1e-3])


	plt.tight_layout()
	plt.savefig('plots/2MPZ_clgg_dl'+str(args.delta_ell)+'_lmin_'+str(args.lmin)+'_lmax'+str(args.lmax)+'_KS_min_'+str(args.K_S_min)+'_KS_max_'+str(args.K_S_max)+'_nside'+str(args.nside)+'_zmin_'+str(args.zmin)+'_zmax'+str(args.zmax)+'_split_spec.pdf', bbox_inches='tight')
	# plt.show()
	# plt.close()

	# embed()

	np.savetxt('spectra/2MPZ_clgg_dl'+str(args.delta_ell)+'_lmin_'+str(args.lmin)+'_lmax'+str(args.lmax)+'_KS_min_'+str(args.K_S_min)+'_KS_max_'+str(args.K_S_max)+'_nside'+str(args.nside)+'_zmin_'+str(args.zmin)+'_zmax'+str(args.zmax)+'_split_spec.dat', np.c_[est.lb, gg, err_gg])

if __name__=='__main__':
		parser = argparse.ArgumentParser(description='')
		# parser.add_argument('-reso', dest='reso', action='store', help='pixel resolution in arcmin', type=float, required=True)
		parser.add_argument('-nside', dest='nside', action='store', help='healpix resolution', type=int, default=256)
		parser.add_argument('-deltal', dest='delta_ell', action='store', help='binsize', type=int, default=20)
		parser.add_argument('-lmin', dest='lmin', action='store', help='minimum ell', type=int, default=10)
		parser.add_argument('-lmax', dest='lmax', action='store', help='maximum ell', type=int, default=250)
		parser.add_argument('-zmin', dest='zmin', action='store', help='minimum Redshift', type=float, default=0.)
		parser.add_argument('-zmax', dest='zmax', action='store', help='maximum Redshift', type=float, default=0.5)
		parser.add_argument('-K_S_min', dest='K_S_min', action='store', help='minmum K_S flux', type=float, default=0.)
		parser.add_argument('-K_S_max', dest='K_S_max', action='store', help='maximum K_S flux', type=float, default=13.9)
		parser.add_argument('-file_spectrum', dest='file_spectrum', action='store', help='file name containing the extracted spectrum', type=str, default='spectra/clgg_extracted.dat')
		args = parser.parse_args()
		main(args)


