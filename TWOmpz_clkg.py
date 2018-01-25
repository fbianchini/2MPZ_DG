"""
This code performs xc analysis between 2MPZ catalog and PlanckDR2 lensing  (see arXiv:1607.xxxx)
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

from TWOmpz_utils import Load2MPZ, GetdNdzNorm, LoadPlanck

import curvspec.master as cs
import hpyroutines.utils as hpy

from cosmojo.universe import Cosmo
from cosmojo.survey import dNdzInterpolation
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

def GetCrossSpectrum(cosmo, z, dndz, bins, b=1, alpha=1., lmax=1000, sigma_zph=0., i=0):
	DNDZ = dNdzInterpolation(z, dndz, bins=bins, sigma_zph=sigma_zph, z_min=0, z_max=1)
	# DNDZ = dNdzInterpolation(z, dndz, nbins=1, z_min=z[0], z_max=z[-1], sigma_zph=sigma_zph)
	limb = Limber(cosmo, lmin=0, lmax=lmax)
	gals = GalsTomo(cosmo, DNDZ, b=b, alpha=alpha)
	return limb.GetCl(gals, k2=LensCMB(cosmo), i=i)

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

	# Counts 'n' delta map
	cat_tmp = twompz[(twompz.ZPHOTO >= args.zmin) & (twompz.ZPHOTO < args.zmax)]
	counts = hpy.GetCountsMap(cat_tmp.RA, cat_tmp.DEC, nside, coord='G', rad=True)
	delta = hpy.Counts2Delta(counts, mask=mask)

	est = cs.Master(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell, MASTER=1, pixwin=True)

	print('...extracting kg spectra...')
	kg, err_kg = est.get_spectra(planck_map, map2=delta, analytic_errors=True)
	print("\t=> Detection at rougly %3.2f sigma " ) %(np.sum((kg[0:]/err_kg[0:])**2)**.5)

	embed()

	# Initializing Cosmo class
	cosmo = Cosmo()
	cosmo_nl = Cosmo(nonlinear=True)

	nz, z, _ = hist(twompz.ZPHOTO,'knuth', normed=1, histtype='step')
	z = 0.5 * (z[1:]+z[:-1])
	plt.close()

	fig = plt.figure()#figsize=(10, 8))

	ax = fig.add_subplot(1,1,1) 
	plt.title(r'$%.2f < z < %.2f$' %(args.zmin,args.zmax))

	clkg    = GetCrossSpectrum(cosmo,    z, nz, [args.zmin,args.zmax], b=1, alpha=1., lmax=500, sigma_zph=0.015)
	clkg_nl = GetCrossSpectrum(cosmo_nl, z, nz, [args.zmin,args.zmax], b=1, alpha=1., lmax=500, sigma_zph=0.015)

	# z, n_z = GetdNdzNorm(nz=1000)
	# clkg = GetCrossSpectrum(cosmo, z, n_z, [bins_edges[zbin][0],0.3], b=1.24, alpha=1., lmax=1000)
	# # clkg = GetCrossSpectrum(cosmo, z, n_z, [bins_edges[zbin][0],bins_edges[zbin][1]], b=1, alpha=1.24, lmax=1000)
	# clkg_nl = GetCrossSpectrum(cosmo_nl, z, n_z, [bins_edges[zbin][0],0.5], b=1.24, alpha=1., lmax=1000)
	# # clkg_nl = GetCrossSpectrum(cosmo_nl, z, n_z, [bins_edges[zbin][0],bins_edges[zbin][1]], b=1.24, alpha=1., lmax=1000)
	# # clgg_pherrz = GetAutoSpectrum(cosmo, z, dndz, [bins_edges[zbin][0],bins_edges[zbin][1]], b=1.24, alpha=1., lmax=1000, sigma_zph=0.03)

	lab = r'2MPZ  $%.1f < K_S < %.1f$' %(K_S_min, K_S_max)
	ax.plot(clkg, color='grey', label=r'$b=1$ $\sigma_z=0.015$')
	ax.plot(clkg_nl, color='darkgrey', ls='--', label=r'NL $b=1$ $\sigma_z=0.015$')
	ax.errorbar(est.lb, kg, yerr=err_kg, fmt='o', capsize=0, ms=5, label=lab)
	ax.set_ylabel(r'$C_{\ell}^{\kappa g}$')
	ax.set_xlabel(r'Multipole $\ell$')
	ax.legend(loc='best')
	ax.axhline(ls='--', color='grey')
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax.set_xlim([2., 200])
	ax.set_yscale('log', nonposy='clip')
	ax.set_ylim([1e-8, 1e-5])


	plt.tight_layout()
	plt.savefig('plots/2MPZ_Planck_clkg_dl'+str(args.delta_ell)+'_lmin_'+str(args.lmin)+'_lmax'+str(args.lmax)+'_KS_min_'+str(args.K_S_min)+'_KS_max_'+str(args.K_S_max)+'_nside'+str(args.nside)+'_zmin_'+str(args.zmin)+'_zmax'+str(args.zmax)+'_split.pdf', bbox_inches='tight')
	# plt.show()
	# plt.close()

	# embed()

	np.savetxt('spectra/2MPZ_Planck2015_clkg_dl'+str(args.delta_ell)+'_lmin_'+str(args.lmin)+'_lmax'+str(args.lmax)+'_KS_min_'+str(args.K_S_min)+'_KS_max_'+str(args.K_S_max)+'_nside'+str(args.nside)+'_zmin_'+str(args.zmin)+'_zmax'+str(args.zmax)+'_split.dat', np.c_[est.lb, kg, err_kg])

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


