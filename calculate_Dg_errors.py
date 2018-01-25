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
import seaborn as sns
import sys, pickle, gzip
from astropy.io import fits
from astropy.visualization import hist

from TWOmpz_utils import Load2MPZ, GetdNdzNorm

import curvspec.master as cs
import hpyroutines.utils as hpy

from cosmojo.universe import Cosmo
from cosmojo.survey import dNdzInterpolation, dNdzMagLim
from cosmojo.limber import Limber
from cosmojo.kernels import *

from IPython import embed

from tqdm import tqdm

def LoadPlanckMask(fitsfile, nside, do_plots=False):
	print("...using Planck CMB Lensing mask (= SMICA mask + stuff)...")
	print("...degrading Planck mask to N_side = %d ..." %nside)

	planck_mask = hp.read_map(fitsfile, verbose=False)
	planck_mask = hp.ud_grade(planck_mask, nside, pess=True)
	planck_mask[planck_mask < 1.] = 0. # Pessimistc cut, retains only M(n) = 1 pixels

	return planck_mask

def GetCrossSpectrumMagLim(cosmo, a, z0, b, bias=1, alpha=1., lmax=1000, sigma_zph=0., compute_at_z0=False):
	DNDZ = dNdzMagLim(a, z0, b, sigma_zph=sigma_zph, z_min=0, z_max=1)
	limb = Limber(cosmo, lmin=0, lmax=lmax, compute_at_z0=compute_at_z0)
	gals = GalsTomo(cosmo, DNDZ, b=bias, alpha=alpha)
	return limb.GetCl(gals, k2=LensCMB(cosmo))

def GetAutoSpectrumMagLim(cosmo, a, z0, b, bias=1, alpha=1., lmax=1000, sigma_zph=0., compute_at_z0=False):
	# DNDZ = dNdzInterpolation(z, dndz, nbins=1, z_min=z[0], z_max=z[-1])
	DNDZ = dNdzMagLim(a, z0, b, sigma_zph=sigma_zph, z_min=0, z_max=1)
	limb = Limber(cosmo, lmin=0, lmax=lmax, compute_at_z0=compute_at_z0)
	gals = GalsTomo(cosmo, DNDZ, b=bias, alpha=alpha)
	return limb.GetCl(gals)

def GetCrossSpectrum(cosmo, z, dndz, bins, b=1, alpha=1., lmax=1000, sigma_zph=0., i=0):
	DNDZ = dNdzInterpolation(z, dndz, bins=bins, sigma_zph=sigma_zph, z_min=0, z_max=1)
	# DNDZ = dNdzInterpolation(z, dndz, nbins=1, z_min=z[0], z_max=z[-1], sigma_zph=sigma_zph)
	limb = Limber(cosmo, lmin=0, lmax=lmax)
	gals = GalsTomo(cosmo, DNDZ, b=b, alpha=alpha)
	return limb.GetCl(gals, k2=LensCMB(cosmo), i=i)

def GetAutoSpectrum(cosmo, z, dndz, bins, b=1, alpha=1., lmax=1000, sigma_zph=0., compute_at_z0=False):
	# DNDZ = dNdzInterpolation(z, dndz, nbins=1, z_min=z[0], z_max=z[-1])
	DNDZ = dNdzInterpolation(z, dndz, bins=bins, sigma_zph=sigma_zph, z_min=0, z_max=1)
	limb = Limber(cosmo, lmin=0, lmax=lmax, compute_at_z0=compute_at_z0)
	gals = GalsTomo(cosmo, DNDZ, b=b, alpha=alpha)
	return limb.GetCl(gals)
	
def GetNlgg(counts, mask=None, lmax=None, return_ngal=False):
    """
    Returns galaxy shot-noise spectra given a number counts Healpix map. 
    If return_ngal is True, it returns also the galaxy density in gal/ster.

    Note
    ----
    1. Use only binary mask.
    2. If mask is not None, yielded spectrum is not pseudo
    """
    counts = np.asarray(counts)

    if lmax is None: lmax = hp.npix2nside(counts.size) * 2
    if mask is not None: 
        mask = np.asarray(mask)
        fsky = np.mean(mask)
    else: 
        mask = 1.
        fsky = 1.

    N_tot = np.sum(counts * mask)
    ngal  = N_tot / 4. / np.pi / fsky

    if return_ngal:
        return np.ones(lmax+1) / ngal, ngal
    else:
        return np.ones(lmax+1) / ngal

def GetDg(clkg_data, clgg_data, clkg_th, clgg_th, lbmin=None, lbmax=None):

	global GetDg
	kg = clkg_data[lbmin:lbmax]/clkg_th[lbmin:lbmax]
	gg = clgg_th[lbmin:lbmax]/clgg_data[lbmin:lbmax]
	return np.mean(kg*np.sqrt(gg))

def GetAllDg(clkg_sims, clgg_sims, clkg_th, clgg_th, lbmin=None, lbmax=None):
	global GetDg
	return np.asarray( [GetDg(clkg_sims[i], clgg_sims[i], clkg_th, clgg_th, lbmin=lbmin, lbmax=lbmax) for i in xrange(clkg_sims.shape[0])] )


# Params
nside     = 256
delta_ell = 10
lmin      = 10
lmax      = 250
nsims     = 500
lmax_map  = 500
K_S_min   = 0.
K_S_max   = 13.9
zmin      = 0.0
zmax      = 0.24
bias      = 1.
# ngal      = 1.23 #1.29446416e-05 # gal/ster

np.random.seed(123)

print K_S_min, K_S_max, zmin, zmax

# Paths
fits_planck_mask = '../Data/mask_convergence_planck_2015_512.fits'
twoMPZ_mask      = 'fits/2MPZ_mask_tot_256.fits' # N_side = 256
twoMPZ_cat       = 'fits/results16_23_12_14_73.fits'

# Load Cat 
print("...loading 2MPZ catalogue...")
twompz = Load2MPZ(twoMPZ_cat, K_S_min=K_S_min, K_S_max=K_S_max)
print("...done...")

# Read masks
mask_planck_2015 = LoadPlanckMask(fits_planck_mask, nside, do_plots=False)
mask_2MPZ        = hp.read_map(twoMPZ_mask, verbose=False)
mask             = mask_planck_2015 * mask_2MPZ

# Get the gal number density
cat_tmp = twompz[(twompz.ZPHOTO >= zmin) & (twompz.ZPHOTO < zmax)]
counts = hpy.GetCountsMap(cat_tmp.RA, cat_tmp.DEC, nside, coord='G', rad=True)
counts = hp.ma(counts)
counts.mask = np.logical_not(mask)
ngal = counts.mean()
ngal_sr = ngal/hp.nside2pixarea(nside)

print ngal, ngal_sr

# exit()

# Initializing Cosmo class
cosmo_nl = Cosmo(nonlinear=True)

# 2MPZ dN/dz
# z, n_z = GetdNdzNorm(nz=1000)
nz, z, _ = hist(twompz.ZPHOTO,'knuth', normed=1, histtype='step')
z = 0.5 * (z[1:]+z[:-1])


# Get theoretical quantities
# clkg_th = GetCrossSpectrumMagLim(cosmo_nl, 2.21, 0.053, 1.43, bias=1., alpha=1., lmax=2000)
# clgg_th = GetAutoSpectrumMagLim( cosmo_nl, 2.21, 0.053, 1.43, bias=1., alpha=1., lmax=2000)
clkg_th = GetCrossSpectrum(cosmo_nl, z, nz, [zmin,zmax], b=1.24, alpha=1., lmax=500, sigma_zph=0.015)
clgg_th = GetAutoSpectrum(cosmo_nl, z, nz,  [zmin,zmax], b=1.24, alpha=1., lmax=500, sigma_zph=0.015)
# clkk_th = cosmo_nl.cmb_spectra(2000)[:,4]
lcmb = LensCMB(cosmo_nl)
limb = Limber(cosmo_nl, lmax=2000)
clkk_th = limb.GetCl(lcmb)
nlkk    = np.loadtxt('/Users/fbianchini/Research/CosMojo/data/nlkk_planck2015.dat', unpack=1, usecols=[1])[:2001]


# embed()
# exit()

# Power spectrum estimator
# est = cs.Master(np.ones_like(mask), lmin=lmin, lmax=lmax, delta_ell=delta_ell, MASTER=1, pixwin=False)
est = cs.Master(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell, MASTER=0, pixwin=False)

clkg = np.zeros((nsims,est.lb.size))
clgg = np.zeros((nsims,est.lb.size))

for n in tqdm(xrange(nsims)):
	map_kk, map_gg = hpy.GetCorrMaps(clkg_th, clkk_th, clgg_th, nside, lmax=lmax_map, pixwin=False)
	map_nkk = hp.synfast(nlkk, nside, lmax_map, pixwin=False)
	map_kk_tot = map_kk + map_nkk
	# counts = hpy.GetCountsTot(map_gg.copy(), ngal, dim='pix',)
	# counts = hpy.GetCountsTot(map_gg.copy(), ngal/2., dim='pix')
	map_gg_tot = map_gg+hp.synfast(np.ones(lmax_map+1)*1./(ngal_sr/2.), nside, lmax_map, pixwin=False) #hpy.Counts2Delta(counts, mask=mask)
	
	# counts2 = hpy.GetCountsTot(map_gg.copy(), ngal/2., dim='pix')
	map_gg_tot2 = map_gg+hp.synfast(np.ones(lmax_map+1)*1./(ngal_sr/2.), nside, lmax_map, pixwin=False) #hpy.Counts2Delta(counts2, mask=mask)

	# clkg[n] = est.get_spectra(map_kk_tot, map2=map_gg_tot, analytic_errors=0)
	clkg[n] = est.get_spectra(map_kk_tot, map2=(map_gg_tot+map_gg_tot2)/2., analytic_errors=0)
	# clgg[n] = est.get_spectra(map_gg_tot, analytic_errors=0)
	# clgg[n] = est.get_spectra(map_gg_tot, nl=1./(ngal/hp.nside2pixarea(nside))*np.ones(lmax+1), pseudo=0., analytic_errors=0)
	# clgg[n] = est.get_spectra(map_gg_tot, nl=GetNlgg(counts, mask=mask), pseudo=0, analytic_errors=0)
	clgg[n] = est.get_spectra((map_gg_tot+map_gg_tot2)/2., analytic_errors=0) - est.get_spectra((map_gg_tot-map_gg_tot2)/2., analytic_errors=0)

	del map_gg, map_kk, map_nkk, map_gg_tot, map_kk_tot, #map_gg_tot2

gg_dic = {}
kg_dic = {}
kg_dic['lb']   = est.lb
gg_dic['lb']   = est.lb
kg_dic['sims'] = clkg
gg_dic['sims'] = clgg

fname_gg = 'clgg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zmin)+'_zmax'+str(zmax)+'_nsims'+str(nsims)+'.pkl.gz'
fname_kg = 'clkg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zmin)+'_zmax'+str(zmax)+'_nsims'+str(nsims)+'.pkl.gz'

pickle.dump(kg_dic, gzip.open('spectra/sims/'+fname_kg,'wb'), protocol=2)
pickle.dump(gg_dic, gzip.open('spectra/sims/'+fname_gg,'wb'), protocol=2)

embed()



# clkg_slash = GetCrossSpectrumMagLim(cosmo_nl, 2.21, 0.053, 1.43, bias=1., alpha=1., lmax=1000, compute_at_z0=True)
# clgg_slash = GetAutoSpectrumMagLim( cosmo_nl, 2.21, 0.053, 1.43, bias=1., alpha=1., lmax=1000, compute_at_z0=True)

# clkg_slash_binned = est.bin_spectra(clkg_slash)
# clgg_slash_binned = est.bin_spectra(clgg_slash)
