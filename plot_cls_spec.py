"""
This code performs xc analysis between 2MPZ catalog and PlanckDR2 lensing  (see arXiv:1607.xxxx)
"""

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
# import seaborn as sns
import sys, pickle, gzip

from astropy.visualization import hist

from TWOmpz_utils import Load2MPZ, GetdNdzNorm

import curvspec.master as cs

from cosmojo.universe import Cosmo
from cosmojo.survey import dNdzInterpolation, dNdzMagLim
from cosmojo.limber import Limber
from cosmojo.kernels import *

import emcee

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
   plt.rcParams['legend.fontsize']  = 17
   plt.rcParams['legend.frameon']  = False

   plt.rcParams['xtick.major.width'] = 1
   plt.rcParams['ytick.major.width'] = 1
   plt.rcParams['xtick.minor.width'] = 1
   plt.rcParams['ytick.minor.width'] = 1
   # plt.clf()
   # sns.set(rc('font',**{'family':'serif','serif':['Computer Modern']}))
   # sns.set_style("ticks", {'figure.facecolor': 'grey'})

SetPlotStyle()

def GetCrossSpectrumMagLim(cosmo, a, z0, b, bias=1, alpha=1., lmax=1000, sigma_zph=0., compute_at_z0=False):
	DNDZ = dNdzMagLim(a, z0, b, sigma_zph=sigma_zph)
	limb = Limber(cosmo, lmin=0, lmax=lmax, compute_at_z0=compute_at_z0)
	gals = GalsTomo(cosmo, DNDZ, b=bias, alpha=alpha)
	return limb.GetCl(gals, k2=LensCMB(cosmo))

def GetAutoSpectrumMagLim(cosmo, a, z0, b, bias=1, alpha=1., lmax=1000, sigma_zph=0., compute_at_z0=False):
	# DNDZ = dNdzInterpolation(z, dndz, nbins=1, z_min=z[0], z_max=z[-1])
	DNDZ = dNdzMagLim(a, z0, b, sigma_zph=sigma_zph)
	limb = Limber(cosmo, lmin=0, lmax=lmax, compute_at_z0=compute_at_z0)
	gals = GalsTomo(cosmo, DNDZ, b=bias, alpha=alpha)
	return limb.GetCl(gals)

def GetCrossSpectrum(cosmo, z, dndz, bins, b=1, alpha=1., lmax=1000, sigma_zph=0., i=0, compute_at_z0=False):
	DNDZ = dNdzInterpolation(z, dndz, bins=bins, sigma_zph=sigma_zph, z_min=0, z_max=1)
	limb = Limber(cosmo, lmin=0, lmax=lmax, compute_at_z0=compute_at_z0)
	gals = GalsTomo(cosmo, DNDZ, b=b, alpha=alpha)
	return limb.GetCl(gals, k2=LensCMB(cosmo), i=i)

def GetAutoSpectrum(cosmo, z, dndz, bins, b=1, alpha=1., lmax=1000, sigma_zph=0., compute_at_z0=False):
	DNDZ = dNdzInterpolation(z, dndz, bins=bins, sigma_zph=sigma_zph, z_min=0, z_max=1)
	limb = Limber(cosmo, lmin=0, lmax=lmax, compute_at_z0=compute_at_z0)
	gals = GalsTomo(cosmo, DNDZ, b=b, alpha=alpha)
	return limb.GetCl(gals)

def lnlike(theta, lbin, cl, err_cl, cl_th, lbmin=None, lbmax=None):
    b = theta
    model = b * cl_th
    inv_sigma2 = 1.0/(err_cl[lbmin:lbmax]**2)
    return -0.5*(np.sum((cl[lbmin:lbmax]-model[lbmin:lbmax])**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
    b = theta
    if 0.0 < b < 10.0:
        return 0.0
    return -np.inf

def lnprob(theta, lbin, cl, err_cl, cl_th, lbmin=None, lbmax=None):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta,  lbin, cl, err_cl, cl_th, lbmin=lbmin, lbmax=lbmax)


# Params
delta_ell = 20
lmin      = 10
lmax      = 250
K_S_min   = 0.
K_S_max   = 13.9
# zmin      = 0.08
# zmax      = 0.4
zbins     = [(0.,0.24)]#, (0.,0.08), (0.08,0.4)]
bias      = 1.
lbmax     = 4
lbmin     = 1
nside = 256

# Load Cat 
print("...loading 2MPZ catalogue...")
twoMPZ_cat = 'fits/results16_23_12_14_73.fits'
twompz = Load2MPZ(twoMPZ_cat, K_S_min=K_S_min, K_S_max=K_S_max)
print("...done...")

# Data spectra
clkg = {}
clgg = {}
err_clkg = {}
err_clgg = {}

for zbin in zbins:
	file_clkg = 'spectra/2MPZ_Planck2015_clkg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split_spec.dat' 
	file_clgg = 'spectra/2MPZ_clgg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split_spec.dat' 
	lb, clgg[zbin], err_clgg[zbin] = np.loadtxt(file_clgg, unpack=1)
	lb, clkg[zbin], err_clkg[zbin] = np.loadtxt(file_clkg, unpack=1)


# Get theoretical quantities
cosmo_nl = Cosmo(nonlinear=True)
cosmo = Cosmo()

# 2MPZ dN/dz
nz, z, _ = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step')
z = 0.5 * (z[1:]+z[:-1])
plt.close()

binner = cs.Binner(lmin=lmin, lmax=lmax, delta_ell=delta_ell)

clkg_th = {}
clgg_th = {}
clkg_th_binned = {}
clgg_th_binned = {}

for zbin in zbins:
	clkg_th[zbin] = GetCrossSpectrum(cosmo_nl, z, nz, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, compute_at_z0=0)
	clgg_th[zbin] = GetAutoSpectrum(cosmo_nl,  z, nz, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, compute_at_z0=0)
	clkg_th_binned[zbin] = binner.bin_spectra(clkg_th[zbin])
	clgg_th_binned[zbin] = binner.bin_spectra(clgg_th[zbin])


ndim, nwalkers = 1, 100
pos = [1. + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

for zbin in zbins:
	sampler_kg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(lb, clkg[zbin], err_clkg[zbin], clkg_th_binned[zbin], lbmin, lbmax))
	sampler_kg.run_mcmc(pos, 500)
	samples_kg = sampler_kg.chain[:, 100:, :].reshape((-1, 1))
	b_kg = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_kg, [16, 50, 84],axis=0)))

	sampler_gg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(lb, clgg[zbin], err_clgg[zbin], clgg_th_binned[zbin], lbmin, lbmax))
	sampler_gg.run_mcmc(pos, 500)
	samples_gg = sampler_gg.chain[:, 100:, :].reshape((-1, 1))
	b_gg = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_gg, [16, 50, 84],axis=0)))

	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(1,2,1) 
	plt.suptitle(r'2MPZ - $%.2f < z < %.2f$' %(zbin[0],zbin[1]), size=20)

	clkg_tmp    = GetCrossSpectrum(cosmo,    z, nz, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500,)
	clkg_nl_tmp = GetCrossSpectrum(cosmo_nl, z, nz, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500,)

	lab = r'$K_S < %.1f$' %(K_S_max)
	# lab = r'$%.1f < K_S < %.1f$' %(K_S_min, K_S_max)
	ax.plot(b_kg[0][0]*clkg_tmp, color='grey', label=r'$b=%.2f$' %(b_kg[0][0]), ls='--',)
	# ax.plot(b_kg[0][0]*clkg_tmp, color='grey', label=r'$b=%.2f$ $\sigma_z=0.015$' %(b_kg[0][0]), ls='--',)
	ax.fill_between(np.arange(501), (b_kg[0][0]+b_kg[0][1])*clkg_nl_tmp, (b_kg[0][0]-b_kg[0][2])*clkg_nl_tmp,  color='darkgrey', alpha=0.2)#, ls='--',)
	ax.plot(b_kg[0][0]*clkg_nl_tmp, color='darkgrey',  label=r'NL $b=%.2f$' %(b_kg[0][0]))
	# ax.plot(b_kg[0][0]*clkg_nl_tmp, color='darkgrey',  label=r'NL $b=%.2f$ $\sigma_z=0.015$' %(b_kg[0][0]))
	ax.errorbar(lb, clkg[zbin], yerr=err_clkg[zbin], fmt='o', capsize=0, ms=5, label=lab)
	ax.set_ylabel(r'$C_{\ell}^{\kappa g}$')
	ax.set_xlabel(r'Multipole $\ell$')
	ax.legend(loc='lower left')
	ax.axhline(ls='--', color='grey')
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax.set_xlim([2., lb[lbmax]+10])
	ax.set_yscale('log')
	ax.set_ylim([1e-8, 1e-5])


	ax = fig.add_subplot(1,2,2) 
	# plt.title(r'$%.2f < z < %.2f$' %(zbin[0],zbin[1]))

	clgg_tmp    = GetAutoSpectrum(cosmo,    z, nz, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500,)
	clgg_nl_tmp = GetAutoSpectrum(cosmo_nl, z, nz, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500,)

	lab = r'$K_S < %.1f$' %(K_S_max)
	# lab = r'$%.1f < K_S < %.1f$' %(K_S_min, K_S_max)
	ax.plot(b_gg[0][0]**2*clgg_tmp, color='grey', label=r'$b=%.2f$' %(b_gg[0][0]), ls='--',)
	# ax.plot(b_gg[0][0]**2*clgg_tmp, color='grey', label=r'$b=%.2f$ $\sigma_z=0.015$' %(b_gg[0][0]), ls='--',)
	ax.fill_between(np.arange(501), (b_gg[0][0]+b_gg[0][1])**2*clgg_nl_tmp, (b_gg[0][0]-b_gg[0][2])**2*clgg_nl_tmp,  color='darkgrey', alpha=0.2)#, ls='--',)
	ax.plot(b_gg[0][0]**2*clgg_nl_tmp, color='darkgrey',  label=r'NL $b=%.2f$' %(b_gg[0][0]))
	# ax.plot(b_gg[0][0]**2*clgg_nl_tmp, color='darkgrey',  label=r'NL $b=%.2f$ $\sigma_z=0.015$' %(b_gg[0][0]))
	ax.errorbar(lb, clgg[zbin], yerr=err_clgg[zbin], fmt='o', capsize=0, ms=5, label=lab)
	ax.set_ylabel(r'$C_{\ell}^{gg}$')
	ax.set_xlabel(r'Multipole $\ell$')
	ax.legend(loc='lower left')
	ax.axhline(ls='--', color='grey')
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
	ax.set_xlim([2., lb[lbmax]+10])
	ax.set_yscale('log')
	ax.set_ylim([1e-5, 1e-3])

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	# plt.savefig('plots/2MPZ_Planck_clkg_dl'+str(args.delta_ell)+'_lmin_'+str(args.lmin)+'_lmax'+str(args.lmax)+'_KS_min_'+str(args.K_S_min)+'_KS_max_'+str(args.K_S_max)+'_nside'+str(args.nside)+'_zmin_'+str(args.zmin)+'_zmax'+str(args.zmax)+'_split.pdf', bbox_inches='tight')
	# plt.show()
	# plt.close()

	embed()
