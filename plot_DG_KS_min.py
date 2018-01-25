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

from TWOmpz_utils import Load2MPZ, GetdNdzNorm, Giannantonio15Params

import curvspec.master as cs

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

def GetDg(clkg_data, clgg_data, clkg_th, clgg_th, err_kg=None, err_gg=None, lbmin=None, lbmax=None):
	kg = clkg_data[lbmin:lbmax+1]/clkg_th[lbmin:lbmax+1]
	gg = clgg_th[lbmin:lbmax+1]/(clgg_data[lbmin:lbmax+1])

	if (err_kg is None) and (err_gg is None):
		w = np.ones_like(kg)
	else:
		sigma2 = np.abs(clgg_th[lbmin:lbmax+1]/(clgg_data[lbmin:lbmax+1]*clkg_th[lbmin:lbmax+1]**2)) * (err_kg[lbmin:lbmax+1]**2 + 0.25*clkg_data[lbmin:lbmax+1]**2*err_gg[lbmin:lbmax+1]**2/clgg_data[lbmin:lbmax+1]**2)
		w = 1./sigma2
		# print w

	x = kg*np.sqrt(gg)

	return np.sum(x*w)/np.sum(w)

def GetAllDg(clkg_sims, clgg_sims, clkg_th, clgg_th, err_kg=None, err_gg=None, lbmin=None, lbmax=None):
	return np.asarray( [GetDg(clkg_sims[i], clgg_sims[i], clkg_th, clgg_th, err_kg=err_kg, err_gg=err_gg, lbmin=lbmin, lbmax=lbmax) for i in xrange(clkg_sims.shape[0])] )

# Params
delta_ell = 10
lmin      = 10
lmax      = 250
K_S_mins  = [0.,12., 13.]
K_S_max   = 13.9
zmin      = 0.0
zmax      = 0.24
bias      = 1.
nsims     = 500
lbmax     = 4

# Initializing Cosmo class
cosmo = Cosmo(Giannantonio15Params, nonlinear=True)
cosmo_lin = Cosmo(Giannantonio15Params)

# To bin theoretical spectra
binner = cs.Binner(lmin=lmin, lmax=lmax, delta_ell=delta_ell)

DGs = []
err_DGs = []
DGs_lin = []
err_DGs_lin = []

for K_S_min in K_S_mins:
	# Load Cat 
	print("...loading 2MPZ catalogue...")
	twoMPZ_cat = 'fits/results16_23_12_14_73.fits'
	twompz = Load2MPZ(twoMPZ_cat, K_S_min=K_S_min, K_S_max=K_S_max)
	print("...done...")

	file_clkg = 'spectra/2MPZ_Planck2015_clkg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zmin)+'_zmax'+str(zmax)+'_split.dat' 
	file_clgg = 'spectra/2MPZ_clgg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zmin)+'_zmax'+str(zmax)+'_split.dat' 
	lb, clgg, err_clgg = np.loadtxt(file_clgg, unpack=1)
	lb, clkg, err_clkg = np.loadtxt(file_clkg, unpack=1)

	# Sims spectra
	# fname_gg = 'clgg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zmin)+'_zmax'+str(zmax)+'_nsims'+str(nsims)+'.pkl.gz'
	# fname_kg = 'clkg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zmin)+'_zmax'+str(zmax)+'_nsims'+str(nsims)+'.pkl.gz'
	fname_gg = 'clgg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_0.0_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zmin)+'_zmax'+str(zmax)+'_nsims'+str(nsims)+'.pkl.gz'
	fname_kg = 'clkg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_0.0_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zmin)+'_zmax'+str(zmax)+'_nsims'+str(nsims)+'.pkl.gz'

	clkg_sims = pickle.load(gzip.open('spectra/sims/'+fname_kg,'rb'))
	clgg_sims = pickle.load(gzip.open('spectra/sims/'+fname_gg,'rb'))

	# 2MPZ dN/dz
	nz, z, _ = hist(twompz.ZPHOTO,'knuth', normed=1, histtype='step')
	z = 0.5 * (z[1:]+z[:-1])
	plt.close()

	# Get theoretical quantities
	clkg_slash = GetCrossSpectrum(cosmo, z, nz, [zmin,zmax], b=bias, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
	clgg_slash = GetAutoSpectrum(cosmo,  z, nz, [zmin,zmax], b=bias, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
	clkg_slash_binned = binner.bin_spectra(clkg_slash)
	clgg_slash_binned = binner.bin_spectra(clgg_slash)

	clkg_slash_lin = GetCrossSpectrum(cosmo_lin, z, nz, [zmin,zmax], b=bias, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
	clgg_slash_lin = GetAutoSpectrum(cosmo_lin,  z, nz, [zmin,zmax], b=bias, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
	clkg_slash_binned_lin = binner.bin_spectra(clkg_slash_lin)
	clgg_slash_binned_lin = binner.bin_spectra(clgg_slash_lin)

	# D_G errors from sims
	dgs_err = [np.std(GetAllDg(clkg_sims['sims'],clgg_sims['sims'],clkg_slash_binned,clgg_slash_binned, err_kg=np.std(clkg_sims['sims'], axis=(0)), err_gg=np.std(clgg_sims['sims'], axis=(0)), lbmax=i)) for i in xrange(2,clkg_slash_binned.size+1)]
	dgs_err_lin = [np.std(GetAllDg(clkg_sims['sims'],clgg_sims['sims'],clkg_slash_binned_lin,clgg_slash_binned_lin, err_kg=np.std(clkg_sims['sims'], axis=(0)), err_gg=np.std(clgg_sims['sims'], axis=(0)), lbmax=i)) for i in xrange(2,clkg_slash_binned.size+1)]

	# D_G Data 
	dgs_data = [GetDg(clkg,clgg,clkg_slash_binned,clgg_slash_binned, err_kg=err_clkg, err_gg=err_clgg, lbmax=i) for i in xrange(2,clgg.size+1)]
	dgs_data_lin = [GetDg(clkg,clgg,clkg_slash_binned_lin,clgg_slash_binned_lin, err_kg=err_clkg, err_gg=err_clgg, lbmax=i) for i in xrange(2,clgg.size+1)]

	DGs.append(dgs_data[lbmax])
	DGs_lin.append(dgs_data_lin[lbmax])
	err_DGs.append(dgs_err[lbmax])
	err_DGs_lin.append(dgs_err_lin[lbmax])
	# embed()


DNDZ = dNdzInterpolation(z, nz, bins=[zmin,zmax], sigma_zph=0.015, z_min=0, z_max=1)
zmed = DNDZ.z_med_bin(0)

# fig, ax = plt.subplots(figsize=(5,5))
# ax.set_title(r'2MPZ - $%.2f < z < %.2f$'%(zmin,zmax))
# # plt.fill_between(lb[1:], [cosmo.D_z_norm(zmed) for i in xrange(len(lb[1:]))]+np.asarray(dgs_err[zbin]), [cosmo.D_z_norm(zmed) for i in xrange(len(lb[1:]))]-np.asarray(dgs_err[zbin]),alpha=0.6, color='lightgrey', label=r'$1\sigma$ from sims')
# ax.axhline(cosmo.D_z_norm(zmed), ls='--', color='k')#, label=r'$D_G(z_{\rm med}=%.3f)$'%zmed)
# ax.errorbar([0,1,2,3], DGs, yerr=err_DGs, fmt='o')#, label='2MPZ Data')
# ax.legend()
# labels = ['0','10','12','13']#[str(ks) for ks in K_S_mins]
# # ax.set_xticklabels(labels)
# plt.xticks([0,1,2,3], labels)
# ax.set_ylim(0.75,1.5)
# ax.set_xlabel(r'Minimum $K_S$')#, size=15)
# ax.set_ylabel(r'$D_G$')#, size=15)
# plt.tight_layout()
# # plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_various_KS_min_KS_max_'+str(K_S_max)+'_nside256_2.pdf')
# plt.show()
# plt.close()

embed()

fig, ax = plt.subplots(figsize=(6,5))
ax.set_title(r'2MPZ - $%.2f < z < %.2f$'%(zmin,zmax))
# plt.fill_between(lb[1:], [cosmo.D_z_norm(zmed) for i in xrange(len(lb[1:]))]+np.asarray(dgs_err[zbin]), [cosmo.D_z_norm(zmed) for i in xrange(len(lb[1:]))]-np.asarray(dgs_err[zbin]),alpha=0.6, color='lightgrey', label=r'$1\sigma$ from sims')
ax.axhline(cosmo.D_z_norm(zmed), ls='--', color='k')#, label=r'$D_G(z_{\rm med}=%.3f)$'%zmed)
ax.errorbar(0, DGs[0], yerr=err_DGs[0], fmt='o', label='Fiducial', color='#083D77', ms=8)
ax.errorbar(1, DGs[-2], yerr=err_DGs[-2], fmt='s', label=r'$K_S^{\rm min}=12$', color='#8C2F39', ms=8)
ax.errorbar(2, DGs[-1], yerr=err_DGs[-1], fmt='^', label=r'$K_S^{\rm min}=13$', color='#7B2493', ms=8)
ax.errorbar(3, DGs_lin[0], yerr=err_DGs_lin[0], fmt='*', label='Linear', color='#F18F01', ms=8)
ax.errorbar(4, 1.17, yerr=0.2, fmt='v', label=r'Spec $dN/dz$', color='#152614', ms=8)
ax.legend(ncol=2, loc='upper left')
# labels = ['0','10','12','13']#[str(ks) for ks in K_S_mins]
ax.set_xticklabels('')
# plt.xticks([0,1,2,3], labels)
# ax.set_ylim(0.9,1.5)
ax.set_xlabel(r'')#, size=15)
ax.set_ylabel(r'$D_G$')#, size=15)
plt.tight_layout()
plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_various_KS_min_KS_max_'+str(K_S_max)+'_nside256_2.pdf')
plt.show()
# plt.close()




