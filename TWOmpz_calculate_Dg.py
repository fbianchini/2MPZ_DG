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
   plt.rcParams['legend.fontsize']  = 15
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

def lnlike(theta, D_G, err_D_G, D_th):
	A = theta
	model = A * D_th
	inv_sigma2 = 1.0/err_D_G**2
	return -0.5*((D_G - model)**2*inv_sigma2 - np.log(inv_sigma2))

def lnprior(theta):
	A = theta
	if -10 < A < 10.0:
		return 0.0
	return -np.inf

def lnprob(theta, D_G, err_D_G, D_th):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, D_G, err_D_G, D_th)

# Params
delta_ell = 10
lmin      = 10
lmax      = 250
K_S_min   = 0.
K_S_max   = 13.9
# zmin      = 0.08
# zmax      = 0.4
zbins     = [(0.,0.24)]#, (0.,0.08), (0.08,0.4)]
# zbins     = [(0.,0.4), (0.,0.08), (0.08,0.4)]
bias      = 1.24
nsims     = 500
# nside. 
fsky = 0

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
	if fsky:
		file_clkg = 'spectra/2MPZ_Planck2015_clkg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split_fsky.dat' 
		file_clgg = 'spectra/2MPZ_clgg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split_fsky.dat' 
	else:
		file_clkg = 'spectra/2MPZ_Planck2015_clkg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split.dat' 
		file_clgg = 'spectra/2MPZ_clgg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split.dat' 
	lb, clgg[zbin], err_clgg[zbin] = np.loadtxt(file_clgg, unpack=1)
	lb, clkg[zbin], err_clkg[zbin] = np.loadtxt(file_clkg, unpack=1)

# Sims spectra
clkg_sims = {}
clgg_sims = {}

for zbin in zbins:
	if fsky:
		fname_gg = 'clgg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_nsims'+str(nsims)+'_fsky.pkl.gz'
		fname_kg = 'clkg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_nsims'+str(nsims)+'_fsky.pkl.gz'
	else:
		fname_gg = 'clgg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_nsims'+str(nsims)+'.pkl.gz'
		fname_kg = 'clkg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_nsims'+str(nsims)+'.pkl.gz'
	clkg_sims[zbin] = pickle.load(gzip.open('spectra/sims/'+fname_kg,'rb'))
	clgg_sims[zbin] = pickle.load(gzip.open('spectra/sims/'+fname_gg,'rb'))

# assert( clkg_sims[zbins[0]]['lb'] == lb )

# To bin theoretical spectra
binner = cs.Binner(lmin=lmin, lmax=lmax, delta_ell=delta_ell)

# Initializing Cosmo class
# cosmo = Cosmo()
cosmo_nl = Cosmo(Giannantonio15Params, nonlinear=True)
# cosmo_w  = Cosmo(params={'w':-0.5})

# 2MPZ dN/dz
nz, z, _ = hist(twompz.ZPHOTO,'knuth', normed=1, histtype='step')
# nz, z, _ = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step')
z = 0.5 * (z[1:]+z[:-1])
plt.close()

# Get theoretical quantities
clkg_slash = {}
clgg_slash = {}
clkg_slash_binned = {}
clgg_slash_binned = {}

for zbin in zbins:
	clkg_slash[zbin] = GetCrossSpectrum(cosmo_nl, z, nz, [zbin[0],zbin[1]], b=bias, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
	clgg_slash[zbin] = GetAutoSpectrum(cosmo_nl,  z, nz, [zbin[0],zbin[1]], b=bias, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
	clkg_slash_binned[zbin] = binner.bin_spectra(clkg_slash[zbin])
	clgg_slash_binned[zbin] = binner.bin_spectra(clgg_slash[zbin])


# D_G errors from sims w/o weights
dgs_err_uw = {}
for zbin in zbins:
	dgs_err_uw[zbin] = [np.std(GetAllDg(clkg_sims[zbin]['sims'],
										clgg_sims[zbin]['sims'],
										clkg_slash_binned[zbin],
										clgg_slash_binned[zbin],lbmax=i)) 
										for i in xrange(1,clkg_slash_binned[zbin].size)]

# D_G errors from sims w/ weights
dgs_err = {}
for zbin in zbins:
	dgs_err[zbin] = [np.std(GetAllDg(clkg_sims[zbin]['sims'],
									 clgg_sims[zbin]['sims'],
									 clkg_slash_binned[zbin],
									 clgg_slash_binned[zbin],
									 # err_kg=err_clkg[zbin], 
									 # err_gg=err_clgg[zbin], lbmax=i)) 
									 err_kg=np.std(clkg_sims[zbin]['sims'], axis=(0)), 
									 err_gg=np.std(clgg_sims[zbin]['sims'], axis=(0)), lbmax=i)) 
									 for i in xrange(1,clkg_slash_binned[zbin].size)]

# D_G sims w/ weights
# dgs_sims = {}
for zbin in zbins:
	dgs_sims = [GetAllDg(clkg_sims[zbin]['sims'],
							   clgg_sims[zbin]['sims'],
							   clkg_slash_binned[zbin],
							   clgg_slash_binned[zbin], 
							   err_kg=np.std(clkg_sims[zbin]['sims'], axis=(0)), 
							   err_gg=np.std(clgg_sims[zbin]['sims'], axis=(0)),
							   # err_kg=err_clkg[zbin], 
							   # err_gg=err_clkg[zbin],
							   lbmax=i) for i in xrange(1,clgg[zbin].size)]

# D_G sims w/o weights
for zbin in zbins:
	dgs_sims_uw = [GetAllDg(clkg_sims[zbin]['sims'],
								  clgg_sims[zbin]['sims'],
								  clkg_slash_binned[zbin],
								  clgg_slash_binned[zbin], lbmax=i) 
								  for i in xrange(1,clgg[zbin].size)]


# D_G Data w/ weights
dgs_data = {}
for zbin in zbins:
	dgs_data[zbin] = [GetDg(clkg[zbin],
							clgg[zbin],
							clkg_slash_binned[zbin],
							clgg_slash_binned[zbin], 
							err_kg=err_clkg[zbin], 
							err_gg=err_clgg[zbin],lbmax=i) for i in xrange(1,clgg[zbin].size)]

# D_G Data w/o weights
dgs_data_uw = {}
for zbin in zbins:
	dgs_data_uw[zbin] = [GetDg(clkg[zbin],
							   clgg[zbin],
							   clkg_slash_binned[zbin],
							   clgg_slash_binned[zbin], lbmax=i) 
							   for i in xrange(1,clgg[zbin].size)]

embed()

exit()

cw = '#E9724C'
cnow = '#255F85'

plt.figure(figsize=(5.5,5.5))
# # Plot D_G vs L_max
for i,zbin in enumerate(zbins):
	DNDZ = dNdzInterpolation(z, nz, bins=[zbin[0],zbin[1]], sigma_zph=0.015, z_min=0, z_max=1)
	zmed = DNDZ.z_med_bin(0)
	plt.subplot(1,1,i+1)
	plt.title(r'$%.2f < z < %.2f$'%(zbin[0],zbin[1]))
	plt.fill_between(lb[1:], dgs_data_uw[zbin]+np.asarray(dgs_err_uw[zbin]), dgs_data_uw[zbin]-np.asarray(dgs_err_uw[zbin]),alpha=0.2, color=cnow, linewidth=0.0)#, label=r'$1\sigma$ from sims w/o weights')
	plt.plot(lb[1:], dgs_data_uw[zbin], color=cnow, ls='-.', label='2MPZ Data (no weights)')
	plt.fill_between(lb[1:], dgs_data[zbin]+np.asarray(dgs_err[zbin]), dgs_data[zbin]-np.asarray(dgs_err[zbin]),alpha=0.2, color=cw, linewidth=0.0)#, label=r'$1\sigma$ from sims w/o weights')
	plt.plot(lb[1:], dgs_data[zbin], color=cw, label='2MPZ Data')
	plt.axhline(cosmo_nl.D_z_norm(zmed), ls='--', color='k', label=r'$D_G(z_{\rm med}=%.2f)$'%zmed)
	# plt.axvline(180./np.rad2deg(1./cosmo_nl.k_NL(zmed)/cosmo_nl.f_K(zmed)),ls=':', color='darkgrey', label=r'$\ell_{\rm NL}(z_{\rm med}=%.2f)$'%zmed)
	plt.axvline(70,ls=':', color='darkgrey', label=r'$\ell_{\rm NL}(z_{\rm med}=%.2f)$'%zmed)
	# plt.axvline(cosmo_nl.k_NL(zmed)*cosmo_nl.f_K(zmed),ls=':', color='darkgrey', label=r'$\ell_{\rm NL}(z_{\rm med}=%.2f)$'%zmed)
	plt.legend(loc='upper right')
	plt.ylim(0.7,2.2)
	plt.xlabel(r'$\ell_{\rm max}$')#, size=15)
	# plt.xlabel(r'Maximum $L$-bin')#, size=15)
	# if i == 0: 
	plt.ylabel(r'$D_G$')#, size=15)
plt.tight_layout()
if fsky:
	plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_single_weighted_cooler_fsky.pdf')
else:
	plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_single_weighted_cooler_spec.pdf')
# plt.show()
plt.close()

# exit( )


# plt.figure(figsize=(5.5,5.5))
# # # Plot D_G vs L_max
# for i,zbin in enumerate(zbins):
# 	DNDZ = dNdzInterpolation(z, nz, bins=[zbin[0],zbin[1]], sigma_zph=0.015, z_min=0, z_max=1)
# 	zmed = DNDZ.z_med_bin(0)
# 	plt.subplot(1,1,i+1)
# 	plt.title(r'$%.2f < z < %.2f$'%(zbin[0],zbin[1]))
# 	plt.fill_between(lb[1:], [cosmo_nl.D_z_norm(zmed) for i in xrange(len(lb[1:]))]+np.asarray(dgs_err_uw[zbin]), [cosmo_nl.D_z_norm(zmed) for i in xrange(len(lb[1:]))]-np.asarray(dgs_err_uw[zbin]),alpha=0.6, color='lightgrey', label=r'$1\sigma$ from sims w/o weights')
# 	plt.fill_between(lb[1:], [cosmo_nl.D_z_norm(zmed) for i in xrange(len(lb[1:]))]+np.asarray(dgs_err[zbin]), [cosmo_nl.D_z_norm(zmed) for i in xrange(len(lb[1:]))]-np.asarray(dgs_err[zbin]),alpha=0.6, color='darkgrey', label=r'$1\sigma$ from sims w/ weights')
# 	plt.axhline(cosmo_nl.D_z_norm(zmed), ls='--', color='k', label=r'$D_G(z_{\rm med}=%.2f)$'%zmed)
# 	# plt.axvline(180./np.rad2deg(1./cosmo_nl.k_NL(zmed)/cosmo_nl.f_K(zmed)),ls='--', color='darkgrey', label=r'$\ell_{\rm NL}(z_{\rm med}=%.2f)$'%zmed)
# 	plt.plot(lb[1:], dgs_data[zbin], label='2MPZ Data')
# 	plt.plot(lb[1:], dgs_data_uw[zbin], label='2MPZ Data (no weights)')
# 	plt.legend(loc='upper left')
# 	plt.ylim(0.5,2)
# 	plt.xlabel(r'$\ell_{\rm max}$')#, size=15)
# 	# plt.xlabel(r'Maximum $L$-bin')#, size=15)
# 	# if i == 0: 
# 	plt.ylabel(r'$D_G$')#, size=15)
# plt.tight_layout()
# if fsky:
# 	plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_single_weighted_fsky.pdf')
# else:
# 	plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_single_weighted_.pdf')
# # plt.show()
# plt.close()

# embed()
# exit()

# plt.figure(figsize=(5,5))
# # plt.figure(figsize=(13,5))
# for i,zbin in enumerate(zbins):
# 	DNDZ = dNdzInterpolation(z, nz, bins=[zbin[0],zbin[1]], sigma_zph=0.015, z_min=0, z_max=1)
# 	zmed = DNDZ.z_med_bin(0)
# 	plt.subplot(1,1,i+1)
# 	plt.title(r'$%.2f < z < %.2f$'%(zbin[0],zbin[1]))
# 	plt.axhline(cosmo_nl.D_z_norm(zmed), ls='--', color='k', label=r'$D_G(z_{\rm med}=%.2f)$'%zmed)
# 	plt.errorbar(lb[1:], dgs_data[zbin], yerr=np.asarray(dgs_err[zbin]), fmt='-o', label='2MPZ Data')
# 	# plt.axvline(180./np.rad2deg(1./cosmo_nl.k_NL(zmed)/cosmo_nl.f_K(zmed)),ls='--', color='darkgrey', label=r'$\ell_{\rm NL}(z_{\rm med}=%.2f)$'%zmed)
# 	plt.legend(loc='upper left')
# 	plt.ylim(0.5,2)
# 	plt.xlabel(r'Maximum $L$-bin')#, size=15)
# 	# if i == 0: 
# 	plt.ylabel(r'$D_G$')#, size=15)
# plt.tight_layout()
# if fsky:
# 	plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_v2_single_weighted_fsky.pdf')
# else:
# 	plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_v2_single_weighted.pdf')
# # plt.show()
# plt.close()


# exit()

z_ = np.logspace(-4,3,1000)
cosmo = Cosmo(Giannantonio15Params)

d0 = cosmo.D_z_norm(z_)   
s8 = cosmo.sigma_Rz(8./cosmo.h)  
om = cosmo.omegam

gamma0 = 0.25
gammaa = -1.
w      = -1.2

dgamma0 = cosmo.D_z_norm(z_, gamma0=gamma0)   
dgamma0 = dgamma0*d0[-1]/dgamma0[-1] #* (cos)/(om*s8)

dgammaa = cosmo.D_z_norm(z_, gammaa=gammaa)   
dgammaa = dgammaa*d0[-1]/dgamma0[-1]

params_w = Giannantonio15Params.copy()
params_w['w'] = w
cosmo_w  = Cosmo(params_w)
dw = cosmo.D_z_norm(z_,)
dw = dw*d0[-1]/dw[-1] * (cosmo_w.omegam*cosmo_w.sigma_Rz(8./cosmo_w.h))/(om*s8)

cosmo_nl = Cosmo(Giannantonio15Params,nonlinear=True)

des_z       = [0.3,0.5,0.7,0.9,1.1]
des_DG      = [0.75,0.70,0.47,0.20,0.52]
des_DG_errs = [0.27,0.17,0.17,0.15,0.12]

# Plot D_G vs redshift
bmax = 5
_z_ = np.linspace(0.,1.4)
plt.figure(figsize=(8,6))
for i,zbin in enumerate(zbins):
	DNDZ = dNdzInterpolation(z, nz, bins=[zbin[0],zbin[1]], sigma_zph=0.015, z_min=0, z_max=1)
	zmed = DNDZ.z_med_bin(0)
	plt.errorbar(zmed, dgs_data[zbin][bmax], yerr=dgs_err[zbin][bmax], fmt='o', color='#F45D01', label='2MPZ')
plt.errorbar(des_z, des_DG, yerr=des_DG_errs, label='DES (Giannantonio+16)', color='grey', fmt='s')
plt.plot(_z_, np.asarray(cosmo_nl.D_z_norm(_z_)), label=r'$\Lambda$CDM')#,color='#345995')
plt.plot(z_, dw, '-.', label=r'$w_0=%.2f$'%w)#, color='#E40066')
plt.plot(z_, dgamma0, '--', label=r'$\gamma_0=%.2f$'%gamma0)#, color='#03CEA4')
plt.plot(z_, dgammaa, ':', label=r'$\gamma_a=%.2f$'%gammaa)#, color='#EAC435')
# plt.plot(z_, np.asarray(cosmo_w.D_z(z_))/np.asarray(cosmo_w.D_z(0.)), '--', label=r'$w=-0.5$')
# plt.plot(z_, cosmo.D_z_norm(z_), label=r'$\Lambda$CDM')
# plt.plot(z_, cosmo_w.D_z_norm(z_), '--', label=r'$w=-0.5$')
# plt.plot(z, cosmo.D_z_norm(z,gamma0=0.85), label=r'$\gamma_0=0.85$')
# plt.plot(z, cosmo.D_z_norm(z,gamma0=0.55, gammaa=0.1), label=r'$\gamma_a=0.1$')
plt.xlim([0,1.4])
# plt.xlim([0,0.2])
plt.legend(loc='upper right')
plt.ylabel(r'$D_G$')#, size=15)
plt.xlabel(r'$z$')#, size=15)
plt.tight_layout()
plt.savefig('plots/D_G_vs_z_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_weighted_spec.pdf')

plt.show()

embed()

ndim, nwalkers = 1, 100
pos = [1. + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(dgs_data[zbin][bmax], dgs_err[zbin][bmax],cosmo_nl.D_z_norm(zmed)))
sampler.run_mcmc(pos, 500)
samples = sampler.chain[:, 100:, :].reshape((-1, 1))
A = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0)))

embed()

"""
# Data spectra
file_clkg      = 'spectra/clkg_2MPZ_Planck2015_lmin10_lmax250_deltal20.dat'
file_clgg_shot = 'spectra/clgg_2MPZ_shotnoise_lmin10_lmax250_deltal20_new.dat'
file_clgg_half = 'spectra/clgg_2MPZ_halfdiff_lmin10_lmax250_deltal20.dat'

lb, clgg_shot, err_clgg_shot = np.loadtxt(file_clgg_shot, unpack=1)
lb, clgg_half, err_clgg_half = np.loadtxt(file_clgg_half, unpack=1)
lb, clkg, err_clkg           = np.loadtxt(file_clkg, unpack=1)

# Sims spectra
clkg_sims = pickle.load(gzip.open('spectra/clkg_sims_lmin10_lmax_250_deltal20.pk.gz','rb'))
clgg_sims = pickle.load(gzip.open('spectra/clgg_sims_lmin10_lmax_250_deltal20.pk.gz','rb'))

# To bin theoretical spectra
binner = cs.Binner(lmin=lmin, lmax=lmax, delta_ell=delta_ell)

# Initializing Cosmo class
cosmo = Cosmo()
cosmo_nl = Cosmo(nonlinear=True)

# 2MPZ dN/dz
# z, n_z = GetdNdzNorm(nz=1000)


# Get theoretical quantities
clkg_slash = GetCrossSpectrum(cosmo_nl, z, nz, [zmin,zmax], b=1, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
clgg_slash = GetAutoSpectrum(cosmo_nl, z, nz,  [zmin,zmax], b=1., alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
# clkg_slash = GetCrossSpectrumMagLim(cosmo_nl, 2.21, 0.053, 1.43, bias=1., alpha=1., lmax=1000, compute_at_z0=True)
# clgg_slash = GetAutoSpectrumMagLim( cosmo_nl, 2.21, 0.053, 1.43, bias=1., alpha=1., lmax=1000, compute_at_z0=True)

clkg_slash_binned = binner.bin_spectra(clkg_slash)
clgg_slash_binned = binner.bin_spectra(clgg_slash)


dgs_err = [np.std(GetAllDg(clkg_sims,clgg_sims,clkg_slash_binned,clgg_slash_binned,lbmax=i)) for i in xrange(2,clkg_slash_binned.size+1)]

dgs_data_half = [GetDg(clkg,clgg_half,clkg_slash_binned,clgg_slash_binned,lbmax=i) for i in xrange(2,clgg_half.size+1)]
dgs_data_shot = [GetDg(clkg,clgg_shot,clkg_slash_binned,clgg_slash_binned,lbmax=i) for i in xrange(2,clgg_half.size+1)]

embed()

plt.errorbar(lb[1:]-1.5, dgs_data_shot, yerr=np.asarray(dgs_err), label='Shot-noise')
plt.errorbar(lb[1:]+1.5, dgs_data_half, yerr=np.asarray(dgs_err), label='Half-diff')


plt.fill_between(lb[1:], [cosmo.D_z_norm(0.09) for i in xrange(len(lb[1:]))]+np.asarray(dgs_err)/2., [cosmo.D_z_norm(0.09) for i in xrange(len(lb[1:]))]-np.asarray(dgs_err)/2.,alpha=0.6, color='lightgrey', label=r'$1\sigma$ from sims')
plt.plot(lb[1:]-1.5, dgs_data_shot, label='Shot-noise')
plt.plot(lb[1:]+1.5, dgs_data_half, label='Half-diff')
plt.axhline(cosmo.D_z_norm(0.09), ls='--', label=r'$D_G(z=0.09)$')
plt.legend()
plt.xlabel(r'Maximum $L$-bin')#, size=15)
plt.ylabel(r'$D_G$')#, size=15)
plt.show()

plt.plot(z, cosmo.D_z_norm(z), label=r'$\Lambda$CDM')
plt.plot(z, cosmo.D_z_norm(z,gamma0=0.85), label=r'$\gamma_0=0.85$')
plt.plot(z, cosmo.D_z_norm(z,gamma0=0.55, gammaa=0.1), label=r'$\gamma_a=0.1$')
plt.errorbar(0.09, GetDg(clkg,clgg_half,clkg_slash_binned,clgg_slash_binned), yerr=dgs_err[-1], fmt='o')
# plt.xlim([0,0.2])
plt.legend()
plt.ylabel(r'$D_G$')#, size=15)
plt.xlabel(r'$z$')#, size=15)
"""
plt.figure(figsize=(5.5,5.5))
# # Plot D_G vs L_max
for i,zbin in enumerate(zbins):
	DNDZ = dNdzInterpolation(z, nz, bins=[zbin[0],zbin[1]], sigma_zph=0.015, z_min=0, z_max=1)
	zmed = DNDZ.z_med_bin(0)
	plt.subplot(1,1,i+1)
	plt.title(r'$%.2f < z < %.2f$'%(zbin[0],zbin[1]))
	plt.fill_between(lb[1:], np.asarray(dgs_sims_uw).mean(1)+np.asarray(dgs_err_uw[zbin]), np.asarray(dgs_sims_uw).mean(1)-np.asarray(dgs_err_uw[zbin]),alpha=0.2, color=cnow, linewidth=0.0)#, label=r'$1\sigma$ from sims w/o weights')
	plt.plot(lb[1:], dgs_sims_uw[zbin][0], color=cnow, ls='-.', label='2MPZ Data (no weights)')
	plt.fill_between(lb[1:], np.asarray(dgs_sims).mean(1)+np.asarray(dgs_err[zbin]), dgs_sims[zbin]-np.asarray(dgs_err[zbin]),alpha=0.2, color=cw, linewidth=0.0)#, label=r'$1\sigma$ from sims w/o weights')
	plt.plot(lb[1:], dgs_sims[zbin], color=cw, label='2MPZ Data')
	plt.axhline(cosmo_nl.D_z_norm(zmed), ls='--', color='k', label=r'$D_G(z_{\rm med}=%.2f)$'%zmed)
	# plt.axvline(180./np.rad2deg(1./cosmo_nl.k_NL(zmed)/cosmo_nl.f_K(zmed)),ls=':', color='darkgrey', label=r'$\ell_{\rm NL}(z_{\rm med}=%.2f)$'%zmed)
	plt.axvline(70,ls=':', color='darkgrey', label=r'$\ell_{\rm NL}(z_{\rm med}=%.2f)$'%zmed)
	# plt.axvline(cosmo_nl.k_NL(zmed)*cosmo_nl.f_K(zmed),ls=':', color='darkgrey', label=r'$\ell_{\rm NL}(z_{\rm med}=%.2f)$'%zmed)
	plt.legend(loc='upper right')
	plt.ylim(0.7,2.2)
	plt.xlabel(r'$\ell_{\rm max}$')#, size=15)
	# plt.xlabel(r'Maximum $L$-bin')#, size=15)
	# if i == 0: 
	plt.ylabel(r'$D_G$')#, size=15)
plt.tight_layout()
if fsky:
	plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_single_weighted_cooler_fsky.pdf')
else:
	plt.savefig('plots/D_G_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside256_single_weighted_cooler_sims.pdf')
# plt.show()
plt.close()



#~~~~~~~~~~~~~~~~~
count = 1
for i in xrange(4):
	for j in xrange(i+1,6):
		plt.subplot(3,5,count)
		if i==0 and j == 5:
			sns.distplot(1.1*dgs_sims[i,:]-1.1*dgs_sims[j,:], norm_hist=1, hist_kws={'histtype':'step', 'linewidth':2},label='(%d,%d)'%(lb[i],lb[j]), color='r')
		else:
			sns.distplot(1.1*dgs_sims[i,:]-1.1*dgs_sims[j,:], norm_hist=1, hist_kws={'histtype':'step', 'linewidth':2},label='(%d,%d)'%(lb[i],lb[j]))
		plt.axvline(dgs_data[zbin][i] - dgs_data[zbin][j])
		count += 1
		plt.legend()
		plt.xticks([])
		plt.yticks([])
		plt.axvline(ls='--',color='grey')




