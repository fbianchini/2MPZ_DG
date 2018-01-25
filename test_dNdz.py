
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

def lnlike(theta, cl, err_cl, cl_th, lbmax=None, kind='kg'):
	b = theta
	if kind == 'kg':
		model = b * cl_th
	elif kind == 'gg':
		model = b**2 * cl_th

	inv_sigma2 = 1.0/(err_cl[:lbmax]**2)
	# print np.sum((cl[:lbmax]-model[:lbmax])**2*inv_sigma2)
	return -0.5*(np.sum((cl[:lbmax]-model[:lbmax])**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
	b = theta
	if 0.0 < b < 10.0:
		return 0.0
	return -np.inf

def lnprob(theta, cl, err_cl, cl_th, lbmax=None, kind='kg'):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta, cl, err_cl, cl_th, lbmax=lbmax, kind=kind)


# Params
delta_ell = 10
lmin      = 10
lmax      = 250
K_S_min   = 0.
K_S_max   = 13.9
# zmin      = 0.08
# zmax      = 0.4
zbins     = [(0.,0.24)]#, (0.,0.08), (0.08,0.4)]
bias      = 1.
lbmax     = 5
nside     = 256
nsims     = 200

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
	file_clkg = 'spectra/2MPZ_Planck2015_clkg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split.dat' 
	file_clgg = 'spectra/2MPZ_clgg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split.dat' 
	# file_clkg = 'spectra/2MPZ_Planck2015_clkg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split_spec.dat' 
	# file_clgg = 'spectra/2MPZ_clgg_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zbin[0])+'_zmax'+str(zbin[1])+'_split_spec.dat' 
	lb, clgg[zbin], err_clgg[zbin] = np.loadtxt(file_clgg, unpack=1)
	lb, clkg[zbin], err_clkg[zbin] = np.loadtxt(file_clkg, unpack=1)
	
	# fname_gg  = 'clgg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zbins[0][0])+'_zmax'+str(zbins[0][1])+'_nsims'+str(nsims)+'.pkl.gz'
	# fname_kg  = 'clkg_sims_dl'+str(delta_ell)+'_lmin_'+str(lmin)+'_lmax'+str(lmax)+'_KS_min_'+str(K_S_min)+'_KS_max_'+str(K_S_max)+'_nside'+str(nside)+'_zmin_'+str(zbins[0][0])+'_zmax'+str(zbins[0][1])+'_nsims'+str(nsims)+'.pkl.gz'
	# clkg_sims = pickle.load(gzip.open('spectra/sims/'+fname_kg,'rb'))#, protocol=2)
	# clgg_sims = pickle.load(gzip.open('spectra/sims/'+fname_gg,'rb'))#, protocol=2)
	# bias_gg   = np.loadtxt('bias_estimator_gg.dat')#clgg_sims['sims'].mean(0)
	# bias_kg   = clkg_sims['sims'].mean(0)


# Get theoretical quantities
cosmo_nl = Cosmo(Giannantonio15Params, nonlinear=True)
cosmo = Cosmo(Giannantonio15Params)

# Create 4 different dN/dz
# 1. Spec-z
# 2. Photo-z of the spec-z
# 3. Convolved #3
# 4. Convolved #1

nz1, z1, _ = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label='Spec-z')
z1 = 0.5 * (z1[1:]+z1[:-1])
nz2, z2, _ = hist(twompz.ZPHOTO,'knuth', normed=1, histtype='step', label='Photo-z')
# nz2, z2, _ = hist(twompz.ZPHOTO[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step', label='Photo-z')
z2 = 0.5 * (z2[1:]+z2[:-1])
# nz3, z3, _ = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step',)# label='Convolved Photo-z')
# z3 = 0.5 * (z3[1:]+z3[:-1])
# nz4, z4, _ = hist(twompz.ZSPEC[twompz.ZSPEC>0.],'knuth', normed=1, histtype='step',)# label='Convolved Spec-z')
# z4 = 0.5 * (z4[1:]+z4[:-1])
# plt.tight_layout()
# plt.savefig('plots/many_dndz.pdf')
plt.close()

# embed()
# exit()

binner = cs.Binner(lmin=lmin, lmax=lmax, delta_ell=delta_ell)

clkg_th_1 = {}
clgg_th_1 = {}
clkg_th_binned_1 = {}
clgg_th_binned_1 = {}
clkg_th_2 = {}
clgg_th_2 = {}
clkg_th_binned_2 = {}
clgg_th_binned_2 = {}
clkg_th_3 = {}
clgg_th_3 = {}
clkg_th_binned_3 = {}
clgg_th_binned_3 = {}
clkg_th_4 = {}
clgg_th_4 = {}
clkg_th_binned_4 = {}
clgg_th_binned_4 = {}

clkg_slash_binned_1 = {}
clgg_slash_binned_1 = {}
clkg_slash_binned_2 = {}
clgg_slash_binned_2 = {}
clkg_slash_binned_3 = {}
clgg_slash_binned_3 = {}
clkg_slash_binned_4 = {}
clgg_slash_binned_4 = {}

for zbin in zbins:
	clkg_th_1[zbin] = GetCrossSpectrum(cosmo_nl, z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=0)
	clgg_th_1[zbin] = GetAutoSpectrum(cosmo_nl,  z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=0)
	clkg_th_binned_1[zbin] = binner.bin_spectra(clkg_th_1[zbin])
	clgg_th_binned_1[zbin] = binner.bin_spectra(clgg_th_1[zbin])
	clkg_th_2[zbin] = GetCrossSpectrum(cosmo_nl, z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=0)
	clgg_th_2[zbin] = GetAutoSpectrum(cosmo_nl,  z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=0)
	clkg_th_binned_2[zbin] = binner.bin_spectra(clkg_th_2[zbin])
	clgg_th_binned_2[zbin] = binner.bin_spectra(clgg_th_2[zbin])
	clkg_th_3[zbin] = GetCrossSpectrum(cosmo_nl, z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.07, compute_at_z0=0)
	clgg_th_3[zbin] = GetAutoSpectrum(cosmo_nl,  z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.07, compute_at_z0=0)
	clkg_th_binned_3[zbin] = binner.bin_spectra(clkg_th_3[zbin])
	clgg_th_binned_3[zbin] = binner.bin_spectra(clgg_th_3[zbin])
	clkg_th_4[zbin] = GetCrossSpectrum(cosmo_nl, z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=0)
	clgg_th_4[zbin] = GetAutoSpectrum(cosmo_nl,  z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=0)
	clkg_th_binned_4[zbin] = binner.bin_spectra(clkg_th_4[zbin])
	clgg_th_binned_4[zbin] = binner.bin_spectra(clgg_th_4[zbin])

	clkg_slash_binned_1[zbin] = binner.bin_spectra(GetCrossSpectrum(cosmo_nl, z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=1))
	clgg_slash_binned_1[zbin] = binner.bin_spectra(GetAutoSpectrum(cosmo_nl,  z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=1))
	clkg_slash_binned_2[zbin] = binner.bin_spectra(GetCrossSpectrum(cosmo_nl, z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=1))
	clgg_slash_binned_2[zbin] = binner.bin_spectra(GetAutoSpectrum(cosmo_nl,  z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=1))
	clkg_slash_binned_3[zbin] = binner.bin_spectra(GetCrossSpectrum(cosmo_nl, z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.07, compute_at_z0=1))
	clgg_slash_binned_3[zbin] = binner.bin_spectra(GetAutoSpectrum(cosmo_nl,  z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.07, compute_at_z0=1))
	clkg_slash_binned_4[zbin] = binner.bin_spectra(GetCrossSpectrum(cosmo_nl, z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=1))
	clgg_slash_binned_4[zbin] = binner.bin_spectra(GetAutoSpectrum(cosmo_nl,  z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=1))


ndim, nwalkers = 1, 100
pos = [1. + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

for zbin in zbins:
	# KG ~~~~~~~~~~~~~~~~~~~~~
	sampler_kg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(clkg[zbin], err_clkg[zbin], clkg_th_binned_1[zbin], lbmax, 'kg'))
	sampler_kg.run_mcmc(pos, 500)
	samples_kg = sampler_kg.chain[:, 100:, :].reshape((-1, 1))
	b_kg_1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_kg, [16, 50, 84],axis=0)))

	sampler_kg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(clkg[zbin], err_clkg[zbin], clkg_th_binned_2[zbin], lbmax, 'kg'))
	sampler_kg.run_mcmc(pos, 500)
	samples_kg = sampler_kg.chain[:, 100:, :].reshape((-1, 1))
	b_kg_2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_kg, [16, 50, 84],axis=0)))

	sampler_kg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(clkg[zbin], err_clkg[zbin], clkg_th_binned_3[zbin], lbmax, 'kg'))
	sampler_kg.run_mcmc(pos, 500)
	samples_kg = sampler_kg.chain[:, 100:, :].reshape((-1, 1))
	b_kg_3 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_kg, [16, 50, 84],axis=0)))

	sampler_kg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(clkg[zbin], err_clkg[zbin], clkg_th_binned_4[zbin], lbmax, 'kg'))
	sampler_kg.run_mcmc(pos, 500)
	samples_kg = sampler_kg.chain[:, 100:, :].reshape((-1, 1))
	b_kg_4 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_kg, [16, 50, 84],axis=0)))

	# GG ~~~~~~~~~~~~~~~~~~~~~
	sampler_gg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(clgg[zbin], err_clgg[zbin], clgg_th_binned_1[zbin], 3, 'gg'))
	sampler_gg.run_mcmc(pos, 500)
	samples_gg = sampler_gg.chain[:, 100:, :].reshape((-1, 1))
	b_gg_1 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_gg, [16, 50, 84],axis=0)))

	sampler_gg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(clgg[zbin], err_clgg[zbin], clgg_th_binned_2[zbin], 3, 'gg'))
	sampler_gg.run_mcmc(pos, 500)
	samples_gg = sampler_gg.chain[:, 100:, :].reshape((-1, 1))
	b_gg_2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_gg, [16, 50, 84],axis=0)))

	sampler_gg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(clgg[zbin], err_clgg[zbin], clgg_th_binned_3[zbin], 3, 'gg'))
	sampler_gg.run_mcmc(pos, 500)
	samples_gg = sampler_gg.chain[:, 100:, :].reshape((-1, 1))
	b_gg_3 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_gg, [16, 50, 84],axis=0)))

	sampler_gg = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(clgg[zbin], err_clgg[zbin], clgg_th_binned_4[zbin], 3, 'gg'))
	sampler_gg.run_mcmc(pos, 500)
	samples_gg = sampler_gg.chain[:, 100:, :].reshape((-1, 1))
	b_gg_4 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_gg, [16, 50, 84],axis=0)))


	fig = plt.figure(figsize=(10,5))
	ax = fig.add_subplot(1,2,1) 
	plt.suptitle(r'2MPZ - $%.2f < z < %.2f$' %(zbin[0],zbin[1]), size=20)

	clkg_th_1_tmp = GetCrossSpectrum(cosmo_nl, z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=0)
	clgg_th_1_tmp = GetAutoSpectrum(cosmo_nl,  z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=0)
	clkg_th_2_tmp = GetCrossSpectrum(cosmo_nl, z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=0)
	clgg_th_2_tmp = GetAutoSpectrum(cosmo_nl,  z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.0, compute_at_z0=0)
	clkg_th_3_tmp = GetCrossSpectrum(cosmo_nl, z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.07, compute_at_z0=0)
	clgg_th_3_tmp = GetAutoSpectrum(cosmo_nl,  z2, nz2, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.07, compute_at_z0=0)
	clkg_th_4_tmp = GetCrossSpectrum(cosmo_nl, z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=0)
	clgg_th_4_tmp = GetAutoSpectrum(cosmo_nl,  z1, nz1, [zbin[0],zbin[1]], b=1, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=0)


	lab = r'$K_S < %.1f$' %(K_S_max)
	ax.plot(b_kg_1[0][0]*clkg_th_1_tmp, color='darkgrey', ls='-', label=r'spec-z $b=%.2f$' %(b_kg_1[0][0]))
	ax.plot(b_kg_2[0][0]*clkg_th_2_tmp, color='darkgrey', ls='--', label=r'phot-z $b=%.2f$' %(b_kg_2[0][0]))
	ax.plot(b_kg_3[0][0]*clkg_th_3_tmp, color='darkgrey', ls=':', label=r'conv phot-z $b=%.2f$' %(b_kg_3[0][0]))
	ax.plot(b_kg_4[0][0]*clkg_th_4_tmp, color='darkgrey', ls='-.', label=r'conv spec-z $b=%.2f$' %(b_kg_4[0][0]))
	ax.errorbar(lb, clkg[zbin], yerr=err_clkg[zbin], fmt='o', capsize=0, ms=5, label=lab)
	ax.set_ylabel(r'$C_{\ell}^{\kappa g}$')
	ax.set_xlabel(r'Multipole $\ell$')
	ax.legend(loc='best')#lower left')
	ax.axhline(ls='--', color='grey')
	ax.set_xlim([10, lb[lbmax]+5])
	ax.set_yscale('linear')#,nonposy='clip')
	ax.set_xscale('linear')
	# ax.set_ylim([5e-8, 1e-5])
	# ax.set_ylim([-1e-7, 4e-6])
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


	ax = fig.add_subplot(1,2,2) 
	# plt.title(r'$%.2f < z < %.2f$' %(zbin[0],zbin[1]))

	l = np.arange(0,500)

	lab = r'$K_S < %.1f$' %(K_S_max)
	ax.plot(b_gg_1[0][0]**2*clgg_th_1_tmp, color='darkgrey', ls='-', label=r'spec-z $b=%.2f$' %(b_gg_1[0][0]))
	ax.plot(b_gg_2[0][0]**2*clgg_th_2_tmp, color='darkgrey', ls='--', label=r'phot-z $b=%.2f$' %(b_gg_2[0][0]))
	ax.plot(b_gg_3[0][0]**2*clgg_th_3_tmp, color='darkgrey', ls=':', label=r'conv phot-z $b=%.2f$' %(b_gg_3[0][0]))
	ax.plot(b_gg_4[0][0]**2*clgg_th_4_tmp, color='darkgrey', ls='-.', label=r'conv spec-z $b=%.2f$' %(b_gg_4[0][0]))
	ax.errorbar(lb, clgg[zbin], yerr=err_clgg[zbin], fmt='o', capsize=0, ms=5, label=lab)
	ax.set_ylabel(r'$C_{\ell}^{gg}$')
	ax.set_xlabel(r'Multipole $\ell$')
	# ax.legend(loc='lower left')
	ax.legend(loc='best')#lower left')
	ax.axhline(ls='--', color='grey')
	ax.set_xlim([10, lb[lbmax]+5])
	ax.set_yscale('linear',nonposy='clip')
	ax.set_xscale('linear')
	# ax.set_ylim([1e-5, 1e-3])
	# ax.set_ylim([1e-5, 8e-4])
	ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	# plt.savefig('plots/2MPZ_Planck_clkg_dl'+str(args.delta_ell)+'_lmin_'+str(args.lmin)+'_lmax'+str(args.lmax)+'_KS_min_'+str(args.K_S_min)+'_KS_max_'+str(args.K_S_max)+'_nside'+str(args.nside)+'_zmin_'+str(args.zmin)+'_zmax'+str(args.zmax)+'_split.pdf', bbox_inches='tight')
	# plt.show()
	# plt.close()

	embed()

z = np.linspace(0,1)
DNDZ_1 = dNdzInterpolation(z1, nz1, bins=[0.,0.24], sigma_zph=0.0, z_min=0, z_max=1)
DNDZ_2 = dNdzInterpolation(z2, nz2, bins=[0.,0.24], sigma_zph=0.0, z_min=0, z_max=1)
DNDZ_3 = dNdzInterpolation(z2, nz2, bins=[0.,0.24], sigma_zph=0.07, z_min=0, z_max=1)
DNDZ_4 = dNdzInterpolation(z1, nz1, bins=[0.,0.24], sigma_zph=0.015, z_min=0, z_max=1)
plt.plot(z, DNDZ_1.raw_dndz_bin(z,0), label=r'Spec-z')
plt.plot(z, DNDZ_2.raw_dndz_bin(z,0), label=r'Phot-z')
plt.plot(z, DNDZ_3.raw_dndz_bin(z,0), label=r'Conv Phot-z')
plt.plot(z, DNDZ_4.raw_dndz_bin(z,0), label=r'Conv Spec-z')
plt.xlabel(r'$z$')
# plt.xlabel(r'$z_{\rm ph}$')
plt.ylabel(r'$dN/dz$')
plt.xlim(0,0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.subplots_adjust(left=0.2)
plt.savefig('plots/many_dndz2.pdf', bboxes_inches='tight')
plt.show()


lbmax = 6
lbmin = 0
print "#1 => D_G = %.3f" %GetDg(clkg[zbin],clgg[zbin], clkg_slash_binned_1[zbin], clgg_slash_binned_1[zbin], err_kg=err_clkg[zbin], err_gg=err_clgg[zbin],lbmax=lbmax, lbmin=lbmin)
print "#2 => D_G = %.3f" %GetDg(clkg[zbin],clgg[zbin], clkg_slash_binned_2[zbin], clgg_slash_binned_2[zbin], err_kg=err_clkg[zbin], err_gg=err_clgg[zbin],lbmax=lbmax, lbmin=lbmin)
print "#3 => D_G = %.3f" %GetDg(clkg[zbin],clgg[zbin], clkg_slash_binned_3[zbin], clgg_slash_binned_3[zbin], err_kg=err_clkg[zbin], err_gg=err_clgg[zbin],lbmax=lbmax, lbmin=lbmin)
print "#4 => D_G = %.3f" %GetDg(clkg[zbin],clgg[zbin], clkg_slash_binned_4[zbin], clgg_slash_binned_4[zbin], err_kg=err_clkg[zbin], err_gg=err_clgg[zbin],lbmax=lbmax, lbmin=lbmin)
