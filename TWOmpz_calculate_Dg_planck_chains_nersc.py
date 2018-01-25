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

n = int(sys.argv[1])

assert(n > 0)

# Params
delta_ell = 20
lmin      = 10
lmax      = 250
K_S_min   = 0.
K_S_max   = 13.9
zbins     = [(0.,0.24)]#, (0.,0.08), (0.08,0.4)]
bias      = 1.24
nsims     = 200
ncosmo    = 100 
fsky      = 0

# Planck chains 
planck_chains_file = 'base_planck_lowl_lowLike_highL/base/planck_lowl_lowLike_highL/base_planck_lowl_lowLike_highL_4.txt'
planck_chains = np.loadtxt(planck_chains_file)

# Load Cat 
print("...loading 2MPZ catalogue...")
twoMPZ_cat = 'results16_23_12_14_73.fits'
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

# embed()

# exit()

cw = '#E9724C'
cnow = '#255F85'

z_ = np.logspace(-4,3,1000)
cosmo_ = Cosmo(Giannantonio15Params)

d0 = cosmo_.D_z_norm(z_)   
s8 = cosmo_.sigma_Rz(8./cosmo_.h)  
om = cosmo_.omegam


def D_z_norm(z, omegam, gamma=0.55):
	if np.isscalar(z) or (np.size(z) == 1):
		def func(x, omegam, gamma): 
			return f_z(x, omegam=omegam, gamma=gamma)/(1+x)
		return np.exp( -integrate.quad( func, 0, z, args=(omegam,gamma,))[0])
	else:
		return np.asarray([ D_z_norm(tz, omegam, gamma=gamma) for tz in z ])

def f_z(z, omegam, gamma=0.55): # [unitless]
	return (omegam*(1+z)**3/E_z(z, omegam)**2)**gamma

def E_z(z, omegam):
	return np.sqrt(omegam*(1+z)**3 + (1-omegam))


cnt = 0

facs = []
dgs_cosmo = []
samps_cosmo = []
samps_resc = []
s8_values = []
om_values = []

clkg_cosmo = []
clgg_cosmo = []
Dz_cosmo = []

for i in xrange(500*(n-1),500*n): #np.random.choice(planck_chains.shape[0], size=5000, replace=0):
	ombh2  = planck_chains[i,2]
	omch2  = planck_chains[i,3]
	H0     = planck_chains[i,44]
	s8_tmp = planck_chains[i,41]
	om_tmp = (ombh2+omch2)/(H0/100.)**2

	s8_values.append(s8_tmp)
	om_values.append(om_tmp)

	# samples_copy = samples.copy()

	facs.append((s8_tmp*om_tmp*(H0/100.)**2)/(s8*om*cosmo_.h**2))

	pars = {}
	pars['ombh2'] = planck_chains[i,2]
	pars['omch2'] = planck_chains[i,3]
	# pars['tau']   = #planck_chains[i,5]
	pars['ns']    = planck_chains[i,6]
	pars['As']    = 1e-10*(np.exp(planck_chains[i,7]))
	pars['H0']    = planck_chains[i,44]

	cosmo_tmp = Cosmo(pars, nonlinear=True) # DON'T FORGET TO USE NL CORRECTIONS!!!
	ndim, nwalkers = 1, 100
	pos = [1. + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

	clkg_slash_tmp = GetCrossSpectrum(cosmo_tmp, z, nz, [zbin[0],zbin[1]], b=bias, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
	clgg_slash_tmp = GetAutoSpectrum(cosmo_tmp,  z, nz, [zbin[0],zbin[1]], b=bias, alpha=1., lmax=500, sigma_zph=0.015, compute_at_z0=True)
	clkg_slash_binned_tmp = binner.bin_spectra(clkg_slash_tmp)
	clgg_slash_binned_tmp = binner.bin_spectra(clgg_slash_tmp)

	clkg_cosmo.append(clkg_slash_tmp)
	clgg_cosmo.append(clgg_slash_tmp)

	dgs_data_tmp = GetDg(clkg[zbin],clgg[zbin],clkg_slash_binned_tmp,clgg_slash_binned_tmp,lbmax=4)#, err_kg=err_clkg[zbin], err_gg=err_clgg[zbin]) 
	dgs_cosmo.append(dgs_data_tmp)

	Dz_cosmo.append(cosmo_tmp.D_z_norm(zmed))

	sampler_tmp = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(dgs_data_tmp, dgs_err[zbin][bmax], cosmo_tmp.D_z_norm(zmed)))
	sampler_tmp.run_mcmc(pos, 400)
	samps_cosmo.append(sampler_tmp.chain[:, 100:, :].reshape((-1, 1)))
	# A = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0)))

	# print cnt + 1
	# cnt += 1

	del pars, cosmo_tmp, sampler_tmp

clkg_cosmo = np.asarray(clkg_cosmo)
clgg_cosmo = np.asarray(clgg_cosmo)

samps_cosmo = np.asarray(samps_cosmo)
samples_cosmo = np.concatenate([samps_cosmo[i] for i in xrange(samps_cosmo.shape[0])])

pickle.dump(samps_cosmo, gzip.open('A_D_samps_'+str(n)+'.pkl.gz','wb'))

A_cosmo = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples_cosmo, [16, 50, 84],axis=0)))

