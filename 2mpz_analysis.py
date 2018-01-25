import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import healpy as hp
from astropy.io import fits
from master import Master

from IPython import embed

def Sky2Hpx(sky1, sky2, nside, galactic=True, nest=False, rad=False):
	"""
	Converts sky coordinates, i.e. (RA,DEC), to Healpix pixel at a given resolution nside.
	By default, it rotates from EQ -> GAL
	"""
	if rad == False: # deg -> rad
		theta = np.deg2rad(90.-sky2) 
		phi   = np.deg2rad(sky1) 
	else: 
		theta = np.pi/2. - sky2
		phi   = sky1 	     

	# Apply rotation from EQ -> GAL if needed
	if galactic:
		r = hp.Rotator(coord=['C','G'], deg=False)
		theta, phi = r(theta, phi)

	# Converting galaxy coordinates -> pixel 
	npix = hp.nside2npix(nside)
	return hp.ang2pix(nside, theta, phi, nest=nest)

def GetCountsMap(ra, dec, nside, galactic=True, nest=False, rad=False, sqdeg=False):
	"""
	Creates an Healpix map with galaxy number counts at resolution nside given two arrays containing
	RA and DEC positions 
	"""
	# Get sources position (pixels)
	pix = Sky2Hpx(ra, dec, nside, galactic=galactic, nest=nest, rad=rad)

	# Create counts map
	counts_map = np.bincount(pix, minlength=hp.nside2npix(nside))*1.

	if sqdeg:
		counts_map *= hp.nside2pixarea(nside)

	return counts_map

def GetNstarMap(ra, dec, nstar, nside, galactic=True, nest=False, rad=False):
	"""
	Creates a star density map at resolution nside given arrays containing RA, DEC, and STARDENSITY
	from 2MPZ catalog.
	"""
	ra  = ra[nstar>0.]
	dec = dec[nstar>0.]
	dec = dec[nstar>0.]

	# Get nstar pixels
	pix = Sky2Hpx(ra, dec, nside, galactic=galactic, nest=nest, rad=rad)

	# Create nstar map
	dens = np.zeros(hp.nside2npix(nside))
	dens[pix] = nstar

	return dens

def Counts2Delta(counts, mask=None):
    """
    Converts a number counts Healpix map into a density contrast map.

    Note
    ----
    Use only binary mask.
    """
    counts = np.asarray(counts)
    
    if mask is not None:
        mask   = np.asarray(mask)
        counts = hp.ma(counts)
        counts.mask = np.logical_not(mask)
    
    mean  = np.mean(counts)
    delta = (counts - mean) / mean
    delta = delta.filled(counts.fill_value) # To avoid pixels with fill_value = 1e+20
    
    return delta

def GuessMask(ra, dec, nside, galactic=True, nest=False, rad=False):
	"""
	Creates an Healpix mask at resolution nside given two arrays containing RA and DEC positions 
	"""
	# Get sources positions
	pix = Sky2Hpx(ra, dec, nside, galactic=galactic, nest=nest, rad=rad)

	# Create Healpix map
	mask = np.zeros(hp.nside2npix(nside))
	mask[pix] = 1.

	return mask

def Load2MPZ(fits_file, K_S_min=0., K_S_max=20.):
	"""
	Returns dictionary with 2MPZ catalog info.
	"""
	hdu = fits.open(fits_file)
	cat = hdu[1].data
	cat = cat[(cat['KCORR'] > K_S_min) & (cat['KCORR'] < K_S_max)]
	return cat

def Get2MPZMask(A_K, nstar, nside=64, A_K_max=0.06, lognstar_max=3.5, maskcnt=None):
	"""
	Returns 2MPZ mask given K-band correction and star density maps with associated cuts.
	Optionally the counts based mask can be applied.
	"""
	assert (hp.npix2nside(A_K.size) == hp.npix2nside(nstar.size))
	if maskcnt is not None: assert (hp.npix2nside(A_K.size) == hp.npix2nside(maskcnt.size))

	mask = np.ones(A_K.size)
	mask[A_K > A_K_max] = 0.
	mask[nstar > lognstar_max] = 0.
	
	if maskcnt is not None: 
		mask *= maskcnt

	if nside != 64: 
		mask = hp.ud_grade(mask, nside, pess=True)

	return mask

def PlotTemplateMaps(A_K, nstar, logA_K_min=-2.85, logA_K_max=1.3, lognstar_min=2.4, lognstar_max=4., cmap=cm.copper):
	cmap.set_under("w") # sets background to white
	hp.mollview(np.log10(A_K), cmap=cmap, min=logA_K_min, max=logA_K_max, sub=(211), title=r'$\log_{10}A_K$')
	hp.graticule()
	hp.mollview(nstar, cmap=cmap, min=lognstar_min, max=lognstar_max, sub=(212), title=r'$\log_{10}n_{star}$')
	hp.graticule()
	pass

def PlotAKvsGalDensity(A_K, cat, fname='A_K_systamtic.pdf'):
	K_S = [(12.,12.5), (12.5,13.), (13.,13.5), (13.5,13.9)]

	ak       = np.logspace(np.log10(A_K.min()), np.log10(A_K.max()), 20)
	akbin    = (ak[:-1]+ak[1:])/2.
	ak_edges = zip(ak[:-1],ak[1:])

	for k_s  in K_S:
		cat_     = cat[(cat['KCORR'] > k_s[0]) & (cat['KCORR'] < k_s[1])]
		counts   = GetCountsMap(cat_['RA'], cat_['DEC'], hp.npix2nside(A_K.size), rad=True, sqdeg=True)
		cnt_mean = np.zeros(ak.size-1)
		cnt_std  = np.zeros(ak.size-1)
		for i, edges in enumerate(ak_edges):
			counts_ = counts[(A_K > edges[0]) & (A_K < edges[1])]
			# hp.mollview(counts_, title=r'%3.1f $< K_S <$ %3.1f' %(k_s[0],k_s[1]))
			# plt.show()
			cnt_mean[i] = np.mean(counts_)
			cnt_std[i]  = np.std(counts_)
		plt.plot(akbin, cnt_mean, label=r'%3.1f $< K_S <$ %3.1f' %(k_s[0],k_s[1]))
	plt.legend(loc='best')
	plt.xlim([3e-3,0.5])
	plt.xlabel(r'$A_K$', size=20)
	plt.ylabel(r'$n_{\Omega}$ deg$^2$', size=20)
	plt.xscale('log')
	plt.yscale('log')
	# plt.show()
	plt.savefig(fname, bbox_inches=True)
	pass

"""
This code performs analysis on 2MPZ catalog based on Alonso+15 (arXiv:1412.5151)
"""
# Params 'n' stuff
fits_planck_mask = 'fits/mask_convergence_planck_2015_512.fits'
fits_planck_map  = 'fits/convergence_planck_2015_512.fits'
fits_ebv = 'fits/lambda_sfd_ebv.fits'
fits_cat = 'fits/results16_23_12_14_73.fits'
nside    = 256
nsidelow = 64
K_S_min  = 12.
K_S_max  = 13.9

# Load Cat
twompz = Load2MPZ(fits_cat, K_S_min=K_S_min, K_S_max=K_S_max)

# Planck CMB lensing map 'n' mask
planck_mask = hp.read_map(fits_planck_mask, verbose=False)
planck_map  = hp.read_map(fits_planck_map, verbose=False)
planck_mask = hp.ud_grade(planck_mask, nside, pess=True)
planck_mask[planck_mask < 1.] = 0. # Pessimistc cut, retains only M(n) = 1 pixels
planck_map  = hp.ud_grade(planck_map, nside, pess=True)

# E(B-V) reddening map by Schlegel+ in gal coord at nside=512
ebv = hp.read_map(fits_ebv, verbose=False)
ebv = hp.ud_grade(ebv, nsidelow, pess=True)
A_K = ebv*0.367 # K-band correction for Galactic extinction
print("...Reddening map loaded...")

# Get nstar map
nstar = GetNstarMap(twompz['RA'], twompz['DEC'], twompz['STARDENSITY'], nsidelow, galactic=True, rad=True)

# Nice "systematics" plots
# PlotTemplateMaps(A_K, nstar, cmap=cm.copper)
# plt.show()
# plt.savefig('template.pdf', bbox_inches='tight')

embed()

exit()

# Get 2MPZ mask (A_K + nstar + counts) 
# !!! "torque rod gashes" still need to be removed !!!
mask = Get2MPZMask(A_K, nstar, nside=nside, maskcnt=GuessMask(twompz['RA'], twompz['DEC'], nsidelow, galactic=True, rad=True))
mask *= planck_mask 
print("Mask f_sky is %3.2f" %np.mean(mask))
# hp.mollview(mask, title='w/ cnt')
# plt.show()
hp.write_map('2MPZ_mask_tot_%d.fits' %nside, mask)

# Counts map
cnt = GetCountsMap(twompz['RA'], twompz['DEC'], nside, galactic=True, rad=True)
cnt_deg = cnt*hp.nside2pixarea(nside, degrees=True)
# hp.mollview(cnt_deg*mask)
# plt.show()
print("Total number of sources is ", np.sum(cnt*mask))

# Converting to overdensity
dlt = Counts2Delta(cnt, mask=mask)
hp.mollview(dlt, title='2MPZ overdensity')
hp.graticule()
# plt.show()
plt.savefig('2MPZ_delta.pdf')#, bbox_inches=True)
plt.close()
# PlotAKvsGalDensity(A_K, twompz)


dsf
# XC estimator
est = Master(lmin=10, lmax=100, delta_ell=10, mask=mask)
# estm = Master(lmin=10, lmax=250, delta_ell=20, mask=mask, fsky_approx=False)

kg, err_kg   = est.get_spectra(planck_map, map2=dlt, analytic_errors=True)
# kgm, err_kgm = estm.get_spectra(planck_map, map2=dlt, analytic_errors=True)

print("Detection at rougly %3.2f sigma") %(np.sum((kg/err_kg)**2)**.5)
# print np.sum((kgm/err_kgm)**2)**.5

l, clkg = np.loadtxt('2MPZ_clkg.dat', unpack=True)

plt.errorbar(est.ell_binned, kg, yerr=err_kg, fmt='o', capsize=0, label='2MPZ x Planck2015')
plt.plot(l, clkg, label=r'Theory $b=1$')
plt.plot(l, clkg*1.35, label=r'Theory $b=1.35$')
# plt.errorbar(estm.ell_binned+2, kgm, yerr=err_kgm, label='MASTER')
plt.legend()
plt.axhline(ls='--', color='grey')
plt.xlabel(r'$\ell$', size=20)
plt.ylabel(r'$C_{\ell}^{\kappa g}$', size=20)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlim([2., est.lmax+1])
# plt.xscale('log')
# plt.yscale('log')
# plt.show()
plt.savefig('2MPZ_clkg.pdf')#, bbox_inches=True)
plt.close()

def fit_dNdz(z, alpha=2.21, beta=1.43, z0=0.053):
	return z**alpha * np.exp(-(z/z0)**beta)

def GetdNdzNorm(alpha=2.21, beta=1.43, z0=0.053, nz=100):
	from scipy.integrate import simps
	zeta = np.linspace(0.,2., nz)
	n_z  = fit_dNdz(zeta, alpha=alpha, beta=beta, z0=z0)
	norm = simps(n_z, zeta)
	return zeta, n_z/norm

z, n_z = GetdNdzNorm(nz=1000)
plt.plot(z, n_z, label='Fit Alonso+15')
plt.hist(twompz['ZSPEC'][twompz['ZSPEC']>0.], bins=50, histtype='step', normed=True, label='Spec 2MPZ')
plt.hist(twompz['ZPHOTO'][twompz['ZPHOTO']>0.], bins=50, histtype='step', normed=True, label='Photo 2MPZ')
plt.xlim([0., 0.3])
plt.xlabel(r'Redshift $z$', size=20)
plt.ylabel(r'$dN/dz$', size=20)
plt.legend()
# plt.show()
plt.savefig('2MPZ_nz.pdf')#, bbox_inches=True)
plt.close()

