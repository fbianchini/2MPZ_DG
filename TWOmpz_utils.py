import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import healpy as hp
import hpyroutines.utils as hpy
from astropy.io import fits
from astropy.visualization import hist
from scipy.optimize import curve_fit

def Load2MPZ(fits_file, K_S_min=0., K_S_max=20.):
	"""
	Returns dictionary with 2MPZ catalog info.
	"""
	hdu = fits.open(fits_file)
	cat = hdu[1].data
	cat = cat[(cat['KCORR'] > K_S_min) & (cat['KCORR'] < K_S_max)]
	return cat

def GetNstarMap(ra, dec, nstar, nside, galactic=True, nest=False, rad=False):
	"""
	Creates a star density map at resolution nside given arrays containing RA, DEC, and STARDENSITY
	from 2MPZ catalog.
	"""
	ra  = ra[nstar>0.]
	dec = dec[nstar>0.]
	# dec = dec[nstar>0.]

	# Get nstar pixels
	pix = hpy.Sky2Hpx(ra, dec, nside, nest=nest, rad=rad)

	# Create nstar map
	dens = np.zeros(hp.nside2npix(nside))
	dens[pix] = nstar[nstar>0.]

	return dens

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
	plt.show()
	# plt.savefig(fname, bbox_inches=True)
	pass

def fit_dNdz(z, alpha=2.21, beta=1.43, z0=0.053):
	return z**alpha * np.exp(-(z/z0)**beta)

def GetdNdzNorm(alpha=2.21, beta=1.43, z0=0.053, nz=100):
	from scipy.integrate import simps
	zeta = np.linspace(0.,2., nz)
	n_z  = fit_dNdz(zeta, alpha=alpha, beta=beta, z0=z0)
	norm = simps(n_z, zeta)
	return zeta, n_z/norm

def FitTheHist(z, nbins=20, func=fit_dNdz, normed=1):
	from scipy.optimize import curve_fit, leastsq
	from astropy.visualization import hist

	n, bins, _ = hist(z, nbins, normed=normed, histtype='step')
	# plt.close()
	bin_centers = 0.5*(bins[1:] + bins[:-1])#bins[:-1] + 0.5 * (bins[1:] - bins[:-1])

	fitfunc  = lambda p, x: x**p[0] * np.exp( -(x/p[2])**p[1] )
	errfunc  = lambda p, x, y: (y - fitfunc(p, x))


	out   = leastsq( errfunc, [2.2,1.4,0.05], args=(bin_centers, n))
	c = out[0]

	print "Fit Coefficients:"
	print c[0],c[1],c[2]#,c[3]

	plt.plot(bin_centers, fitfunc(c, bin_centers))
	# plot(xdata, ydata)


	# popt, pcov = curve_fit(func, bin_centers, n)#p0 = [1., 1., 0.01])
	# plt.scatter(bin_centers, func(bin_centers, *popt))
	# print popt

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

def LoadPlanck(filemap, filemask, nside, do_plots=False, filt_plank_lmin=0, pess=True):
	print("...reading Planck map & mask...")
	planck_mask = hp.read_map(filemask, verbose=False)
	planck_map  = hp.read_map(filemap, verbose=False)
	print("...done...")
	print("...degrading Planck map & mask to N_side = %d ..." %nside)
	planck_mask = hp.ud_grade(planck_mask, nside, pess=pess)
	planck_mask[planck_mask < 1.] = 0. # Pessimistc cut, retains only M(n) = 1 pixels
	planck_map  = hp.ud_grade(planck_map, nside, pess=pess)
	if do_plots:
		planck_map = hp.ma(planck_map)
		planck_map.mask = np.logical_not(planck_mask)
		hp.mollview(planck_map, title='Planck 2015 No filtering')
		hp.graticule()
		# plt.tight_layout()
		plt.savefig('figs/planck.pdf')#, bbox_inches='tight')
		plt.close()
	if filt_plank_lmin != 0:
		planck_map_alm = hp.map2alm(planck_map)
		filt = np.ones(hp.Alm.getlmax(len(planck_map_alm)))
		filt[:filt_plank_lmin+1] = 0.
		planck_map_alm = hp.almxfl(planck_map_alm, filt)
		planck_map = hp.alm2map(planck_map_alm, nside)
	if do_plots:
		planck_map = hp.ma(planck_map)
		planck_map.mask = np.logical_not(planck_mask)
		hp.mollview(planck_map, title=r'Planck 2015 $\ell_{\rm min}^{\rm filt} = %d$'%filt_plank_lmin)
		hp.graticule()
		plt.savefig('figs/planck_filt.pdf')
		plt.close()
	print("...done...")

	return planck_map, planck_mask


Giannantonio15Params = {
    "ombh2"                 : 0.0222,  # baryon physical density at z=0
    "omch2"                 : 0.119,   # cold dark matter physical density at z=0
    "omk"                   : 0.,     # Omega_K curvature paramter
    "mnu"                   : 0.06,   # sum of neutrino masses [eV]
    "nnu"                   : 3.046,  # N_eff, # of effective relativistic dof
    "TCMB"                  : 2.725,  # temperature of the CMB in K at z=0
    "H0"                    : 67.8,   # Hubble's constant at z=0 [km/s/Mpc]
    "w"                     : -1.0,   # dark energy equation of state (fixed throughout cosmic history)
    "wa"                    : 0.,     # dark energy equation of state (fixed throughout cosmic history)
    "cs2"                   : 1.0,    # dark energy equation of state (fixed throughout cosmic history)
    "tau"                   : 0.0952,   # optical depth
    "deltazrei"             : None,   # z-width of reionization
    "YHe"                   : None,   # Helium mass fraction
    "As"                    : 2.21e-9,   # comoving curvature power at k=piveo_scalar
    "ns"                    : 0.961,   # scalar spectral index
    "nrun"                  : 0.,     # running of scalar spectral index
    "nrunrun"               : 0.,     # running of scalar spectral index
    "r"                     : 0.,     # tensor to scalar ratio at pivot scale
    "nt"                    : None,   # tensor spectral index
    "ntrun"                 : 0.,     # running of tensor spectral index
    "pivot_scalar"          : 0.05,   # pivot scale for scalar spectrum 
    "pivot_tensor"          : 0.05,   # pivot scale for tensor spectrum 
    "meffsterile"           : 0.,     # effective mass of sterile neutrinos
    "num_massive_neutrinos" : 1,      # number of massive neutrinos (ignored unless hierarchy == 'degenerate')
    "standard_neutrino_neff": 3.046,
    "gamma0"                : 0.55,   # growth rate index
    "gammaa"                : 0.,     # growth rate index (series expansion term)
    "neutrino_hierarchy": 'degenerate', # degenerate', 'normal', or 'inverted' (1 or 2 eigenstate approximation)
    }





