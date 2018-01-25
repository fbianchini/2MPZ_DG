import numpy as np
from scipy.stats import norm,chi2
import matplotlib.pyplot as plt
from numpy import linalg as la
from matplotlib import rc, rcParams
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator
import healpy as hp
# import seaborn as sns
import sys,pickle,gzip

from TWOmpz_utils import Load2MPZ, GetdNdzNorm

import curvspec.master as cs
import hpyroutines.utils as hpy

from IPython import embed 
from tqdm import tqdm

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

def LoadPlanckMask(fitsfile, nside, do_plots=False):
	print("...using Planck CMB Lensing mask (= SMICA mask + stuff)...")
	print("...degrading Planck mask to N_side = %d ..." %nside)

	planck_mask = hp.read_map(fitsfile, verbose=False)
	planck_mask = hp.ud_grade(planck_mask, nside, pess=True)
	planck_mask[planck_mask < 1.] = 0. # Pessimistc cut, retains only M(n) = 1 pixels

	return planck_mask

try:
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	myid, nproc = comm.Get_rank(), comm.Get_size()
except ImportError:
	myid, nproc = 0, 1


# STuff 
z_bins = ['0_24']
bins_edges = {'0_24': (0.0,0.24)}
Nsim        = 100 
nside       = 256
do_plots    = False 
delta_ell   = 10#20
lmin        = 10
lmax        = 250
K_S_min     = 12.0
K_S_max     = 14.0


bmin = 0
bmax = None

# Paths 
twoMPZ_mask = 'fits/2MPZ_mask_tot_256.fits' # N_side = 256
twoMPZ_cat  = 'fits/results16_23_12_14_73.fits'
fits_planck_mask = '../Data/mask_convergence_planck_2015_512.fits'
path_maps   = '/Volumes/LACIE_SHARE/Data/PlanckDR2/Lensing/sims/obs_klms/'
output_path = 'outputsMC/deltal_'+str(delta_ell)+'/'

# Spectra 
cov_kg  = {}
mean_kg = {}
sims_kg = {}

try: 
	# Cycle over z-bin
	for z_bin in z_bins:
		lb, mean_kg[z_bin] = np.loadtxt(output_path+'kg_spectra_mean_2MPZ_planck2015_z'+z_bin+'.dat', unpack=True)
		cov_kg[z_bin]      = np.loadtxt(output_path+'kg_spectra_cov_2MPZ_planck2015_z'+z_bin+'.dat')
		sims_kg[z_bin]     = np.loadtxt(output_path+'kg_spectra_sims_2MPZ_planck2015_z'+z_bin+'.dat')
	if myid == 0: print '...Mean spectra & covariance matrices found...'
	found = True
except:
	# pass
	if myid == 0: print '...Files not found...\n...Starting evaluation...'
	found = False

	print("...loading 2MPZ catalogue...")
	twompz = Load2MPZ(twoMPZ_cat, K_S_min=K_S_min, K_S_max=K_S_max)
	print("...done...")

	# Masks
	mask_planck_2015 = LoadPlanckMask(fits_planck_mask, nside, do_plots=False)
	mask_2MPZ        = hp.read_map(twoMPZ_mask, verbose=False)
	mask             = mask_planck_2015 * mask_2MPZ

	# Estimator class
	kg = cs.Master(mask, lmin=lmin, lmax=lmax, delta_ell=delta_ell, MASTER=True, pixwin=True)

	for z_bin in z_bins:
		if myid == 0: print '\tz-bin: ', z_bin
		
		# Load WISExSCOS contrast density map 
		cat_tmp = twompz[(twompz.ZPHOTO >= bins_edges[z_bin][0]) & (twompz.ZPHOTO < bins_edges[z_bin][1])]
		counts  = hpy.GetCountsMap(cat_tmp.RA, cat_tmp.DEC, nside, coord='G', rad=True)
		delta   = hpy.Counts2Delta(counts, mask=mask)

		dim = (Nsim) / nproc
		if nproc > 1:
			if myid < (Nsim) % nproc:
				dim += 1

		kg_spectra_sims = np.zeros((dim, kg.lb.size))

		# Cycle over 100 Planck Lensing simulations 2015
		k = 0
		for n in tqdm(xrange(Nsim)):
		# for n in range(myid, Nsim, nproc): # Cycle over simulated alms
			kappa_sim_lm = hp.read_alm(path_maps + 'sim_%04d_klm.fits' %n)
			kappa_sim    = hp.alm2map(kappa_sim_lm, nside=nside, pixwin=True)
			cl_kg        = kg.get_spectra(kappa_sim, map2=delta)
			kg_spectra_sims[k, :] = cl_kg
			k += 1
		assert (k == dim)

		if nproc > 1:
			kg_spectra_sims_tot = comm.gather(kg_spectra_sims, root=0)
			if myid == 0:
				kg_spectra_sims_tot = np.vstack((kg_sims for kg_sims in kg_spectra_sims_tot)) 
		else:
			kg_spectra_sims_tot = kg_spectra_sims
		
		sims_kg[z_bin] = kg_spectra_sims_tot

		if myid == 0:
			# Sims
			np.savetxt(output_path+'kg_spectra_sims_2MPZ_planck2015_z'+z_bin+'.dat', kg_spectra_sims_tot, header = 'Each row is C_{\ell}^{kg} between 2MPZ map and 100 Planck simulations 2015')  
			
			# Mean (as a null-test)
			mean_kg[z_bin] = np.mean(kg_spectra_sims_tot, axis = 0)

			np.savetxt(output_path+'kg_spectra_mean_2MPZ_planck2015_z'+z_bin+'.dat', np.c_[kg.lb, mean_kg[z_bin]], header = '<C_{\ell}^{kg}>_MC averaged over MC simulations')

			# Covariance
			cov_kg[z_bin]  = np.cov(kg_spectra_sims_tot.T)
			
			np.savetxt(output_path+'kg_spectra_cov_2MPZ_planck2015_z'+z_bin+'.dat', cov_kg[z_bin], header = 'Cov^{kg}_ll from MC simulations')
		
		# comm.Barrier()

mock = pickle.load(gzip.open('spectra/sims/null_test_mock_gals_dl20_lmin_10_lmax250_KS_min_0.0_KS_max_13.9_nside256_zmin_0.0_zmax0.24_nsims100.pkl.gz','rb'))

bmin = 0 
bmax = 5
if myid == 0:
	plt.title('Null Tests')
	plt.axhline(ls='--', color='k')
	lb  = lb[bmin:bmax+1]
	PTE = {}
	for z_bin in z_bins:
		mean_kg[z_bin] = mean_kg[z_bin][bmin:bmax+1]
		mean_mock = mock['sims'].mean(0)[bmin:bmax+1]
		# mean_kg[z_bin][3] += 2*6e-9 
		cov_mock = np.cov(mock['sims'].T)[bmin:bmax+1,bmin:bmax+1]
		cov_kg[z_bin]  = cov_kg[z_bin][bmin:bmax+1,bmin:bmax+1]
		PTE[z_bin]     = (1. - chi2.cdf(np.dot(mean_kg[z_bin], np.dot(la.inv(cov_kg[z_bin]/Nsim), mean_kg[z_bin])), len(mean_kg[z_bin])))*100.
		PTE_mock  = (1. - chi2.cdf(np.dot(mean_mock, np.dot(la.inv(cov_mock/100.), mean_mock)), len(mean_mock)))*100.
	if found:
		plt.errorbar(lb-1, mean_kg['0_24']*1e7, yerr=(np.diag(cov_kg['0_24'])/Nsim)**.5*1e7,  color='tomato', fmt='o', elinewidth=2, capsize=0, ms=6, label=r'$\kappa^{\rm sim} \times \delta_g^{\rm 2MPZ}$  - PTE = %.1f' %PTE['0_24'])
		plt.errorbar(lb+1, mean_mock*1e7, yerr=(np.diag(cov_mock/100.))**.5*1e7,  color='royalblue', fmt='s', elinewidth=2, capsize=0, ms=6, label=r'$\kappa^{\rm Planck} \times \delta_g^{\rm sim}$ - PTE = %.1f' %PTE_mock)
		# plt.errorbar(lb-1.5, mean_kg['01_04'], yerr=(np.diag(cov_kg['01_04'])/Nsim)**.5,  color='orange', fmt='o', elinewidth=2, capsize=0, ms=5, label=r'$0.1\le z <0.4$ PTE = %.2f' %PTE['01_04'])
		# plt.errorbar(lb-0.5, mean_kg['01_02'], yerr=(np.diag(cov_kg['01_02'])/Nsim)**.5,  color='tomato', fmt='d', elinewidth=2, capsize=0, ms=5, label=r'$0.1\le z <0.2$ PTE = %.2f' %PTE['01_02'])
		# plt.errorbar(lb+0.5, mean_kg['02_03'], yerr=(np.diag(cov_kg['02_03'])/Nsim)**.5, color='royalblue', fmt='^', elinewidth=2, capsize=0, ms=5, label=r'$0.2 \le z < 0.3$ PTE = %.2f' %PTE['02_03'])
		# plt.errorbar(lb+1.5, mean_kg['03_04'], yerr=(np.diag(cov_kg['03_04'])/Nsim)**.5,  color='seagreen', fmt='s', elinewidth=2, capsize=0, ms=5, label=r'$0.3 \le z < 0.4$ PTE = %.2f' %PTE['03_04'])
	plt.xlabel(r'Multipole $\ell$')
	plt.ylabel(r'$C_{\ell}^{\kappa g} (\times 10^7)$')#' (\times 10^{-8})$')
	plt.legend(loc='lower right')
	# plt.xlim([100,800])
	# plt.ylim([-3,3])
	plt.tight_layout()
	plt.savefig('plots/null_test_planck_gals_sims.pdf', bbox_inches='tight')
	plt.show()
	# plt.savefig('figs/kg_null_test_WISExSCOS_planck_sims_2015_deltal'+str(delta_ell)+'_bmin'+str(bmin)+'.pdf', bbox_inches='tight')
	# plt.close()

embed()

# sys.exit()

# Chi^2 for null tests + Covariance Matrix plots
for z_bin in z_bins:
	print 'z_bin:', z_bin
	mean = mean_kg[z_bin][bmin:bmax]
	cov  = cov_kg[z_bin][bmin:bmax,bmin:bmax]
	plt.errorbar(lb[bmin:bmax], mean, yerr=np.diag(cov/Nsim)**.5)
	plt.axhline(ls='--')
	plt.show()


	fig, ax = plt.subplots()
	sns.heatmap(np.corrcoef(sims_kg[z_bin].T), annot=True)
	# cax = ax.imshow(np.corrcoef(cov), interpolation='nearest',cmap='RdBu', vmin=-1,vmax=1)
	# ax.set_title(r'Corr[$\hat{C}^{\kappa g}_{L}\hat{C}^{\kappa g}_{L^{\prime}}$] Planck Sims',size=27)
	# ax.set_xlabel('Multipole Bin',size=28)
	# ax.set_ylabel('Multipole Bin',size=28)
	# labels = '1234567'
	# for axis in [ax.xaxis, ax.yaxis]:
	#     axis.set(ticks=np.arange(0., len(labels)), ticklabels=labels)
	# cbar = fig.colorbar(cax)
	plt.show()
	print 'chi_square   =', np.dot(mean, np.dot(la.inv(cov/Nsim), mean))
	print 'p_value      =', 1.-chi2.cdf(np.dot(mean, np.dot(la.inv(cov/Nsim), mean)), mean.size)


