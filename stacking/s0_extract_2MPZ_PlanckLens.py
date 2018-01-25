import numpy as np, sys
import os
import random
import matplotlib.pyplot as plt
from IPython import embed
import cPickle as pickle
import gzip
import healpy as hp
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits

start, end = int(sys.argv[1]), int(sys.argv[2])
#smica = 0
which_map = 0# 0- CONVERGENCE; 1 - SMICA; 2 - LGMCA
which_map_dic = {0: 'CONVERGENCE', 1:'SMICA', 2:'LGMCA'}

plt.switch_backend('Agg')

def Load2MPZ(fits_file, K_S_min=0., K_S_max=20.):
	"""
	Returns dictionary with 2MPZ catalog info.
	"""
	hdu = fits.open(fits_file)
	cat = hdu[1].data
	cat = cat[(cat['KCORR'] > K_S_min) & (cat['KCORR'] < K_S_max)]
	return cat

def Sky2Hpx(sky1, sky2, nside, coord='G', nest=False, rad=False):
	"""
	Converts sky coordinates, i.e. (RA,DEC), to Healpix pixel at a given resolution nside.
	By default, it rotates from EQ -> GAL
	Parameters
	----------
	sky1 : array-like
		First coordinate, it can be RA, LON, ...
	sky2 : array-like
		Second coordinate, it can be DEC, LAT, ...
		
	nside : int
		Resolution of the Healpix pixelization scheme
	coord : str 'C','E' or 'G' [def]
		Coordinate system of the output. If coord='C', *no* rotation is applied
	nest : bool [def=False]
		If True, nested pixelation scheme is considered for the output pixels
	rad : bool [def=False]
		If True, input coordinates are expressed in radians, otherwise in degree
	Returns
	-------
	ipix : array-like
		Pixel indices corresponding to (sky1,sky2) coordinates
	"""
	sky1, sky2 = np.asarray(sky1), np.asarray(sky2)

	if rad == False: # deg -> rad
		theta = np.deg2rad(90.-sky2) 
		phi   = np.deg2rad(sky1) 
	else: 
		theta = np.pi/2. - sky2
		phi   = sky1 	     

	# Apply rotation if needed (default EQ -> GAL)
	r = hp.Rotator(coord=['C',coord], deg=False)
	theta, phi = r(theta, phi)

	npix = hp.nside2npix(nside)

	return hp.ang2pix(nside, theta, phi, nest=nest) # Converting galaxy coordinates -> pixel 

# Params 'n' stuff
'''
WISExSCOS_cat = '/data38/fbianchini/data/wiseScosPhotoz160708.csv.gz'#'/Volumes/LACIE_SHARE/Data/WISExSCOS/wiseScosPhotoz160708.csv.gz'
planck_alms  = '/data38/fbianchini/data/data_PlanckLens_DR2/dat_klm.fits'
planck_mask  = '/data38/fbianchini/data/data_PlanckLens_DR2/mask.fits'
cutout_dic_name = '/data38/fbianchini/data/WISExSCOS_PlanckLens_TEST'
'''

twoMPZ_cat      = 'data/results16_23_12_14_73.fits'#'/Volumes/LACIE_SHARE/Data/WISExSCOS/wiseScosPhotoz160708.csv.gz'
twoMPZ_mask     = 'data/mask_2MPZ_nside2048.fits'
planck_alms     = 'data/data_PlanckLens_DR2/dat_klm.fits'
planck_mask     = 'data/data_PlanckLens_DR2/mask.fits'
cutout_dic_name = 'data/%s/2MPZ_%s_%s' %(which_map_dic[which_map],start, end)

#20170813 - SMICA stuff
smica_map_file = 'data/SMICA/COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits'
smica_mask_file = 'data/SMICA/COM_Mask_CMB-confidence-Tmask-IQU-smica_1024_R2.02_full.fits'

#20170817 - LGMCA stuff
lgmca_map_file = 'data/LGMCA/WPR2_CMB_muK.fits'

nside = 2048
boxsize = 60#20 # arcmin
boxsize_mask = 30 #arcmin

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("...loading 2MPZ catalogue...")
cat = Load2MPZ(twoMPZ_cat)
print("...done...")

if which_map == 0:
	print("...loading PlanckLens alms...")
	kappa_lm = hp.read_alm(planck_alms)
	print("...done...")

	print("...converting PlanckLens alms to map...")
	kappa = hp.alm2map(kappa_lm, nside)
	print("...done...")

elif which_map ==1:
	print("...reading SMICA map now: %s..." %(smica_map_file))
	kappa = hp.read_map(smica_map_file, field=0, verbose=0) #not kappa but SMICA: just T now.
	print("...done...")
elif which_map ==2:
	print("...reading LGMCA map now: %s..." %(lgmca_map_file))
	kappa = hp.read_map(lgmca_map_file, field=0, verbose=0) #not kappa but LGMCA: just T now.
	print("...done...")

print("...loading PlanckLens mask...")
mask_planck = hp.read_map(planck_mask, verbose=0)
print("...done...")

print("...loading 2MPZ mask...")
mask_twompz = hp.read_map(twoMPZ_mask, verbose=0)
print("...done...")

mask = mask_twompz * mask_planck

if which_map>=1: #include more mask
	print("...loading SMICA mask and multiplying with lens mask...")
	smica_mask = hp.read_map(smica_mask_file, verbose=0)
	smica_mask = hp.ud_grade(smica_mask, nside) #convert to required nside
	mask = mask * smica_mask
	print("...done...")

twomassX = cat.TWOMASSX[start:end]
ra       = cat.RA[start:end]
dec      = cat.DEC[start:end]
ebv      = cat.EBV[start:end]
jcorr    = cat.JCORR[start:end]
hcorr    = cat.HCORR[start:end]
kcorr    = cat.KCORR[start:end]
bcalcorr = cat.BCALCORR[start:end]
rcalcorr = cat.RCALCORR[start:end]
icalcorr = cat.ICALCORR[start:end]
w1mCorr  = cat.W1MCORR[start:end]
w2mCorr  = cat.W2MCORR[start:end]
z        = cat.ZPHOTO[start:end]
zspec    = cat.ZPHOTO[start:end]

cuts = {}
cuts['cutouts'] = []
cuts['RA'] = []
cuts['DEC'] = []
cuts['Z'] = []
cuts['ZSPEC'] = []
cuts['twomassX'] = []
cuts['EBV'] = []
cuts['W1'] = []
cuts['W2'] = []
cuts['J'] = []
cuts['H'] = []
cuts['K'] = []
cuts['B'] = []
cuts['R'] = []
cuts['I'] = []

coord = SkyCoord(ra=ra, dec=dec, unit='deg').transform_to('galactic')
l = coord.l.value
b = coord.b.value

#logfile = open('logs/%s_%s.txt' %(start, end), 'w')
#def GetCutout(i):
datafolder = 'data/%s' %(which_map_dic[which_map])
logfolder = 'logs/%s' %(which_map_dic[which_map])

if not os.path.exists(datafolder):
	os.system('mkdir %s' %(datafolder))

if not os.path.exists(logfolder):
	os.system('mkdir %s' %(logfolder))

for i in range(len(ra)):
	logfile = open('logs/%s/%s_%s.txt' %(which_map_dic[which_map], start, end), 'a')

	ipix = Sky2Hpx(ra[i], dec[i], nside)
	ivec = hp.pix2vec(nside, ipix)
	disc = hp.query_disc(nside, ivec, np.deg2rad(boxsize_mask/60.))
	if mask[disc].all() == 1:
		logline = '%s-Good\n' %(i)
		cut = hp.gnomview(kappa, rot=[l[i],b[i]], xsize=boxsize, reso=1., return_projected_map=True)
		plt.close()
		cuts['cutouts'].append(cut)
		cuts['RA'].append(ra[i])
		cuts['DEC'].append(dec[i])
		cuts['Z'].append(z[i])
		cuts['ZSPEC'].append(zspec[i])
		cuts['twomassX'].append(twomassX[i])
		cuts['EBV'].append(ebv[i])
		cuts['W1'].append(w1mCorr[i])
		cuts['W2'].append(w2mCorr[i])	
		cuts['J'].append(jcorr[i])	
		cuts['H'].append(hcorr[i])	
		cuts['K'].append(kcorr[i])	
		cuts['B'].append(bcalcorr[i])	
		cuts['R'].append(rcalcorr[i])	
		cuts['I'].append(icalcorr[i])	
		#return [cut, ra[i], dec[i], z[i]]
	else:
		logline = '%s-Bad\n' %(i)
		pass#return [0, 0, 0, 0] #[np.nan, np.nan, np.nan, np.nan]

	logfile.writelines(logline)
	logfile.close()

print("...done...")
pickle.dump(cuts, gzip.open(cutout_dic_name+'.pkl.gz', 'w'), protocol=2)

