import os, gzip, glob, pickle, sys, numpy as np
import healpy as hp
#import matplotlib; matplotlib.use('Agg')

make_zbins = 0
rnd_30percent_stck = 0
flog = 'logs/CONVERGENCE/d.txt'#RND_30percent_stacking.txt'
frac = 0.3


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


if make_zbins:

	start, end = int(sys.argv[1]), int(sys.argv[2])

	which_bin = int(sys.argv[3])

	apply_cluster_mask = int(sys.argv[4])

	implement_redmapper_cuts = 0
	if implement_redmapper_cuts:
		file_cuts = 'data/wisescosmos_redmapper_cuts.txt'
		ra_cuts, dec_cuts, z_cuts = np.loadtxt(file_cuts,unpack=1)	

	#folder = 'data/60_60_am_box/'
	folder = 'data/CONVERGENCE'#/RANDOM/%d' %(id_sim)

	if not os.path.exists(folder):
		os.system('mkdir %s' %folder)

	files = glob.glob('%s/2MPZ_*' %(folder))[start:end]

	flog = 'logs/CONVERGENCE/z_binning_sigma_cuts.txt'
	lf = open(flog, 'w');lf.close()

	zarr = [(0.1, 0.178), (0.178, 0.246), (0.246, 0.4)] #3 bins in z
	
	zarr = zarr[which_bin]

	m1 = hp.read_map('data/mask_RM_clusters_only_01_0178_nside512.fits')
	m2 = hp.read_map('data/mask_RM_clusters_only_0178_0246_nside512.fits')
	m3 = hp.read_map('data/mask_RM_clusters_only_0246_04_nside512.fits')

	sdss_mask = hp.read_map('data/mask_sdss_nside512.fits')

	zarr_masks = {(0.1, 0.178):m1,
				(0.178, 0.246):m2,
				  (0.246, 0.4):m3}

	for f in files:#[0:1]:
		fname = f.split('/')[-1]

		lf = open(flog, 'a')
		lf.writelines('%s\n' %(f))
		lf.close()
		print f
		fdic = pickle.load(gzip.open(f))
		redshifts = np.asarray( fdic['Z'] )
		for z1z2 in [zarr]:

			z1,z2 = z1z2
			if apply_cluster_mask:
				newfolder = '%s/RM_clusters_only_512/woclusters/z_%s_%s' %(folder, z1, z2)
			else:
				newfolder = '%s/RM_clusters_only_512/wclusters/z_%s_%s' %(folder, z1, z2)
			os.system('mkdir %s/RM_clusters_only_512' %(folder))
			if apply_cluster_mask:
				os.system('mkdir %s/RM_clusters_only_512/woclusters' %(folder))
			else:
				os.system('mkdir %s/RM_clusters_only_512/wclusters' %(folder))
			os.system('mkdir %s' %(newfolder))
			newf = '%s/%s_stacked' %(newfolder, fname)

			if os.path.exists(newf): 

				logline = '\t\t\t%s already exists - skipping\n' %(newf)
			 	lf = open(flog, 'a')
				lf.writelines(logline)
				lf.close()
				print logline

				continue


			inds = np.where( (redshifts>z1) & (redshifts<=z2) )[0]
			if len(inds) == 0: continue	
                        
			ra = np.asarray(fdic['RA'])[inds]
                        dec = np.asarray(fdic['DEC'])[inds]
                        z = np.asarray(fdic['Z'])[inds]
                        cutouts = np.asarray(fdic['cutouts'])[inds]

			if apply_cluster_mask:
				inds = np.where(zarr_masks[(z1,z2)][Sky2Hpx(ra, dec, hp.npix2nside(len(zarr_masks[(z1,z2)])))] == 1)[0]	
			else:
				inds = np.where(sdss_mask[Sky2Hpx(ra, dec, hp.npix2nside(len(sdss_mask)))] == 1)[0]			

			if len(inds) == 0: continue

                        ra = ra[inds]
                        dec = dec[inds]
                        z = z[inds]
                        cutouts = cutouts[inds]

			logline = '\t\tz:(%s,%s): %s objects\n' %(z1,z2,len(inds))
		 	lf = open(flog, 'a')
			lf.writelines(logline)
			lf.close()
			print logline

			
			newfdic = {}
			totfiles = len(inds)

			if implement_redmapper_cuts:
				#newfdic['RA'] = np.asarray(fdic['RA'])[inds]#.tolist()
				#newfdic['DEC'] = np.asarray(fdic['DEC'])[inds]#.tolist()
				#newfdic['Z'] = np.asarray(fdic['Z'])[inds]#.tolist()
				#newfdic['cutouts'] = np.asarray(fdic['cutouts'])[inds].tolist()
				#newfdic['stacked'] = np.mean(np.asarray(fdic['cutouts'])[inds], axis = 0)
			
				indeces = []
				radec_cuts = ['(%.4f,%4f)'%(x1,x2) for x1,x2 in zip(ra_cuts,dec_cuts)]

				for i in xrange(len(ra)):
					mystr = '(%.4f,%4f)'%(ra[i],dec[i])	
					if mystr in radec_cuts:
						indeces.append(i)					

				indeces = np.asarray(indeces)

				ra = ra[indeces]
				dec = dec[indeces]
				z = z[indeces]
				cutouts = cutouts[indeces]
				totfiles = len(indeces)

                       	means = np.mean(cutouts, axis=(1,2), keepdims=1)
			cutouts = cutouts - means

                        newfdic['RA'] = ra
                        newfdic['DEC'] = dec
                        newfdic['Z'] = z
			newfdic['totfiles'] = totfiles
			newfdic['summed'] = np.sum(cutouts, axis = 0)			

			pickle.dump(newfdic, gzip.open(newf, 'wb'), protocol = 2)

			del newfdic

		#cmd = 'mv %s %s/archive/%s' %(f, folder, fname)
		#os.system(cmd)
		#sys.exit()

	sys.exit()
