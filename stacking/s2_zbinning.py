import os, gzip, glob, pickle, sys, numpy as np
#import matplotlib; matplotlib.use('Agg')

K_S_min = 0.
K_S_max = 14.

make_zbins = 1
if make_zbins:

	folder = 'data/CONVERGENCE/'
	files = glob.glob('%s/2MPZ*' %(folder))

	lf = open('logs/K_S_binning.txt', 'w');lf.close()

	# zarr = [(0., 0.12), (0.12, 0.24), (0.24, 0.6)] #3 bins in z
	for f in files:
		fname = f.split('/')[-1]
		if f.find('.pkl.gz')>-1:continue
		lf = open('logs/K_S_binning.txt', 'a');
		lf.writelines('%s\n' %(f))
		lf.close()
		print f
		fdic = pickle.load(gzip.open(f))
		
		K_S = np.asarray( fdic['K'] )

		inds = np.where( (K_S > K_S_min) & (K_S <= K_S_max) )[0]

		lf = open('logs/K_S_binning.txt', 'a');
		logline = '\t\tK_S:(%s,%s): %s objects\n' %(K_S_min,K_S_max,len(inds))
		lf.writelines(logline)
		lf.close()
		print logline

		newfdic = {}
		newfdic['RA'] = np.asarray(fdic['RA'])[inds]#.tolist()
		newfdic['DEC'] = np.asarray(fdic['DEC'])[inds]#.tolist()
		newfdic['Z'] = np.asarray(fdic['Z'])[inds]#.tolist()
		newfdic['K'] = np.asarray(fdic['K'])[inds]#.tolist()
		
		totfiles = len(inds)

		cutouts = np.asarray(fdic['cutouts'])[inds]

		means = np.mean(cutouts, axis=(1,2), keepdims=1)
		cutouts = cutouts - means

		newfdic['totfiles'] = totfiles
		newfdic['summed'] = np.sum(cutouts, axis = 0)			

		#newfdic['cutouts'] = np.asarray(fdic['cutouts'])[inds].tolist()
		# newfdic['stacked'] = np.mean(np.asarray(fdic['cutouts'])[inds], axis = 0)

		newfolder = '%s/K_S_%s_%s' %(folder, K_S_min, K_S_max)
		os.system('mkdir %s' %(newfolder))
		newf = '%s/%s_stacked' %(newfolder, fname)
		pickle.dump(newfdic, gzip.open(newf, 'wb'), protocol = 2)

		del newfdic

		# fname = f.split('/')[-1]
		# if f.find('.pkl.gz')>-1:continue
		# lf = open('logs/K_S_binning.txt', 'a');
		# lf.writelines('%s\n' %(f))
		# lf.close()
		# print f
		# fdic = pickle.load(gzip.open(f))
		# redshifts = np.asarray( fdic['Z'] )
		# for z1z2 in zarr:

		# 	z1,z2 = z1z2
		# 	inds = np.where( (redshifts>z1) & (redshifts<=z2) )[0]

		# 	lf = open('logs/K_S_binning.txt', 'a');
		# 	logline = '\t\tz:(%s,%s): %s objects\n' %(z1,z2,len(inds))
		# 	lf.writelines(logline)
		# 	lf.close()
		# 	print logline

		# 	newfdic = {}
		# 	newfdic['RA'] = np.asarray(fdic['RA'])[inds]#.tolist()
		# 	newfdic['DEC'] = np.asarray(fdic['DEC'])[inds]#.tolist()
		# 	newfdic['Z'] = np.asarray(fdic['Z'])[inds]#.tolist()
		# 	#newfdic['cutouts'] = np.asarray(fdic['cutouts'])[inds].tolist()
		# 	newfdic['stacked'] = np.mean(np.asarray(fdic['cutouts'])[inds], axis = 0)

		# 	newfolder = '%s/z_%s_%s' %(folder, z1, z2)
		# 	os.system('mkdir %s' %(newfolder))
		# 	newf = '%s/%s_stacked' %(newfolder, fname)
		# 	pickle.dump(newfdic, gzip.open(newf, 'wb'), protocol = 2)

		# 	del newfdic

		#cmd = 'mv %s %s/archive/%s' %(f, folder, fname)
		#os.system(cmd)
		#sys.exit()

	sys.exit()

####