import numpy as np, os, sys

zbins = [(0.,0.24)]#,(0.0,0.08),(0.08,0.24)]
K_S_mins = [12,13]#[0,10,11,12,13]


for idz in xrange(len(zbins)):
	for K_S_min in K_S_mins:

		# print zbins[idz][0]

		cmd = 'pythonw TWOmpz_clkg.py -deltal 10 -K_S_min %s -nside 256 -zmin %s -zmax %s' %(K_S_min, zbins[idz][0], zbins[idz][1])

		print cmd

		os.system(cmd)
