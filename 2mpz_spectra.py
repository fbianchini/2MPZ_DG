import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import camb
from camb import model, initialpower
from scipy import interpolate
from scipy import integrate
from scipy import constants
from astropy import constants as const
from astropy import units as u

from numba import jit

def fit_dNdz(z, alpha=2.21, beta=1.43, z0=0.053):
	return z**alpha * np.exp(-(z/z0)**beta)

def GetdNdzNorm(alpha=2.21, beta=1.43, z0=0.053, nz=100):
	from scipy.integrate import simps
	zeta = np.linspace(0.,2., nz)
	n_z  = fit_dNdz(zeta, alpha=alpha, beta=beta, z0=z0)
	norm = simps(n_z, zeta)
	return zeta, n_z/norm

def integrand(zprime,z):
    return  (1.-com_dist(z)/com_dist(zprime)) * dndz(zprime)

def magnification_integral(z, **cosmo): # !!! WITHOUT THE (\alpha - 1) FACTOR !!!
    z_ = np.linspace(z, 10, 1000)
    a  = integrate.simps(integrand(z_, z), x=z_)
    return (3.*(cosmo['omegab']+cosmo['omegac'])*cosmo['H0']**2. * (u.km).to(u.Mpc)) /(2*const.c.value * hubble_z(z,**cosmo)) * (1.+z) * com_dist(z)*(u.Mpc).to(u.m) * a

def w_kappa_cmb(z,**cosmo):
    return (3.*omega_m*cosmo['H0']**2 * (u.km).to(u.Mpc))/(2.*const.c.to('Mpc/s').value * hubble_z(z,**cosmo)) * com_dist(z) * (1.+z) * ((chi_rec-com_dist(z))/chi_rec)

# KG Integrands

def kg_integrand(z,ell):
    return (hubble_z(z,**cosmo)* (u.km).to(u.Mpc))/(com_dist(z)**2 * const.c.to('Mpc/s').value) * w_kappa_cmb(z,**cosmo) * dndz(z) * PK.P(z, ell/(com_dist(z)*(cosmo['H0']/100.)))
    # return (hubble_z(z,**cosmo)* (u.km).to(u.Mpc))/(com_dist(z)**2 * const.c.to('Mpc/s').value) * w_kappa_cmb(z,**cosmo) * b_andrea(z) * dndz(z) * pkz_nl(ell/(com_dist(z)*(cosmo['H0']/100.)),z)

def kmu_integrand(z,ell):
    return (hubble_z(z,**cosmo)* (u.km).to(u.Mpc))/(com_dist(z)**2 * const.c.to('Mpc/s').value) * w_kappa_cmb(z,**cosmo) * magnification_integral(z,**cosmo) * PK.P(z, ell/(com_dist(z)*(cosmo['H0']/100.)))

# GG Integrands

def gg_integrand(z,ell):
    return (hubble_z(z,**cosmo)* (u.km).to(u.Mpc))/(com_dist(z)**2 * const.c.to('Mpc/s').value) * ((dndz(z))**2) * PK.P(z, ell/(com_dist(z)*(cosmo['H0']/100.)))
    # return (hubble_z(z,**cosmo)* (u.km).to(u.Mpc))/(com_dist(z)**2 * const.c.to('Mpc/s').value) * ((b_andrea(z)*dndz(z))**2) * pkz_nl(ell/(com_dist(z)*(cosmo['H0']/100.)),z)

def gmu_integrand(z,ell): # CROSS TERM WITHOUT A FACTOR 2
    return (hubble_z(z,**cosmo)* (u.km).to(u.Mpc))/(com_dist(z)**2 * const.c.to('Mpc/s').value) * dndz(z) * magnification_integral(z,**cosmo) * PK.P(z, ell/(com_dist(z)*(cosmo['H0']/100.)))
    # return (hubble_z(z,**cosmo)* (u.km).to(u.Mpc))/(com_dist(z)**2 * const.c.to('Mpc/s').value) * b_andrea(z)*dndz(z) * magnification_integral(z,**cosmo) * pkz_nl(ell/(com_dist(z)*(cosmo['H0']/100.)),z)

def mumu_integrand(z,ell):
    return (hubble_z(z,**cosmo)* (u.km).to(u.Mpc))/(com_dist(z)**2 * const.c.to('Mpc/s').value) * (magnification_integral(z,**cosmo))**2 * PK.P(z, ell/(com_dist(z)*(cosmo['H0']/100.)))

#====================================================
# Cosmological Functions

def hubble_z(z,**cosmo): # [km/sMpc] 
    return cosmo['H0']*(omega_m*(1.+z)**3. + (1.-omega_m))**0.5

def crit_rho(z,**cosmo): # [kg/m^3]
    return 3.*(hubble_z(z,**cosmo)*(u.km).to(u.Mpc))**2/(8.*np.pi*const.G.value)

def comoving_dist(z,**cosmo): # Mpc
    integrand_com_dist = lambda x: 1./hubble_z(x,**cosmo)
    return integrate.quad(integrand_com_dist,0.,z)[0] * const.c.to('km/s').value

# Vectorizing Comoving Dist, etc functions
comoving_dist_vect       = np.vectorize(comoving_dist)
# dndz_vect                = np.vectorize(dndz)
kg_integrand_vect        = np.vectorize(kg_integrand)
kmu_integrand_vect       = np.vectorize(kmu_integrand)
gg_integrand_vect        = np.vectorize(gg_integrand)
gmu_integrand_vect       = np.vectorize(gmu_integrand)
mumu_integrand_vect      = np.vectorize(mumu_integrand)


z, n_z = GetdNdzNorm(nz=1000)
dndz = interpolate.interp1d(z, n_z, bounds_error=False, fill_value=0.)

nz   = 100 #number of steps to use for the radial/redshift integration
kmax = 10  #kmax to use

cosmo   = {'H0':67.94,
         'omegab':0.0480,
         'omegac':0.24225,#0.255,
         'omegak':0.,
         'omegav':0.683,         
         'omegan':0.,
         'TCMB':2.7255,
         'yhe':0.24,
         'Num_Nu_massless':3.046,
         'Num_Nu_massive':0.,
         'scalar_index':0.9624,
         'reion__redshift':11.42,
         'reion__optical_depth':0.0943,
         'scalar_index':0.9582,
         'scalar_amp':2.2107e-9,
         'scalar_running':0,
         'reion__use_optical_depth':1,
         'DoLensing':0
         }
omega_m = (cosmo['omegab'] + cosmo['omegac'])
chi_rec = comoving_dist(1090,**cosmo)

z_spline = np.linspace(0,120,10000)
com_dist = interpolate.InterpolatedUnivariateSpline(z_spline,comoving_dist_vect(z_spline,**cosmo),k=1)

#First set up parameters as usual
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(ns=0.965)

#For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
#so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
results = camb.get_background(pars)
chistar = results.conformal_time(0)- model.tau_maxvis.value
chis    = np.linspace(0,chistar,nz)
zs      = results.redshift_at_comoving_radial_distance(chis)


#Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
#Here for lensing we want the power spectrum of the Weyl potential.
PK = camb.get_matter_power_interpolator(pars, nonlinear=True, 
	hubble_units=True, k_hunit=True, kmax=kmax,
    var1=model.Transfer_cdm,var2=model.Transfer_cdm, zmax=zs[-1])

kg_spline   = []
kmu_spline  = []
gg_spline   = []
gmu_spline  = []
mumu_spline = []



zmin = 0.001
zmax = 1

delta_z = 0.01
N_step = (zmax-zmin)/delta_z
print N_step
zeta = np.linspace(zmin,zmax,N_step)

# NEW =============================
# plt.plot(zeta, kg_integrand_vect(zeta,100), lw = 2, label = r'$\ell=100$')
# plt.plot(zeta, kg_integrand_vect(zeta,200), lw = 2, label = r'$\ell=200$')
# plt.plot(zeta, kg_integrand_vect(zeta,400), lw = 2, label = r'$\ell=400$')
# plt.xlabel(r'$z$')
# plt.ylabel(r'$\frac{\text{d}C_{\ell}^{\kappa g}}{\text{d}z}$')
# plt.legend()
# plt.show()
# NEW =============================

l = np.arange(2,300)
kg = np.zeros(l.shape)

# Logspace in ell and linspace in z    
# @jit(no_python=True)
def getKG(l):
    for i, ell in enumerate(l):
        print ell
        
        kg_integrand_spline_points      = kg_integrand_vect(zeta,ell)
        kg_integrand_spline_points      = np.insert(kg_integrand_spline_points,0,0.)
        # kmu_integrand_spline_points     = kmu_integrand_vect(zeta,ell)
        # kmu_integrand_spline_points     = np.insert(kmu_integrand_spline_points,0,0.)
        # gg_integrand_spline_points      = gg_integrand_vect(zeta,ell)
        # gg_integrand_spline_points      = np.insert(gg_integrand_spline_points,0,0.)
        # gmu_integrand_spline_points     = gmu_integrand_vect(zeta,ell)
        # gmu_integrand_spline_points     = np.insert(gmu_integrand_spline_points,0,0.)
        # mumu_integrand_spline_points    = mumu_integrand_vect(zeta,ell)
        # mumu_integrand_spline_points    = np.insert(mumu_integrand_spline_points,0,0.)

        zeta_new = np.insert(zeta,0,0.)

        kg[i] = integrate.simps(kg_integrand_spline_points,x=zeta_new)
        # kmu_spline.append(integrate.simps(kmu_integrand_spline_points,x=zeta_new))
        # gg_spline.append(integrate.simps(gg_integrand_spline_points,x=zeta_new))
        # gmu_spline.append(integrate.simps(gmu_integrand_spline_points,x=zeta_new))
        # mumu_spline.append(integrate.simps(mumu_integrand_spline_points,x=zeta_new))

    kg = kg/(cosmo['H0']/100.)**3
    return kg

def getKG(l):
    for i, ell in enumerate(l):
        print ell
        
        kg_integrand_spline_points      = kg_integrand_vect(zeta,ell)
        kg_integrand_spline_points      = np.insert(kg_integrand_spline_points,0,0.)
        zeta_new = np.insert(zeta,0,0.)
        kg[i] = integrate.simps(kg_integrand_spline_points,x=zeta_new)
    return kg

# kmu_spline     = np.asarray(kmu_spline)/(cosmo['H0']/100.)**3
# gg_spline      = np.asarray(gg_spline)/(cosmo['H0']/100.)**3
# gmu_spline     = np.asarray(gmu_spline)/(cosmo['H0']/100.)**3
# mumu_spline    = np.asarray(mumu_spline)/(cosmo['H0']/100.)**3

# kg     = interpolate.InterpolatedUnivariateSpline(l_log,kg_spline,k=2)
# kmu    = interpolate.InterpolatedUnivariateSpline(l_log,kmu_spline,k=2)
# gg     = interpolate.InterpolatedUnivariateSpline(l_log,gg_spline,k=2)
# gmu    = interpolate.InterpolatedUnivariateSpline(l_log,gmu_spline,k=2)
# mumu   = interpolate.InterpolatedUnivariateSpline(l_log,mumu_spline,k=2)

# np.savetxt('2MPZ_clkg.dat', np.c_[l, kg])
kg = getKG(l)

plt.plot(l, kg)
plt.show()


