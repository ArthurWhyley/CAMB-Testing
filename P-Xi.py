import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower, correlations
import scipy

#Define model variables
H0 = 67.5
ombh2 = 0.022
omch2 = 0.122
OM = ombh2 + omch2
c = 3e5

#CAMB Matter Power Spectrum----------------------------------------------------

#For calculating large-scale structure and lensing results yourself, get a power spectrum
#interpolation object. In this example we calculate the CMB lensing potential power
#spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
#calling PK(z, k) will then get power spectrum at any k and redshift z in range.

nz = 100 #number of steps to use for the radial/redshift integration
kmax=10  #kmax to use
#First set up parameters as usual
pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
pars.InitPower.set_params(ns=0.965)

#For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
#so get background results to find chistar, set up a range in chi, and calculate corresponding redshifts
results= camb.get_background(pars)
chistar = results.conformal_time(0)- results.tau_maxvis #Just follows eq to get horizon distance
chis = np.linspace(0,chistar,nz)
zs=results.redshift_at_comoving_radial_distance(chis)
#Calculate array of delta_chi, and drop first and last points where things go singular
dchis = (chis[2:]-chis[:-2])/2
chis = chis[1:-1]
zs = zs[1:-1]

#Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
#Here for lensing we want the power spectrum of the Weyl potential.
PD = camb.get_matter_power_interpolator(pars, nonlinear=True, 
    hubble_units=True, k_hunit=False, kmax=kmax,
    var1="delta_tot",var2="delta_tot", zmax=zs[-1])

#Have a look at interpolated power spectrum results for a range of redshifts
#Expect linear potentials to decay a bit when Lambda becomes important, and change from non-linear growth
plt.figure(figsize=(8,5))
k=np.exp(np.log(10)*np.linspace(-4,2,200))
zplot = [0, 0.5, 1, 4 ,20]
for z in zplot:
    plt.loglog(k, PD.P(z,k))
plt.xlim([1e-4,kmax])
plt.xlabel('k Mpc')
plt.ylabel('$P_m(k) (h^{-3} Mpc^{3}$)')
plt.legend(['z=%s'%z for z in zplot]);
plt.show()

#Transform to Kappa Power Spectrum---------------------------------------------

#define a function for the galaxy distribution
def galaxy_dist(chi,length):
    z = results.redshift_at_comoving_radial_distance(chi)
    n = (z**2) / (2 * 0.5**3) * np.e**(-1*z/0.5) #From Huterer (2001)
    n = n / 15.15
    return(n)

n = np.zeros(len(chis))
for i in range(len(chis)):
    n[i] = galaxy_dist(chis[i],len(chis))
print(np.sum(n))

#define the weight function
def weight(z):
    chi = results.comoving_radial_distance(z)
    const = 1.5 * H0**2 * OM * (1+z) * chi / (c**2)
    chirange = np.arange(chi,chistar)
    integrand = galaxy_dist(chirange,len(chirange)) * (chirange - chi) / chirange
    return(const * np.sum(integrand))
  
W = np.zeros(len(zs))
for i in range(len(zs)):
    W[i] = weight(zs[i])

plt.plot(zs,W)
plt.show()

#define a function to calculate P_kappa
def P_kappa(l,zrange):
    integrand = np.zeros(len(zrange))
    for i in range(len(zrange)):
        r = results.comoving_radial_distance(zrange[i])
        integrand[i] = ((weight(zrange[i]))**2 / r**2 * PD.P(zrange[i],(l/r)))
    return(np.sum(integrand))

ls = np.arange(1,2501)
Pk = np.zeros(2500)
for i in range(len(Pk)):
    Pk[i] = P_kappa(ls[i], zs)
    
m = 2 * np.pi / ls
    
plt.loglog(m,Pk)
plt.xlabel("2pi / l")
plt.ylabel("P kappa")
plt.show()


#Fourier transform
def Fourier(f,l,theta,pm):
    const = 1 / (2 * np.pi)
    if pm == "plus":
        v = 0
    elif pm == "minus":
        v = 4
    else:
        print("Must specify plus or minus")
        return()
    integrand = f * l * scipy.special.jv(v, l*theta)
    return(const * np.sum(integrand))
    
arcmins = np.arange(1000) * 0.1
rads = arcmins / 60 * 2 * np.pi / 360
cors_p = np.zeros(len(rads))
cors_m = np.zeros(len(rads))

for i  in range(len(rads)):
    cors_p[i] = Fourier(Pk, ls, rads[i], "plus")
    
plt.plot(arcmins,cors_p)
plt.show()

for i  in range(len(rads)):
    cors_m[i] = Fourier(Pk, ls, rads[i], "minus")
    
plt.plot(arcmins,cors_m)
plt.show()