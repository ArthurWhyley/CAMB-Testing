import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower

lmax = 2500

#Get angular power spectrum for galaxy number counts and lensing
from camb.sources import GaussianSourceWindow, SplinedSourceWindow

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
pars.InitPower.set_params(As=2e-9, ns=0.965)
pars.set_for_lmax(lmax, lens_potential_accuracy=1)
#set Want_CMB to true if you also want CMB spectra or correlations
pars.Want_CMB = False 
#NonLinear_both or NonLinear_lens will use non-linear corrections
pars.NonLinear = model.NonLinear_both
#Set up W(z) window functions, later labelled W1, W2. Gaussian here.
pars.SourceWindows = [
    GaussianSourceWindow(redshift=0.17, source_type='counts', bias=1.2, sigma=0.04, dlog10Ndm=-0.2),
    GaussianSourceWindow(redshift=0.5, source_type='lensing', sigma=0.07)]

results = camb.get_results(pars)
cls = results.get_source_cls_dict()

#Note that P is CMB lensing, as a deflection angle power (i.e. PxP is [l(l+1)]^2C_l\phi\phi/2\pi)
#lensing window functions are for kappa (and counts for the fractional angular number density)
ls=  np.arange(2, lmax+1)
for spectrum in ['W1xW1','W2xW2','W1xW2',"PxW1", "PxW2"]:
    plt.loglog(ls, cls[spectrum][2:lmax+1], label=spectrum)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi$')
plt.legend();
plt.show()

# For number counts you can give a redshift-dependent bias (the underlying code supports general b(z,k))
# toy model for single-bin LSST/Vera Rubin [using numbers from 1705.02332]
z0=0.311
zs = np.arange(0, 10, 0.02)
W = np.exp(-zs/z0)*(zs/z0)**2/2/z0
bias = 1 + 0.84*zs
pars.SourceWindows = [SplinedSourceWindow(dlog10Ndm=0, z=zs, W=W, bias_z = bias)]
lmax=3000
pars.set_for_lmax(lmax, lens_potential_accuracy=5)
results = camb.get_results(pars)

#e.g. plot the cross-correlation with CMB lensing
cls = results.get_cmb_unlensed_scalar_array_dict()

nbar = 40/(np.pi/180/60)**2 # Poission noise
ls= np.arange(2,lmax+1)
Dnoise = 1/nbar*ls*(ls+1)/2/np.pi

correlation=cls["PxW1"][2:lmax+1]/np.sqrt(cls["PxP"][2:lmax+1]*(cls["W1xW1"][2:lmax+1]+Dnoise))
plt.plot(np.arange(2,lmax+1), correlation)
plt.xlabel(r'$L$')
plt.ylabel('correlation')
plt.xlim(2,lmax)
plt.title('CMB lensing - LSST correlation (single redshift bin)');

