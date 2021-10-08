# RAiSERed module
# Ross Turner, 17 May 2020

# import packages
import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import time as ti
import os, warnings
from astropy import constants as const
from astropy import units as u
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy import wcs
from matplotlib import pyplot as plt
from matplotlib import cm, rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, NullFormatter, LogLocator
from numba import jit
from scipy.optimize import least_squares
from scipy.special import gamma, zeta


## Define global variables that can be adjusted to customise model output
# basic constants
year = 365.2422*24*3600 # average year in seconds
maverage = (0.6*const.m_p.value) # kg average particle mass
hubble = 0.7 # dimensionless Hubble parameter
OmegaM = 0.27 # fraction of matter in the flat universe
OmegaD = 0.73 # fraction of dark energy in the flat universe
freq_cmb = 5.879e10 # frequency of cosmic microwave background at z = 0
temp_cmb = 2.725 # temperature of cosmic microwave background at z = 0

c_speed = const.c.value # speed of light
e_charge = const.e.value # electron charge
k_B = const.k_B.value # Boltzmann constant
m_e = const.m_e.value # electron mass
mu0 = const.mu0.value # vacuum permeability
sigma_T = const.sigma_T.value # electron scattering cross-section

# model parameters that can be optimised for efficiency
nangles = 16 # number of angles to calculate expansion rate along (must be greater than 1)
betaRegions = 64 # set maximum number of beta regions
limTime = (year) # the FR-II limit must be used before this time
stepRatio = 1.01 # ratio to increase time/radius

# shocked gas and lobe parameters
chi = 2*np.pi/3.0 # lobe geometry parameter
shockAxisRatio = 0.5875 # exponent relating the cocoon axis ratio to the shocked gas axis ratio
shockRadius = 1.072 # fraction of the radius the shocked gas is greater than the lobe
gammaX = (5./3) # lorentz factor of external gas
gammaJ = (4./3) # lorentz factor of jet plasma
    
# set electron energy distribution constants
Lorentzmin = 780. # minimum Lorentz factor of injected electrons AT HOTSPOT for Cygnus A
Lorentzmax = 1e6 # effectively infinity

# density and temperature profiles
rCutoff = 0.01 # minimum radius to match profiles as a fraction of r200
betaMax = 2 # set critical value above which the cocoon expands balistically

# average and standard deviation of Vikhlinin model parameters
alphaAvg = 1.64 # corrected for removal of second core term
alphaStdev = 0.30
betaPrimeAvg = 0.56
betaPrimeStdev = 0.10
gammaPrimeAvg = 3
gammaPrimeStdev = 0
epsilonAvg = 3.23
epsilonStdev = 0 # 1.93; this parameter has little effect on profile
rCoreAvg = 0.087 # this is ratio of rc to r200
rCoreStdev = 0.028
rSlopeAvg = 0.73 # this is ratio of rs to r200
rSlopeStdev = 0 # 0.39; this parameter has little effect on profile

# temperature parameters
TmgConst = (-2.099)
TmgSlope = 0.6678
TmgError = 0.0727
# new temperature parameters assuming heating from AGN during expansion
TmgAvg = 7.00
TmgStdev = 0.28

# approximate halo to gas fraction conversion
# for halo masses between 10^12 and 10^15 and redshifts 0 < z < 5
halogasfracCONST1z0 = (-0.881768418)
halogasfracCONST1z1 = (-0.02832004)
halogasfracCONST2z0 = (-0.921393448)
halogasfracCONST2z1 = 0.00064515
halogasfracSLOPE = 0.053302276
# uncertainties, in dex
dhalogasfracz0 = 0.05172769
dhalogasfracz1 = (-0.00177947)
# correction to SAGE densities
SAGEdensitycorr = (-0.1)


## Define functions for run-time user output
def __join(*values):
    return ";".join(str(v) for v in values)
    
def __color_text(s, c, base=30):
    template = '\x1b[{0}m{1}\x1b[0m'
    t = __join(base+8, 2, __join(*c))
    return template.format(t, s)
    
class Colors:
    DogderBlue = (30, 144, 255)
    Green = (0,200,0)
    Orange = (255, 165, 0)


## Define main function to run RAiSE HD
def RAiSE_run(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass=None, rand_profile=False, betas=None, regions=None, rho0Value=None, temperature=None, active_age=10.14, jet_lorentz=5, equipartition=-1.5, spectral_index=0.7, gammaCValue=4./3, lorentz_min=Lorentzmin, brightness=True, angle=0., resolution='standard', seed=None, aj_star=0.23, crit_mach=1., jet_angle=0.58):
    
    # record start time of code
    start_time = ti.time()
    
    # function to test type of inputs and convert type where appropriate
    frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz, nenvirons = __test_inputs(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz)
    
    # download and pre-process particles from hydrodynamical simulation
    if not resolution == None:
        print(__color_text('Reading particle data from file.', Colors.Green))
        time, shock_time, major, minor, x1, x2, x3, tracer, vx3, volume, pressure, press_minor, alphaP_hyd, alphaP_henv, hotspot_ratio = __PLUTO_particles('RAiSE_particles.hdf5')
    # set seed for quasi-random profiles
    if not seed == None:
        __set_seed(seed)
    
    if not resolution == None:
        print(__color_text('Running RAiSE dynamics and emissivity.', Colors.Green))
    else:
        print(__color_text('Running RAiSE dynamics.', Colors.Green))

    for i in range(0, len(redshift)):
        for j in range(0, len(axis_ratio)):
            for k in range(0, len(jet_power)):
                for l in range(0, nenvirons):
                    for m in range(0, len(active_age)):
                        for n in range(0, len(equipartition)):
                            for o in range(0, len(jet_lorentz)):
                                # set correct data types for halo mass and core density
                                if isinstance(halo_mass, (list, np.ndarray)):
                                    new_halo_mass = halo_mass[l]
                                else:
                                    new_halo_mass = halo_mass
                                if isinstance(rho0Value, (list, np.ndarray)):
                                    new_rho0Value = rho0Value[l]
                                    new_temperature = temperature[l]
                                    new_betas = betas[l]
                                    new_regions = regions[l]
                                else:
                                    new_rho0Value = rho0Value
                                    new_temperature = temperature
                                    new_betas = betas
                                    new_regions = regions
                                    
                                # calculate dynamical evolution of lobe and shocked shell using RAiSE dynamics
                                lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda = __RAiSE_environment(redshift[i], axis_ratio[j], jet_power[k], source_age, halo_mass=new_halo_mass, rand_profile=rand_profile, rho0Value=new_rho0Value, regions=new_regions, betas=new_betas, temperature=new_temperature, active_age=active_age[m], jet_lorentz=jet_lorentz[o], gammaCValue=gammaCValue, aj_star=aj_star, crit_mach=crit_mach, jet_angle=jet_angle)
                                
                                # calculate synchrotron emission from lobe using particles and RAiSE model
                                if not resolution == None:
                                    location, luminosity, magnetic_field = __RAiSE_emissivity(frequency, redshift[i], time, shock_time, major, minor, x1, x2, x3, tracer, vx3, volume, pressure, press_minor, alphaP_hyd, alphaP_henv, hotspot_ratio, source_age, lobe_lengths, lobe_minor, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, active_age[m], equipartition[n], spectral_index, gammaCValue=gammaCValue, lorentz_min=lorentz_min, resolution=resolution)
                                
                                # create pandas dataframe for integrated emission
                                df = pd.DataFrame()
                                df['Time (yrs)'] = 10**np.asarray(source_age).astype(np.float_)
                                df['Size (kpc)'] = 2*lobe_lengths[0,:]/const.kpc.value
                                df['Pressure (Pa)'] = shock_pressures[0,:]
                                df['Axis Ratio'] = lobe_lengths[0,:]/lobe_lengths[-1,:]
                                if not resolution == None:
                                    for q in range(0, len(frequency)):
                                        df['B{:.2f} (T)'.format(frequency[q])] = magnetic_field[:,q]
                                        df['L{:.2f} (W/Hz)'.format(frequency[q])] = np.nansum(luminosity[:,:,q], axis=1)
                                    
                                # write data to file
                                if isinstance(rho0Value, (list, np.ndarray)):
                                    df.to_csv('LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), np.abs(np.log10(rho0Value[l])), jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i]), index=False)
                                elif isinstance(halo_mass, (list, np.ndarray)):
                                    df.to_csv('LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), halo_mass[l], jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i]), index=False)
                                else:
                                    raise Exception('Either the halo mass or full density profile must be provided as model inputs.')
                                
                                # calculate brightness per pixel across the source
                                if brightness == True and not resolution == None:
                                    x_values, y_values, brightness_list = __RAiSE_brightness_map(frequency, redshift[i], source_age, lobe_lengths, location, luminosity, angle, resolution=resolution)
                                    
                                    for p in range(0, len(source_age)):
                                        for q in range(0, len(frequency)):
                                            # create pandas dataframe for spatially resolved emission
                                            if isinstance(x_values[p][q], (list, np.ndarray)):
                                                df = pd.DataFrame(index=x_values[p][q]/const.kpc.value, columns=y_values[p][q]/const.kpc.value, data=brightness_list[p][q])
                                        
                                                # write surface brightness map to file
                                                if isinstance(rho0Value, (list, np.ndarray)):
                                                    df.to_csv('LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), np.abs(np.log10(rho0Value[l])), jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], frequency[q], source_age[p], resolution), header=True, index=True)
                                                elif isinstance(halo_mass, (list, np.ndarray)):
                                                    df.to_csv('LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), halo_mass[l], jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], frequency[q], source_age[p], resolution), header=True, index=True)
                                                else:
                                                    raise Exception('Either the halo mass or full density profile must be provided as model inputs.')
                                            else:
                                                if isinstance(rho0Value, (list, np.ndarray)):
                                                    warnings.warn('The following file was not created as no emission is present: LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), np.abs(np.log10(rho0Value[l])), jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], frequency[q], source_age[p], resolution), category=UserWarning)
                                                elif isinstance(halo_mass, (list, np.ndarray)):
                                                    warnings.warn('The following file was not created as no emission is present: LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), halo_mass[l], jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], frequency[q], source_age[p], resolution), category=UserWarning)
                                                else:
                                                    raise Exception('Either the halo mass or full density profile must be provided as model inputs.')
    
    # print total run time to screen
    print(__color_text('RAiSE completed running after {:.2f} seconds.'.format(ti.time() - start_time), Colors.Green))

    
# Define function to test type of inputs and convert type where appropriate
def __test_inputs(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz):

    # convert redshift, axis ratio and jet power to correct data types
    if not isinstance(frequency, (list, np.ndarray)):
        frequency = [frequency]
    for i in range(0, len(frequency)):
        if not isinstance(frequency[i], (int, float)) or not (5 < frequency[i] and frequency[i] < 20):
            raise Exception('Frequency must be provided as a float or list/array of floats in units of log10 Hertz.')
            
    if not isinstance(redshift, (list, np.ndarray)):
        redshift = [redshift]
    for i in range(0, len(redshift)):
        if not isinstance(redshift[i], (int, float)) or not (0 < redshift[i] and redshift[i] < 20):
            raise Exception('Redshift must be provided as a float or list/array of floats.')
            
    if not isinstance(axis_ratio, (list, np.ndarray)):
        axis_ratio = [axis_ratio]
    for i in range(0, len(axis_ratio)):
        if not isinstance(axis_ratio[i], (int, float)) or not (1 <= axis_ratio[i] and axis_ratio[i] < 20):
            raise Exception('Axis ratio must be provided as a float or list/array of floats and be greater than 1.')
            
    if not isinstance(jet_power, (list, np.ndarray)):
        jet_power = [jet_power]
    for i in range(0, len(jet_power)):
        if not isinstance(jet_power[i], (int, float)) or not (33 < jet_power[i] and jet_power[i] < 46):
            raise Exception('Jet power must be provided as a float or list/array of floats in units of log10 Watts.')
            
    if not isinstance(source_age, (list, np.ndarray)):
        source_age = [source_age]
    for i in range(0, len(source_age)):
        if not isinstance(source_age[i], (int, float)) or not (0 <= source_age[i] and source_age[i] <= 10.14):
            raise Exception('Source age must be provided as a float or list/array of floats in units of log10 years.')
    
    if not isinstance(active_age, (list, np.ndarray)):
        active_age = [active_age]
    for i in range(0, len(active_age)):
        if not isinstance(active_age[i], (int, float)) or not (0 <= active_age[i] and active_age[i] <= 10.14):
            raise Exception('Active age must be provided as a float or list/array of floats in units of log10 years.')
    
    if not isinstance(equipartition, (list, np.ndarray)):
        equipartition = [equipartition]
    for i in range(0, len(equipartition)):
        if not isinstance(equipartition[i], (int, float)) or not (-6 < equipartition[i] and equipartition[i] < 6):
            raise Exception('Equipartition factor must be provided as a float or list/array of floats in units of log10.')
            
    if not isinstance(jet_lorentz, (list, np.ndarray)):
        jet_lorentz = [jet_lorentz]
    for i in range(0, len(jet_lorentz)):
        if not isinstance(jet_lorentz[i], (int, float)) or not (-100 <= jet_lorentz[i] and jet_lorentz[i] < 20):
            raise Exception('Jet bulk lorentz factor factor must be provided as a float or list/array of floats.')
        if (-100 <= jet_lorentz[i] and jet_lorentz[i] <= 1):
            jet_lorentz[i] = 0
            warnings.warn('Jet phase will not be included in this simulation.', category=UserWarning)

    # convert environment to correct data types
    if not isinstance(halo_mass, (list, np.ndarray)) and not halo_mass == None:
        halo_mass = [halo_mass]
        nenvirons_halo = len(halo_mass)
    elif not halo_mass == None:
        nenvirons_halo = len(halo_mass)
    if isinstance(halo_mass, (list, np.ndarray)):
        for i in range(0, len(halo_mass)):
            if not isinstance(halo_mass[i], (int, float)) or not (9 < halo_mass[i] and halo_mass[i] < 17):
                raise Exception('Dark matter halo mass must be provided as a float or list/array of floats in units of log10 stellar mass.')
    
    if not isinstance(rho0Value, (list, np.ndarray)) and not rho0Value == None:
        rho0Value = [rho0Value]
        nenvirons_rho = len(rho0Value)
    elif not rho0Value == None:
        nenvirons_rho = len(rho0Value)
    if isinstance(rho0Value, (list, np.ndarray)):
        if not isinstance(temperature, (list, np.ndarray)) and not temperature == None:
            temperature = [temperature]*nenvirons_rho
        elif temperature == None or not len(temperature) == nenvirons_rho:
            rho0Value = None # full density profile not provided
        if isinstance(betas, (list, np.ndarray)) and not isinstance(betas[0], (list, np.ndarray)):
            betas = [betas]*nenvirons_rho
        elif not isinstance(betas, (list, np.ndarray)) and not betas == None:
            betas = [[betas]]*nenvirons_rho
        elif betas == None or not len(betas) == nenvirons_rho:
            rho0Value = None # full density profile not provided
        if isinstance(regions, (list, np.ndarray)) and not isinstance(regions[0], (list, np.ndarray)):
            regions = [regions]*nenvirons_rho
        elif not isinstance(regions, (list, np.ndarray)) and not betas == None:
            regions = [[regions]]*nenvirons_rho
        elif regions == None or not len(regions) == nenvirons_rho:
            rho0Value = None # full density profile not provided
    if isinstance(rho0Value, (list, np.ndarray)):
        nenvirons = nenvirons_rho
        for i in range(0, len(rho0Value)):
            if not isinstance(rho0Value[i], (int, float)) or not (1e-30 < rho0Value[i] and rho0Value[i] < 1e-15):
                raise Exception('Core gas density must be provided as a float or list/array of floats in units of kg/m^3.')
        for i in range(0, len(temperature)):
            if not isinstance(temperature[i], (int, float)) or not (0 < temperature[i] and temperature[i] < 1e12):
                raise Exception('Gas temperature must be provided as a float or list/array of floats in units of Kelvin.')
    else:
        nenvirons = nenvirons_halo
    
    return frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz, nenvirons


# Define random seed function
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __set_seed(value):
    np.random.seed(value)
    

## Define functions for analytic modelling of the environment
# function to calculate properties of the environment and call RAiSE_evolution
def __RAiSE_environment(redshift, axis_ratio, jet_power, source_age, halo_mass=None, rand_profile=False, betas=None, regions=None, rho0Value=None, temperature=None, active_age=10.14, jet_lorentz=5., gammaCValue=4./3, aj_star=0.23, crit_mach=1., jet_angle=0.58):
    
    # check minimal inputs
    if halo_mass == None and (not isinstance(betas, (list, np.ndarray)) or not isinstance(regions, (list, np.ndarray))):
        raise Exception('Either the halo mass or full density profile must be provided as model inputs.')
    
    # calculate gas mass and virial radius of halo unless density and temperature profile fully specified
    gasfraction = 0
    if not halo_mass == None:
        rVir = (10**halo_mass*const.M_sun.value/(100./const.G.value*(100.*hubble*np.sqrt(OmegaM*(1 + redshift)**3 + OmegaD)/const.kpc.value)**2))**(1./3)
        if rand_profile == False:
            gasfraction = __HalogasfracFunction(halo_mass, redshift)
        else:
            gasfraction = __rand_norm(__HalogasfracFunction(halo_mass, redshift), __dHalogasfracFunction(halo_mass, redshift))
        gasMass = 10**(halo_mass + gasfraction)*const.M_sun.value
    
    # approximate the gas density profile of Vikhlinin 2006 by multiple density profiles with a simple beta dependence
    if not isinstance(betas, (list, np.ndarray)) or not isinstance(regions, (list, np.ndarray)):
        # set maximum number of regions
        nregions = betaRegions
        nregions, new_betas, new_regions = __DensityProfiler(rVir, nregions, rand_profile)
    elif len(betas) == len(regions):
        # set maximum number of regions
        nregions = len(betas)
        new_betas = np.asarray(betas.copy())
        new_regions = np.asarray(regions.copy())
    else:
        raise Exception('Variables betas and regions must be arrays of the same length.')
    
    # calculate the average temperature of the external medium
    if temperature == None:
        if not halo_mass == None:
            if rand_profile == False:
                tempFlat = 10**TmgAvg
                tempCluster = 10**(TmgConst + TmgSlope*np.log10(halo_mass))
            else:
                tempFlat = 10**(__rand_norm(TmgAvg, TmgStdev))
                tempCluster = 10**(__rand_norm(TmgConst + TmgSlope*np.log10(halo_mass), TmgError))
            temperature = max(tempFlat, tempCluster) # take the highest temperature out of the flat profile and cluster model
        else:
            raise Exception('Either the halo mass or temperature must be provided as model inputs.')

    # determine initial value of density parameter given gas mass and density profile
    if not rho0Value == None:
        # determine density parameter in the core
        k0Value = rho0Value*new_regions[0]**new_betas[0]
        # extend first beta region to a radius of zero
        new_regions[0] = 0
    elif not halo_mass == None:
        # extend first beta region to a radius of zero
        new_regions[0] = 0
        # find relative values (i.e. to 1) of density parameter in each beta region
        kValues = __DensityParameter(nregions, 1.0, new_betas, new_regions)
        # determine density parameter in the core
        k0Value = __k0ValueFinder(rVir, gasMass, nregions, new_betas, new_regions, kValues)
    else:
        raise Exception('Either the halo mass or core density must be provided as model inputs.')
    
    # find values of density parameter in each beta region
    kValues = __DensityParameter(nregions, k0Value, new_betas, new_regions)
    
    # call RadioSourceEvolution function to calculate Dt tracks
    return __RAiSE_evolution(redshift, axis_ratio, jet_power, source_age, active_age, gammaCValue, nregions, new_betas, new_regions, kValues, temperature, jet_lorentz, aj_star, crit_mach, jet_angle)

    
# approximate the gas density profile of Vikhlinin 2006 by multiple density profiles with a simple beta dependence
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __DensityProfiler(rVir, nregions, rand_profile):
    
    # instantiate variables
    betas, regions = np.zeros(nregions), np.zeros(nregions)
    
    # set values of Vikhlinin model parameters
    if rand_profile == False:
        alpha = alphaAvg
        betaPrime = betaPrimeAvg
        gammaPrime = gammaPrimeAvg # this value has no uncertainty
        epsilon = epsilonAvg
        rCore = rCoreAvg
        rSlope = rSlopeAvg
    else:
        alpha = __rand_norm(alphaAvg, alphaStdev)
        betaPrime = __rand_norm(betaPrimeAvg, betaPrimeStdev)
        gammaPrime = __rand_norm(gammaPrimeAvg, gammaPrimeStdev) # this value has no uncertainty
        epsilon = __rand_norm(epsilonAvg, epsilonStdev)
        rCore = __rand_norm(rCoreAvg, rCoreStdev)
        rSlope = __rand_norm(rSlopeAvg, rSlopeStdev)
    
    # set minimum and maximum radius for density profile to be matched
    rmin = rCutoff*rVir
    rmax = rVir
    
    # use logarithmic radius scale
    r = rmin
    ratio = (rmax/rmin)**(1./(nregions)) - 1
    
    for count in range(0, nregions):
        # set radius at low end of region
        rlow = r
        # calculate relative density at rlow, i.e. ignoring rho_0 factor
        rhoLow = np.sqrt((rlow/(rCore*rVir))**(-alpha)/((1 + rlow**2/(rCore*rVir)**2)**(3*betaPrime - alpha/2.)*(1 + rlow**gammaPrime/(rSlope*rVir)**gammaPrime)**(epsilon/gammaPrime)))
        
        # increment radius
        dr = r*ratio
        r = r + dr
        
        # set radius at high end of region
        rhigh = r
        # calculate relative density at rlow, i.e. ignoring rho_0 factor
        rhoHigh = np.sqrt((rhigh/(rCore*rVir))**(-alpha)/((1 + rhigh**2/(rCore*rVir)**2)**(3*betaPrime - alpha/2.)*(1 + rhigh**gammaPrime/(rSlope*rVir)**gammaPrime)**(epsilon/gammaPrime)))
        
        # set value of innermost radius of each beta region
        if count == 0:
            # extend first beta region to a radius of zero
            regions[count] = 0
        else:
            regions[count] = rlow
        
        # calculate exponent beta for each region to match density profile, ensuring beta is less than 2
        if (-np.log(rhoLow/rhoHigh)/np.log(rlow/rhigh) < betaMax):
            betas[count] = -np.log(rhoLow/rhoHigh)/np.log(rlow/rhigh)
        else:
            # ensure beta is less than (or equal to) 2
            betas[count] = betaMax
            # set this count to be the number of distinct regions
            nregions = count + 1
            break

    return nregions, betas, regions
    

# find values of density parameter in each beta region
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __DensityParameter(nregions, k0Value, betas, regions):
        
    # instantiate variables
    kValues = np.zeros(nregions)
        
    # calculate density parameters in each region
    for count in range(0, nregions):
        # match tracks between regions `a' and `b'
        if count > 0:
            # find replicating core density in region `b' required to match pressures and times
            kValues[count] = kValues[count - 1]*regions[count]**(betas[count] - betas[count - 1])
        # if first region, set initial value of replicating core density as actual core density
        else:
            kValues[count] = k0Value
    
    return kValues


# determine value of the density parameter at the core given gas mass and density profile
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __k0ValueFinder(rVir, gasMass, nregions, betas, regions, kValues):
    
    # set volume to zero initially
    volume = 0
    
    # calculate weighted volume integral using by analytically integraing the volume in each beta region
    for count in range(0, nregions):
        # set lower bound of analytic integral
        rlow = regions[count]
        
        # set upper bound of analytic integral
        if (count + 1 == nregions):
            rhigh = rVir
        else:
            rhigh = regions[count + 1]
        
        # increment total weighted volume by weigthed volume of this region
        volume = volume + 4*np.pi*(kValues[count]/kValues[0])/(3 - betas[count])*(rhigh**(3 - betas[count]) - rlow**(3 - betas[count]))

    # calculate density parameter at the core from stellar mass and weighted volume
    k0Value = gasMass/volume
    
    return k0Value


# random normal with values truncated to avoid sign changes
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __rand_norm(mean, stdev):
    
    rand_number = np.random.normal(mean, stdev)
    while (mean*rand_number < 0 or np.abs(rand_number - mean) > 2*stdev):
        rand_number = np.random.normal(mean, stdev)

    return rand_number


# gas fraction-halo mass relationship
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __HalogasfracFunction(halo_mass, redshift):
    return max(halogasfracCONST1z0 + halogasfracCONST1z1*redshift, halogasfracCONST2z0 + halogasfracCONST2z1*redshift) + halogasfracSLOPE*(halo_mass - 14) + SAGEdensitycorr # in log space


# gas fraction-halo mass relationship error
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __dHalogasfracFunction(halo_mass, redshift):
    return dhalogasfracz0 + dhalogasfracz1*redshift # in log space


## Define functions required for RAiSE dynamical evolution
# function to calculate dynamical evolution of lobe and shocked shell
def __RAiSE_evolution(redshift, axis_ratio, jet_power, source_age, active_age, gammaCValue, nregions, betas, regions, kValues, temperature, jet_lorentz, aj_star=0.23, crit_mach=1., jet_angle=0.58):
    
    # convert jet power and source age to correct units
    QavgValue = 10**jet_power/2. # set the power of *each* jet; convert from log space
    if isinstance(source_age, (list, np.ndarray)):
        tFinal = np.zeros_like(source_age)
        for i in range(0, len(source_age)):
            tFinal[i] = 10**source_age[i]*year # convert from log space years to seconds
    else:
        tFinal = np.array([10**source_age*year])
    tActive = 10**active_age*year

    # calculate angle of current radial line
    angles = np.arange(0, nangles, 1).astype(np.int_)
    dtheta = (np.pi/2)/nangles
    theta = dtheta*(angles + 0.5)
    
    # calculate opening angle of jet
    open_angle = (jet_angle*np.pi/180)/(axis_ratio/4.)
    angles_j = np.ones_like(theta)
    angles_j[theta > open_angle] = 0
    
    # evaluate the translation coefficients eta_c and eta_s
    eta_c = 1./np.sqrt(axis_ratio**2*(np.sin(theta))**2 + (np.cos(theta))**2)
    eta_s = 1./np.sqrt(axis_ratio**(2*shockAxisRatio)*(np.sin(theta))**2 + (np.cos(theta))**2)
    # evaluate the translation coefficient zeta_s/eta_s at t -> infinity
    zetaeta = np.sqrt(axis_ratio**(2*shockAxisRatio)*(np.sin(theta))**2 + (np.cos(theta))**2)/np.sqrt(axis_ratio**(4*shockAxisRatio)*(np.sin(theta))**2 + (np.cos(theta))**2)
    eta_c[0], eta_s[0], zetaeta[0] = 1., 1., 1,
    
    # calculate the differential volume element coefficient chi
    dchi = 4*np.pi/3.*np.sin(theta)*np.sin(dtheta/2.)
    
    # solve RAiSE dynamics iteratively to find thermal component of lobe pressure
    if jet_lorentz > 1:
        # run code in strong-shock limit to calibrate initial velocity
        x_time = 10**10.14*year
        _, _, _, _, _, _, _, critical_point_1 = __RAiSE_runge_kutta(QavgValue, np.array([x_time]), x_time, axis_ratio, aj_star, jet_lorentz, open_angle, angles, eta_c, eta_s, zetaeta, dchi, nregions, betas, regions, kValues, temperature, gammaCValue, critical_velocity=c_speed, strong_shock=True)

        # run code for full RAiSE HD dynamical model
        lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, critical_point_3 = __RAiSE_runge_kutta(QavgValue, tFinal, tActive, axis_ratio, aj_star, jet_lorentz, open_angle, angles, eta_c, eta_s, zetaeta, dchi, nregions, betas, regions, kValues, temperature, gammaCValue, critical_velocity=c_speed*critical_point_1[2]/critical_point_1[3])
    else:
        # run code for RAiSE X dynamical model
        lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, _ = __RAiSE_runge_kutta(QavgValue, tFinal, tActive, axis_ratio, aj_star, jet_lorentz, open_angle, angles, eta_c, eta_s, zetaeta, dchi, nregions, betas, regions, kValues, temperature, gammaCValue)

    return lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda
    

# function to apply Runge-Kutta method and extract values at requested time steps
#@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_runge_kutta(QavgValue, source_age, active_age, axis_ratio, aj_star, jet_lorentz, open_angle, angles, eta_c, eta_s, zetaeta, dchi, nregions, betas, regions, kValues, temperature, gammaCValue, critical_velocity=0., strong_shock=False):

    # instantiate variables
    X, P = np.zeros((nangles, 5)), np.zeros((nangles, 4))
    X_prev, P_prev, P2_prev = 0, 0, 0
    critical_point = np.zeros(4)
    regionPointer = np.zeros(nangles).astype(np.int_)
    lobe_minor, lambda_crit, alphaP_denv, alpha_lambda = np.zeros(len(source_age)), np.zeros(len(source_age)), np.zeros(len(source_age)), np.zeros(len(source_age))
    lobe_lengths, shock_lengths, shock_pressures = np.zeros((nangles, len(source_age))), np.zeros((nangles, len(source_age))), np.zeros((nangles, len(source_age)))
    
    # calculate the spatially-averaged jet velocity and Lorentz factor
    if jet_lorentz > 1:
        bulk_lorentz = np.sqrt(jet_lorentz**2*aj_star**4 - aj_star**4 + 1)
        bulk_velocity = np.sqrt((jet_lorentz**2*aj_star**4 - aj_star**4)/(jet_lorentz**2*aj_star**4 - aj_star**4 + 1))*c_speed
    else:
        bulk_lorentz, bulk_velocity = -1, -1
    
    for timePointer in range(0, len(source_age)):
        # set initial conditions for each volume element
        if timePointer == 0:
            # calculate initial time and radius for ODE
            FR2time = limTime
            if jet_lorentz > 1:
                FR2radius = bulk_velocity*limTime
                FR2velocity = bulk_velocity # eta_R is very large
            else:
                FR2radius = np.sqrt(1 - 1./100**2)*c_speed*limTime
                FR2velocity = np.sqrt(1 - 1./100**2)*c_speed
            # test if this radius is above start of second region boundary
            if (regions[1] < FR2radius):
                FR2radius = regions[1]
                if jet_lorentz > 1:
                    FR2time = regions[1]/bulk_velocity
                    FR2velocity = bulk_velocity
                else:
                    FR2time = regions[1]/(np.sqrt(1 - 1./100**2)*c_speed)
                    FR2velocity = np.sqrt(1 - 1./100**2)*c_speed
            
            # calculate the initial jet/shock shell radius and velocity for each angle theta
            X[angles,0] = FR2time
            X[angles,1] = FR2radius*eta_s
            X[angles,2] = FR2velocity*eta_s
            if jet_lorentz > 1:
                X[0,3], X[angles[1:],3] = bulk_lorentz, 1./np.sqrt(1 - (FR2velocity*eta_s[angles[1:]]/c_speed)**2)
            else:
                X[0,3], X[angles[1:],3] = 100, 100*eta_s[angles[1:]]
            X[angles,4] = -1 # null value
            #X[angles,5] = 0

            # set region pointer to first (non-zero) region if smaller than FR2 radius
            index = regions[1] < X[angles,1]
            regionPointer[index] = 1
            regionPointer[np.logical_not(index)] = 0
            
            # calculate fraction of jet power injected into each volume element
            injectFrac = dchi*eta_s**(3 - betas[regionPointer[0]])*zetaeta**2
            injectFrac = injectFrac/np.sum(injectFrac) # sum should be equal to unity
            
        # solve ODE to find radius and pressue at each time step
        while (X[0,0] < source_age[timePointer]):
            # calculate the appropriate density profile for each angle theta
            for anglePointer in range(0, nangles):
                while (regionPointer[anglePointer] + 1 < nregions and X[anglePointer,1] > regions[regionPointer[anglePointer] + 1]):
                    regionPointer[anglePointer] = regionPointer[anglePointer] + 1

            # check if next step passes time point of interest
            if (X[0,0]*stepRatio > source_age[timePointer]):
                step = source_age[timePointer] - X[0,0]
            else:
                step = X[0,0]*(stepRatio - 1)

            # update estimates of time, radius and velocity
            X_prev = X[0,0]
            P_prev = P[0,0]
            P2_prev = P[0,3]
            __rk4sys(step, X, P, QavgValue, active_age, aj_star, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
            X[:,3] = np.maximum(1, X[:,3])
            
            # find location of jet--lobe transition
            critical_point[0], critical_point[1], critical_point[2], critical_point[3] = X[0,0], X[0,1], X[0,2]*X[0,3], X[0,4]

        # calculate the lobe and shocked shell length, shock pressure and total pressure as a function of angle
        lobe_lengths[angles,timePointer] = P[angles,0]
        shock_lengths[angles,timePointer] = X[angles,1]
        shock_pressures[angles,timePointer] = P[angles,1]
        lambda_crit[timePointer] = P[0,3]
        
        # calculate lobe minor axis (associated with dimensions of shocked shell) at this time step
        lobe_minor[timePointer] = X[-1,1]*eta_c[-1]/(shockRadius*eta_s[-1])
        
        # calculate the slope of external pressure profile at this time step
        if P_prev == 0:
            alphaP_denv[timePointer] = 0
        else:
            alphaP_denv[timePointer] = np.log(P[0,2]/P_prev)/np.log(X[0,0]/X_prev)
        if P2_prev <= 0:
            alpha_lambda[timePointer] = 0
        else:
            alpha_lambda[timePointer] = np.log(P[0,3]/P2_prev)/np.log(X[0,0]/X_prev)
    
    return lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, critical_point


# Runge-Kutta method to solve ODE in dynamical model
#@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __rk4sys(step, X, P, QavgValue, active_age, aj_star, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock):
    
    # instantiate variables
    Y, K1, K2, K3, K4 = np.zeros((len(angles), 5)), np.zeros((len(angles), 5)), np.zeros((len(angles), 5)), np.zeros((len(angles), 5)), np.zeros((len(angles), 5))
    
    # fouth order Runge-Kutta method
    __xpsys(X, K1, P, QavgValue, active_age, aj_star, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
    Y[:,:] = X[:,:] + 0.5*step*K1[:,:]
    __xpsys(Y, K2, P, QavgValue, active_age, aj_star, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
    Y[:,:] = X[:,:] + 0.5*step*K2[:,:]
    __xpsys(Y, K3, P, QavgValue, active_age, aj_star, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
    Y[:,:] = X[:,:] + 0.5*step*K3[:,:]
    __xpsys(Y, K4, P, QavgValue, active_age, aj_star, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
    X[:,:] = X[:,:] + (step/6.)*(K1[:,:] + 2*K2[:,:] + 2*K3[:,:] + K4[:,:])


# coupled second order differential equations for lobe evolution
#@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __xpsys(X, f, P, QavgValue, active_age, aj_star, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock):
    
    # Differential equations for X[0,1,2,3,4] = (time, radius, velocity, lorentz_factor, thermal_velocity)
    # Additional variable for P[0,1,2,3] = (lobe_length, lobe_pressure, external_pressure, lambda_crit)
    f[angles,0] = 1.
    f[angles,1] = X[angles,2]
    
    # test if the AGN is active at this time-step
    if (X[0,0] <= active_age):
        active_jet = 1
    else:
        active_jet = 0
        
    # calculate the spatially-averaged jet velocity and Lorentz factor
    if jet_lorentz > 1:
        bulk_lorentz = np.sqrt(jet_lorentz**2*aj_star**4 - aj_star**4 + 1)
        bulk_velocity = np.sqrt((jet_lorentz**2*aj_star**4 - aj_star**4)/(jet_lorentz**2*aj_star**4 - aj_star**4 + 1))*c_speed
    else:
        bulk_lorentz, bulk_velocity = -1, -1

    # TWO-PHASE FLUID
    if jet_lorentz > 1:
        # calculate the lobe formation scale
        eta_R = QavgValue*bulk_lorentz**2/(2*np.pi*kValues[regionPointer[0]]*(bulk_lorentz*bulk_velocity)*(bulk_lorentz - 1)*c_speed**2*(1 - np.cos(open_angle))*X[0,1]**(2 - betas[regionPointer[0]]))
        
        # calculate lambda_crit
        lambda_crit = 1/((eta_R/bulk_lorentz**2)**10 + 1)
        #lambda_crit = np.exp(-(eta_R/bulk_lorentz**2)/(2*np.log(2)))
        if np.isinf(lambda_crit):
            lambda_crit = 0
        
        P[0,3] = lambda_crit
    else:
        P[0,3] = 1
        
    # ACCELERATION
    # update fraction of jet power injected into each volume element
    injectFrac_new = dchi*eta_s**(3 - betas[regionPointer[0]])*zetaeta**2
    injectFrac_new = injectFrac/np.sum(injectFrac) # sum should be equal to unity
    if jet_lorentz > 1:
        injectFrac[angles] = (1 - lambda_crit)*injectFrac_new + lambda_crit*injectFrac # keep static at late times
    else:
        injectFrac[angles] = injectFrac_new[angles]
    
    # acceleration of jet-head
    if jet_lorentz > 1:
        jet_acceleration = (betas[regionPointer[0]] - 2)*bulk_velocity*X[0,2]/(2*X[0,1]*(1 + eta_R**(-1./2))**2*eta_R**(1./2))

    # acceleration of lobe (supersonic/subsonic)
    sub_angles = (X[angles,2]*X[angles,3]*zetaeta)**2/(gammaX*(k_B*temperature/maverage)) <= 1
    super_angles = np.logical_not(sub_angles)
    if strong_shock == True:
        f[angles,2] = np.minimum((gammaX + 1)*(gammaCValue - 1)*injectFrac[angles]*(QavgValue*active_jet)*X[angles,1]**(betas[regionPointer[angles]] - 3)/(2*X[angles,2]*(1 + (X[angles,3]*X[angles,2]/c_speed)**2)*dchi[angles]*(X[angles,3]*zetaeta[angles])**2*kValues[regionPointer[angles]]) + (betas[regionPointer[angles]] - 3*gammaCValue)*(X[angles,2])**2/(2*X[angles,1]*(1 + (X[angles,3]*X[angles,2]/c_speed)**2)), (betas[regionPointer[angles]] - 2)/(5 - betas[regionPointer[angles]]) * X[angles,2]*X[angles,3]/(X[0,0] + year)) # ensure model doesn't run slower than limit due to numerics
    else:
        sub_angles = (X[angles,2]*X[angles,3]*zetaeta)**2/(gammaX*(k_B*temperature/maverage)) <= 1
        super_angles = np.logical_not(sub_angles)
        f[super_angles,2] = (gammaX + 1)*(gammaCValue - 1)*injectFrac[super_angles]*(QavgValue*active_jet)*X[super_angles,1]**(betas[regionPointer[super_angles]] - 3)/(2*X[super_angles,2]*(1 + (X[super_angles,3]*X[super_angles,2]/c_speed)**2)*dchi[super_angles]*(X[super_angles,3]*zetaeta[super_angles])**2*kValues[regionPointer[super_angles]]) + (betas[regionPointer[super_angles]] - 3*gammaCValue)*(X[super_angles,2])**2/(2*X[super_angles,1]*(1 + (X[super_angles,3]*X[super_angles,2]/c_speed)**2)) + (gammaX - 1)*(3*gammaCValue - betas[regionPointer[super_angles]])*(k_B*temperature/maverage)/(4*X[super_angles,1]*(1 + (X[super_angles,3]*X[super_angles,2]/c_speed)**2)*(X[super_angles,3]*zetaeta[super_angles])**2)
        f[sub_angles,2] = (betas[regionPointer[sub_angles]] - 2)*(X[sub_angles,2])**2/X[sub_angles,1]

    # combine acceleration from jet-head and lobe as two-phase fluid
    if jet_lorentz > 1:
        if (lambda_crit < 0.01 or X[0,0] < 10*limTime): # improve stability
            f[0,2], f[angles[1:],2] = jet_acceleration, jet_acceleration*eta_s[angles[1:]]
            X[angles[1:],2] = X[0,2]*eta_s[angles[1:]]
        else:
            f[0,2], f[angles[1:],2] = (1 - lambda_crit)*jet_acceleration + lambda_crit*f[0,2], (1 - lambda_crit)*jet_acceleration*eta_s[angles[1:]] + lambda_crit*f[angles[1:],2]
   
    # calculate Lorentz factor of two-phase fluid
    f[angles,3] = X[angles,3]**3*X[angles,2]*f[angles,2]/c_speed**2
   
    # PRESSURES
    # external pressure at each volume element
    P[angles,2] = kValues[regionPointer[angles]]*(k_B*temperature/maverage)*X[angles,1]**(-betas[regionPointer[angles]])
    
    # set velocity associated with thermal component of lobe perssure
    if jet_lorentz > 1 and critical_velocity > 0:
        if X[0,4] < 0:
            X[angles,4] = X[angles,2]*critical_velocity/X[0,2]
        f[angles,4] = (betas[regionPointer[angles]] - 2)/(5 - betas[regionPointer[angles]]) * X[angles,4]/(X[0,0] + year)
    else:
        X[angles,4], f[angles,4] = X[angles,2]*X[angles,3], f[angles,2]

    # jet/lobe pressure at each volume element
    volume = X[angles,1]**3*dchi[angles]
    if jet_lorentz > 1:
        # calculate lobe pressure
        P[super_angles,1] = 2./(gammaX + 1)*zetaeta[super_angles]**2*kValues[regionPointer[super_angles]]*X[super_angles,1]**(-betas[regionPointer[super_angles]])*(np.minimum(X[super_angles,2], X[super_angles,4]))**2 - (gammaX - 1)/(gammaX + 1)*kValues[regionPointer[super_angles]]*(k_B*temperature/maverage)*X[super_angles,1]**(-betas[regionPointer[super_angles]])
        P[sub_angles,1] = P[sub_angles,2]
        
        # calculate average pressure across jet/lobe
        pressure = np.sum(P[angles,1]*volume)/np.sum(volume)
    else:
        # calculate lobe pressure
        P[super_angles,1] = 2./(gammaX + 1)*zetaeta[super_angles]**2*kValues[regionPointer[super_angles]]*X[super_angles,1]**(-betas[regionPointer[super_angles]])*(X[super_angles,2]*X[super_angles,3])**2 - (gammaX - 1)/(gammaX + 1)*kValues[regionPointer[super_angles]]*(k_B*temperature/maverage)*X[super_angles,1]**(-betas[regionPointer[super_angles]])
        P[sub_angles,1] = P[sub_angles,2]
        
        # calculate average pressure across jet/lobe
        pressure = np.sum(P[angles,1]*volume)/np.sum(volume)
    
    # calculate the gradient across the lobe using the empirical relationship of Kaiser+2000
    hotspot_ratio = __RAiSE_hotspot_ratio(X[0,1]/X[-1,1], betas[regionPointer[0]])
    # set pressure at major and minor axes
    P[0,1], P[angles[1:-1],1], P[-1,1] = pressure*8*hotspot_ratio/(5 + 3*hotspot_ratio), pressure, pressure*8/(5 + 3*hotspot_ratio)

    # AXIS RATIO
    if jet_lorentz > 1:
        # calculate total mass of particles from the jet
        particle_mass = QavgValue*np.minimum(active_age, X[0,0])/((jet_lorentz - 1)*c_speed**2)
        
        # calculate volume occupied by particles expanding at speed of light and at sound speed; i.e. maximum distance reached and volume of lobe particles only
        jet_sound = c_speed #c_speed*np.sqrt(gammaJ - 1)
        lobe_volume = particle_mass/(pressure/c_speed**2) # mass / density
        particle_volume = particle_mass/(pressure/jet_sound**2) # mass / density

        # calculate volume averaged fraction of lobe particles
        #f[angles,5] = 3*X[angles,1]**2*X[angles,2]*dchi[angles] * lambda_crit
        #lambda_avg = np.sum(X[angles,5])/np.sum(volume)
        
        # find axis ratio for an ellipsoidal lobe
        P[0,0] = X[0,1]/shockRadius
        if particle_volume > 0:
            lobe_axis_ratio = np.minimum(np.sqrt(2*np.pi*P[0,0]**3/(3*particle_volume)), 1/np.tan(open_angle))
            particle_axis_ratio = np.minimum(np.sqrt(2*np.pi*P[0,0]**3/(3*particle_volume)), 1/np.tan(open_angle))
        else:
            lobe_axis_ratio = 1/np.tan(open_angle)
            particle_axis_ratio = 1/np.tan(open_angle)

        # calculate geometry of each angular volume element
        dtheta = (np.pi/2)/len(angles)
        theta = dtheta*(angles + 0.5)
        lobe_eta_c = 1./np.sqrt(lobe_axis_ratio**2*(np.sin(theta))**2 + (np.cos(theta))**2)
        particle_eta_c = 1./np.sqrt(particle_axis_ratio**2*(np.sin(theta))**2 + (np.cos(theta))**2)
        
        # set length of lobe along each angular volume element, and length of equivalent volume if occupied by lobe particle with filling factor = 1
        P[angles[1:],0] = np.minimum(lobe_eta_c[angles[1:]]*P[0,0], X[angles[1:],1]*eta_c[angles[1:]]/(shockRadius*eta_s[angles[1:]]))
        particle_lengths = np.zeros_like(P[:,0])
        particle_lengths[0], particle_lengths[angles[1:]] = X[0,1]/shockRadius, np.minimum(particle_eta_c[angles[1:]]*P[0,0], X[angles[1:],1]*eta_c[angles[1:]]/(shockRadius*eta_s[angles[1:]]))

        # calculate modified lambda_crit due to partial filling factor
        P[0,3] = P[0,3]*np.sum(particle_lengths[angles]*dchi[angles])/np.sum(P[angles,0]*dchi[angles])
    else:
        # set length of lobe along each angular volume element
        P[0,0], P[angles[1:],0] = X[0,1]/shockRadius, X[angles[1:],1]*eta_c[angles[1:]]/(shockRadius*eta_s[angles[1:]])
    
    
# define function to calculate the expected hotspot to lobe pressure ratio; Kaiser+2000
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_hotspot_ratio(axis_ratio, beta):
    return (2.14 - 0.52*beta)*(axis_ratio/2)**(2.04 - 0.25*beta)


## Define functions to download and preprocess particles from hydrodynamical simulations
def __PLUTO_particles(particle_data_path):
        
    # unpack particle data from hydrodynamical simulations
    particle_dict = h5py.File(particle_data_path, 'r')
    
    # store variables at desired resolution
    time = particle_dict['time'][:]
    shock_time = particle_dict['tinject'][:,:]
    major = particle_dict['major'][:]
    minor = particle_dict['minor'][:]
    x1 = particle_dict['x1'][:,:]
    x2 = particle_dict['x2'][:,:]
    x3 = particle_dict['x3'][:,:]
    tracer = particle_dict['tracer'][:,:]
    vx3 = particle_dict['vx3'][:,:]
    volume = particle_dict['volume'][:,:]
    pressure = particle_dict['pressure'][:,:]
    press_minor = particle_dict['pressminor'][:]
    alphaP_hyd = particle_dict['alphaP'][:,:]
    alphaP_henv = particle_dict['alphaPenv'][:,:]
    hotspot_ratio = particle_dict['hotspotratio'][:]

    return time, shock_time, major, minor, x1, x2, x3, tracer, vx3, volume, pressure, press_minor, alphaP_hyd, alphaP_henv, hotspot_ratio
    

## Define functions to add emissivity from particles in hydrodynamical simulations on top of dynamics
# function to manage orientation and distribution of particles from simulation output
def __RAiSE_emissivity(frequency, redshift, time, shock_time, major, minor, x1, x2, x3, tracer, vx3, volume, pressure, press_minor, alphaP_hyd, alphaP_henv, hotspot_ratio, source_age, lobe_lengths, lobe_minor, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, active_age, equipartition, spectral_index, gammaCValue=4./3, lorentz_min=Lorentzmin, resolution='standard'):
    
    # determine spatial resolution of particles; i.e. overdensity of particles to include in calculations
    if resolution == 'best':
        nsamples = 2048
    elif resolution == 'high':
        nsamples = 512
    elif resolution == 'standard':
        nsamples = 128
    elif resolution == 'poor':
        nsamples = 32
    else:
        raise Exception('Unrecognised keyword for particle resolution. The accepted keywords are: best, high, standard and poor.')

    # randomly generate viewing time in the simulated source age
    timePointer = np.arange(0, nsamples).astype(np.int_)%len(time)
    
    # convert frequency, equipartition factor and spectral index to correct units
    if isinstance(frequency, (list, np.ndarray)):
        rest_frequency = np.zeros_like(frequency)
        inverse_compton = np.zeros_like(frequency).astype(np.int_)
        for freqPointer in range(0, len(frequency)):
            rest_frequency[freqPointer] = 10**frequency[freqPointer]*(1 + redshift)
            if rest_frequency[freqPointer] > 1e12: # assume frequencies greater than 1000 GHz are inverse-Compton
                inverse_compton[freqPointer] = 1
    else:
        rest_frequency = [10**frequency*(1 + redshift)]
        if rest_frequency[freqPointer] > 1e12: # assume frequencies greater than 1000 GHz are inverse-Compton
            inverse_compton = [1]
    if isinstance(source_age, (list, np.ndarray)):
        tFinal = np.zeros_like(source_age)
        for i in range(0, len(source_age)):
            tFinal[i] = 10**source_age[i]*year # convert from log space years to seconds
    else:
        tFinal = [10**source_age*year]
    tActive = 10**active_age*year
    equi_factor = 10**float(-np.abs(equipartition)) # ensure sign is correct
    s_index = 2*float(np.abs(spectral_index)) + 1 # ensure sign is correct
    
    # derive redshift dependent ancillary variables used by every analytic model
    Ks = __RAiSE_Ks(s_index, gammaCValue, lorentz_min)
    blackbody = __RAiSE_blackbody(s_index)
    
    return __RAiSE_particles(timePointer, rest_frequency, inverse_compton, redshift, time, shock_time, major, minor, x1, x2, x3, tracer, vx3, volume, pressure, press_minor, alphaP_hyd, alphaP_henv, hotspot_ratio, tFinal, lobe_lengths, lobe_minor, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, tActive, equi_factor, s_index, gammaCValue, lorentz_min, Ks, blackbody)


# function to calculate emissivity from each particle using RAiSE model
@jit(nopython=True, parallel=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_particles(timePointer, rest_frequency, inverse_compton, redshift, time, shock_time, major, minor, x1, x2, x3, tracer, vx3, volume, pressure, press_minor, alphaP_hyd, alphaP_henv, hotspot_ratio, tFinal, lobe_lengths, lobe_minor, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, tActive, equi_factor, s_index, gammaCValue, lorentz_min, Ks, blackbody):
    
    # instantiate variables
    luminosity = np.zeros((len(tFinal), len(timePointer)*len(pressure[:,0]), len(rest_frequency)))
    magnetic_field = np.zeros((len(tFinal), len(rest_frequency)))
    magnetic_particle, magnetic_weighting = np.zeros((len(tFinal), len(timePointer), len(rest_frequency))), np.zeros((len(tFinal), len(timePointer), len(rest_frequency)))
    location = np.zeros((len(tFinal), len(timePointer)*len(pressure[:,0]), 3))
    
    # derive emissivity at each time step
    for i in range(0, len(tFinal)):
        # derive emissivity for random variations in particle distribution
        for j in range(0, len(timePointer)):
        
            # SHOCK ACCELERATION TIMES
            new_shock_time = shock_time[:,timePointer[j]]*(tFinal[i]/time[timePointer[j]])*np.minimum(1., (tActive/tFinal[i])) # scale the last acceleration time to active age if source is a remnant
            
            # PRESSURES
            new_pressure = pressure[:,timePointer[j]]*(shock_pressures[-1,i]/press_minor[timePointer[j]])
            # correct the hotspot/lobe pressure ratio based on the dynamical model
            new_pressure = new_pressure*((shock_pressures[0,i]/shock_pressures[-1,i])/hotspot_ratio[timePointer[j]] - 1)*(np.abs(x3[:,timePointer[j]])/major[timePointer[j]]) + new_pressure # increase log-space pressure linearly along lobe
            # correct the evolutionary histories of the particles based on the dynamical model
            alphaP_dyn = np.maximum(-2, np.minimum(0, alphaP_denv[i] + alphaP_hyd[:,timePointer[j]] - alphaP_henv[:,timePointer[j]]))
            
            # VOLUMES
            volume_fraction = volume[:,timePointer[j]]*tracer[:,timePointer[j]]/(4*np.pi/3.*major[timePointer[j]]*minor[timePointer[j]]**2)
            # cap the largest volumes at the 90th percentile to outliers in surface brightness map; minimal effect on total luminosity
            volume_fraction[volume_fraction > np.nanpercentile(volume_fraction, 90)] = np.nanpercentile(volume_fraction, 90)
            new_volume = volume_fraction*(4*np.pi/3.*lobe_lengths[0,i]*lobe_minor[i]**2)
            
            # RELATIVISTIC BEAMING
            doppler_factor = np.sqrt(np.maximum(1e-6, 1 - vx3[:,timePointer[j]]**2))**(3 - (s_index - 1)/2.) # Doppler boosting of particles in jet; 1e-6 ensures some very low level emission
            
            # LOBE PARTICLES
            # find angle and radius of each particle from core
            new_angles = np.arctan(np.sqrt(x1[:,timePointer[j]]**2 + x2[:,timePointer[j]]**2)/x3[:,timePointer[j]])
            new_radii = np.sqrt(x1[:,timePointer[j]]**2 + x2[:,timePointer[j]]**2 + x3[:,timePointer[j]]**2)/major[timePointer[j]]
            # find particles within lobe region; particles outside this region will not emit. Particle map is set to axis ratio based on shocked shell to maintain geometry of jet
            new_eta_c = 1./np.sqrt((lobe_lengths[0,i]/lobe_lengths[-1,i])**2*(np.sin(new_angles))**2 + (np.cos(new_angles))**2)
            lobe_particles = np.zeros_like(x1[:,timePointer[j]])
            lobe_particles[new_radii < new_eta_c] = 1.

            # TWO PHASE FLUID
            # fraction of jet particles that have reached location in lobe
            two_phase_weighting = np.maximum(0, np.minimum(1, lambda_crit[i]*(new_shock_time/np.minimum(tActive, tFinal[i]))**alpha_lambda[i]))
            if tActive/tFinal[i] >= 1:
                # keep jet particles visible at all times
                two_phase_weighting = np.maximum(two_phase_weighting, np.minimum(1, np.abs(vx3[:,timePointer[j]]*np.sqrt(3)))) # assume sound speed is critical value for relativisitic particles
            else:
                # suppress emission from jet particle
                two_phase_weighting = np.minimum(two_phase_weighting, 1 - np.minimum(1, np.abs(vx3[:,timePointer[j]]*np.sqrt(3))))
            
            # PARTICLE EMISSIVITY
            for k in range(0, len(rest_frequency)):
                # calculate losses due to adiabatic expansion, and synchrotron/iC radiation
                lorentz_ratio, pressure_ratio = __RAiSE_loss_mechanisms(rest_frequency[k], inverse_compton[k], redshift, tFinal[i], new_shock_time, new_pressure, alphaP_dyn, equi_factor, gammaCValue)
                
                # calculate luminosity associated with each particle
                temp_luminosity = None
                if inverse_compton[k] == 1:
                    # inverse-Compton
                    sync_frequency = (3*e_charge*rest_frequency[k]*np.sqrt(2*mu0*( equi_factor*new_pressure/((gammaCValue - 1)*(equi_factor + 1)) ))/(2*np.pi*m_e*(freq_cmb*temp_cmb*(1 + redshift)))) # assuming emission at CMB frequency only
                    temp_luminosity = Ks/blackbody*sync_frequency**((1 - s_index)/2.)*(sync_frequency/rest_frequency[k])*(gammaCValue - 1)*__RAiSE_uC(redshift) * (equi_factor**((s_index + 1)/4. -  1 )/(equi_factor + 1)**((s_index + 5)/4. -  1 ))*new_volume*new_pressure**((s_index +  1 )/4.)*pressure_ratio**(1 - 4./(3*gammaCValue))*lorentz_ratio**(2 - s_index)/len(timePointer) * doppler_factor*lobe_particles*two_phase_weighting
                else:
                    # synchrotron
                    temp_luminosity = Ks*rest_frequency[k]**((1 - s_index)/2.)*(equi_factor**((s_index + 1)/4.)/(equi_factor + 1)**((s_index + 5)/4.))*new_volume*new_pressure**((s_index + 5)/4.)*pressure_ratio**(1 - 4./(3*gammaCValue))*lorentz_ratio**(2 - s_index)/len(timePointer) * doppler_factor*lobe_particles*two_phase_weighting
                # remove any infs
                index = np.isinf(temp_luminosity)
                temp_luminosity[index] = np.nan
                luminosity[i,j*len(pressure[:,0]):(j+1)*len(pressure[:,0]),k] = temp_luminosity
                
                # calculate luminosity weighted magnetic field strength
                magnetic_particle[i,j,k] = np.nansum(np.sqrt(2*mu0*new_pressure*equi_factor/(gammaCValue - 1)*(equi_factor + 1))*luminosity[i,j*len(pressure[:,0]):(j+1)*len(pressure[:,0]),k])
                magnetic_weighting[i,j,k] = np.nansum(luminosity[i,j*len(pressure[:,0]):(j+1)*len(pressure[:,0]),k])
            
            # CARTESIAN LOCATIONS
            location[i,j*len(pressure[:,0]):(j+1)*len(pressure[:,0]),0] = x1[:,timePointer[j]]*lobe_minor[i]/minor[timePointer[j]] *np.sign(timePointer[j]%8 - 3.5)
            location[i,j*len(pressure[:,0]):(j+1)*len(pressure[:,0]),1] = x2[:,timePointer[j]]*lobe_minor[i]/minor[timePointer[j]] *np.sign(timePointer[j]%4 - 1.5)
            location[i,j*len(pressure[:,0]):(j+1)*len(pressure[:,0]),2] = x3[:,timePointer[j]]*lobe_lengths[0,i]/major[timePointer[j]] *np.sign(timePointer[j]%2 - 0.5)
            
        # calculate luminosity weighted magnetic field strength for time step
        for k in range(0, len(rest_frequency)):
            if np.nansum(magnetic_weighting[i,:,k]) == 0:
                magnetic_field[i,k] = 0
            else:
                magnetic_field[i,k] = np.nansum(magnetic_particle[i,:,k])/np.nansum(magnetic_weighting[i,:,k])

    return location, luminosity, magnetic_field


# find ratio of the lorentz factor and the pressure at the time of acceleration to that at the time of emission
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_loss_mechanisms(rest_frequency, inverse_compton, redshift, time, shock_time, pressure, alphaP, equipartition, gammaCValue=4./3):
    
    # calculate lorentz factor at time of emission
    if inverse_compton == 1:
        # inverse-Compton
        lorentz_factor = np.sqrt(rest_frequency/(freq_cmb*temp_cmb*(1 + redshift)))*np.ones(len(pressure)) # assuming emission at CMB frequency only
    else:
        # synchrotron
        lorentz_factor = np.sqrt(2*np.pi*m_e*rest_frequency/(3*e_charge*np.sqrt(2*mu0*pressure/(gammaCValue - 1)*(equipartition/(equipartition + 1))))) # assuming emission at Larmor frequency only
    
    # calculate pressure and volume at time of acceleration
    pressure_inject = pressure*(shock_time/time)**alphaP
    
    # calculate RAiSE constant a2
    a2 = __RAiSE_a2(redshift, time, shock_time, pressure, pressure_inject, equipartition, alphaP, gammaCValue)
    
    # calculate lorentz factor at time of acceleration, and remove invalid points
    lorentz_inject = lorentz_factor*shock_time**(alphaP/(3*gammaCValue))/(time**(alphaP/(3*gammaCValue)) - a2*lorentz_factor) # second time is i becasue is time_high
    lorentz_inject[lorentz_inject < 1] = np.nan
    
    return lorentz_inject/lorentz_factor, pressure_inject/pressure
 

# find RAiSE constant a2 for synchrotron and iC radiative losses
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_a2(redshift, time, shock_time, pressure, pressure_inject, equipartition, alphaP, gammaCValue=4./3):
    
    return 4*sigma_T/(3*m_e*c_speed)*(pressure_inject/(gammaCValue - 1)*(equipartition/(equipartition + 1))/(1 + alphaP*(1 + 1./(3*gammaCValue)))*shock_time**(-alphaP)*(time**(1 + alphaP*(1 + 1./(3*gammaCValue))) - shock_time**(1 + alphaP*(1 + 1./(3*gammaCValue)))) + __RAiSE_uC(redshift)/(1 + alphaP/(3*gammaCValue))*(time**(1 + alphaP/(3*gammaCValue)) - shock_time**(1 + alphaP/(3*gammaCValue)))) # array is shorter by one element


# find CMB radiation energy density
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_uC(redshift):
    
    uC0 = 0.25*1e6*e_charge # J m-3 CMB energy density at z = 0 (Longair, 1981)
    return uC0*(redshift + 1)**4 # assuming uC prop to (z + 1)^4 as in KDA97


# find RAiSE constant K(s) for the absolute scaling of the emissivity
def __RAiSE_Ks(s_index, gammaCValue=4./3, lorentz_min=Lorentzmin):
    
    kappa = (gamma(s_index/4. + 19./12)*gamma(s_index/4. - 1./12)*gamma(s_index/4. + 5./4)/gamma(s_index/4. + 7./4))

    return kappa/(m_e**((s_index + 3)/2.)*c_speed*(s_index + 1))*(e_charge**2*mu0/(2*(gammaCValue - 1)))**((s_index + 5)/4.)*(3./np.pi)**(s_index/2.)/((lorentz_min**(2 - s_index) - Lorentzmax**(2 - s_index))/(s_index - 2) - (lorentz_min**(1 - s_index) - Lorentzmax**(1 - s_index))/(s_index - 1))
    

# find RAiSE blackbody constant to convert cosmic microwave background emission from single frequency to blackbody spectrum
def __RAiSE_blackbody(s_index):

    return np.pi**4/(15*gamma((s_index + 5)/2.)*zeta((s_index + 5)/2.))


## Define functions to produce surface brightness maps of radio lobes
# define function to manage the discretisation of particles down to pixels
def __RAiSE_brightness_map(frequency, redshift, source_age, lobe_lengths, location, luminosity, angle, resolution='standard'):
    
    # determine spatial resolution of particles; i.e. overdensity of particles to include in calculations
    if resolution == 'best':
        npixels = 2048/4
    elif resolution == 'high':
        npixels = 512/2
    elif resolution == 'standard':
        npixels = 128/1
    elif resolution == 'poor':
        npixels = 32*2
    else:
        raise Exception('Unrecognised keyword for particle resolution. The accepted keywords are: best, high, standard and poor.')
        
    # convert frequency, equipartition factor and spectral index to correct units
    if isinstance(frequency, (list, np.ndarray)):
        rest_frequency = np.zeros_like(frequency)
        for freqPointer in range(0, len(frequency)):
            rest_frequency[freqPointer] = 10**frequency[freqPointer]*(1 + redshift)
    else:
        rest_frequency = [10**frequency*(1 + redshift)]
    if isinstance(source_age, (list, np.ndarray)):
        tFinal = np.zeros_like(source_age)
        for i in range(0, len(source_age)):
            tFinal[i] = 10**source_age[i]*year # convert from log space years to seconds
    else:
        tFinal = [10**source_age*year]

    return __RAiSE_pixels(rest_frequency, redshift, tFinal, lobe_lengths, location, luminosity, angle, npixels)


# define function to discretise particles down to pixels
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_pixels(rest_frequency, redshift, tFinal, lobe_lengths, location, luminosity, angle, npixels):

    # instantiate variables to store brightness map variables
    x_list = []
    y_list = []
    brightness_list = []
    
    for i in range(0, len(tFinal)):
        x_col = []
        y_col = []
        brightness_col = []
        sim_x, sim_y, sim_z = location[i,:,0], location[i,:,1], location[i,:,2] # x, y, z (i.e. 0, 1, 2) in simulations
        for j in range(0, len(rest_frequency)):
            # separate location array into components
            index = np.logical_and(np.logical_and(np.logical_not(np.isnan(luminosity[i,:,j])), np.logical_not(np.isinf(luminosity[i,:,j]))), np.logical_not(np.isnan(sim_x)))
            location_x = np.sin(angle*np.pi/180.)*sim_y[index] + np.cos(angle*np.pi/180.)*sim_z[index]
            location_y = sim_x[index]
            new_luminosity = luminosity[i,:,j]
            new_luminosity = new_luminosity[index]
            
            if len(location_x > 0):
                # discretise particles
                location_x = np.floor(location_x/lobe_lengths[0,i]*(npixels//2)).astype(np.int_)
                location_y = np.floor(location_y/lobe_lengths[0,i]*(npixels//2)).astype(np.int_)
                min_x, min_y = np.min(location_x), np.min(location_y)
                location_x = location_x - min_x
                location_y = location_y - min_y
                
                # instantiate variables to store discrete particles
                x_values = np.arange(np.min(location_x), np.max(location_x) + 0.1, 1).astype(np.int_)
                y_values = np.arange(np.min(location_y), np.max(location_y) + 0.1, 1).astype(np.int_)
                brightness = np.zeros((len(x_values), len(y_values)))
                
                # add luminosity from each particle to correct pixel
                for k in range(0, len(new_luminosity)):
                    brightness[location_x[k],location_y[k]] = brightness[location_x[k],location_y[k]] + new_luminosity[k]
                
                # add x and y pixel values, and brightnesses to arrays
                x_col.append((x_values + min_x + 0.5)*lobe_lengths[0,i]/(npixels//2)) # add 0.5 to get pixel centres and scale back to physical dimensions
                y_col.append((y_values + min_y + 0.5)*lobe_lengths[0,i]/(npixels//2))
                brightness_col.append(brightness)
            else:
                x_col.append(None)
                y_col.append(None)
                brightness_col.append(None)
                
        x_list.append(x_col)
        y_list.append(y_col)
        brightness_list.append(brightness_col)
    
    return x_list, y_list, brightness_list


## SUPPLEMENTARY FUNCTIONS


# define function to apply a gaussian filter to existing (or newly created) brightness map; bmaj/bmin are the FWHM in arcsec, bpa is angle of 'major' axis anticlockwise from the x axis
def RAiSE_beam_convolver(filename, redshift, bmaj, bmin, bpa=0.):
    
    # read-in data from file (must be RAiSE output of correct format)
    dataframe = pd.read_csv(filename, index_col=0)
    # assign dataframe contents to variables
    x, y = (dataframe.index).astype(np.float_), (dataframe.columns).astype(np.float_)
    luminosity = dataframe.values
    
    # convert physical dimensions into angular sizes
    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    dL = (cosmo.luminosity_distance(redshift).to(u.kpc)).value
    x = x/(dL/(1 + redshift)**2)*(3600*180/np.pi) # arcsecs
    y = y/(dL/(1 + redshift)**2)*(3600*180/np.pi)
    
    # calculate beam size in pixels; assume that spacing in x and y may differ (though it should not)
    beam_x = np.sqrt((np.cos(bpa*np.pi/180.)*bmaj/(x[1] - x[0]))**2 + (np.sin(bpa*np.pi/180.)*bmaj/(y[1] - y[0]))**2)
    beam_y = np.sqrt((np.sin(bpa*np.pi/180.)*bmaj/(x[1] - x[0]))**2 + (-np.cos(bpa*np.pi/180.)*bmaj/(y[1] - y[0]))**2)

    # apply gaussian filter based on shape of beam
    gauss_kernel = Gaussian2DKernel(x_stddev=beam_x/(2*np.sqrt(2*np.log(2))), y_stddev=beam_y/(2*np.sqrt(2*np.log(2))), theta=bpa*np.pi/180.) # pixels
    surface_brightness = convolve(luminosity, gauss_kernel)
    # convert physical brightness into flux density
    surface_brightness = surface_brightness/(4*np.pi*(dL*const.kpc.value)**2)*1e26 * (np.pi*beam_x*beam_y/(4*np.log(2))) #convolve function already oper beam
    
    # write gaussian filtered maps to file
    df = pd.DataFrame(index=x, columns=y, data=surface_brightness)
    df.to_csv(os.path.splitext(filename)[0]+'_surf.csv', header=True, index=True)


## Define functions to plot brightness maps and LD tracks
# Define function to plot brightness map
def RAiSE_brightness_plot(filename):
    
    # read-in data from file (must be RAiSE output of correct format)
    dataframe = pd.read_csv(filename, index_col=0)
    # assign dataframe contents to variables
    x, y = (dataframe.index).astype(np.float_), (dataframe.columns).astype(np.float_)
    Y, X = np.meshgrid(y, x)
    Z = dataframe.values
    
    # set up plot
    fig, axs = plt.subplots(1, 1, figsize=(2.5*len(x)/float(len(y)), 2.5))
    fig.subplots_adjust(hspace=0)
    
    rc('text', usetex=True)
    rc('font', size=14.5)
    rc('legend', fontsize=14.5)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    cmap = cm.get_cmap('viridis')
    
    #warnings.filterwarnings('always')
    #warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    h = axs.pcolormesh(X, Y, Z, shading='nearest')
    axs.set_aspect('equal')
    cbar = plt.colorbar(h)
    
    # set axis labels and scale
    try:
        if (os.path.splitext(filename)[0])[-4:] == 'surf':
            axs.set_xlabel(r'Jet axis (arcsec)')
            axs.set_ylabel(r'Transverse axis (arcsec)')
            cbar.set_label(r'Flux density (Jy/beam)')
        else:
            axs.set_xlabel(r'Jet axis (kpc)')
            axs.set_ylabel(r'Transverse axis (kpc)')
            cbar.set_label(r'Luminosity (W/Hz/pixel)')
    except:
        axs.set_xlabel(r'Jet axis (kpc)')
        axs.set_ylabel(r'Transverse axis (kpc)')
        cbar.set_label(r'Luminosity (W/Hz/pixel)')

    x_crit = np.max(np.abs(x))
    y_crit = np.max(np.abs(y))
    axs.set_xlim([-x_crit, x_crit])
    axs.set_ylim([-y_crit, y_crit])
    
    axs.set_facecolor(cmap(0))

    plt.show()
    
# Define function to plot LD tracks
def RAiSE_evolution_plot(filename, seriesname=None, hydro_time=None, hydro_size=None, hydro_pressure=None, hydro_volume=None):
    
    # read-in data from file (must be RAiSE output of correct format)
    dataframe, series, time, size, pressure, volume = [], [], [], [], [], []
    if isinstance(filename, (list, np.ndarray)):
        for i in range(0, len(filename)):
            dataframe.append(pd.read_csv(filename[i], index_col=None))
            try:
                series.append(seriesname[i])
            except:
                series.append(None)
    else:
        dataframe.append(pd.read_csv(filename, index_col=None))
        series.append(seriesname)
    
    # assign dataframe contents to variables
    for i in range(0, len(dataframe)):
        time.append((dataframe[i].iloc[:,0]).astype(np.float_))
        size.append((dataframe[i].iloc[:,1]).astype(np.float_))
        pressure.append((dataframe[i].iloc[:,2]).astype(np.float_))

    # set up plot
    if not isinstance(hydro_pressure, (list, np.ndarray)) and hydro_pressure == None:
        fig, axs = plt.subplots(1, 1, figsize=(6, 6), sharex=True)
        axs = [axs]
    else:
        fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    fig.subplots_adjust(hspace=0)
    
    colour_scheme = ['crimson', 'darkorange', 'darkorchid', 'mediumblue']
    
    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    # add hydrosims to plot
    if isinstance(hydro_time, (list, np.ndarray)) and isinstance(hydro_size, (list, np.ndarray)):
        if isinstance(hydro_time[0], (list, np.ndarray)) and isinstance(hydro_size[0], (list, np.ndarray)):
            for i in range(0, len(hydro_time)):
                axs[0].plot(hydro_time[i]/1e6, hydro_size[i]/2, color=colour_scheme[i%4], linewidth=3.5, alpha=0.3)
                if isinstance(hydro_pressure, (list, np.ndarray)):
                    axs[1].plot(hydro_time[i]/1e6, hydro_pressure[i], color=colour_scheme[i%4], linewidth=3.5, alpha=0.3)
        else:
            axs[0].plot(hydro_time/1e6, hydro_size/2, color=colour_scheme[0], linewidth=3.5, alpha=0.3)
            if isinstance(hydro_pressure, (list, np.ndarray)):
                axs[1].plot(hydro_time/1e6, hydro_pressure, color=colour_scheme[i%4], linewidth=3.5, alpha=0.3)
    
    # add RAiSE HD models to plot
    for i in range(0, len(dataframe)):
        axs[0].plot(time[i]/1e6, size[i]/2, colour_scheme[i%4], linewidth=1.25, label=series[i])
        if isinstance(hydro_pressure, (list, np.ndarray)):
            axs[1].plot(time[i]/1e6, pressure[i], colour_scheme[i%4], linewidth=1.25, label=series[i])
            axs[1].set_xlabel(r'Source age (Myr)')
            axs[1].set_ylabel(r'Pressure (Pa)')
        else:
            axs[0].set_xlabel(r'Source age (Myr)')
        axs[0].set_ylabel(r'Lobe length (kpc)')
    
    # set axes limits
    x_min = np.min(np.abs(np.asarray(time)/1e6))
    y_min = np.min(np.abs(np.asarray(size)/2))
    x_max = np.max(np.abs(np.asarray(time)/1e6))
    y_max = np.max(np.abs(np.asarray(size)/2))
    axs[0].set_xlim([x_min, x_max])
    axs[0].set_ylim([y_min, y_max])
    
    # set nicely labelled log axes
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    if isinstance(hydro_pressure, (list, np.ndarray)):
        axs[1].set_yscale('log')
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%g'))
    axs[0].xaxis.set_minor_formatter(NullFormatter())
    if x_max/x_min < 10:
        axs[0].xaxis.set_major_locator(LogLocator(base=10, subs=[1,2,3,5]))
        axs[0].xaxis.set_minor_locator(LogLocator(base=10, subs=range(100)))
    elif x_max/x_min < 100:
        axs[0].xaxis.set_major_locator(LogLocator(base=10, subs=[1,3]))
        axs[0].xaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
    else:
        axs[0].xaxis.set_major_locator(LogLocator(base=10, subs=[1]))
        axs[0].xaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%g'))
    axs[0].yaxis.set_minor_formatter(NullFormatter())
    if y_max/y_min < 10:
        axs[0].yaxis.set_major_locator(LogLocator(base=10, subs=[1,2,3,5]))
        axs[0].yaxis.set_minor_locator(LogLocator(base=10, subs=range(100)))
    elif y_max/y_min < 100:
        axs[0].yaxis.set_major_locator(LogLocator(base=10, subs=[1,3]))
        axs[0].yaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
    else:
        axs[0].yaxis.set_major_locator(LogLocator(base=10, subs=[1]))
        axs[0].yaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
    
    # add legend
    if not seriesname == None:
        axs[0].legend()

    plt.show()
    
    
# Define functions to compare dynamics of analytic model to hydrodynamical simulations
def __PLUTO_sim_A():
    # Model A simulation
    # input variables
    jet_power = 38.8
    rho0Value = 2.40921860484895E-24
    temperature = 3.46e7
    regions = np.array([0.1, 5.09949995, 10.0989999, 15.09849985, 20.0979998, 25.09749975, 30.0969997, 35.09649965, 40.0959996, 45.09549955, 50.0949995, 55.09449945, 60.0939994, 65.09349935, 70.0929993, 75.09249925, 80.0919992, 85.09149915, 90.0909991, 95.09049905, 100.089999, 105.0894989, 110.0889989, 115.0884988, 120.0879988, 125.0874987, 130.0869987, 135.0864986, 140.0859986, 145.0854985, 150.0849985, 155.0844984, 160.0839984, 165.0834983, 170.0829983, 175.0824982, 180.0819982, 185.0814981, 190.0809981, 195.080498, 200.079998, 205.0794979, 210.0789979, 215.0784978, 220.0779978, 225.0774977, 230.0769977, 235.0764976, 240.0759976, 245.0754975, 250.0749975, 255.0744974, 260.0739974, 265.0734973, 270.0729973, 275.0724972, 280.0719972, 285.0714971, 290.0709971, 295.070497, 300.069997, 305.0694969, 310.0689969, 315.0684968, 320.0679968, 325.0674967, 330.0669967, 335.0664966, 340.0659966, 345.0654965, 350.0649965, 355.0644964, 360.0639964, 365.0634963, 370.0629963, 375.0624962, 380.0619962, 385.0614961, 390.0609961, 395.060496, 400.059996, 405.0594959, 410.0589959, 415.0584958, 420.0579958, 425.0574957, 430.0569957, 435.0564956, 440.0559956, 445.0554955, 450.0549955, 455.0544954, 460.0539954, 465.0534953, 470.0529953, 475.0524952, 480.0519952, 485.0514951, 490.0509951, 495.050495])*const.kpc.value
    betas = [5.39541E-06, 0.00157771, 0.00588552, 0.012867226, 0.022424118, 0.034424032, 0.0487058, 0.065084443, 0.083356817, 0.103307401, 0.124713959, 0.147352834, 0.171003696, 0.19545361, 0.220500362, 0.245955018, 0.271643754, 0.297408991, 0.323109938, 0.348622621, 0.373839496, 0.398668737, 0.423033298, 0.446869801, 0.470127346, 0.492766272, 0.514756921, 0.536078449, 0.556717678, 0.576668043, 0.595928608, 0.614503183, 0.632399521, 0.649628613, 0.666204063, 0.682141537, 0.697458296, 0.712172784, 0.726304283, 0.739872618, 0.75289791, 0.76540037, 0.777400133, 0.788917113, 0.799970897, 0.810580656, 0.820765074, 0.830542299, 0.839929906, 0.848944865, 0.857603535, 0.865921645, 0.873914301, 0.881595986, 0.888980569, 0.896081318, 0.90291091, 0.909481455, 0.915804506, 0.921891086, 0.927751698, 0.933396355, 0.938834593, 0.944075494, 0.949127707, 0.953999464, 0.958698603, 0.963232585, 0.967608508, 0.971833134, 0.975912894, 0.979853915, 0.983662026, 0.987342779, 0.990901462, 0.994343109, 0.997672517, 1.000894256, 1.004012679, 1.007031938, 1.009955988, 1.012788603, 1.015533381, 1.018193754, 1.020772998, 1.023274238, 1.02570046, 1.028054514, 1.03033912, 1.032556882, 1.034710283, 1.0368017, 1.038833405, 1.040807571, 1.042726278, 1.044591513, 1.046405183, 1.04816911, 1.049885041, 1.051554649]
    
    # output dynamics
    time = np.array([ 0.99995208,  1.99997134,  2.99993711,  3.99991659,  4.99997727, 5.99992706,  6.99998449,  7.99992449,  8.99991342,  9.99999368, 10.99993043, 11.99994219, 12.9999833 , 13.99992657, 14.9999579 , 15.99999576, 16.99998143, 17.99992796, 18.99995277, 19.99993191, 20.99997955, 21.99995544, 22.99994437, 23.99993982, 24.99998746, 25.99993399, 26.99996206, 27.99994121, 28.99996928, 29.99999082, 30.99997649, 31.99995238])*1e6
    size = np.array([ 82.80000305, 109.79999542, 129.6000061 , 142.19999695, 154.79998779, 163.79998779, 171.        , 180.        , 189.        , 196.20001221, 203.3999939 , 212.3999939 , 217.79998779, 225.        , 230.3999939 , 237.6000061 , 244.79998779, 252.        , 257.3999939 , 261.        , 266.3999939 , 273.6000061 , 277.19998169, 282.6000061 , 289.80001831, 293.3999939 , 297.        , 304.19998169, 307.80001831, 313.19998169, 318.6000061 , 320.3999939 ])
    axis_ratio = [32e6, 2.83]
    open_angle = 10.
    
    time_fit = np.arange(4.5, 7.50001, 0.05)
    size_fit = [1.57302243654115, 0.51365606050089, -0.22218253972033, 0.07952252535386, 0.12956323278409, -0.01833328248534, -0.18278214996110, 0.00356151437991, 0.15391348327574, -0.00686612824889, -0.05335183342284, 0.00184766238885, 0.00638876158595]
    pressure_fit = [-10.39207947622930, -0.27075123982130, -1.25453805273312, 0.77519589642022, 1.61776953948815, -1.39872049557852, -1.26408572246346, 0.93122863720828, 0.66334035452180, -0.29981398756446, -0.19629404650729, 0.03827473545327, 0.02387878059118]
    volume_fit = [3.87598507538180, 1.47939034335789, 0.42018655082979, -0.11765495191812, -0.99477548151592, 0.07720270143391, 1.33569247754429, -0.12209041655205, -0.86531713538419, 0.07919527994336, 0.26135525059559, -0.01534599197152, -0.02983083017576]

    return jet_power, rho0Value, temperature, regions, betas, time, size, axis_ratio, open_angle, time_fit, size_fit, pressure_fit, volume_fit

def __PLUTO_sim_B():
    # Model B simulation
    # input variables
    jet_power = 38.8
    rho0Value = 1.62224669316853E-24
    temperature = 3.46e7
    regions = np.array([0.1, 5.09949995, 10.0989999, 15.09849985, 20.0979998, 25.09749975, 30.0969997, 35.09649965, 40.0959996, 45.09549955, 50.0949995, 55.09449945, 60.0939994, 65.09349935, 70.0929993, 75.09249925, 80.0919992, 85.09149915, 90.0909991, 95.09049905, 100.089999, 105.0894989, 110.0889989, 115.0884988, 120.0879988, 125.0874987, 130.0869987, 135.0864986, 140.0859986, 145.0854985, 150.0849985, 155.0844984, 160.0839984, 165.0834983, 170.0829983, 175.0824982, 180.0819982, 185.0814981, 190.0809981, 195.080498, 200.079998, 205.0794979, 210.0789979, 215.0784978, 220.0779978, 225.0774977, 230.0769977, 235.0764976, 240.0759976, 245.0754975, 250.0749975, 255.0744974, 260.0739974, 265.0734973, 270.0729973, 275.0724972, 280.0719972, 285.0714971, 290.0709971, 295.070497, 300.069997, 305.0694969, 310.0689969, 315.0684968, 320.0679968, 325.0674967, 330.0669967, 335.0664966, 340.0659966, 345.0654965, 350.0649965, 355.0644964, 360.0639964, 365.0634963, 370.0629963, 375.0624962, 380.0619962, 385.0614961, 390.0609961, 395.060496, 400.059996, 405.0594959, 410.0589959, 415.0584958, 420.0579958, 425.0574957, 430.0569957, 435.0564956, 440.0559956, 445.0554955, 450.0549955, 455.0544954, 460.0539954, 465.0534953, 470.0529953, 475.0524952, 480.0519952, 485.0514951, 490.0509951, 495.050495])*const.kpc.value
    betas = [0.001107261, 0.021197709, 0.040958797, 0.060586688, 0.080035719, 0.099264847, 0.118239207, 0.136929742, 0.155312631, 0.173368694, 0.19108285, 0.208443608, 0.225442608, 0.242074212, 0.258335132, 0.274224101, 0.28974158, 0.3048895, 0.319671032, 0.334090388, 0.348152642, 0.361863577, 0.375229553, 0.388257385, 0.400954246, 0.413327577, 0.425385012, 0.437134313, 0.448583316, 0.459739879, 0.470611849, 0.481207024, 0.491533124, 0.50159777, 0.511408463, 0.520972569, 0.530297306, 0.539389737, 0.548256756, 0.55690509, 0.565341289, 0.573571727, 0.5816026, 0.589439926, 0.597089546, 0.604557124, 0.611848153, 0.618967951, 0.625921673, 0.632714304, 0.639350673, 0.64583545, 0.652173151, 0.658368144, 0.664424655, 0.670346764, 0.676138421, 0.681803439, 0.687345505, 0.692768184, 0.698074918, 0.703269037, 0.708353757, 0.713332186, 0.71820733, 0.722982091, 0.727659278, 0.732241605, 0.736731693, 0.741132081, 0.74544522, 0.749673482, 0.753819162, 0.757884477, 0.761871574, 0.765782529, 0.769619352, 0.773383988, 0.777078318, 0.780704164, 0.78426329, 0.787757404, 0.79118816, 0.794557159, 0.797865954, 0.801116048, 0.804308898, 0.807445917, 0.810528472, 0.813557892, 0.816535462, 0.81946243, 0.822340006, 0.825169365, 0.827951645, 0.830687952, 0.833379358, 0.836026905, 0.838631604, 0.841194435]
    
    # output dynamics
    time = np.array([ 0.99996186,  1.99994753,  2.99992374,  3.99994594,  4.99999684, 5.99998903,  6.99993882,  7.99995059,  8.99997213,  9.99996433, 10.99997935, 11.99996502, 12.99995395, 13.99997876, 14.9999155, 15.99995336, 16.99998469, 17.99997362, 18.99999843, 19.99999388, 20.99994041, 21.99999458, 22.99999655, 23.99994308, 24.99992549, 25.99998944, 26.99991966, 27.99995425, 28.99999211, 29.99998756, 30.9999504 , 31.99997195, 32.99998045, 33.99994329, 35.00000397, 35.99999943])*1e6
    size = np.array([ 91.80000305, 117.        , 136.79998779, 151.20001221, 162.        , 172.79998779, 181.80000305, 194.3999939 , 201.6000061 , 210.6000061 , 216.        , 223.20001221, 232.20001221, 237.6000061 , 244.80000305, 252.        , 259.20001221, 266.40002441, 270.        , 273.6000061 , 280.80001831, 286.20001221, 289.79998779, 297.        , 302.40002441, 306.        , 311.3999939 , 315.        , 320.3999939 , 325.80001831, 329.3999939 , 334.79998779, 340.20001221, 343.79998779, 349.20001221, 352.79998779])
    axis_ratio = [32e6, 2.55]
    open_angle = 10.
    
    return jet_power, rho0Value, temperature, regions, betas, time, size, axis_ratio, open_angle, None, None, None, None

def RAiSE_dynamics_test():
    
    # hydrodynamical simulation data
    jet_power_A, rho0Value_A, temperature_A, regions_A, betas_A, time_A, size_A, axis_ratio_A, open_angle_A, time_fit, size_fit, pressure_fit, volume_fit = __PLUTO_sim_A()

    source_age = time_fit
    hydro_size = np.zeros_like(time_fit)
    hydro_pressure = np.zeros_like(time_fit)
    for l in range(0, len(size_fit)):
        hydro_size = hydro_size + size_fit[l]*(time_fit - 6)**l
    for l in range(0, len(pressure_fit)):
        hydro_pressure = hydro_pressure + pressure_fit[l]*(time_fit - 6)**l
                    
    RAiSE_run(9.146, 0.05, axis_ratio_A[1] - 0.2, jet_power_A, source_age, spectral_index=0.7, rho0Value=rho0Value_A, equipartition=-1.5, regions=regions_A, betas=betas_A, temperature=3.46e7, resolution=None, brightness=False)
    
    filename_A = 'LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}.csv'.format(axis_ratio_A[1] - 0.2, 1.5, np.abs(np.log10(rho0Value_A)), jet_power_A, 2*0.7 + 1, 10.14, 5., 0.05)
    
    RAiSE_evolution_plot([filename_A], seriesname=['Model A'], hydro_time=10**time_fit, hydro_size=2*10**hydro_size, hydro_pressure=10**hydro_pressure, hydro_volume=None)
    
def RAiSE_dynamics_test2():
    
    # hydrodynamical simulation data
    jet_power_A, rho0Value_A, temperature_A, regions_A, betas_A, time_A, size_A, axis_ratio_A, open_angle_A, n, n, n, n = __PLUTO_sim_A()
    jet_power_B, rho0Value_B, temperature_B, regions_B, betas_B, time_B, size_B, axis_ratio_B, open_angle_B, n, n, n, n = __PLUTO_sim_B()

    source_age = [5.9, 5.95, 6, 6.05, 6.1, 6.15, 6.2, 6.25, 6.3, 6.35, 6.4, 6.45, 6.5, 6.55, 6.6, 6.65, 6.7, 6.75, 6.8, 6.85, 6.9, 6.95, 7, 7.05, 7.1, 7.15, 7.2, 7.25, 7.3, 7.35, 7.4, 7.45, 7.5, 7.55, 7.6]

    RAiSE_run(9.146, 0.05, axis_ratio_A[1] - 0.2, jet_power_A, source_age, spectral_index=0.7, rho0Value=rho0Value_A, equipartition=-1.5, regions=regions_A, betas=betas_A, temperature=3.46e7, resolution=None, brightness=False)
    RAiSE_run(9.146, 0.05, axis_ratio_B[1] - 0.2, jet_power_B, source_age, spectral_index=0.7, rho0Value=rho0Value_B, equipartition=-1.5, regions=regions_B, betas=betas_B, temperature=3.46e7, resolution=None, brightness=False)
    
    filename_A = 'LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}.csv'.format(axis_ratio_A[1] - 0.2, 1.5, np.abs(np.log10(rho0Value_A)), jet_power_A, 2*0.7 + 1, 10.14, 5., 0.05)
    filename_B = 'LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}.csv'.format(axis_ratio_B[1] - 0.2, 1.5, np.abs(np.log10(rho0Value_B)), jet_power_B, 2*0.7 + 1, 10.14, 5., 0.05)
    
    RAiSE_evolution_plot([filename_A, filename_B], seriesname=['Model A', 'Model B'], hydro_time=[time_A, time_B], hydro_size=[size_A, size_B], hydro_pressure=None, hydro_volume=None)
    
def RAiSE_parameter_calib(filename=None):
    
    # define number of hydrosims
    nhydrosims = 1
    # define grid in aj_star and crit_mach
    aj_star = np.arange(0.5, 100, 5)
    crit_mach = np.arange(0.0375, 1.5, 0.25)

    # define array to store results
    residuals = np.zeros((len(aj_star), len(crit_mach)))
    jet_angles = np.zeros((len(aj_star), len(crit_mach)))
    
    for i in range(0, len(aj_star)):
        for j in range(0, len(crit_mach)):
            # find best fit initial axis ratio for current set of parameters
            jet_angle = 0
            count = 0
            for k in range(0, nhydrosims):
                if k == 0:
                    # this simulation is used to set relationship between opening angle of jet and initial axis ratio
                    jet_power, rho0Value, temperature, regions, betas, time, size, axis_ratio, open_angle, time_fit, size_fit, pressure_fit, volume_fit = __PLUTO_sim_A()
                else:
                    pass # add extra simulations
                
                # find best fit initial axis ratio
                result = least_squares(__axis_ratio_residuals, x0=(axis_ratio[1]), args=(axis_ratio[1], jet_power, np.log10(axis_ratio[0]), rho0Value, regions, betas, temperature, aj_star[i], crit_mach[j], jet_angle, open_angle), bounds=(1, axis_ratio[1]), ftol=1e-6)
                axis_ratio = result.x[0]
                if k == 0:
                    jet_angle = open_angle*(axis_ratio/4.)
                
                # run RAiSE code for best fit initial axis ratio
                RAiSE_run(9.146, 0.05, axis_ratio, jet_power, time_fit, spectral_index=0.7, rho0Value=rho0Value, equipartition=-1.5, regions=regions, betas=betas, temperature=temperature, resolution=None, brightness=False, aj_star=aj_star[i], crit_mach=crit_mach[j], jet_angle=jet_angle)
                
                # obtain present axis ratio from output file
                RAiSE_filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}.csv'.format(axis_ratio, 1.5, np.abs(np.log10(rho0Value)), jet_power, 2*0.7 + 1, 10.14, 5., 0.05)
                df = pd.read_csv(RAiSE_filename)
                RAiSE_size = df['Size (kpc)'].values # compare source size, not lobe size
                RAiSE_pressure = df['Pressure (Pa)'].values # compare source size, not lobe size
                hydro_size = np.zeros_like(time_fit)
                hydro_pressure = np.zeros_like(time_fit)
                for l in range(0, len(size_fit)):
                    hydro_size = hydro_size + size_fit[l]*(time_fit - 6)**l
                for l in range(0, len(pressure_fit)):
                    hydro_pressure = hydro_pressure + pressure_fit[l]*(time_fit - 6)**l
                
                count = count + 2*len(np.asarray(time_fit))
                residuals[i,j] = residuals[i,j] + np.nansum((np.log10(RAiSE_size) - hydro_size)**2) + np.nansum((np.log10(RAiSE_pressure) - hydro_pressure)**2)
                if k == 0:
                    jet_angles[i,j] = jet_angle
            
            residuals[i,j] = residuals[i,j]/count
            
    df = pd.DataFrame(data=residuals, index=aj_star, columns=crit_mach)
    dg = pd.DataFrame(data=jet_angles, index=aj_star, columns=crit_mach)
    if not filename == None:
        df.to_csv(filename+'-residuals.csv')
        dg.to_csv(filename+'-angles.csv')
    else:
        df.to_csv('aj_star-crit_mach-residuals.csv')
        dg.to_csv('aj_star-crit_mach-angles.csv')

def __axis_ratio_residuals(params, axis_ratio, jet_power, source_age, rho0Value, regions, betas, temperature, aj_star, crit_mach, jet_angle, open_angle):

    # run RAiSE code for trial initial axis ratio
    if jet_angle <= 0:
        RAiSE_run(9.146, 0.05, params[0], jet_power, source_age, spectral_index=0.7, rho0Value=rho0Value, equipartition=-1.5, regions=regions, betas=betas, temperature=temperature, resolution=None, brightness=False, aj_star=aj_star, crit_mach=crit_mach, jet_angle=open_angle*(params[0]/4.))
    else:
        RAiSE_run(9.146, 0.05, params[0], jet_power, source_age, spectral_index=0.7, rho0Value=rho0Value, equipartition=-1.5, regions=regions, betas=betas, temperature=temperature, resolution=None, brightness=False, aj_star=aj_star, crit_mach=crit_mach, jet_angle=jet_angle)
    
    # obtain present axis ratio from output file
    filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}.csv'.format(axis_ratio, 1.5, np.abs(np.log10(rho0Value)), jet_power, 2*0.7 + 1, 10.14, 5., 0.05)
    df = pd.read_csv(filename)
    
    return (df['Axis Ratio'][0] - axis_ratio)


def RAiSE_parameter_plot(filename=None):

    # read-in data from file (must be RAiSE output of correct format)
    if filename == None:
        df = pd.read_csv('aj_star-crit_mach-residuals.csv', index_col=0, header=0)
    else:
        df = pd.read_csv(filename+'-residuals.csv', index_col=0, header=0)
    
    # assign dataframe contents to variables
    aj_star = np.array(df.index)
    crit_mach = np.array([np.float(x) for x in df.columns])
    residuals = df.to_numpy()
    
    # set up plot
    fig, axs = plt.subplots(1, 1, figsize=(6, 5.5), sharex=True)
    fig.subplots_adjust(hspace=0)
    
    colour_scheme = ['crimson', 'darkorange', 'darkorchid', 'mediumblue']
    
    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    # add contours to plot
    c = axs.pcolormesh(aj_star, crit_mach, np.transpose(residuals), cmap='Blues_r', shading='nearest', norm=LogNorm())
    
    axs.set_xlabel(r'$a_{j*}$', fontsize=14.5)
    axs.set_ylabel(r'$\mathcal{M}_{s,crit}$', fontsize=14.5)
    axs.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    axs.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    #axs.set_xlim([np.min(aj_star) - (aj_star[1] - aj_star[0])/2., np.max(aj_star) + (aj_star[1] - aj_star[0])/2.])
    #axs.set_ylim([np.min(crit_mach) - (crit_mach[1] - crit_mach[0])/2., np.max(crit_mach) + (crit_mach[1] - crit_mach[0])/2.])
    axs.set_xlim([0,100])
    axs.set_ylim([0.025,1.5])
    
    cb = plt.colorbar(c, ax=axs, pad=0.025)
    cb.set_label(r'RSS/$n$', fontsize=14)
    #cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))

    plt.show()
    

# Define functions to plot emissivity maps throughout source evolutionary history
def RAiSE_evolution_maps(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass=None, rand_profile=False, betas=None, regions=None, rho0Value=None, temperature=None, active_age=10.14, jet_lorentz=5., equipartition=-1.5, spectral_index=0.7, gammaCValue=4./3, lorentz_min=Lorentzmin, resolution='standard', seed=None, surface_brightness=True, rerun=False):
    
    # function to test type of inputs and convert type where appropriate
    frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz, nenvirons = __test_inputs(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz)
    
    # set up plot
    fig, axs = plt.subplots(len(source_age), 1, figsize=(12, 1 + (10/axis_ratio[0] + 0.8)*len(source_age)))
    if len(source_age) <= 1: # handle case of single image
        axs = [axs]
    fig.subplots_adjust(hspace=0)
    
    cmap = cm.get_cmap('binary')
    
    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    for i in range(0, len(source_age)):
        if isinstance(rho0Value, (list, np.ndarray)):
            filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}'.format(axis_ratio[0], np.abs(equipartition[0]), np.abs(np.log10(rho0Value[0])), jet_power[0], 2*np.abs(spectral_index) + 1, active_age[0], jet_lorentz[0], redshift[0], frequency[0], source_age[i])
        elif isinstance(halo_mass, (list, np.ndarray)):
            filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}'.format(axis_ratio[0], np.abs(equipartition[0]), halo_mass[0], jet_power[0], 2*np.abs(spectral_index) + 1, active_age[0], jet_lorentz[0], redshift[0], frequency[0], source_age[i])
        
        # read-in data from file (must be RAiSE output of correct format)
        if rerun == False:
            try:
                dataframe = pd.read_csv(filename+'_best.csv', index_col=0)
            except:
                # run RAiSE HD for set of parameters at requested resolution
                RAiSE_run(frequency[0], redshift[0], axis_ratio[0], jet_power[0], source_age[i], halo_mass=halo_mass, rand_profile=rand_profile, betas=betas, regions=regions, rho0Value=rho0Value, temperature=temperature, active_age=active_age[0], jet_lorentz=jet_lorentz[0], equipartition=equipartition[0], spectral_index=spectral_index, gammaCValue=gammaCValue, lorentz_min=Lorentzmin, brightness=True, resolution='best', seed=seed)
                dataframe = pd.read_csv(filename+'_best.csv', index_col=0)

        else:
            # run RAiSE HD for set of parameters at requested resolution
            RAiSE_run(frequency[0], redshift[0], axis_ratio[0], jet_power[0], source_age[i], halo_mass=halo_mass, rand_profile=rand_profile, betas=betas, regions=regions, rho0Value=rho0Value, temperature=temperature, active_age=active_age[0], jet_lorentz=jet_lorentz[0], equipartition=equipartition[0], spectral_index=spectral_index, gammaCValue=gammaCValue, lorentz_min=Lorentzmin, brightness=True, resolution='best', seed=seed)
            dataframe = pd.read_csv(filename+'_best.csv', index_col=0)
            
        # assign dataframe contents to variables
        x, y = (dataframe.index).astype(np.float_), (dataframe.columns).astype(np.float_)
        #x, y = x/np.max(x), y/np.max(x)
        Y, X = np.meshgrid(y, x)
        Z = dataframe.values
        Z = Z/np.max(Z)
        
        if surface_brightness == True:
            # angular size of beam
            if 10**frequency[0]*(1 + redshift[0]) > 1e12:
                beam = 15*(2.41e17/10**frequency[0])
            else:
                beam = 6*(1.5e8/10**frequency[0])

            # convert physical dimensions into angular sizes
            cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
            dL = (cosmo.luminosity_distance(redshift[0]).to(u.kpc)).value
            size = np.max(x)/(dL/(1 + redshift[0])**2)*(3600*180/np.pi) # arcsecs
            
            # ensure at least one beam per lobe
            if size > 2*beam:
                if rerun == False:
                    try:
                        convframe = pd.read_csv(filename+'_'+resolution+'_surf.csv', index_col=0)
                    except:
                        # run RAiSE HD for set of parameters at requested resolution
                        RAiSE_run(frequency[0], redshift[0], axis_ratio[0], jet_power[0], source_age[i], halo_mass=halo_mass, rand_profile=rand_profile, betas=betas, regions=regions, rho0Value=rho0Value, temperature=temperature, active_age=active_age[0], jet_lorentz=jet_lorentz[0], equipartition=equipartition[0], spectral_index=spectral_index, gammaCValue=gammaCValue, lorentz_min=Lorentzmin, brightness=True, resolution=resolution, seed=seed)
                        # convolve image with 1 arcsec^2 (lambda/lambda_1.5)^2 beam with 10 uJy sensitivity
                        RAiSE_beam_convolver(filename+'_'+resolution+'.csv', redshift[0], beam, beam, bpa=0.)
                        convframe = pd.read_csv(filename+'_'+resolution+'_surf.csv', index_col=0)
                else:
                    # run RAiSE HD for set of parameters at requested resolution
                    RAiSE_run(frequency[0], redshift[0], axis_ratio[0], jet_power[0], source_age[i], halo_mass=halo_mass, rand_profile=rand_profile, betas=betas, regions=regions, rho0Value=rho0Value, temperature=temperature, active_age=active_age[0], jet_lorentz=jet_lorentz[0], equipartition=equipartition[0], spectral_index=spectral_index, gammaCValue=gammaCValue, lorentz_min=Lorentzmin, brightness=True, resolution=resolution, seed=seed)
                    # convolve image with 1 arcsec^2 (lambda/lambda_1.5)^2 beam with 10 uJy sensitivity
                    RAiSE_beam_convolver(filename+'_'+resolution+'.csv', redshift[0], beam, beam, bpa=0.)
                    convframe = pd.read_csv(filename+'_'+resolution+'_surf.csv', index_col=0)
                    
                # assign dataframe contents to variables
                xc, yc = (convframe.index).astype(np.float_), (convframe.columns).astype(np.float_)
                xc, yc = xc*(dL/(1 + redshift[0])**2)/(3600*180/np.pi), yc*(dL/(1 + redshift[0])**2)/(3600*180/np.pi)
                Yc, Xc = np.meshgrid(yc, xc)
                Zc = convframe.values
    
        h = axs[i].pcolormesh(X, Y, Z, shading='nearest', cmap=cmap)
        #if surface_brightness == True and size > 2*beam:
        #    if 10**frequency[0]*(1 + redshift[0]) > 1e12:
        #        c = axs[i].contour(Xc, Yc, Zc, levels=14e-9*np.array([1, np.sqrt(2)*1, 2, np.sqrt(2)*2, 4, np.sqrt(2)*4, 8, np.sqrt(2)*8, 16, np.sqrt(2)*16, 32, np.sqrt(2)*32, 64, np.sqrt(2)*64, 128, np.sqrt(2)*128, 256, np.sqrt(2)*256, 512, np.sqrt(2)*512]), colors='crimson', linewidths=0.5)
        #    else:
        #        c = axs[i].contour(Xc, Yc, Zc, levels=71e-6*np.array([1, np.sqrt(2)*1, 2, np.sqrt(2)*2, 4, np.sqrt(2)*4, 8, np.sqrt(2)*8, 16, np.sqrt(2)*16, 32, np.sqrt(2)*32, 64, np.sqrt(2)*64, 128, np.sqrt(2)*128, 256, np.sqrt(2)*256, 512, np.sqrt(2)*512]), colors='crimson', linewidths=0.5)
                
        axs[i].set_aspect('equal')
        axs[i].set_xlim([-1.05*np.max(x), 1.05*np.max(x)])
        axs[i].set_ylim([-1.05*np.max(x)/axis_ratio, 1.05*np.max(x)/axis_ratio])
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%g'))
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%g'))
        
        axs[i].plot(np.NaN, np.NaN, '-', color='none', label=str('{:g}'.format(float('{:.2g}'.format(10**source_age[i]/1e6))))+' Myr')
        axs[i].legend(frameon=False)
    
    # add a big axes for labels, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)

    plt.xlabel(r'Jet axis (kpc)', fontsize=14.5, labelpad=-5)
    plt.ylabel(r'Transverse axis (kpc)', fontsize=14.5, labelpad=25)

    # show plot and return handle to plot
    plt.show()
    return fig


# Define function to plot Dt and LD tracks
def RAiSE_evolution_tracks(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass=None, rand_profile=False, betas=None, regions=None, rho0Value=None, temperature=None, active_age=10.14, jet_lorentz=5., equipartition=-1.5, spectral_index=0.7, gammaCValue=4./3, lorentz_min=Lorentzmin, resolution='standard', seed=None, rerun=False, labels=None, colors=None, linestyles=None):

    # function to test type of inputs and convert type where appropriate
    frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz, nenvirons = __test_inputs(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz)
    
    if len(source_age) <= 1:
        raise Exception('Evolutionary tracks require more than one source age; provide a list/array of ages.')
    if len(frequency) > 1:
        warnings.warn('First frequency in list/array will be plotted for every set of parameters.', category=UserWarning)
    if not isinstance(colors, (list, np.ndarray)) and not colors == None:
        colors = [colors]
    elif colors == None:
        colors = ['crimson', 'darkorange', 'darkorchid', 'mediumblue']
    if not isinstance(linestyles, (list, np.ndarray)) and not linestyles == None:
        linestyles = [linestyles]
    elif linestyles == None:
        linestyles = ['-']
    
    # set up plot
    if resolution == None:
        fig, axs = plt.subplots(2, 1, figsize=(6, 10), sharex=True)
    else:
        fig, axs = plt.subplots(3, 1, figsize=(6, 14), sharex=True)
        fig2, axs2 = plt.subplots(1, 1, figsize=(6, 6))
    fig.subplots_adjust(hspace=0)
    
    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    if resolution == None:
        axs[1].set_xlabel(r'Source age (Myr)')
    else:
        axs[2].set_xlabel(r'Source age (Myr)')
        axs[2].set_ylabel(r'Lobe luminosity (W/Hz)')
        axs2.set_xlabel(r'Lobe length (kpc)')
        axs2.set_ylabel(r'Lobe luminosity (W/Hz)')
    axs[0].set_ylabel(r'Lobe length (kpc)')
    axs[1].set_ylabel(r'Pressure (Pa)')

    # calculate number of plots
    nplots = np.max(np.array([len(redshift), len(axis_ratio), len(jet_power), nenvirons, len(active_age), len(equipartition), len(jet_lorentz)]))
    time, size, pressure, luminosity, y_min, y_max = [], [], [], [], [], []
    
    for i in range(0, nplots):
        if isinstance(rho0Value, (list, np.ndarray)):
            filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}'.format(axis_ratio[min(len(axis_ratio) - 1, i)], np.abs(equipartition[min(len(equipartition) - 1, i)]), np.abs(np.log10(rho0Value[min(len(rho0Value) - 1, i)])), jet_power[min(len(jet_power) - 1, i)], 2*np.abs(spectral_index) + 1, active_age[min(len(active_age) - 1, i)], jet_lorentz[min(len(jet_lorentz) - 1, i)], redshift[min(len(redshift) - 1, i)])
        elif isinstance(halo_mass, (list, np.ndarray)):
            filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}'.format(axis_ratio[min(len(axis_ratio) - 1, i)], np.abs(equipartition[min(len(equipartition) - 1, i)]), halo_mass[min(len(halo_mass) - 1, i)], jet_power[min(len(jet_power) - 1, i)], 2*np.abs(spectral_index) + 1, active_age[min(len(active_age) - 1, i)], jet_lorentz[min(len(jet_lorentz) - 1, i)], redshift[min(len(redshift) - 1, i)])
        
        # read-in data from file (must be RAiSE output of correct format)
        if rerun == False:
            try:
                dataframe = pd.read_csv(filename+'.csv', index_col=None)
            except:
                # run RAiSE HD for set of parameters at requested resolution
                RAiSE_run(frequency, redshift[min(len(redshift) - 1, i)], axis_ratio[min(len(axis_ratio) - 1, i)], jet_power[min(len(jet_power) - 1, i)], source_age, halo_mass=halo_mass, rand_profile=rand_profile, betas=betas, regions=regions, rho0Value=rho0Value, temperature=temperature, active_age=active_age[min(len(active_age) - 1, i)], jet_lorentz=jet_lorentz[min(len(jet_lorentz) - 1, i)], equipartition=equipartition[min(len(equipartition) - 1, i)], spectral_index=spectral_index, gammaCValue=gammaCValue, lorentz_min=Lorentzmin, brightness=False, resolution=resolution, seed=seed)
                dataframe = pd.read_csv(filename+'.csv', index_col=None)
        else:
            # run RAiSE HD for set of parameters at requested resolution
            RAiSE_run(frequency, redshift[min(len(redshift) - 1, i)], axis_ratio[min(len(axis_ratio) - 1, i)], jet_power[min(len(jet_power) - 1, i)], source_age, halo_mass=halo_mass, rand_profile=rand_profile, betas=betas, regions=regions, rho0Value=rho0Value, temperature=temperature, active_age=active_age[min(len(active_age) - 1, i)], jet_lorentz=jet_lorentz[min(len(jet_lorentz) - 1, i)], equipartition=equipartition[min(len(equipartition) - 1, i)], spectral_index=spectral_index, gammaCValue=gammaCValue, lorentz_min=Lorentzmin, brightness=False, resolution=resolution, seed=seed)
            dataframe = pd.read_csv(filename+'.csv', index_col=None)
        
        time.append((dataframe.iloc[:,0]).astype(np.float_))
        size.append((dataframe.iloc[:,1]).astype(np.float_))
        pressure.append((dataframe.iloc[:,2]).astype(np.float_))
        if not resolution == None:
            luminosity.append((dataframe.iloc[:,5]).astype(np.float_))
            index = luminosity[i] <= 1e10
            luminosity[i][index] = np.nan
        try:
            if isinstance(labels, (list, np.ndarray)):
                axs[0].plot(time[i]/1e6, size[i]/2, colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25, label=labels[i])
                axs[1].plot(time[i]/1e6, pressure[i], colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25, label=labels[i])
                if not resolution == None:
                    axs[2].plot(time[i]/1e6, luminosity[i]/2, colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25, label=labels[i])
                    axs2.plot(size[i]/2, luminosity[i]/2, colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25, label=labels[i])
            else:
                axs[0].plot(time[i]/1e6, size[i]/2, colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25)
                axs[1].plot(time[i]/1e6, pressure[i], colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25)
                if not resolution == None:
                    axs[2].plot(time[i]/1e6, luminosity[i]/2, colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25)
                    axs2.plot(size[i]/2, luminosity[i]/2, colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25)
        except:
            axs[0].plot(time[i]/1e6, size[i]/2, colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25)
            axs[1].plot(time[i]/1e6, pressure[i], colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25)
            if not resolution == None:
                axs[2].plot(time[i]/1e6, luminosity[i]/2, colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25)
                axs2.plot(size[i]/2, luminosity[i]/2, colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)], linewidth=1.25)

    # set axes limits
    x_min = np.nanmin(np.abs(np.asarray(time)/1e6))
    y_min.append(np.nanmin(np.abs(np.asarray(size)/2)))
    y_min.append(np.nanmin(np.abs(np.asarray(pressure))))
    x_max = np.nanmax(np.abs(np.asarray(time)/1e6))
    y_max.append(np.nanmax(np.abs(np.asarray(size)/2)))
    y_max.append(np.nanmax(np.abs(np.asarray(pressure))))
    axs[0].set_xlim([x_min, x_max])
    axs[0].set_ylim([y_min[0], y_max[0]])
    axs[1].set_ylim([y_min[1], y_max[1]])
    if not resolution == None:
        y_max.append(np.nanmax(np.abs(np.asarray(luminosity)/2))*1.2)
        y_min.append(np.maximum(np.nanmin(np.abs(np.asarray(luminosity)/2))/1.2, y_max[2]/1e4))
        axs[2].set_ylim([y_min[2], y_max[2]])
        axs2.set_xlim([0, np.nanmax(np.abs(np.asarray(size)[~np.isnan(np.asarray(luminosity))]/2))])
        axs2.set_ylim([y_min[2], y_max[2]])

    # set nicely labelled log axes
    for i in range(0, len(axs)):
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].xaxis.set_major_formatter(FormatStrFormatter('%g'))
        axs[i].xaxis.set_minor_formatter(NullFormatter())
        if x_max/x_min < 10:
            axs[i].xaxis.set_major_locator(LogLocator(base=10, subs=[1,2,3,5]))
            axs[i].xaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
        elif x_max/x_min < 100:
            axs[i].xaxis.set_major_locator(LogLocator(base=10, subs=[1,3]))
            axs[i].xaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
        else:
            axs[i].xaxis.set_major_locator(LogLocator(base=10, subs=[1]))
            axs[i].xaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
        axs[i].yaxis.set_major_formatter(FormatStrFormatter('%g'))
        axs[i].yaxis.set_minor_formatter(NullFormatter())
        if y_max[i]/y_min[i] < 10:
            axs[i].yaxis.set_major_locator(LogLocator(base=10, subs=[1,2,3,5]))
            axs[i].yaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
        elif y_max[i]/y_min[i] < 100:
            axs[i].yaxis.set_major_locator(LogLocator(base=10, subs=[1,3]))
            axs[i].yaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
        else:
            axs[i].yaxis.set_major_locator(LogLocator(base=10, subs=[1]))
            axs[i].yaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
    if not resolution == None:
        axs2.set_yscale('log')
        axs2.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        axs2.yaxis.set_minor_formatter(NullFormatter())
        if y_max[2]/y_min[2] < 10:
            axs2.yaxis.set_major_locator(LogLocator(base=10, subs=[1,2,3,5]))
            axs2.yaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
        elif y_max[2]/y_min[2] < 100:
            axs2.yaxis.set_major_locator(LogLocator(base=10, subs=[1,3]))
            axs2.yaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
        else:
            axs2.yaxis.set_major_locator(LogLocator(base=10, subs=[1]))
            axs2.yaxis.set_minor_locator(LogLocator(base=10, subs=range(10)))
    
    # add legend
    if isinstance(labels, (list, np.ndarray)):
        axs[0].legend()
        if not resolution == None:
            axs2.legend()

    plt.show()
