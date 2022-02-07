# RAiSERHD module
# Ross Turner, 19 Jan 2022

# import packages
import h5py
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
from copy import copy
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
crit_age = 0.95 # fraction of source age for lower end of power law approximations
pressure_corr = 1.#2.96 # manual correction to calibrate particles

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
def RAiSE_run(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass=None, rand_profile=False, betas=None, regions=None, rho0Value=None, temperature=None, active_age=10.14, jet_lorentz=5, equipartition=-1.5, spectral_index=0.7, gammaCValue=5./3, lorentz_min=Lorentzmin, brightness=True, resolution='standard', seed=None, aj_star=0.2305, fill_factor=0.1545, jet_angle=0.5375):
    
    # record start time of code
    start_time = ti.time()
    
    # function to test type of inputs and convert type where appropriate
    if nangles <= 1:
        raise Exception('Private variable nangles must be greater than 1.')
    frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz, nenvirons = __test_inputs(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz)
    
    # download and pre-process particles from hydrodynamical simulation
    if not resolution == None:
        print(__color_text('Reading particle data from file.', Colors.Green))
        time, shock_time, major, minor, x1, x2, x3, tracer, vx3, volume, pressure, press_minor, alphaP_hyd, alphaP_henv, hotspot_ratio = __PLUTO_particles('RAiSE_particles.hdf5')
    # set seed for quasi-random profiles
    if not seed == None:
        __set_seed(seed)
        
    # create folder for output files if not present
    if not os.path.exists('LDtracks'):
        os.mkdir('LDtracks')
    
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
                                lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda = __RAiSE_environment(redshift[i], axis_ratio[j], jet_power[k], source_age, halo_mass=new_halo_mass, rand_profile=rand_profile, rho0Value=new_rho0Value, regions=new_regions, betas=new_betas, temperature=new_temperature, active_age=active_age[m], jet_lorentz=jet_lorentz[o], gammaCValue=gammaCValue, aj_star=aj_star, fill_factor=fill_factor, jet_angle=jet_angle)
                                
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
                                        if frequency[q] > 0:
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
                                    x_values, y_values, brightness_list = __RAiSE_brightness_map(frequency, redshift[i], source_age, lobe_lengths, location, luminosity, resolution=resolution)
                                    
                                    for p in range(0, len(source_age)):
                                        for q in range(0, len(frequency)):
                                            # create pandas dataframe for spatially resolved emission
                                            if isinstance(x_values[p][q], (list, np.ndarray)):
                                                df = pd.DataFrame(index=x_values[p][q]/const.kpc.value, columns=y_values[p][q]/const.kpc.value, data=brightness_list[p][q])
                                        
                                                # write surface brightness map to file
                                                if isinstance(rho0Value, (list, np.ndarray)):
                                                    if frequency[q] > 0:
                                                        df.to_csv('LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), np.abs(np.log10(rho0Value[l])), jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], frequency[q], source_age[p], resolution), header=True, index=True)
                                                    else:
                                                        df.to_csv('LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), np.abs(np.log10(rho0Value[l])), jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], source_age[p], resolution), header=True, index=True)
                                                elif isinstance(halo_mass, (list, np.ndarray)):
                                                    if frequency[q] > 0:
                                                        df.to_csv('LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), halo_mass[l], jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], frequency[q], source_age[p], resolution), header=True, index=True)
                                                    else:
                                                        df.to_csv('LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), halo_mass[l], jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], source_age[p], resolution), header=True, index=True)
                                                else:
                                                    raise Exception('Either the halo mass or full density profile must be provided as model inputs.')
                                            else:
                                                if isinstance(rho0Value, (list, np.ndarray)):
                                                    warnings.warn('The following file was not created as no emission is present: LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), np.abs(np.log10(rho0Value[l])), jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], frequency[q], source_age[p], resolution), category=UserWarning)
                                                elif isinstance(halo_mass, (list, np.ndarray)):
                                                    warnings.warn('The following file was not created as no emission is present: LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}_{:s}.csv'.format(axis_ratio[j], np.abs(equipartition[n]), halo_mass[l], jet_power[k], 2*np.abs(spectral_index) + 1, active_age[m], jet_lorentz[o], redshift[i], frequency[q], source_age[p], resolution), category=UserWarning)
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
        if not isinstance(frequency[i], (int, float)):
            if frequency[i] <= 0:
                frequency[i] = -1.
                warnings.warn('Pressure map will be produced instead of surface brightness image.', category=UserWarning)
            elif not (5 < frequency[i] and frequency[i] < 20):
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
def __RAiSE_environment(redshift, axis_ratio, jet_power, source_age, halo_mass=None, rand_profile=False, betas=None, regions=None, rho0Value=None, temperature=None, active_age=10.14, jet_lorentz=5., gammaCValue=5./3, aj_star=0.2305, fill_factor=0.1545, jet_angle=0.5375):
    
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
                tempCluster = 10**(TmgConst + TmgSlope*halo_mass)
            else:
                tempFlat = 10**(__rand_norm(TmgAvg, TmgStdev))
                tempCluster = 10**(__rand_norm(TmgConst + TmgSlope*halo_mass, TmgError))
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
    return __RAiSE_evolution(redshift, axis_ratio, jet_power, source_age, active_age, gammaCValue, nregions, new_betas, new_regions, kValues, temperature, jet_lorentz, aj_star, fill_factor, jet_angle)

    
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
def __RAiSE_evolution(redshift, axis_ratio, jet_power, source_age, active_age, gammaCValue, nregions, betas, regions, kValues, temperature, jet_lorentz, aj_star=0.2305, fill_factor=0.1545, jet_angle=0.5375):
    
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
        _, _, _, _, _, _, _, critical_point_1 = __RAiSE_runge_kutta(QavgValue, np.array([x_time]), x_time, axis_ratio, aj_star, fill_factor, jet_lorentz, open_angle, angles, eta_c, eta_s, zetaeta, dchi, nregions, betas, regions, kValues, temperature, gammaCValue, critical_velocity=c_speed, strong_shock=True)

        # run code for full RAiSE HD dynamical model
        lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, critical_point_3 = __RAiSE_runge_kutta(QavgValue, tFinal, tActive, axis_ratio, aj_star, fill_factor, jet_lorentz, open_angle, angles, eta_c, eta_s, zetaeta, dchi, nregions, betas, regions, kValues, temperature, gammaCValue, critical_velocity=c_speed*critical_point_1[2]/critical_point_1[3])
    else:
        # run code for RAiSE X dynamical model
        lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, _ = __RAiSE_runge_kutta(QavgValue, tFinal, tActive, axis_ratio, aj_star, fill_factor, jet_lorentz, open_angle, angles, eta_c, eta_s, zetaeta, dchi, nregions, betas, regions, kValues, temperature, gammaCValue)

    return lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda
    

# function to apply Runge-Kutta method and extract values at requested time steps
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_runge_kutta(QavgValue, source_age, active_age, axis_ratio, aj_star, fill_factor, jet_lorentz, open_angle, angles, eta_c, eta_s, zetaeta, dchi, nregions, betas, regions, kValues, temperature, gammaCValue, critical_velocity=0., strong_shock=False):

    # instantiate variables
    X, P = np.zeros((nangles, 5)), np.zeros((nangles, 4))
    critical_point = np.zeros(4)
    regionPointer = np.zeros(nangles).astype(np.int_)
    lobe_minor, lambda_crit, alphaP_denv, alpha_lambda = np.zeros(len(source_age)), np.zeros(len(source_age)), np.zeros(len(source_age)), np.zeros(len(source_age))
    lobe_lengths, shock_lengths, shock_pressures = np.zeros((nangles, len(source_age))), np.zeros((nangles, len(source_age))), np.zeros((nangles, len(source_age)))
    
    # calculate injection ages to derive time-average power-law indices for external pressure and filling factor
    inject_age = np.zeros(2*len(source_age))
    inject_axis_ratios, inject_pressures, inject_lambdas = np.zeros(2*len(source_age)), np.zeros(2*len(source_age)), np.zeros(2*len(source_age))
    for timePointer in range(0, len(source_age)):
        inject_age[2*timePointer:2*(timePointer + 1)] = np.asarray([crit_age*source_age[timePointer], source_age[timePointer]])
    inject_index = np.argsort(inject_age) # sort ages in ascending order
    
    # calculate the spatially-averaged jet velocity and Lorentz factor
    if jet_lorentz > 1:
        bulk_lorentz = np.sqrt(jet_lorentz**2*aj_star**4 - aj_star**4 + 1)
        bulk_velocity = np.sqrt((jet_lorentz**2*aj_star**4 - aj_star**4)/(jet_lorentz**2*aj_star**4 - aj_star**4 + 1))*c_speed
    else:
        bulk_lorentz, bulk_velocity = -1, -1
    
    i = 0
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

            # set region pointer to first (non-zero) region if smaller than FR2 radius
            index = regions[1] < X[angles,1]
            regionPointer[index] = 1
            regionPointer[np.logical_not(index)] = 0
            
            # calculate fraction of jet power injected into each volume element
            injectFrac = dchi*eta_s**(3 - betas[regionPointer[0]])*zetaeta**2
            injectFrac = injectFrac/np.sum(injectFrac) # sum should be equal to unity
            
        # solve ODE to find radius and pressue at each time step
        while (X[0,0] < source_age[timePointer]):
            while (X[0,0] < inject_age[inject_index[i]]):
                # calculate the appropriate density profile for each angle theta
                for anglePointer in range(0, nangles):
                    while (regionPointer[anglePointer] + 1 < nregions and X[anglePointer,1] > regions[regionPointer[anglePointer] + 1]):
                        regionPointer[anglePointer] = regionPointer[anglePointer] + 1

                # check if next step passes time point of interest
                if (X[0,0]*stepRatio > inject_age[inject_index[i]]):
                    step = inject_age[inject_index[i]] - X[0,0]
                else:
                    step = X[0,0]*(stepRatio - 1)

                # update estimates of time, radius and velocity
                __rk4sys(step, X, P, QavgValue, active_age, aj_star, fill_factor, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
                X[:,3] = np.maximum(1, X[:,3])
                
                # find location of jet--lobe transition
                critical_point[0], critical_point[1], critical_point[2], critical_point[3] = X[0,0], X[0,1], X[0,2]*X[0,3], X[0,4]
             
            # record axis ratio, external pressure and filling factor and injection times
            if P[-1,0] > 0:
                inject_axis_ratios[inject_index[i]] = 1./(P[0,0]/P[-1,0])**2 # inverted to match alpha_lambda definition
            else:
                inject_axis_ratios[inject_index[i]] = 1
            inject_pressures[inject_index[i]] = P[0,2]
            inject_lambdas[inject_index[i]] = P[0,3]
            
            # update injection age if not a requested source age
            if inject_age[inject_index[i]] < source_age[timePointer]:
                i = i + 1

        # calculate the lobe and shocked shell length, shock pressure and total pressure as a function of angle
        lobe_lengths[angles,timePointer] = P[angles,0]
        shock_lengths[angles,timePointer] = X[angles,1]
        shock_pressures[angles,timePointer] = P[angles,1]
        lambda_crit[timePointer] = P[0,3]
        
        # calculate lobe minor axis (associated with dimensions of shocked shell) at this time step
        lobe_minor[timePointer] = X[-1,1]*eta_c[-1]/(shockRadius*eta_s[-1])
        
        # calculate the slope of external pressure profile at this time step
        if inject_pressures[inject_index[2*timePointer]] <= 0:
            alphaP_denv[timePointer] = 0
        else:
            alphaP_denv[timePointer] = np.log(inject_pressures[2*timePointer + 1]/inject_pressures[2*timePointer])/np.log(inject_age[2*timePointer + 1]/inject_age[2*timePointer])
        if inject_lambdas[2*timePointer] <= 0:
             alpha_lambda[timePointer] = 1e9 # no emission from this injection time
        else:
            alpha_lambda[timePointer] = np.log(inject_lambdas[2*timePointer + 1]/inject_lambdas[2*timePointer])/np.log(inject_age[2*timePointer + 1]/inject_age[2*timePointer]) + np.log(inject_axis_ratios[2*timePointer + 1]/inject_axis_ratios[2*timePointer])/np.log(inject_age[2*timePointer + 1]/inject_age[2*timePointer]) # filling factor and changing volume/axis ratio
    
    return lobe_lengths, lobe_minor, shock_lengths, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, critical_point


# Runge-Kutta method to solve ODE in dynamical model
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __rk4sys(step, X, P, QavgValue, active_age, aj_star, fill_factor, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock):
    
    # instantiate variables
    Y, K1, K2, K3, K4 = np.zeros((len(angles), 5)), np.zeros((len(angles), 5)), np.zeros((len(angles), 5)), np.zeros((len(angles), 5)), np.zeros((len(angles), 5))
    
    # fouth order Runge-Kutta method
    __xpsys(X, K1, P, QavgValue, active_age, aj_star, fill_factor, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
    Y[:,:] = X[:,:] + 0.5*step*K1[:,:]
    __xpsys(Y, K2, P, QavgValue, active_age, aj_star, fill_factor, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
    Y[:,:] = X[:,:] + 0.5*step*K2[:,:]
    __xpsys(Y, K3, P, QavgValue, active_age, aj_star, fill_factor, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
    Y[:,:] = X[:,:] + 0.5*step*K3[:,:]
    __xpsys(Y, K4, P, QavgValue, active_age, aj_star, fill_factor, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock)
    X[:,:] = X[:,:] + (step/6.)*(K1[:,:] + 2*K2[:,:] + 2*K3[:,:] + K4[:,:])


# coupled second order differential equations for lobe evolution
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __xpsys(X, f, P, QavgValue, active_age, aj_star, fill_factor, jet_lorentz, open_angle, angles, injectFrac, eta_c, eta_s, zetaeta, dchi, regionPointer, betas, kValues, temperature, gammaCValue, critical_velocity, strong_shock):
    
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
        if (eta_R/bulk_lorentz**2) > 1:
            lambda_crit = 0
        else:
            lambda_crit = 1
        #lambda_crit = np.exp(-(eta_R/bulk_lorentz**2)/(2*np.log(2)))
        
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
    if jet_lorentz > 1 and strong_shock == True:
        f[angles,2] = np.minimum((gammaCValue - 1)*injectFrac[angles]*(QavgValue*active_jet)*X[angles,1]**(betas[regionPointer[angles]] - 3)/(X[angles,2]*(1 + (X[angles,3]*X[angles,2]/c_speed)**2)*dchi[angles]*(X[angles,3]*zetaeta[angles])**2*kValues[regionPointer[angles]]) + (betas[regionPointer[angles]] - 3*gammaCValue)*(X[angles,2])**2/(2*X[angles,1]*(1 + (X[angles,3]*X[angles,2]/c_speed)**2)), (betas[regionPointer[angles]] - 2)/(5 - betas[regionPointer[angles]]) * X[angles,2]*X[angles,3]/(X[0,0] + year)) # ensure model doesn't run slower than limit due to numerics
    elif jet_lorentz > 1:
        f[angles,2] = (gammaCValue - 1)*injectFrac[angles]*(QavgValue*active_jet)*X[angles,1]**(betas[regionPointer[angles]] - 3)/(X[angles,2]*(1 + (X[angles,3]*X[angles,2]/c_speed)**2)*dchi[angles]*(X[angles,3]*zetaeta[angles])**2*kValues[regionPointer[angles]]) + (betas[regionPointer[angles]] - 3*gammaCValue)*(X[angles,2])**2/(2*X[angles,1]*(1 + (X[angles,3]*X[angles,2]/c_speed)**2)) - (3*gammaCValue - betas[regionPointer[angles]])*(k_B*temperature/maverage)/(2*X[angles,1]*(1 + (X[angles,3]*X[angles,2]/c_speed)**2)*(X[angles,3]*zetaeta[angles])**2)
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
        P[angles,1] = zetaeta[angles]**2*kValues[regionPointer[angles]]*X[angles,1]**(-betas[regionPointer[angles]])*(np.minimum(X[angles,2], X[angles,4]))**2 + kValues[regionPointer[angles]]*(k_B*temperature/maverage)*X[angles,1]**(-betas[regionPointer[angles]])
        
        # calculate average pressure across jet/lobe
        pressure = np.sum(P[angles,1]*volume)/np.sum(volume)
        # set average pressure in all of lobe other than hotspot
        P[angles[1:],1] = pressure
    else:
        # calculate lobe pressure
        P[super_angles,1] = 2./(gammaX + 1)*zetaeta[super_angles]**2*kValues[regionPointer[super_angles]]*X[super_angles,1]**(-betas[regionPointer[super_angles]])*(X[super_angles,2]*X[super_angles,3])**2 - (gammaX - 1)/(gammaX + 1)*kValues[regionPointer[super_angles]]*(k_B*temperature/maverage)*X[super_angles,1]**(-betas[regionPointer[super_angles]])
        P[sub_angles,1] = P[sub_angles,2]
        
        # calculate average pressure across jet/lobe
        pressure = np.sum(P[angles,1]*volume)/np.sum(volume)
        # set average pressure in all of lobe other than hotspot
        P[angles[1:],1] = pressure

    # AXIS RATIO
    if jet_lorentz > 1:
        # calculate total mass of particles from the jet
        particle_mass = QavgValue*np.minimum(active_age, X[0,0])/((jet_lorentz - 1)*c_speed**2)
        
        # calculate volume occupied by particles expanding at sound speed and maximum fillable volume within shocked shell
        jet_sound = c_speed*np.sqrt(gammaJ - 1)
        particle_volume = particle_mass/(gammaJ*pressure/jet_sound**2) # mass / density
        shell_volume = np.sum(volume*eta_c/(shockRadius*eta_s))
        
        # calculate (optimal) lobe volume as weighted sum of particle volume and maximum fillable volume (i.e. enable sound speed to reduce as lobe approaches size of shocked shell)
        lobe_volume = 1./np.sqrt(1./particle_volume**2 + 1./(fill_factor*shell_volume)**2) / fill_factor
        
        # find axis ratio for an ellipsoidal lobe
        if lobe_volume > 0 and lambda_crit >= 0.01:
            lobe_axis_ratio = np.minimum(np.sqrt(2*np.pi*(X[0,1]/shockRadius)**3/(3*lobe_volume)), 1/np.tan(open_angle))
        else:
            lobe_axis_ratio = 1/np.tan(open_angle)
        # update lobe length along let axis and axis ratio of shocked shell
        P[0,0] = X[0,1]/shockRadius

        # calculate geometry of each angular volume element
        dtheta = (np.pi/2)/len(angles)
        theta = dtheta*(angles + 0.5)
        lobe_eta_c = 1./np.sqrt(lobe_axis_ratio**2*(np.sin(theta))**2 + (np.cos(theta))**2)
        # set length of lobe along each angular volume element
        P[angles[1:],0] = np.minimum(lobe_eta_c[angles[1:]]*P[0,0], X[angles[1:],1]*eta_c[angles[1:]]/(shockRadius*eta_s[angles[1:]])) # second condition should rarely be met
    else:
        # set length of lobe along each angular volume element
        P[0,0], P[angles[1:],0] = X[0,1]/shockRadius, X[angles[1:],1]*eta_c[angles[1:]]/(shockRadius*eta_s[angles[1:]])


## Define functions to download and preprocess particles from hydrodynamical simulations
def __PLUTO_particles(particle_data_path):
        
    # unpack particle data from hydrodynamical simulations
    particle_dict = h5py.File(os.path.join(os.path.dirname(os.path.realpath(__file__)), particle_data_path), 'r')
    
    # store variables at desired resolution
    time = particle_dict['time'][:].astype(np.float32)
    shock_time = particle_dict['tinject'][:,:].astype(np.float32)
    major = particle_dict['major'][:].astype(np.float32)
    minor = particle_dict['minor'][:].astype(np.float32)
    x1 = particle_dict['x1'][:,:].astype(np.float32)
    x2 = particle_dict['x2'][:,:].astype(np.float32)
    x3 = particle_dict['x3'][:,:].astype(np.float32)
    tracer = particle_dict['tracer'][:,:].astype(np.float32)
    vx3 = particle_dict['vx3'][:,:].astype(np.float32)
    volume = particle_dict['volume'][:,:].astype(np.float32)
    pressure = particle_dict['pressure'][:,:].astype(np.float32)
    press_minor = particle_dict['pressminor'][:].astype(np.float32)
    alphaP_hyd = particle_dict['alphaP'][:,:].astype(np.float32)
    alphaP_henv = particle_dict['alphaPenv'][:,:].astype(np.float32)
    hotspot_ratio = particle_dict['hotspotratio'][:].astype(np.float32)

    return time, shock_time, major, minor, x1, x2, x3, tracer, vx3, volume, pressure, press_minor, alphaP_hyd, alphaP_henv, hotspot_ratio
    

## Define functions to add emissivity from particles in hydrodynamical simulations on top of dynamics
# function to manage orientation and distribution of particles from simulation output
def __RAiSE_emissivity(frequency, redshift, time, shock_time, major, minor, x1, x2, x3, tracer, vx3, volume, pressure, press_minor, alphaP_hyd, alphaP_henv, hotspot_ratio, source_age, lobe_lengths, lobe_minor, shock_pressures, lambda_crit, alphaP_denv, alpha_lambda, active_age, equipartition, spectral_index, gammaCValue=5./3, lorentz_min=Lorentzmin, resolution='standard'):
    
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
            new_pressure = pressure[:,timePointer[j]]*(shock_pressures[-1,i]/press_minor[timePointer[j]]) * pressure_corr # correction factor to match Model A
            # correct the hotspot/lobe pressure ratio based on the dynamical model
            new_pressure = new_pressure*((shock_pressures[0,i]/shock_pressures[-1,i])/hotspot_ratio[timePointer[j]] - 1)*(np.abs(x3[:,timePointer[j]])/major[timePointer[j]]) + new_pressure # increase log-space pressure linearly along lobe
            # correct the evolutionary histories of the particles based on the dynamical model
            alphaP_dyn = np.maximum(-2, np.minimum(0, alphaP_denv[i] + alphaP_hyd[:,timePointer[j]] - alphaP_henv[:,timePointer[j]]))
            
            # VOLUMES
            volume_fraction = volume[:,timePointer[j]]/(4*np.pi/3.*major[timePointer[j]]*minor[timePointer[j]]**2)
            #volume_sum = np.nansum(volume_fraction[~np.isinf(volume_fraction)])
            # cap the largest volumes at the 95th percentile to outliers in surface brightness map; minimal effect on total luminosity
            volume_fraction[volume_fraction > np.nanpercentile(volume_fraction, 95)] = np.nanpercentile(volume_fraction, 95)
            new_volume = volume_fraction*(4*np.pi/3.*lobe_lengths[0,i]*lobe_minor[i]**2)*tracer[:,timePointer[j]] #/volume_sum
            
            # RELATIVISTIC BEAMING
            doppler_factor = np.sqrt(np.maximum(1e-6, 1 - vx3[:,timePointer[j]]**2))**(3 - (s_index - 1)/2.) # Doppler boosting of particles in jet; 1e-6 ensures some very low level emission
            doppler_factor[np.logical_and(np.abs(x3[:,timePointer[j]])/major[timePointer[j]] < 0.1, np.logical_and(np.abs(x1[:,timePointer[j]])/major[timePointer[j]] < 0.01, np.abs(x2[:,timePointer[j]])/major[timePointer[j]] < 0.01))] = 0 # completely remove very bright particles clumped at start of jet
            
            # LOBE PARTICLES
            # find angle and radius of each particle from core
            new_angles = np.arctan((np.sqrt(x1[:,timePointer[j]]**2 + x2[:,timePointer[j]]**2)*lobe_minor[i]/minor[timePointer[j]])/(x3[:,timePointer[j]]*lobe_lengths[0,i]/major[timePointer[j]])) # rescale axes to correct axis ratio
            new_radii = np.sqrt((x1[:,timePointer[j]]**2 + x2[:,timePointer[j]]**2)*(lobe_minor[i]/minor[timePointer[j]])**2 + (x3[:,timePointer[j]]*lobe_lengths[0,i]/major[timePointer[j]])**2)/lobe_lengths[0,i]
            # find particles within lobe region; particles outside this region will not emit. Particle map is set to axis ratio based on shocked shell to maintain geometry of jet
            new_eta_c = 1./np.sqrt((lobe_lengths[0,i]/lobe_lengths[-1,i])**2*(np.sin(new_angles))**2 + (np.cos(new_angles))**2)
            lobe_particles = np.zeros_like(x1[:,timePointer[j]])
            lobe_particles[np.abs(vx3[:,timePointer[j]]) > 1./np.sqrt(3)] = 1 # assume sound speed is critical value for relativisitic particles
            lobe_particles[new_radii < new_eta_c] = 1.
            
            # TWO PHASE FLUID
            # fraction of jet particles that have reached location in lobe
            two_phase_weighting = np.maximum(0, np.minimum(1, lambda_crit[i]*(new_shock_time/np.minimum(tActive, tFinal[i]))**np.maximum(0, alpha_lambda[i])))
            if tActive/tFinal[i] >= 1:
                # keep jet particles visible at all times
                two_phase_weighting = np.maximum(two_phase_weighting, np.minimum(1, np.abs(vx3[:,timePointer[j]]*np.sqrt(3)))) # assume sound speed is critical value for relativisitic particles
            else:
                # suppress emission from jet particle
                two_phase_weighting = np.minimum(two_phase_weighting, 1 - np.minimum(1, np.abs(vx3[:,timePointer[j]]*np.sqrt(3))))
            
            # PARTICLE EMISSIVITY
            for k in range(0, len(rest_frequency)):
                if rest_frequency[k] > 100:
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
            
            # PARTICLE PRESSURE
                else:
                    luminosity[i,j*len(pressure[:,0]):(j+1)*len(pressure[:,0]),k] = new_pressure*lobe_particles
            
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
def __RAiSE_loss_mechanisms(rest_frequency, inverse_compton, redshift, time, shock_time, pressure, alphaP, equipartition, gammaCValue=5./3):
    
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
def __RAiSE_a2(redshift, time, shock_time, pressure, pressure_inject, equipartition, alphaP, gammaCValue=5./3):
    
    return 4*sigma_T/(3*m_e*c_speed)*(pressure_inject/(gammaCValue - 1)*(equipartition/(equipartition + 1))/(1 + alphaP*(1 + 1./(3*gammaCValue)))*shock_time**(-alphaP)*(time**(1 + alphaP*(1 + 1./(3*gammaCValue))) - shock_time**(1 + alphaP*(1 + 1./(3*gammaCValue)))) + __RAiSE_uC(redshift)/(1 + alphaP/(3*gammaCValue))*(time**(1 + alphaP/(3*gammaCValue)) - shock_time**(1 + alphaP/(3*gammaCValue)))) # array is shorter by one element


# find CMB radiation energy density
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_uC(redshift):
    
    uC0 = 0.25*1e6*e_charge # J m-3 CMB energy density at z = 0 (Longair, 1981)
    return uC0*(redshift + 1)**4 # assuming uC prop to (z + 1)^4 as in KDA97


# find RAiSE constant K(s) for the absolute scaling of the emissivity
def __RAiSE_Ks(s_index, gammaCValue=5./3, lorentz_min=Lorentzmin):
    
    kappa = (gamma(s_index/4. + 19./12)*gamma(s_index/4. - 1./12)*gamma(s_index/4. + 5./4)/gamma(s_index/4. + 7./4))

    return kappa/(m_e**((s_index + 3)/2.)*c_speed*(s_index + 1))*(e_charge**2*mu0/(2*(gammaCValue - 1)))**((s_index + 5)/4.)*(3./np.pi)**(s_index/2.)/((lorentz_min**(2 - s_index) - Lorentzmax**(2 - s_index))/(s_index - 2) - (lorentz_min**(1 - s_index) - Lorentzmax**(1 - s_index))/(s_index - 1))
    

# find RAiSE blackbody constant to convert cosmic microwave background emission from single frequency to blackbody spectrum
def __RAiSE_blackbody(s_index):

    return np.pi**4/(15*gamma((s_index + 5)/2.)*zeta((s_index + 5)/2.))


## Define functions to produce surface brightness maps of radio lobes
# define function to manage the discretisation of particles down to pixels
def __RAiSE_brightness_map(frequency, redshift, source_age, lobe_lengths, location, luminosity, resolution='standard'):
    
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

    return __RAiSE_pixels(rest_frequency, redshift, tFinal, lobe_lengths, location, luminosity, npixels)


# define function to discretise particles down to pixels
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def __RAiSE_pixels(rest_frequency, redshift, tFinal, lobe_lengths, location, luminosity, npixels):

    angle = 0 # hard-wired in for this version
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
            
            if len(location_x) > 0:
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
                    if rest_frequency[j] > 100:
                        brightness[location_x[k],location_y[k]] = brightness[location_x[k],location_y[k]] + new_luminosity[k]
                    else:
                        brightness[location_x[k],location_y[k]] = max(brightness[location_x[k],location_y[k]], new_luminosity[k])
                
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


# Define functions to plot emissivity maps throughout source evolutionary history
def RAiSE_evolution_maps(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass=None, rand_profile=False, betas=None, regions=None, rho0Value=None, temperature=None, active_age=10.14, jet_lorentz=5., equipartition=-1.5, spectral_index=0.7, gammaCValue=5./3, lorentz_min=Lorentzmin, seed=None, rerun=False, cmap='RdPu'):
    
    # function to test type of inputs and convert type where appropriate
    frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz, nenvirons = __test_inputs(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass, betas, regions, rho0Value, temperature, active_age, equipartition, jet_lorentz)
    
    # set up plot
    fig, axs = plt.subplots(len(source_age), 1, figsize=(12, 1 + (10/axis_ratio[0] + 0.8)*len(source_age)))
    if len(source_age) <= 1: # handle case of single image
        axs = [axs]
    fig.subplots_adjust(hspace=0)
    
    #cmap = cm.get_cmap('binary')
    colour_scheme = cm.get_cmap(cmap)

    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    
    for i in range(0, len(source_age)):
        if isinstance(rho0Value, (list, np.ndarray)):
            if frequency[0] > 0:
                filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}'.format(axis_ratio[0], np.abs(equipartition[0]), np.abs(np.log10(rho0Value[0])), jet_power[0], 2*np.abs(spectral_index) + 1, active_age[0], jet_lorentz[0], redshift[0], frequency[0], source_age[i])
            else:
                filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_p={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_t={:.2f}'.format(axis_ratio[0], np.abs(equipartition[0]), np.abs(np.log10(rho0Value[0])), jet_power[0], 2*np.abs(spectral_index) + 1, active_age[0], jet_lorentz[0], redshift[0], source_age[i])
        elif isinstance(halo_mass, (list, np.ndarray)):
            if frequency[0] > 0:
                filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}_nu={:.2f}_t={:.2f}'.format(axis_ratio[0], np.abs(equipartition[0]), halo_mass[0], jet_power[0], 2*np.abs(spectral_index) + 1, active_age[0], jet_lorentz[0], redshift[0], frequency[0], source_age[i])
            else:
                 filename = 'LDtracks/LD_A={:.2f}_eq={:.2f}_H={:.2f}_Q={:.2f}_s={:.2f}_T={:.2f}_y={:.2f}_z={:.2f}__t={:.2f}'.format(axis_ratio[0], np.abs(equipartition[0]), halo_mass[0], jet_power[0], 2*np.abs(spectral_index) + 1, active_age[0], jet_lorentz[0], redshift[0], source_age[i])
        
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
        if frequency[0] > 0:
            Z = Z/np.nanmax(Z)
        else:
            Z = Z*1e12
        Z[Z <= 0] = np.nan
        
        if frequency[0] > 0:
            h = axs[i].pcolormesh(X, Y, Z, shading='nearest', cmap=colour_scheme, vmin=0, vmax=1)
        else:
            h = axs[i].pcolormesh(X, Y, Z, shading='nearest', cmap=colour_scheme, vmin=np.nanmin(Z[0:len(x)//3,:]), vmax=np.nanmax(Z[0:len(x)//3,:]))

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

    axs[-1].set_xlabel(r'Jet axis (kpc)', fontsize=14.5, labelpad=10)
    plt.ylabel(r'Transverse axis (kpc)', fontsize=14.5, labelpad=15)
    
    if frequency[0] <= 0:
        if len(axs) == 1:
            cax = fig.add_axes([axs[0].get_position().x1+0.01,axs[0].get_position().y0,0.02,axs[0].get_position().height])
            c = plt.colorbar(h, cax=cax, pad=0.025)
        else:
            cax = fig.add_axes([axs.ravel().tolist().get_position().x1+0.01,axs.ravel().tolist().get_position().y0,0.02,axs.ravel().tolist().get_position().height]) # haven't tested this yet
            c = plt.colorbar(h, cax=cax, pad=0.015)
        c.set_label(r'Pressure (pPa)', labelpad=12.5)

    # show plot and return handle to plot
    plt.show()
    return fig


# Define function to plot Dt and LD tracks
def RAiSE_evolution_tracks(frequency, redshift, axis_ratio, jet_power, source_age, halo_mass=None, rand_profile=False, betas=None, regions=None, rho0Value=None, temperature=None, active_age=10.14, jet_lorentz=5., equipartition=-1.5, spectral_index=0.7, gammaCValue=5./3, lorentz_min=Lorentzmin, resolution='standard', seed=None, rerun=False, labels=None, colors=None, linestyles=None):

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

    # show plot and return handle to plot
    plt.show()
    return fig
