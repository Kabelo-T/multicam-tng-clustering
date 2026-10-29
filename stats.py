from astropy.table import Table
import numpy as np
from numpy.random import default_rng
from halotools.mock_observables import return_xyz_formatted_array
from constants import TNG_H

def generate_randoms(lbox, n_randoms, rng=None):
    """Generate randoms to be used for tpcf_jackknife
    Args:
        lbox: size of TNG simulation box (usually Mpc/h)
        n_randoms: desired number of randoms, i.e. 10,000
        rng: a numpy Generator, an int seed, or None (defaults to a fixed
            seed of 42 for reproducibility across runs)
    Return:
        Nran: xyz formatted array of uniform random variables
    """
    if rng is None:
        _rng = default_rng(42)
    elif isinstance(rng, int):
        _rng = default_rng(rng)
    else:
        _rng = rng

    Xran = _rng.uniform(0, lbox, n_randoms)
    Yran = _rng.uniform(0, lbox, n_randoms)
    Zran = _rng.uniform(0, lbox, n_randoms)
    Nran = return_xyz_formatted_array(Xran, Yran, Zran)
    return Nran


def compute_tpcf_jackknife_errors(xi_cov):
    """Computes jackknife errors on tpcf for plotting
        Args:
            xi_cov: covariance of tpcf computed by tpcf_jackknife   
        Return:
            errors: jackknife errors for error shading
    """
    var = np.diag(xi_cov)
    errors = np.sqrt(var)
    return errors


def _create_bounds(N3, Lbox):
    """Creates bounds for subsets used to compute jackknife errors
    Args: 
        N3: cube root of the desired number of total boxes for jackknife
        Lbox: size of the TNG simulation box 
    Return:
        Lists x_min, x_max, y_min, y_max, z_min, and z_max
    """
    x_min = []
    x_max = []
    y_min = []
    y_max = []
    z_min = []
    z_max = []

    for ii in range (0, N3):
        for jj in range (0, N3):
            for kk in range(0, N3):
                x_min_ii = ii*(Lbox/N3) 
                x_max_ii = (Lbox/N3) * (1 + ii)
                x_min.append(x_min_ii)
                x_max.append(x_max_ii)
                
                y_min_jj = jj*(Lbox/N3)
                y_max_jj = (Lbox/N3) * (1 + jj)
                y_min.append(y_min_jj)
                y_max.append(y_max_jj)
                
                z_min_kk = kk*(Lbox/N3)
                z_max_kk = (Lbox/N3) * (1 + kk)
                z_min.append(z_min_kk)
                z_max.append(z_max_kk)       
    return x_min, x_max, y_min, y_max, z_min, z_max


def _create_subsets(N3, bounds, table):
    """Creates subsets used to compute jackknife errors
    Args: 
        N_3: cube root of the desired number of total boxes for jackknife
        bounds: bounds found using _create_bounds
        table: astropy table of the form created by import_data
    Return:
         Outputs a list of subsets
    """
    subsets = []
    for i in range(0, N3**3):
        xcond = (table['x']>bounds[0][i])&(table['x']<bounds[1][i])
        ycond = (table['y']>bounds[2][i])&(table['y']<bounds[3][i])
        zcond = (table['z']>bounds[4][i])&(table['z']<bounds[5][i])
        si = table[~(xcond&ycond&zcond)]
        subsets.append(si)
    return subsets


def compute_subhalo_mass_mean(table):
    """Computes mean of the 'SubhaloMass' column of astropy table
    Args: 
        table: astropy table created with import_data 
    Return:
        Mean of the subhalo mass Mh  
    """    
    Mh = np.mean(table['SubhaloMass'])
    return Mh


def jackknife_errors(N3, Lbox, table, fctn):
    """Computes jackknife_errors of a given statistic
    Args: 
        N_3: cube root of the desired number of total boxes for jackknife
        Lbox: length of the TNG simulation box
        table: astropy table created with import_data
        fctn: a function used to compute statistic that will be jackknifed
    Return:
         The jackknife error (std) corresponding to the statistic
    """    
    bounds = _create_bounds(N3, Lbox)
    subsets = _create_subsets(N3, bounds, table)
    
    statistic_subset = []
    for subset in subsets:
        stat_subs = fctn(subset)
        statistic_subset.append(stat_subs)

    statistic_subset = np.array(statistic_subset)
    
    statistic_bar = np.mean(statistic_subset)
    
    Var = ((N3**3-1)/N3**3) * np.sum((statistic_subset - statistic_bar)**2)
    
    std = np.sqrt(Var)
    return std

