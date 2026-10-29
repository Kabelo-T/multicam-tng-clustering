from astropy.table import Table
import numpy as np
from constants import TNG_H

def mass_sample(table, mass, mass_min, mass_max):
    """Create mass samples in for TNG data
    Args: 
        table: astropy table
        mass: string 'StellarMass' or 'SubhaloMass'
        mass_min: minimum mass, i.e. 10**7
        mass_max: maximum mass, i.e. float('inf')
    Return:
        Astropy table with desired mass range
        i.e. 8 <= stellar_mass < 8.5
    """
    t = table[(table[mass] >= mass_min)&(table[mass] < mass_max)]
    return t
