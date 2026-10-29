from tqdm import tqdm
from astropy.table import Table
import numpy as np
import h5py
import pandas as pd
from constants import TNG_H

def _convert_tng_position(tng_pos):
    """Convert TNG positions from from kiloparsecs to megaparsecs
    Args: 
        tng_pos: column of an astropy table, i.e. t['x']
    Return:
        Column in new units of megaparsecs
    """
    pos = tng_pos/1000
    return pos


def _check_flag(table):
    """Filters astropy table such that 'SubhaloFlag' is True
    Args:
        table: astropy table created with import_subhalo
    Return: 
        Outputs the filtered astropy table
    """
    t = table[(table['SubhaloFlag'] == 1)]
    return t


def import_hydro(file_number, directory, check_subhalo_flag):
    """Reads in data in the form of h5py files from TNG100
    Args:
        file_number: the desired number of files to be read in
        directory: string that details path from location 
        of notebook to dataset, i.e. 'Data/hydro_data'
    Return:
        Outputs an astropy table with 'ID', 'SubhaloMass', 
        'SubhaloFlag','StellarMass', 'x', 'y', 'z',
        'logSubhaloMass' and 'logStellarMass' as keys
    Note:
        'StellarMass' units Msun/h
        'logStellarMass' = log10('StellarMass')
    """
    subhalo_ids = []
    subhalo_mass = []
    subhalo_flag = []
    stellar_mass = []
    x_position = []
    y_position = []
    z_position = []
    count = 0
    
    for ii in tqdm(range(0, file_number + 1)):
        filename = f'{directory}/fof_subhalo_tab_099.{ii}.hdf5' 
        with h5py.File(filename, 'r') as f:
            if 'SubhaloMass' in f['Subhalo'].keys(): 
                for s in f['Subhalo']['SubhaloMass']:
                    subhalo_mass.append(s)
                    subhalo_ids.append(count)
                    count+=1
                for flag in f['Subhalo']['SubhaloFlag']:
                    subhalo_flag.append(flag)
                for star in f['Subhalo']['SubhaloMassInRadType'][:,4]: 
                    stellar_mass.append(star)
                for (x, y, z) in f['Subhalo']['SubhaloPos'][:]:
                    x_position.append(x)
                    y_position.append(y)
                    z_position.append(z) 
                             
    t = Table(data = [subhalo_ids, subhalo_mass, subhalo_flag, stellar_mass, 
                      x_position, y_position, z_position], 
              names = ['subfind_id', 'SubhaloMass', 'SubhaloFlag', 'StellarMass', 'x', 'y', 'z'])
    
    if check_subhalo_flag == True:
        t = _check_flag(t)

    # Convert from TNG units to Msun/h
    t['StellarMass'] = t['StellarMass']*1e10 
    t['SubhaloMass'] = t['SubhaloMass']*1e10 

    # Create logSubhaloMass and logStellarMass 
    t['logSubhaloMass'] = np.log10(t['SubhaloMass'])
    t['logStellarMass'] = np.log10(t['StellarMass'])
    
    t['x'] = _convert_tng_position(t['x'])
    t['y'] = _convert_tng_position(t['y'])
    t['z'] = _convert_tng_position(t['z'])
    return t

    
def import_hydro_prop(file_number, directory, check_subhalo_flag):
    """Same as import_hydro but returned astropy table contains
    additional properties.
    
    Return:
        Outputs an astropy table with 'subfind_id', 'SubhaloMass', 
        'SubhaloFlag','StellarMass', 'x', 'y', 'z', 'SFR',
        'Vmax', 'Vdisp', 'SubhaloSpin', 'SubhaloHalfmassRad',
        'logSubhaloMass' and 'logStellarMass' as keys
    """
    subhalo_ids = []
    subhalo_mass = []
    subhalo_flag = []
    stellar_mass = []
    x_position = []
    y_position = []
    z_position = []
    sfr_list = []
    vmax_list = []
    spin_list = []
    vdisp_list = []
    halfmass_list = []
    
    count = 0
    
    for ii in tqdm(range(0, file_number + 1)):
        filename = f'{directory}/fof_subhalo_tab_099.{ii}.hdf5' 
        with h5py.File(filename, 'r') as f:
            if 'SubhaloMass' in f['Subhalo'].keys(): 
                for s in f['Subhalo']['SubhaloMass']:
                    subhalo_mass.append(s)
                    subhalo_ids.append(count)
                    count+=1
                for flag in f['Subhalo']['SubhaloFlag']:
                    subhalo_flag.append(flag)
                for star in f['Subhalo']['SubhaloMassInRadType'][:,4]: 
                    stellar_mass.append(star)
                for (x, y, z) in f['Subhalo']['SubhaloPos'][:]:
                    x_position.append(x)
                    y_position.append(y)
                    z_position.append(z) 
                for sfr in f['Subhalo']['SubhaloSFR']:
                    sfr_list.append(sfr) # units Msun/yr
                for v in f['Subhalo']['SubhaloVmax']:
                    vmax_list.append(v)
                for vd in f['Subhalo']['SubhaloVelDisp']:
                    vdisp_list.append(vd)  
                for (spinx, spiny, spinz) in f['Subhalo']['SubhaloSpin'][:]:
                    spin_list.append(np.sqrt(spinx**2 + spiny**2 + spinz**2))
                for halfmass in f['Subhalo']['SubhaloHalfmassRad']:
                    halfmass_list.append(halfmass)
                    
    t = Table(data = [subhalo_ids, subhalo_mass, subhalo_flag, stellar_mass, 
                      x_position, y_position, z_position, sfr_list, vmax_list, 
                      vdisp_list, spin_list, halfmass_list], 
              names = ['subfind_id', 'SubhaloMass', 'SubhaloFlag', 'StellarMass', 'x', 'y', 'z', 
                       'SFR', 'Vmax', 'Vdisp', 'SubhaloSpin', 'SubhaloHalfmassRad'])
    
    if check_subhalo_flag == True:
        t = _check_flag(t)

    # Convert from TNG units to Msun/h
    t['StellarMass'] = t['StellarMass']*1e10 
    t['SubhaloMass'] = t['SubhaloMass']*1e10 

    # Create logSubhaloMass and logStellarMass 
    t['logSubhaloMass'] = np.log10(t['SubhaloMass'])
    t['logStellarMass'] = np.log10(t['StellarMass'])
    
    t['x'] = _convert_tng_position(t['x'])
    t['y'] = _convert_tng_position(t['y'])
    t['z'] = _convert_tng_position(t['z'])
    return t


def import_dmo(file_number, directory):
    """Reads in data in the form of h5py files from TNG100-Dark
    Args:
        file_number: the desired number of files to be read in
        directory: string that details path from location 
        of notebook to dataset, i.e. 'Data/dmo_data'
    Return:
        Outputs an astropy table with 'subfind_id' and 'SubhaloMass'
        as keys
    Note:
        'SubhaloMass' units Msun/h
    """
    subhalo_ids = []
    subhalo_mass = []
    subhalo_flag = []
    x_position = []
    y_position = []
    z_position = []
    v_max = []
    count = 0
    
    for ii in tqdm(range(0, file_number + 1)):
        filename = f'{directory}/fof_subhalo_tab_099.{ii}.hdf5' 
        with h5py.File(filename, 'r') as f:
            if 'SubhaloMass' in f['Subhalo'].keys(): 
                for s in f['Subhalo']['SubhaloMass']:
                    subhalo_mass.append(s)
                    subhalo_ids.append(count)
                    count += 1
                for v in f['Subhalo']['SubhaloVmax']:
                    v_max.append(v)
                for (x, y, z) in f['Subhalo']['SubhaloPos'][:]:
                    x_position.append(x)
                    y_position.append(y)
                    z_position.append(z)
                          
    t = Table(data = [subhalo_ids, subhalo_mass, x_position, y_position, z_position, v_max], 
              names = ['subfind_id', 'SubhaloMass', 'x', 'y', 'z', 'Vmax'])

    # Convert from TNG units to Msun/h
    t['SubhaloMass'] = t['SubhaloMass']*1e10 

    # Convert units
    t['x'] = _convert_tng_position(t['x'])
    t['y'] = _convert_tng_position(t['y'])
    t['z'] = _convert_tng_position(t['z'])
    return t

    
def import_color(directory):
    """Reads in color data from a given directory
    Args:
        directory: string that details path from location
    Return:
        Outputs pandas data fram with columns "sdss_u",
        "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wfc_acs_f606w", 
        "des_y", "jwst_f150w", and "subfind_id"

    Note: Use to compute "g - i" for red/blue galaxies
    """
    filename = f'{directory}/Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc_099.hdf5' 
    f = h5py.File(filename, "r")
    colnames = (
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_i",
        "sdss_z",
        "wfc_acs_f606w",
        "des_y",
        "jwst_f150w",
    )
    arr = f["Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc"][:]
    arr = np.mean(arr, axis=-1) # averaging over projections
    df_color = pd.DataFrame(arr, columns=colnames)
    df_color["subfind_id"] = f["subhaloIDs"]
    return df_color


def import_tree_dmo(directory):
    """Reads in merger tree data from given directory
    Args: 
        directory: string that details path to file
    Return:
        out: dictionary with keys:
        'subfind_id', 'vpeak_pre', 'vpeak', 
        'mpeak_pre', 'mpeak', 'mah', 'is_sub', 
        'mpeak_mah', 'a12'
    """
    out = {}
    filename = f'{directory}/out-trees-tng100-dmo.npz'
    npzfile = np.load(filename)
    for kk in npzfile.files:
        out[kk] = npzfile[kk]
    return out


def import_tree_bar(directory):
    """Reads in merger tree data from given directory
    Args: 
        directory: string that details path to file
    Return:
        out: dictionary with keys:
        'subfind_id', 'vpeak', 'mpeak_mah', 
        'is_sub', 'a12', 'mpeak_pre', 
        'mpeak', 'stellar_mass'
    """
    out = {}
    filename = f'{directory}/out-trees-tng100-bar.npz'
    npzfile = np.load(filename)
    for kk in npzfile.files:
        out[kk] = npzfile[kk]
    return out

