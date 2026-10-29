from tqdm import tqdm
from astropy.table import Table
import numpy as np
import h5py
import pandas as pd
import data
import paths
from constants import TNG_H



### Create Hydro + Tree + Color Cache File ###

# Read in Hydro Data
file_number = 447
directory = paths.HYDRO_RAW_DIR
check_subhalo_flag = False
hydro_data = data.import_hydro_prop(file_number, directory, check_subhalo_flag)
print("length hydro_data = ", len(hydro_data))

# Read in color data
color_data = data.import_color(paths.COLOR_RAW_DIR)
print("length color_data = ", len(color_data))

# Read in tree data
_tree_data_bar = data.import_tree_bar(paths.TREE_RAW_DIR)

# Make mpeak_mah columns
mpeak_mah = _tree_data_bar.pop('mpeak_mah')
for ii in range(100):
    mpeak_mahii = mpeak_mah[:, ii]
    _tree_data_bar[f'mpeak_mah{ii}'] = mpeak_mahii

# Make dictonary pandas data frame
tree_data_bar = pd.DataFrame(_tree_data_bar)
print("length tree_data = ", len(tree_data_bar))

# Convert Astropy Table to Pandas DataFrame
hydro_df = hydro_data.to_pandas()
hydro_color_data = pd.merge(hydro_df, color_data, on='subfind_id') 
print("length hydro_color_data = ", len(hydro_color_data))

_complete_data = pd.merge(hydro_color_data, tree_data_bar, on='subfind_id') 
print("length hydro_color_tree_data = ", len(_complete_data))

# Add g-i column
_complete_data['g-i'] = _complete_data['sdss_g'] - _complete_data['sdss_i'] 

# Check SubhaloFlag
complete_data = _complete_data[(_complete_data['SubhaloFlag'] == 1)]
print("length hydro_color_tree_data after subhaloflag check = ", len(complete_data))

# Save _complete_data and complete_data to cache files
paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)

_complete_data.to_csv(paths.HYDRO_COLOR_TREE_NO_FLAG_CACHE)
print(f'Hydro + Color + Tree + No SubhaloFlag Check Cache File saved to {paths.HYDRO_COLOR_TREE_NO_FLAG_CACHE}')

complete_data.to_csv(paths.HYDRO_COLOR_TREE_CACHE)
print(f'Hydro + Color + Tree + SubhaloFlag Check Cache File saved to {paths.HYDRO_COLOR_TREE_CACHE}')




### Create DMO + Tree Cache File ###

# Read in TNG100-Dark Data
file_number = 63
directory = paths.DMO_RAW_DIR
dmo_data = data.import_dmo(file_number, directory)

# Create 'logSubhaloMass' 
dmo_data['logSubhaloMass'] = np.log10(dmo_data['SubhaloMass'])
print("length dmo_data = ", len(dmo_data))

# Read in tree data
_tree_data_dmo = data.import_tree_dmo(paths.TREE_RAW_DIR)

# Make mah and mpeak_mah columns
mah = _tree_data_dmo.pop('mah')
for ii in range(100):
    mahii = mah[:, ii]
    _tree_data_dmo[f'mah{ii}'] = mahii

mpeak_mah = _tree_data_dmo.pop('mpeak_mah')
for ii in range(100):
    mpeak_mahii = mpeak_mah[:, ii]
    _tree_data_dmo[f'mpeak_mah{ii}'] = mpeak_mahii

# Make dictonary pandas data frame
tree_data = pd.DataFrame(_tree_data_dmo)
print("length tree_data = ", len(tree_data))

# Convert Astropy Table to Pandas DataFrame
dmo_df = dmo_data.to_pandas()
full_data = pd.merge(dmo_df, tree_data, on='subfind_id') 
print("length dmo_tree_data = ", len(full_data))

# Save full_data to cache file
paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
full_data.to_csv(paths.DMO_TREE_CACHE)
print(f'DMO + Tree Cache File saved to {paths.DMO_TREE_CACHE}')


