import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import plot
import stats
import cam
import paths
from constants import TNG_H, LBOX, HYDRO_STELLAR_MASS_MIN, MASS_BIN_EDGES
from halotools.mock_observables import return_xyz_formatted_array, tpcf_jackknife

COLOR_CUT = 0.85

## Load data ###
dmo_full = pd.read_csv(paths.DMO_TREE_CACHE, usecols=['SubhaloMass', 'logSubhaloMass',
                                                                      'vpeak', 'vpeak_pre',
                                                                      'mpeak_pre', 'mpeak',
                                                                      'x', 'y', 'z', 'a12'])
print("\nDMO Table:")
print(len(dmo_full))

hydro_full = pd.read_csv(paths.HYDRO_COLOR_TREE_CACHE, usecols=['SubhaloMass', 'StellarMass', 'x', 'y', 'z',
                                                                           'logStellarMass', 'logSubhaloMass', 'g-i'])
print("\nHydro Table")
print(len(hydro_full))
print("\nCache files read in!")

# minimum on Stellar Mass for hydro catalog
hydro_full = hydro_full[hydro_full['StellarMass'] >= HYDRO_STELLAR_MASS_MIN]
print("\nNew Hydro Table Length:")
print(len(hydro_full))

# Pick top whatever of DMO and then do SHAM
dmo_full = dmo_full.sort_values('vpeak', ascending=False).head(len(hydro_full))
print("\nNew DMO Table Length:")
print(len(dmo_full))

# Add z12 column to dmo_full

dmo_full['z12'] = (1 / dmo_full['a12']) - 1


### Run CAM predictions for bins ###
mass_bins = {f'bin{i+1}': edges for i, edges in enumerate(MASS_BIN_EDGES)}

cam_results = {}

for key, (low, high) in mass_bins.items(): 
    color_bin, color_pred, mask_gal, mask_halo = cam.cam(
        dmo_full['vpeak'].to_numpy(), 
        hydro_full['logStellarMass'].to_numpy(), 
        dmo_full['z12'].to_numpy(), 
        hydro_full['g-i'].to_numpy(), 
        low, high)

    dmo_bin = dmo_full[mask_halo].copy()
    hydro_bin = hydro_full[mask_gal].copy()
    dmo_bin['color_pred'] = color_pred
    hydro_bin['color'] = color_bin

    cam_results[key] = {'dmo': dmo_bin, 'hydro': hydro_bin}

print("\nCAM predictions complete for all bins.")




## Share Axes
### Histogram of color vs. predicted color ###
plt.figure(figsize=(18, 4))

for i, key in enumerate(mass_bins.keys()):
    plt.subplot(1, len(mass_bins), i+1)

    # Access the right data for this key
    hydro_color = cam_results[key]['hydro']['color']
    dmo_color_pred = cam_results[key]['dmo']['color_pred']

    plt.hist(hydro_color, bins=40, density=True, alpha = 0.4, label='True color (Hydro)')
    plt.hist(dmo_color_pred, bins=40, density=True, alpha = 0.4, label='Predicted color (CAM)')
    plt.axvline(COLOR_CUT, color='k', linestyle='--')
    low, high = mass_bins[key]
    plt.title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    plt.xlabel(r'$g - i$ color')
    plt.ylabel('Normalized frequency')

    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()



         
### Define red/blue galaxies ###
for key, res in cam_results.items():
    dmo = res['dmo']
    hydro = res['hydro']

    res['red_dmo'] = dmo[dmo['color_pred'] > COLOR_CUT]
    res['blue_dmo'] = dmo[dmo['color_pred'] <= COLOR_CUT]
    res['red_hydro'] = hydro[hydro['color'] > COLOR_CUT]
    res['blue_hydro'] = hydro[hydro['color'] <= COLOR_CUT]

print("\nRed and blue galaxies defined for all bins.")





### Histogram of true vs. predicted red/blue gal ###
plt.figure(figsize=(18, 4))

for i, key in enumerate(mass_bins.keys()):
    plt.subplot(1, len(mass_bins), i+1)

    # Access data
    red_true = cam_results[key]['red_hydro']['color']
    blue_true = cam_results[key]['blue_hydro']['color']
    red_pred = cam_results[key]['red_dmo']['color_pred']
    blue_pred = cam_results[key]['blue_dmo']['color_pred']

    # Plot true (Hydro) histograms
    plt.hist(red_true, bins=40, density=True, color='red', alpha=0.7, histtype = 'step', label= r'Truth Red: TNG100 Hydro $\xi(r)$')
    plt.hist(blue_true, bins=40, density=True, color='blue', alpha=0.7, histtype = 'step', label= r'Truth Blue: TNG100 Hydro $\xi(r)$')

    # Plot predicted (CAM) histograms
    plt.hist(red_pred, bins=40, density=True, color='red', alpha=0.4, linewidth=2, label= r'Prediction Red: CAM with $V_{peak}$ and $a_{1/2}$ $\xi(r)$')
    plt.hist(blue_pred, bins=40, density=True, color='blue', alpha=0.4, linewidth=2, label=r'Prediction Blue: CAM with $V_{peak}$ and $a_{1/2}$ $\xi(r)$')

    # Vertical color cut
    plt.axvline(COLOR_CUT, color='k', linestyle='--')
    low, high = mass_bins[key]
    # Labels and layout
    plt.title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    plt.xlabel(r'$g - i$ color')
    plt.ylabel('Normalized frequency')

    if i == 0:
        plt.legend(fontsize=8)

plt.tight_layout()
plt.show()





### Compute TPCF ###
Lbox = LBOX
rbins = np.logspace(np.log10(0.1), np.log10(24.9), 16)
rbin_centers = 0.5 * (rbins[1:] + rbins[:-1])
randoms = stats.generate_randoms(Lbox, 50000)
print("\nGenerated 50000 randoms.")

# Store all xi and cov results here
tpcf_results = {
    'hydro': [],
    'cam': []
}

for key, res in cam_results.items():
    print(f"\nComputing TPCF for {key}.") 

    xi_red_dmo, cov_red_dmo = tpcf_jackknife(
        return_xyz_formatted_array(res['red_dmo']['x'] * TNG_H, res['red_dmo']['y'] * TNG_H, res['red_dmo']['z'] * TNG_H),
        randoms, rbins, Nsub=3, period=Lbox, num_threads='max'
    )
    xi_blue_dmo, cov_blue_dmo = tpcf_jackknife(
        return_xyz_formatted_array(res['blue_dmo']['x'] * TNG_H, res['blue_dmo']['y'] * TNG_H, res['blue_dmo']['z'] * TNG_H),
        randoms, rbins, Nsub=3, period=Lbox, num_threads='max'
    )
    xi_red_hydro, cov_red_hydro = tpcf_jackknife(
        return_xyz_formatted_array(res['red_hydro']['x'] * TNG_H, res['red_hydro']['y'] * TNG_H, res['red_hydro']['z'] * TNG_H),
        randoms, rbins, Nsub=3, period=Lbox, num_threads='max'
    )
    xi_blue_hydro, cov_blue_hydro = tpcf_jackknife(
        return_xyz_formatted_array(res['blue_hydro']['x'] * TNG_H, res['blue_hydro']['y'] * TNG_H, res['blue_hydro']['z'] * TNG_H),
        randoms, rbins, Nsub=3, period=Lbox, num_threads='max'
    )

    tpcf_results['cam'].append(((xi_red_dmo, xi_blue_dmo), (cov_red_dmo, cov_blue_dmo)))
    tpcf_results['hydro'].append(((xi_red_hydro, xi_blue_hydro), (cov_red_hydro, cov_blue_hydro)))

print("\nTPCF computed for all bins.")

#### Print number of gal in each stellar mass bin and by color ###



### Plot results ###
fig1, axes = plt.subplots(1, 3, figsize=(19, 7), sharey=True)
for i, key in enumerate(mass_bins.keys()):
    (xi_red_dmo, xi_blue_dmo), _ = tpcf_results['cam'][i]
    (xi_red_hydro, xi_blue_hydro), _ = tpcf_results['hydro'][i]
    low, high = mass_bins[key]
    ax = axes[i]

    ax.plot(rbin_centers, xi_red_dmo, 'r-o', linestyle = "--", label=r'Prediction Red: CAM with $V_{peak}$ and $a_{1/2}$ $\xi(r)$')
    ax.plot(rbin_centers, xi_blue_dmo, 'b-o', linestyle = "--", label=r'Prediction Blue: CAM with $V_{peak}$ and $a_{1/2}$ $\xi(r)$')
    ax.plot(rbin_centers, xi_red_hydro, 'r-o', label=r'Truth Red: TNG100 Hydro $\xi(r)$')
    ax.plot(rbin_centers, xi_blue_hydro, 'b-o', label=r'Truth Blue: TNG100 Hydro $\xi(r)$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    ax.set_xlabel(r'$r \, [\mathrm{Mpc}/h]$')
    if i == 0:
        ax.set_ylabel(r'$\xi(r)$')
    if i == 2:
        ax.legend()

plt.subplots_adjust(wspace=0, left=0.05, right=0.95, bottom=0.1, top=0.9)
plt.show()


### Plot results ###
fig2, axes = plt.subplots(1, 3, figsize=(19, 3), sharey=True)
for i, key in enumerate(mass_bins.keys()):
    (xi_red_dmo, xi_blue_dmo), (cov_red_dmo, cov_blue_dmo) = tpcf_results['cam'][i]
    (xi_red_hydro, xi_blue_hydro), (cov_red_hydro, cov_blue_hydro) = tpcf_results['hydro'][i]
    
    low, high = mass_bins[key]
    ax = axes[i]
    
    cov = [cov_red_dmo, cov_blue_dmo, cov_red_hydro, cov_blue_hydro]
    err = [stats.compute_tpcf_jackknife_errors(x) for x in cov]

    ax.plot(rbin_centers, xi_red_dmo, 'r-o', linestyle = "--", label= r'Prediction Red: CAM with $V_{peak}$ and $a_{1/2}$ $\xi(r)$')
    ax.fill_between(rbin_centers, xi_red_dmo - err[0], xi_red_dmo + err[0], color= 'red', alpha=0.2)

    ax.plot(rbin_centers, xi_blue_dmo, 'b-o', linestyle = "--", label= r'Prediction Blue: CAM with $V_{peak}$ and $a_{1/2}$ $\xi(r)$')
    ax.fill_between(rbin_centers, xi_blue_dmo - err[1], xi_blue_dmo + err[1], color= 'blue', alpha=0.2)

    ax.plot(rbin_centers, xi_red_hydro, 'r-o', label= r'Truth Red: TNG100 Hydro $\xi(r)$')
    ax.fill_between(rbin_centers, xi_red_hydro - err[2], xi_red_hydro + err[2], color= 'red', alpha=0.3)
    
    ax.plot(rbin_centers, xi_blue_hydro, 'b-o', label= r'Truth Blue: TNG100 Hydro $\xi(r)$')
    ax.fill_between(rbin_centers, xi_blue_hydro - err[3], xi_blue_hydro + err[3], color= 'blue', alpha=0.3)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    ax.set_xlabel(r'$r \, [\mathrm{Mpc}/h]$')
    if i == 0:
        ax.set_ylabel(r'$\xi(r)$')
    if i == 2:
        ax.legend()

plt.subplots_adjust(wspace=0, left=0.05, right=0.95, bottom=0.1, top=0.9)
plt.show()

### Save figures as PNG files ###
figures = [fig1, fig2]
png_filenames = plot.save_figures_as_png(figures, 1, directory=paths.FIGURES_DIR)
print(f'\nFigures saved as PNG files: {png_filenames}')


### Save results with pickle ###
paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
with open(paths.CACHE_DIR / 'tpcf_results_cam.pkl', 'wb') as f:
    pickle.dump({'tpcf': tpcf_results, 'rbins': rbins, 'cam': cam_results}, f)
print("\nSaved TPCF and CAM results to pickle.")

