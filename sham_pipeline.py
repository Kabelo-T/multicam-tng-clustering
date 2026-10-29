import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sample
import stats
import sham
import plot
import paths
from constants import TNG_H, LBOX, HYDRO_STELLAR_MASS_MIN, MASS_BIN_EDGES
from halotools.mock_observables import return_xyz_formatted_array, tpcf_jackknife
import pickle


### Import cache files ###
dmo_path = paths.DMO_TREE_CACHE
dmo_full = pd.read_csv(dmo_path, usecols=['SubhaloMass', 'logSubhaloMass',
                                          'vpeak', 'vpeak_pre', 'mpeak_pre',
                                          'mpeak', 'x', 'y', 'z'])
print("\nDMO Table Length:")
print(len(dmo_full))

hydro_path = paths.HYDRO_COLOR_TREE_CACHE
hydro_full = pd.read_csv(hydro_path, usecols=['SubhaloMass', 'StellarMass', 'x', 'y', 'z',
                                              'logStellarMass', 'logSubhaloMass', 'g-i'])
print("\nHydro Table Length:")
print(len(hydro_full))

print("\nCache files read in!")

# minimum on Stellar Mass for hydro catalog
hydro_full = hydro_full[hydro_full['StellarMass'] >= HYDRO_STELLAR_MASS_MIN]
print("\nNew Hydro Table Length:")
print(len(hydro_full))


### Select top-matched DMO subsets ###
dmo_dict = {
    'vpeak_pre': dmo_full.sort_values('vpeak_pre', ascending=False).head(len(hydro_full)).copy(),
    'vpeak': dmo_full.sort_values('vpeak', ascending=False).head(len(hydro_full)).copy(),
    'mpeak_pre': dmo_full.sort_values('mpeak_pre', ascending=False).head(len(hydro_full)).copy(),
    'mpeak': dmo_full.sort_values('mpeak', ascending=False).head(len(hydro_full)).copy()
}



### SHAM predictions ###
sham_vars = {
    'vpeak_pre': 'vpeak_pre_SM_pred',
    'vpeak': 'vpeak_SM_pred',
    'mpeak_pre': 'mpeak_pre_SM_pred',
    'mpeak': 'mpeak_SM_pred'
}

for var, newcol in sham_vars.items():
    dmo_dict[var][newcol] = sham.sham(
        dmo_dict[var][var].to_numpy(),
        hydro_full['StellarMass'].to_numpy()
    )
    dmo_dict[var][f'log_{newcol}'] = np.log10(dmo_dict[var][newcol].to_numpy())

print("\nSHAM SM predictions computed.")



### Define mass bins ###
bin_edges = MASS_BIN_EDGES

mass_bins = {'hydro': []}
mass_bins.update({key: [] for key in sham_vars.values()})

# Hydro bins
for low, high in bin_edges:
    mass_bins['hydro'].append(sample.mass_sample(hydro_full, 'logStellarMass', low, high))

# SHAM bins
for var, newcol in sham_vars.items():
    for low, high in bin_edges:
        mass_bins[newcol].append(sample.mass_sample(dmo_dict[var], f'log_{newcol}', low, high))

print("\nMass bins defined.")



### Compute TPCF ###
Lbox = LBOX
rbins = np.logspace(np.log10(.1), np.log10(24.9), 16)
rbin_centers = 0.5 * (rbins[1:] + rbins[:-1])
randoms = stats.generate_randoms(Lbox, 50000)

print("\n50000 randoms used!")

tpcf_results = {key: [] for key in mass_bins}

for key, bins in mass_bins.items():
    for i, b in enumerate(bins):
        low, high = bin_edges[i]
        pos = return_xyz_formatted_array(b['x'] * TNG_H, b['y'] * TNG_H, b['z'] * TNG_H) # units Mpc / h
        xi, cov = tpcf_jackknife(pos, randoms, rbins, Nsub=3, period=Lbox, num_threads='max')
        tpcf_results[key].append({
            'bin_range': (low, high),
            'xi': xi,
            'cov': cov
        })
    print(f"\nTPCF computed for {key}")  # Access Example: xi_bin1 = tpcf_results['hydro'][0]['xi']



### Plot settings ###
labels = {
    'hydro': r'Truth: TNG100 Hydro $\xi(r)$',
    'vpeak_pre_SM_pred': r'Prediction: SHAM with $V_{peak\_pre}$ $\xi(r)$',
    'vpeak_SM_pred': r'Prediction: SHAM with $V_{peak}$ $\xi(r)$',
    'mpeak_pre_SM_pred': r'Prediction: SHAM with $M_{peak\_pre}$ $\xi(r)$',
    'mpeak_SM_pred': r'Prediction: SHAM with $M_{peak}$ $\xi(r)$'
}
colors = {
    'hydro': 'k',
    'vpeak_pre_SM_pred': 'g',
    'vpeak_SM_pred': 'r',
    'mpeak_pre_SM_pred': 'b',
    'mpeak_SM_pred': 'm'
}

titles = [
    r'$8.75 \leq \log(M_{*}/h^{-1}M_{\odot}) < 9.3$',
    r'$9.3 \leq \log(M_{*}/h^{-1}M_{\odot}) < 10.0$',
    r'$10.0 \leq \log(M_{*}/h^{-1}M_{\odot}) < 12.5$'
]



### Plotting w/o errors ###
fig1, axes = plt.subplots(1, 3, figsize=(19, 7), sharey=True)

for i, ax in enumerate(axes):
    for key in mass_bins.keys():
        xi = tpcf_results[key][i]['xi']  # Access subkey

        if key != 'hydro':
            ax.plot(rbin_centers, xi, f'{colors[key]}-o', label=labels[key])
        else:
            ax.plot(rbin_centers, xi, f'{colors[key]}-o', linestyle='--', label=labels[key])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(titles[i])
    ax.set_xlabel(r'$r \, \rm{[Mpc/h]}$')
    if i == 0:
        ax.set_ylabel(r'$\xi(r)$')
    if i == 2:
        ax.legend()

plt.subplots_adjust(wspace=0, left=0.05, right=0.95, bottom=0.1, top=0.9)
plt.show()



### Plotting w/ errors ###
fig2, axes = plt.subplots(1, 3, figsize=(19, 7), sharey=True)

for i, ax in enumerate(axes):
    for key in mass_bins.keys():
        xi = tpcf_results[key][i]['xi']
        cov = tpcf_results[key][i]['cov']
        errors = stats.compute_tpcf_jackknife_errors(cov)

        if key != 'hydro':
            ax.plot(rbin_centers, xi, f'{colors[key]}-o', label=labels[key])
            ax.fill_between(rbin_centers, xi - errors, xi + errors, color=colors[key], alpha=0.2)
        else:
            ax.plot(rbin_centers, xi, f'{colors[key]}-o', linestyle='--', label=labels[key])
            ax.fill_between(rbin_centers, xi - errors, xi + errors, color=colors[key], alpha=0.2)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(titles[i])
    ax.set_xlabel(r'$r \, \rm{[Mpc/h]}$')
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



### Cache TPCF with pickle ###
paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
cache_path = paths.CACHE_DIR / 'tpcf_results_sham.pkl'
with open(cache_path, 'wb') as f:
    pickle.dump(tpcf_results, f)
print(f"\ntpcf_results_sham.pkl saved to {cache_path}.")
