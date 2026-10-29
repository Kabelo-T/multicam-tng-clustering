import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr
import stats
import sham
import paths
from constants import TNG_H, LBOX, HYDRO_STELLAR_MASS_MIN, MASS_BIN_EDGES
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables.two_point_clustering import tpcf_jackknife, tpcf
from multicam.multicam import MultiCAM, MultiCamSampling

# Hydro-only comparison of two ways of predicting (M*, g-i) from halo properties.
# No DMO data is used: every halo property comes from the hydro catalog's own trees.
#
# The halo features are set by FEATURES (from the environment, default 'z12'):
#   'z12': color feature z12;         M*/color features (z12, log mpeak)
#   'mah': color feature full MAH;    M*/color features (full MAH, log mpeak)
# where the full MAH is mpeak_mah0..99.
#
# SHAM + MultiCAM: stellar mass is assigned by SHAM of hydro mpeak (peak total SubhaloMass)
# onto the hydro stellar mass distribution. In each SHAM stellar-mass bin, color comes from
# MultiCAM (color features -> g-i) fit on the true stellar-mass bin and applied to the
# SHAM bin (with FEATURES='z12' this is the MultiCAM in mcam_vs_cam.py).
#
# MultiCAM (M*, color): one MultiCAM with the M*/color features and 2 targets
# (g-i, log M*), fit and applied on the full hydro sample. Galaxies are binned on the
# predicted stellar mass and use the predicted color. Unlike SHAM, the stellar mass can
# depend on assembly history at fixed mpeak (assembly bias).
#
# The two methods put different galaxies in each bin, so the halo-by-halo color comparison
# uses the galaxies that fall in the bin under both predicted stellar masses.

COLOR_CUT = 0.85
N_THREADS = 16
N_SAMPLES = 100
SAMPLE_PCTILES = (16, 84)
# MultiCamSampling (scatter) mode; slow because of the N_SAMPLES tpcf draws.
# Set from the environment (e.g. mcam_hydro_only.sbatch) with RUN_SAMPLING=True
RUN_SAMPLING = os.environ.get('RUN_SAMPLING', 'False') == 'True'
# Halo features; set from the environment with FEATURES=mah (see header)
FEATURES = os.environ.get('FEATURES', 'z12')
if FEATURES == 'mah':
    FIGURES_SUBDIR = paths.FIGURES_DIR / 'mcam_full_mah'
else:
    FIGURES_SUBDIR = paths.FIGURES_DIR / ('sampling' if RUN_SAMPLING else 'no_sampling')
MAH_COLS = [f'mpeak_mah{ii}' for ii in range(100)]
COLOR_FEATURES = {'z12': ['z12'], 'mah': MAH_COLS}[FEATURES]
MCAM_FEATURES = COLOR_FEATURES + ['logMpeak']
# output file prefix, so each feature set has its own figures, diagnostics and pickle
OUT_NAME = 'mcam_hydro_only' + ('' if FEATURES == 'z12' else f'_{FEATURES}')
METHODS = ['sham', 'mcam']
FEATURE_LABELS = {'z12': (r'$z_{1/2}$', r'$z_{1/2}$ + $M_{\rm peak}$'),
                  'mah': ('MAH', r'MAH + $M_{\rm peak}$')}[FEATURES]
METHOD_LABELS = {'sham': f'SHAM + MultiCAM ({FEATURE_LABELS[0]})',
                 'mcam': fr'MultiCAM ($M_*$, color; {FEATURE_LABELS[1]})'}

## Load data ###
hydro_full = pd.read_csv(paths.HYDRO_COLOR_TREE_CACHE, usecols=['SubhaloMass', 'StellarMass', 'x', 'y', 'z',
                                                                'logStellarMass', 'logSubhaloMass', 'g-i',
                                                                'a12', 'mpeak', 'is_sub',
                                                                *(MAH_COLS if FEATURES == 'mah' else [])])
print("\nHydro Table")
print(len(hydro_full))
print("\nCache files read in!")

# minimum on Stellar Mass for hydro catalog
hydro_full = hydro_full[hydro_full['StellarMass'] >= HYDRO_STELLAR_MASS_MIN].copy()
print("\nNew Hydro Table Length:")
print(len(hydro_full))

# Add z12 and log mpeak columns
hydro_full['z12'] = (1 / hydro_full['a12']) - 1
hydro_full['logMpeak'] = np.log10(hydro_full['mpeak'])


### Stellar mass predictions for the full sample ###
# SHAM: mpeak rank-matched onto the observed stellar mass distribution
hydro_full['logStellarMass_sham'] = sham.sham(hydro_full['logMpeak'].to_numpy(),
                                              hydro_full['logStellarMass'].to_numpy())

# MultiCAM: M*/color features -> (g-i, log M*), fit and applied on the full sample
features_mcam = hydro_full[MCAM_FEATURES].to_numpy()
targets_mcam = hydro_full[['g-i', 'logStellarMass']].to_numpy()
model_mcam = MultiCAM(len(MCAM_FEATURES), 2)
model_mcam.fit(features_mcam, targets_mcam)
pred_mcam = model_mcam.predict(features_mcam)
hydro_full['color_pred_mcam'] = pred_mcam[:, 0]
hydro_full['logStellarMass_mcam'] = pred_mcam[:, 1]

if RUN_SAMPLING:
    model_mcam_samp = MultiCamSampling(len(MCAM_FEATURES), 2)
    model_mcam_samp.fit(features_mcam, targets_mcam)
    mcam_samp_draws = np.array([model_mcam_samp.sample(features_mcam) for _ in range(N_SAMPLES)])

mstar_diag = {}
for name in METHODS:
    resid = hydro_full[f'logStellarMass_{name}'] - hydro_full['logStellarMass']
    mstar_diag[name] = {'spearman_mstar': spearmanr(hydro_full[f'logStellarMass_{name}'],
                                                    hydro_full['logStellarMass']).statistic,
                        'mstar_resid_std_dex': np.std(resid)}
    print(f"\n{METHOD_LABELS[name]} log M*: Spearman with truth = {mstar_diag[name]['spearman_mstar']:.3f}, "
          f"scatter about truth = {mstar_diag[name]['mstar_resid_std_dex']:.3f} dex")


### Run predictions for bins ###
mass_bins = {f'bin{i+1}': edges for i, edges in enumerate(MASS_BIN_EDGES)}

results = {}
diagnostics = []

for key, (low, high) in mass_bins.items():
    mask_gal = (hydro_full['logStellarMass'] >= low) & (hydro_full['logStellarMass'] < high)
    mask_sham = (hydro_full['logStellarMass_sham'] >= low) & (hydro_full['logStellarMass_sham'] < high)
    mask_mcam = (hydro_full['logStellarMass_mcam'] >= low) & (hydro_full['logStellarMass_mcam'] < high)

    hydro_bin = hydro_full[mask_gal].copy()  # truth and SHAM + MultiCAM training set (true stellar mass)
    sham_bin = hydro_full[mask_sham].copy()  # SHAM + MultiCAM population (SHAM stellar mass)
    mcam_bin = hydro_full[mask_mcam].copy()  # MultiCAM (M*, color) population (predicted stellar mass)
    hydro_bin['color'] = hydro_bin['g-i']

    # SHAM + MultiCAM: fit on true-bin (color features, g-i) pairs, applied to the SHAM bin
    x_train = hydro_bin[COLOR_FEATURES].to_numpy()
    y_train = hydro_bin[['g-i']].to_numpy()
    x_sham = sham_bin[COLOR_FEATURES].to_numpy()

    model = MultiCAM(len(COLOR_FEATURES), 1)
    model.fit(x_train, y_train)
    sham_bin['color_pred_sham'] = model.predict(x_sham)[:, 0]

    samp_draws = None
    if RUN_SAMPLING:
        model_samp = MultiCamSampling(len(COLOR_FEATURES), 1)
        model_samp.fit(x_train, y_train)
        # each draw: (population mask over hydro_full, predicted colors of that population)
        samp_draws = {'sham': [(mask_sham.to_numpy(), model_samp.sample(x_sham)[:, 0]) for _ in range(N_SAMPLES)],
                      'mcam': []}
        for draw in mcam_samp_draws:
            in_bin = (draw[:, 1] >= low) & (draw[:, 1] < high)
            samp_draws['mcam'].append((in_bin, draw[in_bin, 0]))

    pops = {'hydro': (hydro_bin, 'color'),
            'sham': (sham_bin, 'color_pred_sham'),
            'mcam': (mcam_bin, 'color_pred_mcam')}

    # Halo-by-halo color comparison on the overlap of the SHAM and MultiCAM bins
    common = sham_bin.index.intersection(mcam_bin.index)
    sham_common = sham_bin.loc[common, 'color_pred_sham'].to_numpy()
    mcam_common = mcam_bin.loc[common, 'color_pred_mcam'].to_numpy()
    rho_colors = spearmanr(sham_common, mcam_common).statistic
    diff = mcam_common - sham_common

    print(f"\n{key} [{low}, {high}): N_hydro = {len(hydro_bin)}, N_sham = {len(sham_bin)}, N_mcam = {len(mcam_bin)}")
    diag = {'bin': key, 'logmstar_min': low, 'logmstar_max': high,
            'n_hydro': len(hydro_bin), 'n_sham': len(sham_bin), 'n_mcam': len(mcam_bin),
            'n_overlap_sham_mcam': len(common),
            'spearman_train_z12_gi': spearmanr(hydro_bin['z12'], hydro_bin['g-i']).statistic,
            'spearman_sham_mcam_colors': rho_colors,
            'median_diff_mcam_sham_colors': np.median(diff)}
    for name in METHODS:
        # fraction of true-bin galaxies that the method also places in the bin
        diag[f'overlap_frac_{name}'] = len(pops[name][0].index.intersection(hydro_bin.index)) / len(hydro_bin)
    print(f"  true-bin overlap: SHAM {diag['overlap_frac_sham']:.3f}, MultiCAM {diag['overlap_frac_mcam']:.3f}")
    print(f"  Spearman(SHAM + MultiCAM, MultiCAM) colors, overlap of the two bins (N = {len(common)}): {rho_colors:.4f}")
    print(f"  median (MultiCAM - SHAM + MultiCAM) color, overlap:  {np.median(diff):.4f}")

    # Red and satellite fractions of each red/blue sample
    for name, (pop, col) in pops.items():
        red = pop[col] > COLOR_CUT
        diag[f'red_frac_{name}'] = red.mean()
        diag[f'sat_frac_red_{name}'] = pop.loc[red, 'is_sub'].mean()
        diag[f'sat_frac_blue_{name}'] = pop.loc[~red, 'is_sub'].mean()
    print("  red fraction:                     " + ", ".join(
        f"{name} {diag[f'red_frac_{name}']:.3f}" for name in pops))
    print("  satellite fraction (red / blue):  " + ", ".join(
        f"{name} {diag[f'sat_frac_red_{name}']:.3f} / {diag[f'sat_frac_blue_{name}']:.3f}" for name in pops))
    if RUN_SAMPLING:
        for name in METHODS:
            diag[f'red_frac_{name}_samp'] = np.mean(np.concatenate([c for _, c in samp_draws[name]]) > COLOR_CUT)
    diagnostics.append(diag)

    results[key] = {'hydro': hydro_bin, 'sham': sham_bin, 'mcam': mcam_bin, 'common': common,
                    'samp_draws': samp_draws}

print("\nPredictions complete for all bins.")

suffix = '_with_sampling' if RUN_SAMPLING else ''
paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
diagnostics_df = pd.DataFrame(diagnostics)
for name in METHODS:
    for stat, value in mstar_diag[name].items():
        diagnostics_df[f'{stat}_{name}'] = value
diagnostics_file = paths.CACHE_DIR / f'{OUT_NAME}_diagnostics{suffix}.csv'
diagnostics_df.to_csv(diagnostics_file, index=False)
print(f"\nSaved diagnostics to {diagnostics_file}")


### Histogram of color vs. predicted color ###
fig0 = plt.figure(figsize=(18, 4))

for i, key in enumerate(mass_bins.keys()):
    plt.subplot(1, len(mass_bins), i+1)

    res = results[key]
    plt.hist(res['hydro']['color'], bins=40, density=True, alpha=0.4, label='True color (Hydro)')
    plt.hist(res['sham']['color_pred_sham'], bins=40, density=True, histtype='step', linewidth=2, label=METHOD_LABELS['sham'])
    plt.hist(res['mcam']['color_pred_mcam'], bins=40, density=True, histtype='step', linewidth=2, linestyle='--', label=METHOD_LABELS['mcam'])
    plt.axvline(COLOR_CUT, color='k', linestyle='--')
    low, high = mass_bins[key]
    plt.title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    plt.xlabel(r'$g - i$ color')
    plt.ylabel('Normalized frequency')

    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()


### SHAM + MultiCAM vs MultiCAM color, halo by halo on the overlap of their bins ###
fig1, axes = plt.subplots(1, len(mass_bins), figsize=(18, 5))
for i, key in enumerate(mass_bins.keys()):
    res = results[key]
    ax = axes[i]
    common = res['common']
    ax.scatter(res['sham'].loc[common, 'color_pred_sham'], res['mcam'].loc[common, 'color_pred_mcam'], s=1, color='k')
    lims = [res['hydro']['color'].min(), res['hydro']['color'].max()]
    ax.plot(lims, lims, 'r--', linewidth=1)
    low, high = mass_bins[key]
    ax.set_title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    ax.set_xlabel(r'SHAM + MultiCAM $g - i$ (overlap of predicted $M_*$ bins)')
    if i == 0:
        ax.set_ylabel(r'MultiCAM ($M_*$, color) $g - i$')

plt.tight_layout()
plt.show()


### Define red/blue galaxies ###
for key, res in results.items():
    res['red_hydro'] = res['hydro'][res['hydro']['color'] > COLOR_CUT]
    res['blue_hydro'] = res['hydro'][res['hydro']['color'] <= COLOR_CUT]
    for name in METHODS:
        pop = res[name]
        res[f'red_{name}'] = pop[pop[f'color_pred_{name}'] > COLOR_CUT]
        res[f'blue_{name}'] = pop[pop[f'color_pred_{name}'] <= COLOR_CUT]

print("\nRed and blue galaxies defined for all bins.")


### Compute TPCF ###
Lbox = LBOX
rbins = np.logspace(np.log10(0.1), np.log10(24.9), 16)
rbin_centers = 0.5 * (rbins[1:] + rbins[:-1])
randoms = stats.generate_randoms(Lbox, 50000)
print("\nGenerated 50000 randoms.")


def xyz(table):
    return return_xyz_formatted_array(table['x'] * TNG_H, table['y'] * TNG_H, table['z'] * TNG_H)


def jackknife(table):
    return tpcf_jackknife(xyz(table), randoms, rbins, Nsub=3, period=Lbox, num_threads=N_THREADS)


# Store all xi and cov results here; sampling stores (median, lo, hi) over draws
tpcf_results = {
    'hydro': [],
    'sham': [],
    'mcam': [],
    'sham_samp': [],
    'mcam_samp': [],
}

for key, res in results.items():
    print(f"\nComputing TPCF for {key}.")

    for name in ['hydro', *METHODS]:
        xi_red, cov_red = jackknife(res[f'red_{name}'])
        xi_blue, cov_blue = jackknife(res[f'blue_{name}'])
        tpcf_results[name].append(((xi_red, xi_blue), (cov_red, cov_blue)))

    if not RUN_SAMPLING:
        continue

    # MultiCAM Sampling: tpcf of each draw, median and percentile band across draws
    for name in METHODS:
        xi_red_draws, xi_blue_draws = [], []
        for in_bin, color_draw in res['samp_draws'][name]:
            pop = hydro_full[in_bin]
            red = color_draw > COLOR_CUT
            xi_red_draws.append(tpcf(xyz(pop[red]), rbins, randoms=randoms, period=Lbox, num_threads=N_THREADS))
            xi_blue_draws.append(tpcf(xyz(pop[~red]), rbins, randoms=randoms, period=Lbox, num_threads=N_THREADS))

        summary = []
        for draws in [np.array(xi_red_draws), np.array(xi_blue_draws)]:
            summary.append((np.median(draws, axis=0),
                            np.percentile(draws, SAMPLE_PCTILES[0], axis=0),
                            np.percentile(draws, SAMPLE_PCTILES[1], axis=0)))
        tpcf_results[f'{name}_samp'].append(tuple(summary))

print("\nTPCF computed for all bins.")


### Plot results: xi(r) and ratio to hydro truth ###
# The two predictions are offset slightly in r so their jackknife error bars don't overlap
r_offset = {'sham': rbin_centers * 0.97, 'mcam': rbin_centers * 1.03}
styles = {'sham': {'marker': 's', 'linestyle': '--'}, 'mcam': {'marker': 'x', 'linestyle': ':'}}
samp_colors = {('sham', 'red'): 'orange', ('sham', 'blue'): 'green',
               ('mcam', 'red'): 'magenta', ('mcam', 'blue'): 'cyan'}

fig2, axes = plt.subplots(2, len(mass_bins), figsize=(19, 9), sharex=True, sharey='row',
                          gridspec_kw={'height_ratios': [3, 1]})
for i, key in enumerate(mass_bins.keys()):
    low, high = mass_bins[key]
    ax, rax = axes[0, i], axes[1, i]

    for c, color in enumerate(['red', 'blue']):
        xi_hydro = tpcf_results['hydro'][i][0][c]
        err = stats.compute_tpcf_jackknife_errors(tpcf_results['hydro'][i][1][c])
        label = color.capitalize()

        ax.plot(rbin_centers, xi_hydro, marker='o', color=color, label=fr'Truth {label}: TNG100 Hydro $\xi(r)$')
        ax.fill_between(rbin_centers, xi_hydro - err, xi_hydro + err, color=color, alpha=0.2)
        rax.fill_between(rbin_centers, 1 - err / xi_hydro, 1 + err / xi_hydro, color=color, alpha=0.2)

        for name in METHODS:
            xi_pred = tpcf_results[name][i][0][c]
            err_pred = stats.compute_tpcf_jackknife_errors(tpcf_results[name][i][1][c])
            line_color = color if name == 'sham' else ('k' if color == 'red' else 'gray')
            ax.errorbar(r_offset[name], xi_pred, yerr=err_pred, color=line_color, capsize=2, **styles[name],
                        label=fr'Prediction {label}: {METHOD_LABELS[name]} $\xi(r)$')
            rax.errorbar(r_offset[name], xi_pred / xi_hydro, yerr=err_pred / xi_hydro, color=line_color, capsize=2,
                         **styles[name])

            if RUN_SAMPLING:
                xi_s, xi_s_lo, xi_s_hi = tpcf_results[f'{name}_samp'][i][c]
                ax.plot(rbin_centers, xi_s, color=samp_colors[(name, color)], linestyle='-.',
                        label=fr'Prediction {label}: {METHOD_LABELS[name]} Sampling $\xi(r)$ (median of {N_SAMPLES})')
                ax.fill_between(rbin_centers, xi_s_lo, xi_s_hi, color=samp_colors[(name, color)], alpha=0.2)
                rax.plot(rbin_centers, xi_s / xi_hydro, color=samp_colors[(name, color)], linestyle='-.')

    rax.axhline(1, color='k', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    rax.set_xlabel(r'$r \, [\mathrm{Mpc}/h]$')
    if i == 0:
        ax.set_ylabel(r'$\xi(r)$')
        rax.set_ylabel(r'$\xi_{\rm pred} / \xi_{\rm hydro}$')
    if i == len(mass_bins) - 1:
        ax.legend(fontsize=8)

axes[1, 0].set_ylim(0.5, 1.5)
plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95, bottom=0.1, top=0.9)
plt.show()

### Save figures as PNG files ###
figure_names = {
    f'{OUT_NAME}_color_hist': fig0,
    f'{OUT_NAME}_color_scatter': fig1,
    f'{OUT_NAME}_red_blue_tpcf': fig2,
}
FIGURES_SUBDIR.mkdir(parents=True, exist_ok=True)
png_filenames = []
for name, fig in figure_names.items():
    filename = FIGURES_SUBDIR / f'{name}.png'
    fig.savefig(filename, format='png', bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    png_filenames.append(str(filename))
print(f'\nFigures saved as PNG files: {png_filenames}')


### Save results with pickle ###
with open(paths.CACHE_DIR / f'tpcf_results_{OUT_NAME}{suffix}.pkl', 'wb') as f:
    pickle.dump({'tpcf': tpcf_results, 'rbins': rbins, 'results': results}, f)
print("\nSaved TPCF and prediction results to pickle.")
