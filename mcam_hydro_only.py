import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr
import stats
import paths
from constants import LBOX, HYDRO_STELLAR_MASS_MIN, MASS_BIN_EDGES
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables.two_point_clustering import tpcf_jackknife, tpcf
from multicam.multicam import MultiCAM, MultiCamSampling

# Hydro-only test of how the halo features given to MultiCAM change its joint prediction of
# (g-i, log M*). No DMO data is used: every halo property comes from the hydro catalog's own trees.
#
# Feature sets (FEATURE_SETS), where the full MAH is mpeak_mah0..99:
#   'z12':       z12
#   'z12_mpeak': z12, log mpeak
#   'mah_mpeak': full MAH, log mpeak
#   'z12_mah':   z12, full MAH
# mpeak_mah99 is mpeak itself (same ranks), so it is left out of 'mah_mpeak': the duplicate
# feature makes the MultiCamSampling feature covariance singular. 'z12_mah' keeps it, so it
# also carries mpeak.
#
# For each set, one MultiCAM with 2 targets (g-i, log M*) is fit and applied on the full hydro
# sample. Galaxies are binned on that set's predicted stellar mass and use its predicted color,
# so each set puts different galaxies in each bin; the truth uses the true stellar mass and color.
#
# Without sampling, MultiCAM maps its predictions back onto the training targets, so the
# full-sample M* and color distributions match the truth by construction; the sets differ in
# the M* residuals, the per-bin color distributions and the clustering. With RUN_SAMPLING,
# MultiCamSampling adds the conditional scatter: histograms are pooled over N_SAMPLES draws and
# xi(r) uses the first N_TPCF_DRAWS draws.

COLOR_CUT = 0.85
N_THREADS = 16
N_SAMPLES = 100
N_TPCF_DRAWS = 25
SAMPLE_PCTILES = (16, 84)
# xi(r) is skipped (NaN) for samples with fewer than MIN_TPCF_POINTS galaxies; the jackknife
# also needs a galaxy in every one of its Nsub^3 subvolumes
MIN_TPCF_POINTS = 10
JACKKNIFE_NSUB = 3
# MultiCamSampling (scatter) mode; slow because of the N_TPCF_DRAWS tpcf draws per set.
# Set from the environment (e.g. mcam_hydro_only.sbatch) with RUN_SAMPLING=True
RUN_SAMPLING = os.environ.get('RUN_SAMPLING', 'False') == 'True'
MAH_COLS = [f'mpeak_mah{ii}' for ii in range(100)]
FEATURE_SETS = {'z12': ['z12'],
                'z12_mpeak': ['z12', 'logMpeak'],
                'mah_mpeak': MAH_COLS[:-1] + ['logMpeak'],
                'z12_mah': ['z12'] + MAH_COLS}
FEATURE_LABELS = {'z12': r'MultiCAM ($z_{1/2}$)',
                  'z12_mpeak': r'MultiCAM ($z_{1/2}$ + $M_{\rm peak}$)',
                  'mah_mpeak': r'MultiCAM (MAH + $M_{\rm peak}$)',
                  'z12_mah': r'MultiCAM ($z_{1/2}$ + MAH)'}
SET_COLORS = dict(zip(FEATURE_SETS, ['C0', 'C1', 'C2', 'C3']))
OUT_DIR = paths.REPO_ROOT / 'tests' / 'mcam_mstar_color'
OUT_NAME = 'mcam_mstar_color'

## Load data ###
hydro_full = pd.read_csv(paths.HYDRO_COLOR_TREE_CACHE, usecols=['SubhaloMass', 'StellarMass', 'x', 'y', 'z',
                                                                'logStellarMass', 'logSubhaloMass', 'g-i',
                                                                'a12', 'mpeak', 'is_sub', *MAH_COLS])
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
log_mstar_true = hydro_full['logStellarMass'].to_numpy()


### (g-i, log M*) predictions for the full sample, one MultiCAM per feature set ###
targets = hydro_full[['g-i', 'logStellarMass']].to_numpy()
samp_draws_full = {}  # per set: (N_SAMPLES, N_galaxies, 2) array of sampled (g-i, log M*)

for name, cols in FEATURE_SETS.items():
    features = hydro_full[cols].to_numpy()
    model = MultiCAM(len(cols), 2)
    model.fit(features, targets)
    pred = model.predict(features)
    hydro_full[f'color_pred_{name}'] = pred[:, 0]
    hydro_full[f'logStellarMass_{name}'] = pred[:, 1]

    if RUN_SAMPLING:
        model_samp = MultiCamSampling(len(cols), 2)
        model_samp.fit(features, targets)
        samp_draws_full[name] = np.array([model_samp.sample(features) for _ in range(N_SAMPLES)])

mstar_diag = {}
for name in FEATURE_SETS:
    resid = hydro_full[f'logStellarMass_{name}'] - hydro_full['logStellarMass']
    mstar_diag[name] = {'spearman_mstar': spearmanr(hydro_full[f'logStellarMass_{name}'],
                                                    hydro_full['logStellarMass']).statistic,
                        'mstar_resid_std_dex': np.std(resid)}
    if RUN_SAMPLING:
        draws_mstar = samp_draws_full[name][:, :, 1]
        mstar_diag[name]['spearman_mstar_samp'] = np.mean([spearmanr(d, log_mstar_true).statistic
                                                           for d in draws_mstar])
        mstar_diag[name]['mstar_resid_std_dex_samp'] = np.mean(np.std(draws_mstar - log_mstar_true, axis=1))
    print(f"\n{FEATURE_LABELS[name]} log M*: Spearman with truth = {mstar_diag[name]['spearman_mstar']:.3f}, "
          f"scatter about truth = {mstar_diag[name]['mstar_resid_std_dex']:.3f} dex")
    if RUN_SAMPLING:
        print(f"  sampling (mean of {N_SAMPLES} draws): Spearman = {mstar_diag[name]['spearman_mstar_samp']:.3f}, "
              f"scatter = {mstar_diag[name]['mstar_resid_std_dex_samp']:.3f} dex")


### Run predictions for bins ###
mass_bins = {f'bin{i+1}': edges for i, edges in enumerate(MASS_BIN_EDGES)}

results = {}
diagnostics = []

for key, (low, high) in mass_bins.items():
    mask_gal = (hydro_full['logStellarMass'] >= low) & (hydro_full['logStellarMass'] < high)
    hydro_bin = hydro_full[mask_gal].copy()  # truth (true stellar mass and color)
    hydro_bin['color'] = hydro_bin['g-i']

    # each set's population: galaxies whose predicted stellar mass falls in the bin
    pops = {'hydro': (hydro_bin, 'color')}
    for name in FEATURE_SETS:
        mask_pred = (hydro_full[f'logStellarMass_{name}'] >= low) & (hydro_full[f'logStellarMass_{name}'] < high)
        pops[name] = (hydro_full[mask_pred].copy(), f'color_pred_{name}')

    samp_draws = None
    if RUN_SAMPLING:
        # each draw: (population mask over hydro_full, predicted colors of that population)
        samp_draws = {}
        for name in FEATURE_SETS:
            samp_draws[name] = []
            for draw in samp_draws_full[name]:
                in_bin = (draw[:, 1] >= low) & (draw[:, 1] < high)
                samp_draws[name].append((in_bin, draw[in_bin, 0]))

    print(f"\n{key} [{low}, {high}): " + ", ".join(f"N_{name} = {len(pop)}" for name, (pop, _) in pops.items()))
    diag = {'bin': key, 'logmstar_min': low, 'logmstar_max': high,
            'spearman_train_z12_gi': spearmanr(hydro_bin['z12'], hydro_bin['g-i']).statistic}
    for name, (pop, _) in pops.items():
        diag[f'n_{name}'] = len(pop)
    for name in FEATURE_SETS:
        # fraction of true-bin galaxies that the set also places in the bin
        diag[f'overlap_frac_{name}'] = len(pops[name][0].index.intersection(hydro_bin.index)) / len(hydro_bin)
    print("  true-bin overlap:                 " + ", ".join(
        f"{name} {diag[f'overlap_frac_{name}']:.3f}" for name in FEATURE_SETS))

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
        for name in FEATURE_SETS:
            diag[f'n_{name}_samp'] = np.mean([in_bin.sum() for in_bin, _ in samp_draws[name]])
            diag[f'red_frac_{name}_samp'] = np.mean(np.concatenate([c for _, c in samp_draws[name]]) > COLOR_CUT)
        print("  red fraction (sampling, pooled):  " + ", ".join(
            f"{name} {diag[f'red_frac_{name}_samp']:.3f}" for name in FEATURE_SETS))
    diagnostics.append(diag)

    results[key] = {name: pop for name, (pop, _) in pops.items()}
    results[key]['samp_draws'] = samp_draws

print("\nPredictions complete for all bins.")

suffix = '_with_sampling' if RUN_SAMPLING else ''
OUT_DIR.mkdir(parents=True, exist_ok=True)
diagnostics_df = pd.DataFrame(diagnostics)
for name in FEATURE_SETS:
    for stat, value in mstar_diag[name].items():
        diagnostics_df[f'{stat}_{name}'] = value
diagnostics_file = OUT_DIR / f'{OUT_NAME}_diagnostics{suffix}.csv'
diagnostics_df.to_csv(diagnostics_file, index=False)
print(f"\nSaved diagnostics to {diagnostics_file}")


### Histograms of stellar mass: full sample and residual about truth ###
# With sampling, the predictions are pooled over all N_SAMPLES draws
mode_label = f' (sampling, {N_SAMPLES} draws)' if RUN_SAMPLING else ''
fig_mstar, axes = plt.subplots(1, 2, figsize=(13, 5))
mstar_hist_bins = np.linspace(log_mstar_true.min(), log_mstar_true.max(), 50)
axes[0].hist(log_mstar_true, bins=mstar_hist_bins, density=True, color='gray', alpha=0.4,
             label=r'True $M_*$ (Hydro)')
for name in FEATURE_SETS:
    if RUN_SAMPLING:
        mstar_pred = samp_draws_full[name][:, :, 1]
    else:
        mstar_pred = hydro_full[f'logStellarMass_{name}'].to_numpy()
    axes[0].hist(mstar_pred.ravel(), bins=mstar_hist_bins, density=True, histtype='step', linewidth=2,
                 color=SET_COLORS[name], label=FEATURE_LABELS[name] + mode_label)
    axes[1].hist((mstar_pred - log_mstar_true).ravel(), bins=60, density=True, histtype='step', linewidth=2,
                 color=SET_COLORS[name], label=FEATURE_LABELS[name] + mode_label)
for low, high in MASS_BIN_EDGES:
    for edge in (low, high):
        axes[0].axvline(edge, color='k', linestyle=':', linewidth=0.8)
axes[0].set_yscale('log')
axes[0].set_xlabel(r'$\log(M_*/h^{-1}M_\odot)$')
axes[0].set_ylabel('Normalized frequency')
axes[0].legend(fontsize=8)
axes[1].axvline(0, color='k', linewidth=0.8)
axes[1].set_xlabel(r'$\log M_{*,\rm pred} - \log M_{*,\rm true}$ [dex]')
axes[1].set_ylabel('Normalized frequency')
plt.tight_layout()
plt.show()


### Histogram of color vs. predicted color, per stellar-mass bin ###
fig_color = plt.figure(figsize=(18, 4))

for i, key in enumerate(mass_bins.keys()):
    plt.subplot(1, len(mass_bins), i+1)

    res = results[key]
    color_hist_bins = np.linspace(hydro_full['g-i'].min(), hydro_full['g-i'].max(), 40)
    plt.hist(res['hydro']['color'], bins=color_hist_bins, density=True, color='gray', alpha=0.4,
             label='True color (Hydro)')
    for name in FEATURE_SETS:
        if RUN_SAMPLING:
            color_pred = np.concatenate([c for _, c in res['samp_draws'][name]])
        else:
            color_pred = res[name][f'color_pred_{name}']
        plt.hist(color_pred, bins=color_hist_bins, density=True, histtype='step', linewidth=2,
                 color=SET_COLORS[name], label=FEATURE_LABELS[name] + mode_label)
    plt.axvline(COLOR_CUT, color='k', linestyle='--')
    low, high = mass_bins[key]
    plt.title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    plt.xlabel(r'$g - i$ color')
    plt.ylabel('Normalized frequency')

    if i == 0:
        plt.legend(fontsize=8)

plt.tight_layout()
plt.show()


### Define red/blue galaxies ###
for key, res in results.items():
    res['red_hydro'] = res['hydro'][res['hydro']['color'] > COLOR_CUT]
    res['blue_hydro'] = res['hydro'][res['hydro']['color'] <= COLOR_CUT]
    for name in FEATURE_SETS:
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
    # cached positions are already comoving Mpc/h (SubhaloPos ckpc/h / 1000), filling [0, LBOX)
    return return_xyz_formatted_array(table['x'], table['y'], table['z'])


def jackknife(table):
    # e.g. no red galaxies in a bin; NaN xi and cov are skipped when plotting
    subvol = np.floor(xyz(table) / (Lbox / JACKKNIFE_NSUB)).astype(int) % JACKKNIFE_NSUB
    n_occupied = len(np.unique(subvol, axis=0))
    if len(table) < MIN_TPCF_POINTS or n_occupied < JACKKNIFE_NSUB ** 3:
        print(f"  skipping jackknife xi(r): N = {len(table)}, occupied subvolumes = {n_occupied}")
        return np.full(len(rbins) - 1, np.nan), np.full((len(rbins) - 1, len(rbins) - 1), np.nan)
    return tpcf_jackknife(xyz(table), randoms, rbins, Nsub=JACKKNIFE_NSUB, period=Lbox, num_threads=N_THREADS)


def plain_tpcf(table):
    if len(table) < MIN_TPCF_POINTS:
        return np.full(len(rbins) - 1, np.nan)
    return tpcf(xyz(table), rbins, randoms=randoms, period=Lbox, num_threads=N_THREADS)


# Store all xi and cov results here; sampling stores (median, lo, hi) over draws
tpcf_results = {'hydro': []}
for name in FEATURE_SETS:
    tpcf_results[name] = []
    tpcf_results[f'{name}_samp'] = []

for key, res in results.items():
    print(f"\nComputing TPCF for {key}.")

    for name in ['hydro', *FEATURE_SETS]:
        xi_red, cov_red = jackknife(res[f'red_{name}'])
        xi_blue, cov_blue = jackknife(res[f'blue_{name}'])
        tpcf_results[name].append(((xi_red, xi_blue), (cov_red, cov_blue)))

    if not RUN_SAMPLING:
        continue

    # MultiCAM Sampling: tpcf of each draw, median and percentile band across draws
    for name in FEATURE_SETS:
        xi_red_draws, xi_blue_draws = [], []
        for in_bin, color_draw in res['samp_draws'][name][:N_TPCF_DRAWS]:
            pop = hydro_full[in_bin]
            red = color_draw > COLOR_CUT
            xi_red_draws.append(plain_tpcf(pop[red]))
            xi_blue_draws.append(plain_tpcf(pop[~red]))

        summary = []
        for draws in [np.array(xi_red_draws), np.array(xi_blue_draws)]:
            summary.append((np.nanmedian(draws, axis=0),
                            np.nanpercentile(draws, SAMPLE_PCTILES[0], axis=0),
                            np.nanpercentile(draws, SAMPLE_PCTILES[1], axis=0)))
        tpcf_results[f'{name}_samp'].append(tuple(summary))

print("\nTPCF computed for all bins.")


### Plot results: xi(r) and ratio to hydro truth, one figure each for red and blue ###
# The predictions are offset slightly in r so their jackknife error bars don't overlap
r_offset = dict(zip(FEATURE_SETS, [rbin_centers * f for f in [0.94, 0.98, 1.02, 1.06]]))
markers = dict(zip(FEATURE_SETS, ['s', 'x', '^', 'D']))

tpcf_figs = {}
for c, color in enumerate(['red', 'blue']):
    fig, axes = plt.subplots(2, len(mass_bins), figsize=(19, 9), sharex=True, sharey='row',
                             gridspec_kw={'height_ratios': [3, 1]})
    label = color.capitalize()
    for i, key in enumerate(mass_bins.keys()):
        low, high = mass_bins[key]
        ax, rax = axes[0, i], axes[1, i]

        xi_hydro = tpcf_results['hydro'][i][0][c]
        err = stats.compute_tpcf_jackknife_errors(tpcf_results['hydro'][i][1][c])
        ax.plot(rbin_centers, xi_hydro, marker='o', color='k', label=fr'Truth {label}: TNG100 Hydro $\xi(r)$')
        ax.fill_between(rbin_centers, xi_hydro - err, xi_hydro + err, color='k', alpha=0.15)
        rax.fill_between(rbin_centers, 1 - err / xi_hydro, 1 + err / xi_hydro, color='k', alpha=0.15)

        for name in FEATURE_SETS:
            xi_pred = tpcf_results[name][i][0][c]
            err_pred = stats.compute_tpcf_jackknife_errors(tpcf_results[name][i][1][c])
            ax.errorbar(r_offset[name], xi_pred, yerr=err_pred, color=SET_COLORS[name], capsize=2,
                        marker=markers[name], linestyle='--',
                        label=fr'Prediction {label}: {FEATURE_LABELS[name]} $\xi(r)$')
            rax.errorbar(r_offset[name], xi_pred / xi_hydro, yerr=err_pred / xi_hydro, color=SET_COLORS[name],
                         capsize=2, marker=markers[name], linestyle='--')

            if RUN_SAMPLING:
                xi_s, xi_s_lo, xi_s_hi = tpcf_results[f'{name}_samp'][i][c]
                ax.plot(rbin_centers, xi_s, color=SET_COLORS[name], linestyle='-.',
                        label=fr'Prediction {label}: {FEATURE_LABELS[name]} Sampling $\xi(r)$ '
                              fr'(median of {N_TPCF_DRAWS})')
                ax.fill_between(rbin_centers, xi_s_lo, xi_s_hi, color=SET_COLORS[name], alpha=0.2)
                rax.plot(rbin_centers, xi_s / xi_hydro, color=SET_COLORS[name], linestyle='-.')
                rax.fill_between(rbin_centers, xi_s_lo / xi_hydro, xi_s_hi / xi_hydro, color=SET_COLORS[name],
                                 alpha=0.2)

        rax.axhline(1, color='k', linewidth=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(fr'{label}: ${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
        rax.set_xlabel(r'$r \, [\mathrm{Mpc}/h]$')
        if i == 0:
            ax.set_ylabel(r'$\xi(r)$')
            rax.set_ylabel(r'$\xi_{\rm pred} / \xi_{\rm hydro}$')
        if i == len(mass_bins) - 1:
            ax.legend(fontsize=8)

    axes[1, 0].set_ylim(0.5, 1.5)
    plt.subplots_adjust(wspace=0, hspace=0, left=0.05, right=0.95, bottom=0.1, top=0.9)
    plt.show()
    tpcf_figs[color] = fig

### Save figures as PNG files ###
figure_names = {
    f'{OUT_NAME}_mstar_hist{suffix}': fig_mstar,
    f'{OUT_NAME}_color_hist{suffix}': fig_color,
    f'{OUT_NAME}_tpcf_red{suffix}': tpcf_figs['red'],
    f'{OUT_NAME}_tpcf_blue{suffix}': tpcf_figs['blue'],
}
png_filenames = []
for name, fig in figure_names.items():
    filename = OUT_DIR / f'{name}.png'
    fig.savefig(filename, format='png', bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    png_filenames.append(str(filename))
print(f'\nFigures saved as PNG files: {png_filenames}')


### Save results with pickle ###
with open(OUT_DIR / f'tpcf_results_{OUT_NAME}{suffix}.pkl', 'wb') as f:
    pickle.dump({'tpcf': tpcf_results, 'rbins': rbins, 'results': results}, f)
print("\nSaved TPCF and prediction results to pickle.")
