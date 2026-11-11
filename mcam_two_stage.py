import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr
import stats
import sham
import plot
import paths
from constants import LBOX, HYDRO_STELLAR_MASS_MIN, MASS_BIN_EDGES
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables.two_point_clustering import tpcf_jackknife, tpcf
from multicam.multicam import MultiCAM, MultiCamSampling

# Hydro-only two-stage test: stellar mass first, then color within each stellar-mass bin.
# No DMO data is used: every halo property comes from the hydro catalog's own trees.
#
# Stage 1 (M*), from log mpeak alone, in two ways (MSTAR_METHODS):
#   'sham': SHAM of log mpeak onto the hydro stellar mass distribution
#   'mcam': MultiCAM (log mpeak -> log M*), fit and applied on the full sample. Without
#           sampling, one feature and one target is a rank-match, so this reproduces SHAM.
# N_MSTAR_DRAWS sets how the MultiCAM M* is drawn:
#   0: deterministic MultiCAM (reproduces SHAM)
#   1: one MultiCamSampling draw (seeded with MSTAR_SEED), i.e. SHAM with scatter at fixed mpeak
#   N > 1: N seeded draws. Stage 2 and xi(r) are rerun for every draw and shown as the median and
#          SAMPLE_PCTILES band over draws; the first draw (the same as with N_MSTAR_DRAWS=1) also
#          runs the single-population analysis (jackknife xi(r), per-bin diagnostics). Stage-2
#          color sampling (RUN_SAMPLING) is not combined with this mode.
#
# Stage 2 (color): in each predicted-M* bin, one MultiCAM (features -> g-i) per feature set is
# fit on the galaxies in that bin (their features and true g-i) and applied to the same galaxies.
# Feature sets (FEATURE_SETS), where the full MAH is mpeak_mah0..99:
#   'z12':       z12
#   'z12_mpeak': z12, log mpeak
#   'mah_mpeak': full MAH, log mpeak
#   'z12_mah':   z12, full MAH
# mpeak_mah99 is mpeak itself (same ranks), so it is left out of 'mah_mpeak': the duplicate
# feature makes the MultiCamSampling feature covariance singular. 'z12_mah' keeps it, so it
# also carries mpeak.
#
# Without sampling, the color MultiCAM maps its predictions back onto the true colors of the
# bin it was fit on, so each bin's predicted color distribution is the same for every feature
# set; the sets differ in which galaxies get which color, i.e. in the clustering. With
# RUN_SAMPLING, MultiCamSampling adds the conditional color scatter: histograms are pooled over
# N_SAMPLES draws and xi(r) uses the first N_TPCF_DRAWS draws. RUN_SAMPLING only affects
# stage 2; stage-1 sampling is set by N_MSTAR_DRAWS.

COLOR_CUT = 0.85
N_THREADS = 16
N_SAMPLES = 100
N_TPCF_DRAWS = 25
SAMPLE_PCTILES = (16, 84)
# xi(r) is skipped (NaN) for samples with fewer than MIN_TPCF_POINTS galaxies; the jackknife
# also needs a galaxy in every one of its Nsub^3 subvolumes
MIN_TPCF_POINTS = 10
JACKKNIFE_NSUB = 3
# MultiCamSampling (color scatter) mode; slow because of the N_TPCF_DRAWS tpcf draws per set.
# Set from the environment (e.g. mcam_two_stage.sbatch) with RUN_SAMPLING=True
RUN_SAMPLING = os.environ.get('RUN_SAMPLING', 'False') == 'True'
# Stage-1 MultiCAM M* draws (see header); set from the environment with N_MSTAR_DRAWS=1
N_MSTAR_DRAWS = int(os.environ.get('N_MSTAR_DRAWS', '0'))
if N_MSTAR_DRAWS > 1 and RUN_SAMPLING:
    raise ValueError('RUN_SAMPLING is not supported with N_MSTAR_DRAWS > 1')
MSTAR_SEED = 42
MSTAR_METHODS = ['sham', 'mcam']
# with many draws, the single-population analysis uses the first draw
MSTAR_LABELS = {'sham': 'SHAM',
                'mcam': {0: 'MultiCAM', 1: 'MultiCAM (1 draw)'}.get(N_MSTAR_DRAWS, 'MultiCAM (draw 1)')}
MSTAR_DRAWS_LABEL = f'MultiCAM ({N_MSTAR_DRAWS} draws)'
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
MSTAR_COLORS = {'sham': 'C4', 'mcam': 'C5'}
OUT_DIR = paths.REPO_ROOT / 'tests' / 'mcam_two_stage'
OUT_NAME = 'mcam_two_stage'

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


### Stage 1: stellar mass from log mpeak ###
hydro_full['logStellarMass_sham'] = sham.sham(hydro_full['logMpeak'].to_numpy(), log_mstar_true)

x_mstar = hydro_full[['logMpeak']].to_numpy()
if N_MSTAR_DRAWS == 0:
    model_mstar = MultiCAM(1, 1)
    model_mstar.fit(x_mstar, hydro_full[['logStellarMass']].to_numpy())
    hydro_full['logStellarMass_mcam'] = model_mstar.predict(x_mstar)[:, 0]
else:
    # sample() breaks ties in mpeak with the global numpy RNG, so seed it as well as the model's
    np.random.seed(MSTAR_SEED)
    model_mstar = MultiCamSampling(1, 1, rng=np.random.default_rng(MSTAR_SEED))
    model_mstar.fit(x_mstar, hydro_full[['logStellarMass']].to_numpy())
    # (N_MSTAR_DRAWS, N_galaxies) sampled log M*; the first draw is the N_MSTAR_DRAWS=1 draw
    mstar_draws = np.array([model_mstar.sample(x_mstar)[:, 0] for _ in range(N_MSTAR_DRAWS)])
    hydro_full['logStellarMass_mcam'] = mstar_draws[0]

mstar_diag = {}
for method in MSTAR_METHODS:
    resid = hydro_full[f'logStellarMass_{method}'] - hydro_full['logStellarMass']
    mstar_diag[method] = {'spearman_mstar': spearmanr(hydro_full[f'logStellarMass_{method}'],
                                                      hydro_full['logStellarMass']).statistic,
                          'mstar_resid_std_dex': np.std(resid)}
    print(f"\n{MSTAR_LABELS[method]} log M*: Spearman with truth = {mstar_diag[method]['spearman_mstar']:.3f}, "
          f"scatter about truth = {mstar_diag[method]['mstar_resid_std_dex']:.3f} dex")

mass_bins = {f'bin{i+1}': edges for i, edges in enumerate(MASS_BIN_EDGES)}
bin_index = {method: np.digitize(hydro_full[f'logStellarMass_{method}'], np.unique(MASS_BIN_EDGES))
             for method in MSTAR_METHODS}
max_diff = np.max(np.abs(hydro_full['logStellarMass_sham'] - hydro_full['logStellarMass_mcam']))
bin_diff_frac = np.mean(bin_index['sham'] != bin_index['mcam'])
print(f"\nSHAM vs MultiCAM log M*: max |difference| = {max_diff:.2e} dex, "
      f"fraction in a different bin = {bin_diff_frac:.4f}")


### Stage 2: color within each predicted stellar-mass bin ###
def predict_colors(pop):
    """Adds color_pred_{set} to pop: per feature set, MultiCAM fit on pop's features and true g-i,
    applied to pop"""
    y_train = pop[['g-i']].to_numpy()
    for name, cols in FEATURE_SETS.items():
        x = pop[cols].to_numpy()
        model = MultiCAM(len(cols), 1)
        model.fit(x, y_train)
        pop[f'color_pred_{name}'] = model.predict(x)[:, 0]


results = {}
diagnostics = []

for key, (low, high) in mass_bins.items():
    mask_gal = (hydro_full['logStellarMass'] >= low) & (hydro_full['logStellarMass'] < high)
    hydro_bin = hydro_full[mask_gal].copy()  # truth (true stellar mass and color)
    hydro_bin['color'] = hydro_bin['g-i']
    results[key] = {'hydro': hydro_bin, 'samp_draws': {}}

    diag = {'bin': key, 'logmstar_min': low, 'logmstar_max': high, 'n_hydro': len(hydro_bin),
            'spearman_z12_gi_hydro': spearmanr(hydro_bin['z12'], hydro_bin['g-i']).statistic}
    red = hydro_bin['color'] > COLOR_CUT
    diag['red_frac_hydro'] = red.mean()
    diag['sat_frac_red_hydro'] = hydro_bin.loc[red, 'is_sub'].mean()
    diag['sat_frac_blue_hydro'] = hydro_bin.loc[~red, 'is_sub'].mean()
    print(f"\n{key} [{low}, {high}): N_hydro = {len(hydro_bin)}, red fraction {diag['red_frac_hydro']:.3f}, "
          f"satellite fraction (red / blue) {diag['sat_frac_red_hydro']:.3f} / {diag['sat_frac_blue_hydro']:.3f}")

    for method in MSTAR_METHODS:
        mask_pred = (hydro_full[f'logStellarMass_{method}'] >= low) & (hydro_full[f'logStellarMass_{method}'] < high)
        pop = hydro_full[mask_pred].copy()  # galaxies in the predicted-M* bin, with their true g-i

        # fraction of true-bin galaxies that the method also places in the bin
        diag[f'n_{method}'] = len(pop)
        diag[f'overlap_frac_{method}'] = len(pop.index.intersection(hydro_bin.index)) / len(hydro_bin)
        diag[f'spearman_z12_gi_{method}'] = spearmanr(pop['z12'], pop['g-i']).statistic
        print(f"  {MSTAR_LABELS[method]}: N = {len(pop)}, true-bin overlap = {diag[f'overlap_frac_{method}']:.3f}")

        predict_colors(pop)
        y_train = pop[['g-i']].to_numpy()
        for name, cols in FEATURE_SETS.items():
            if RUN_SAMPLING:
                x = pop[cols].to_numpy()
                model_samp = MultiCamSampling(len(cols), 1)
                model_samp.fit(x, y_train)
                # population is fixed, so each draw is just the predicted colors of pop
                results[key]['samp_draws'][f'{method}_{name}'] = np.array(
                    [model_samp.sample(x)[:, 0] for _ in range(N_SAMPLES)])

            # Red and satellite fractions of each red/blue sample
            red = pop[f'color_pred_{name}'] > COLOR_CUT
            diag[f'red_frac_{method}_{name}'] = red.mean()
            diag[f'sat_frac_red_{method}_{name}'] = pop.loc[red, 'is_sub'].mean()
            diag[f'sat_frac_blue_{method}_{name}'] = pop.loc[~red, 'is_sub'].mean()
            if RUN_SAMPLING:
                draws = results[key]['samp_draws'][f'{method}_{name}']
                diag[f'red_frac_{method}_{name}_samp'] = np.mean(draws > COLOR_CUT)
                diag[f'sat_frac_red_{method}_{name}_samp'] = np.mean(
                    [pop.loc[d > COLOR_CUT, 'is_sub'].mean() for d in draws])

        print("    red fraction:                     " + ", ".join(
            f"{name} {diag[f'red_frac_{method}_{name}']:.3f}" for name in FEATURE_SETS))
        print("    satellite fraction (red / blue):  " + ", ".join(
            f"{name} {diag[f'sat_frac_red_{method}_{name}']:.3f} / {diag[f'sat_frac_blue_{method}_{name}']:.3f}"
            for name in FEATURE_SETS))
        if RUN_SAMPLING:
            print("    red fraction (sampling, pooled):  " + ", ".join(
                f"{name} {diag[f'red_frac_{method}_{name}_samp']:.3f}" for name in FEATURE_SETS))
            print("    red satellite fraction (sampling): " + ", ".join(
                f"{name} {diag[f'sat_frac_red_{method}_{name}_samp']:.3f}" for name in FEATURE_SETS))
        results[key][method] = pop

    diagnostics.append(diag)

print("\nPredictions complete for all bins.")


### Stage 2 for every MultiCAM M* draw ###
# per bin, one population per draw: the galaxies the draw puts in the bin, with their true g-i
# and the colors predicted by stage 2 fit on that population
mstar_draw_pops = {key: [] for key in mass_bins}
if N_MSTAR_DRAWS > 1:
    pred_cols = [f'color_pred_{name}' for name in FEATURE_SETS]
    for d, draw in enumerate(mstar_draws):
        for key, (low, high) in mass_bins.items():
            pop = hydro_full[(draw >= low) & (draw < high)].copy()
            predict_colors(pop)
            mstar_draw_pops[key].append(pop[['x', 'y', 'z', 'is_sub', 'g-i', *pred_cols]])
        print(f"  stage 2 done for M* draw {d + 1} / {N_MSTAR_DRAWS}")

    # mean and spread over draws of the per-bin statistics
    for diag, (key, res) in zip(diagnostics, results.items()):
        pops = mstar_draw_pops[key]
        stat_draws = {'n_mcam_draws': [len(p) for p in pops],
                      'overlap_frac_mcam_draws': [len(p.index.intersection(res['hydro'].index)) / len(res['hydro'])
                                                  for p in pops]}
        for name in FEATURE_SETS:
            reds = [p[f'color_pred_{name}'] > COLOR_CUT for p in pops]
            stat_draws[f'red_frac_mcam_{name}_draws'] = [red.mean() for red in reds]
            stat_draws[f'sat_frac_red_mcam_{name}_draws'] = [p.loc[red, 'is_sub'].mean() for p, red in zip(pops, reds)]
            stat_draws[f'sat_frac_blue_mcam_{name}_draws'] = [p.loc[~red, 'is_sub'].mean()
                                                              for p, red in zip(pops, reds)]
        for stat, values in stat_draws.items():
            diag[f'{stat}_mean'] = np.mean(values)
            diag[f'{stat}_std'] = np.std(values)

        print(f"\n{key}, {MSTAR_DRAWS_LABEL} (mean ± std over draws): "
              f"N = {diag['n_mcam_draws_mean']:.0f} ± {diag['n_mcam_draws_std']:.0f}, "
              f"true-bin overlap = {diag['overlap_frac_mcam_draws_mean']:.3f} ± {diag['overlap_frac_mcam_draws_std']:.3f}")
        print("    red fraction:                     " + ", ".join(
            f"{name} {diag[f'red_frac_mcam_{name}_draws_mean']:.3f} ± {diag[f'red_frac_mcam_{name}_draws_std']:.3f}"
            for name in FEATURE_SETS))
        print("    satellite fraction (red / blue):  " + ", ".join(
            f"{name} {diag[f'sat_frac_red_mcam_{name}_draws_mean']:.3f} / "
            f"{diag[f'sat_frac_blue_mcam_{name}_draws_mean']:.3f}" for name in FEATURE_SETS))

    print(f"\nStage 2 complete for all {N_MSTAR_DRAWS} M* draws.")

suffix = ({0: '', 1: '_mstar_1draw'}.get(N_MSTAR_DRAWS, f'_mstar_{N_MSTAR_DRAWS}draws')
          + ('_with_sampling' if RUN_SAMPLING else ''))
OUT_DIR.mkdir(parents=True, exist_ok=True)
diagnostics_df = pd.DataFrame(diagnostics)
for method in MSTAR_METHODS:
    for stat, value in mstar_diag[method].items():
        diagnostics_df[f'{stat}_{method}'] = value
diagnostics_df['max_abs_diff_sham_mcam_mstar'] = max_diff
diagnostics_df['bin_diff_frac_sham_mcam'] = bin_diff_frac
diagnostics_file = OUT_DIR / f'{OUT_NAME}_diagnostics{suffix}.csv'
diagnostics_df.to_csv(diagnostics_file, index=False)
print(f"\nSaved diagnostics to {diagnostics_file}")


### Histograms of stellar mass: full sample and residual about truth ###
fig_mstar, axes = plt.subplots(1, 2, figsize=(13, 5))
mstar_hist_bins = np.linspace(log_mstar_true.min(), log_mstar_true.max(), 50)
axes[0].hist(log_mstar_true, bins=mstar_hist_bins, density=True, color='gray', alpha=0.4,
             label=r'True $M_*$ (Hydro)')
for method, linestyle in zip(MSTAR_METHODS, ['-', '--']):
    mstar_pred = hydro_full[f'logStellarMass_{method}'].to_numpy()
    axes[0].hist(mstar_pred, bins=mstar_hist_bins, density=True, histtype='step', linewidth=2,
                 linestyle=linestyle, color=MSTAR_COLORS[method], label=fr'{MSTAR_LABELS[method]} ($M_{{\rm peak}}$)')
    axes[1].hist(mstar_pred - log_mstar_true, bins=60, density=True, histtype='step', linewidth=2,
                 linestyle=linestyle, color=MSTAR_COLORS[method], label=fr'{MSTAR_LABELS[method]} ($M_{{\rm peak}}$)')
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


### Histogram of color vs. predicted color: one row per stellar-mass method ###
# With sampling, the predictions are pooled over all N_SAMPLES draws
mode_label = f' (sampling, {N_SAMPLES} draws)' if RUN_SAMPLING else ''
fig_color, axes = plt.subplots(len(MSTAR_METHODS), len(mass_bins), figsize=(18, 4 * len(MSTAR_METHODS)))
for row, method in enumerate(MSTAR_METHODS):
    # with many M* draws, the MultiCAM row pools the populations of all draws
    if method == 'mcam' and N_MSTAR_DRAWS > 1:
        pops = {key: pd.concat(mstar_draw_pops[key]) for key in mass_bins}
        method_label = MSTAR_DRAWS_LABEL
    else:
        pops = {key: res[method] for key, res in results.items()}
        method_label = MSTAR_LABELS[method]
    color_preds = []
    for name in FEATURE_SETS:
        if RUN_SAMPLING:
            colors = {key: res['samp_draws'][f'{method}_{name}'].ravel() for key, res in results.items()}
        else:
            colors = {key: pop[f'color_pred_{name}'] for key, pop in pops.items()}
        color_preds.append((FEATURE_LABELS[name] + mode_label, colors, {'color': SET_COLORS[name]}))
    # reference: the true colors of the galaxies the method puts in each bin, which the
    # deterministic color MultiCAM is fit on; gray vs. black is the stage-1 bin membership
    color_preds.append((f'True color of {method_label}-bin galaxies',
                        {key: pop['g-i'] for key, pop in pops.items()},
                        {'color': 'k', 'linewidth': 1.2}))
    plot.color_hist_by_bin(mass_bins, {key: res['hydro']['color'] for key, res in results.items()},
                           color_preds, COLOR_CUT, np.linspace(hydro_full['g-i'].min(), hydro_full['g-i'].max(), 40),
                           axes=axes[row], title_prefix=f'{method_label} $M_*$: ')
plt.show()


### Define red/blue galaxies ###
for key, res in results.items():
    res['red_hydro'] = res['hydro'][res['hydro']['color'] > COLOR_CUT]
    res['blue_hydro'] = res['hydro'][res['hydro']['color'] <= COLOR_CUT]
    for method in MSTAR_METHODS:
        pop = res[method]
        for name in FEATURE_SETS:
            res[f'red_{method}_{name}'] = pop[pop[f'color_pred_{name}'] > COLOR_CUT]
            res[f'blue_{method}_{name}'] = pop[pop[f'color_pred_{name}'] <= COLOR_CUT]

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


# Store all xi and cov results here, keyed 'hydro' and '{method}_{set}';
# sampling stores (median, lo, hi) over draws under '{method}_{set}_samp'
combos = [f'{method}_{name}' for method in MSTAR_METHODS for name in FEATURE_SETS]
tpcf_results = {'hydro': []}
for combo in combos:
    tpcf_results[combo] = []
    tpcf_results[f'{combo}_samp'] = []

for key, res in results.items():
    print(f"\nComputing TPCF for {key}.")

    for combo in ['hydro', *combos]:
        xi_red, cov_red = jackknife(res[f'red_{combo}'])
        xi_blue, cov_blue = jackknife(res[f'blue_{combo}'])
        tpcf_results[combo].append(((xi_red, xi_blue), (cov_red, cov_blue)))

    if not RUN_SAMPLING:
        continue

    # MultiCAM Sampling: tpcf of each color draw, median and percentile band across draws
    for method in MSTAR_METHODS:
        pop = res[method]
        for name in FEATURE_SETS:
            xi_red_draws, xi_blue_draws = [], []
            for color_draw in res['samp_draws'][f'{method}_{name}'][:N_TPCF_DRAWS]:
                red = color_draw > COLOR_CUT
                xi_red_draws.append(plain_tpcf(pop[red]))
                xi_blue_draws.append(plain_tpcf(pop[~red]))

            summary = []
            for draws in [np.array(xi_red_draws), np.array(xi_blue_draws)]:
                summary.append((np.nanmedian(draws, axis=0),
                                np.nanpercentile(draws, SAMPLE_PCTILES[0], axis=0),
                                np.nanpercentile(draws, SAMPLE_PCTILES[1], axis=0)))
            tpcf_results[f'{method}_{name}_samp'].append(tuple(summary))

# MultiCAM M* draws: tpcf of each draw's red/blue populations, median and percentile band across draws;
# the per-draw xi are kept in mstar_draw_xi[set][bin] as (N_MSTAR_DRAWS, 2 colors (red, blue), n_r) arrays
mstar_draw_xi = {}
if N_MSTAR_DRAWS > 1:
    for name in FEATURE_SETS:
        tpcf_results[f'mcam_{name}_mstar_draws'] = []
        mstar_draw_xi[name] = []
    for key in mass_bins:
        print(f"\nComputing TPCF over M* draws for {key}.")
        for name in FEATURE_SETS:
            xi_draws = []
            for pop in mstar_draw_pops[key]:
                red = pop[f'color_pred_{name}'] > COLOR_CUT
                xi_draws.append((plain_tpcf(pop[red]), plain_tpcf(pop[~red])))
            xi_draws = np.array(xi_draws)
            mstar_draw_xi[name].append(xi_draws)
            tpcf_results[f'mcam_{name}_mstar_draws'].append(tuple(
                (np.nanmedian(xi_draws[:, c], axis=0),
                 np.nanpercentile(xi_draws[:, c], SAMPLE_PCTILES[0], axis=0),
                 np.nanpercentile(xi_draws[:, c], SAMPLE_PCTILES[1], axis=0)) for c in range(2)))

print("\nTPCF computed for all bins.")


### Plot results: xi(r) and ratio to hydro truth, one figure per color and stellar-mass method ###
# The predictions are offset slightly in r so their jackknife error bars don't overlap
r_offset = dict(zip(FEATURE_SETS, [rbin_centers * f for f in [0.94, 0.98, 1.02, 1.06]]))
markers = dict(zip(FEATURE_SETS, ['s', 'x', '^', 'D']))

tpcf_figs = {}
for c, color in enumerate(['red', 'blue']):
    label = color.capitalize()
    for method in MSTAR_METHODS:
        fig, axes = plot.tpcf_ratio_grid(len(mass_bins))
        for i in range(len(mass_bins)):
            curves = []
            for name in FEATURE_SETS:
                combo = f'{method}_{name}'
                curves.append({'r': r_offset[name], 'xi': tpcf_results[combo][i][0][c],
                               'err': stats.compute_tpcf_jackknife_errors(tpcf_results[combo][i][1][c]),
                               'color': SET_COLORS[name], 'marker': markers[name], 'linestyle': '--',
                               'label': fr'Prediction {label}: {MSTAR_LABELS[method]} $M_*$ + '
                                        fr'{FEATURE_LABELS[name]} $\xi(r)$'})
                if RUN_SAMPLING:
                    xi_s, xi_s_lo, xi_s_hi = tpcf_results[f'{combo}_samp'][i][c]
                    curves.append({'median': xi_s, 'lo': xi_s_lo, 'hi': xi_s_hi, 'color': SET_COLORS[name],
                                   'label': fr'Prediction {label}: {MSTAR_LABELS[method]} $M_*$ + '
                                            fr'{FEATURE_LABELS[name]} Sampling $\xi(r)$ (median of {N_TPCF_DRAWS})'})
                if method == 'mcam' and N_MSTAR_DRAWS > 1:
                    xi_s, xi_s_lo, xi_s_hi = tpcf_results[f'mcam_{name}_mstar_draws'][i][c]
                    curves.append({'median': xi_s, 'lo': xi_s_lo, 'hi': xi_s_hi, 'color': SET_COLORS[name],
                                   'label': fr'Prediction {label}: {MSTAR_DRAWS_LABEL} $M_*$ + '
                                            fr'{FEATURE_LABELS[name]} $\xi(r)$ (median over draws)'})

            plot.tpcf_ratio_panel(axes[0, i], axes[1, i], rbin_centers, tpcf_results['hydro'][i][0][c],
                                  stats.compute_tpcf_jackknife_errors(tpcf_results['hydro'][i][1][c]),
                                  fr'Truth {label}: TNG100 Hydro $\xi(r)$', 'k', curves, truth_alpha=0.15)

        # short method name in the titles (the legend has the draw details) so adjacent titles don't overlap
        plot.finish_tpcf_ratio_grid(fig, axes, mass_bins,
                                    title_prefix=f"{label}, {MSTAR_LABELS[method].split(' (')[0]} $M_*$: ")
        plt.show()
        tpcf_figs[(color, method)] = fig

### Save figures as PNG files ###
figure_names = {
    f'{OUT_NAME}_mstar_hist{suffix}': fig_mstar,
    f'{OUT_NAME}_color_hist{suffix}': fig_color,
}
for (color, method), fig in tpcf_figs.items():
    figure_names[f'{OUT_NAME}_tpcf_{color}_{method}{suffix}'] = fig
png_filenames = plot.save_named_figures(figure_names, OUT_DIR)
print(f'\nFigures saved as PNG files: {png_filenames}')


### Save results with pickle ###
# per-bin populations with their predicted colors; color draws are left out to keep it small
results_out = {key: {k: v for k, v in res.items() if k != 'samp_draws'} for key, res in results.items()}
with open(OUT_DIR / f'tpcf_results_{OUT_NAME}{suffix}.pkl', 'wb') as f:
    pickle.dump({'tpcf': tpcf_results, 'rbins': rbins, 'results': results_out,
                 'mstar_draw_xi': mstar_draw_xi}, f)
print("\nSaved TPCF and prediction results to pickle.")
