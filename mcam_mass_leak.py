import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sham
import plot
import paths
from constants import HYDRO_STELLAR_MASS_MIN, MASS_BIN_EDGES
from multicam.multicam import MultiCAM

# Row-order leak test for stage 2 (color) of mcam_two_stage.py.
#
# MultiCAM gaussianizes its training features with ordinal ranks, so tied feature values are
# ranked by row order. The cache rows are in subfind_id order (FoF group mass, then mass within
# the group), and the early MAH snapshots are heavily tied (mpeak_mah0 is 0 for ~99.7% of
# galaxies; ties fade out by snapshot ~8). Row order can then put host mass and central/satellite
# information into the fitted color model through the early MAH columns.
#
# Stage 1 is the SHAM M* of mcam_two_stage.py (log mpeak -> log M*), computed once in the cache
# order. Within each predicted-M* bin, the stage-2 color MultiCAM (features -> g-i) is refit with
# the bin's rows in different orders (ROW_ORDERS):
#   'subfind':      cache order, i.e. what mcam_two_stage.py does
#   'shuffle':      N_SEEDS random permutations; random tie-breaking and the seed-noise floor
#   'color_sorted': sorted by true g-i, so tie ranks follow the target (adversarial upper bound)
# and with the MAH starting at snapshot 0 or MAH_CUT_SNAP (MAH_FIRST_SNAPS), the first snapshot
# with essentially no ties. z12 and z12_mpeak have no MAH and serve as controls: their results
# should not depend on row order.
#
# Prediction uses mid-ranks for ties, so the leak can only act through the fitted coefficients;
# the mpeak_mah0..9 coefficients are saved to show whether it is learned.

COLOR_CUT = 0.85
N_SEEDS = 10
MAH_CUT_SNAP = 10
MAH_FIRST_SNAPS = [0, MAH_CUT_SNAP]
N_COEF_SNAPS = 10  # coefficients saved for mpeak_mah0..N_COEF_SNAPS-1
MAH_COLS = [f'mpeak_mah{ii}' for ii in range(100)]
OUT_DIR = paths.REPO_ROOT / 'tests' / 'mass_leak'
OUT_NAME = 'mass_leak'


def feature_sets(first_snap):
    """Feature sets of mcam_two_stage.py with the MAH starting at first_snap; the MAH-free
    controls are only included once (first_snap == 0)"""
    mah = MAH_COLS[first_snap:]
    sets = {'mah_mpeak': mah[:-1] + ['logMpeak'],
            'z12_mah': ['z12'] + mah}
    if first_snap == 0:
        sets = {'z12': ['z12'], 'z12_mpeak': ['z12', 'logMpeak'], **sets}
    return sets


## Load data ###
hydro_full = pd.read_csv(paths.HYDRO_COLOR_TREE_CACHE, usecols=['StellarMass', 'logStellarMass', 'g-i',
                                                                'a12', 'mpeak', 'is_sub', *MAH_COLS])
hydro_full = hydro_full[hydro_full['StellarMass'] >= HYDRO_STELLAR_MASS_MIN].copy()
print(f"\nHydro table after the stellar mass cut: {len(hydro_full)}")

hydro_full['z12'] = (1 / hydro_full['a12']) - 1
hydro_full['logMpeak'] = np.log10(hydro_full['mpeak'])


### Stage 1: SHAM stellar mass, in cache order ###
hydro_full['logStellarMass_sham'] = sham.sham(hydro_full['logMpeak'].to_numpy(),
                                              hydro_full['logStellarMass'].to_numpy())
mass_bins = {f'bin{i+1}': edges for i, edges in enumerate(MASS_BIN_EDGES)}
pops = {}
for key, (low, high) in mass_bins.items():
    mask = (hydro_full['logStellarMass_sham'] >= low) & (hydro_full['logStellarMass_sham'] < high)
    pops[key] = hydro_full[mask].copy()
    tied = [pops[key][col].duplicated(keep=False).mean() for col in MAH_COLS[:MAH_CUT_SNAP]]
    print(f"{key} [{low}, {high}): N = {len(pops[key])}, tied fraction mpeak_mah0..{MAH_CUT_SNAP - 1}: "
          + ", ".join(f"{t:.3f}" for t in tied))


### Stage 2 under each row order ###
def row_orders(pop):
    """Yields (order, seed, reordered pop)"""
    yield 'subfind', -1, pop
    for seed in range(N_SEEDS):
        yield 'shuffle', seed, pop.iloc[np.random.default_rng(seed).permutation(len(pop))]
    yield 'color_sorted', -1, pop.sort_values('g-i', kind='stable')


rows = []
for key, pop in pops.items():
    true_red = pop['g-i'] > COLOR_CUT
    for order, seed, pop_ordered in row_orders(pop):
        y_train = pop_ordered[['g-i']].to_numpy()
        for first_snap in MAH_FIRST_SNAPS:
            for name, cols in feature_sets(first_snap).items():
                model = MultiCAM(len(cols), 1)
                model.fit(pop_ordered[cols].to_numpy(), y_train)
                # back to the bin's original order, to compare with the truth
                color_pred = pd.Series(model.predict(pop_ordered[cols].to_numpy())[:, 0],
                                       index=pop_ordered.index).loc[pop.index]
                red = color_pred > COLOR_CUT
                row = {'bin': key, 'order': order, 'seed': seed, 'first_snap': first_snap, 'set': name,
                       'spearman_color': spearmanr(color_pred, pop['g-i']).statistic,
                       'red_match_frac': np.mean(red == true_red),
                       'red_frac': red.mean(),
                       'sat_frac_red': pop.loc[red, 'is_sub'].mean(),
                       'sat_frac_blue': pop.loc[~red, 'is_sub'].mean()}
                coefs = dict(zip(cols, model.reg.coef_[0]))
                for col in MAH_COLS[:N_COEF_SNAPS]:
                    row[f'coef_{col}'] = coefs.get(col, np.nan)
                rows.append(row)
    print(f"\n{key}: stage 2 done for all row orders")

leak = pd.DataFrame(rows)
OUT_DIR.mkdir(parents=True, exist_ok=True)
leak_file = OUT_DIR / f'{OUT_NAME}_diagnostics.csv'
leak.to_csv(leak_file, index=False)
print(f"\nSaved diagnostics to {leak_file}")


### Summary: cache order and color-sorted vs. the shuffle mean ± std ###
METRICS = ['spearman_color', 'red_match_frac', 'red_frac', 'sat_frac_red', 'sat_frac_blue']
group_cols = ['bin', 'first_snap', 'set']
shuffle = leak[leak['order'] == 'shuffle'].groupby(group_cols)[METRICS].agg(['mean', 'std'])
fixed = {order: leak[leak['order'] == order].set_index(group_cols)[METRICS]
         for order in ['subfind', 'color_sorted']}
variants = list(fixed['subfind'].loc[list(mass_bins)[0]].index)  # (first_snap, set)

for metric in METRICS:
    print(f"\n{metric}: subfind / color_sorted vs. shuffle mean ± std, (subfind - mean) / std")
    for key in mass_bins:
        for first_snap, name in variants:
            idx = (key, first_snap, name)
            mean, std = shuffle.loc[idx, (metric, 'mean')], shuffle.loc[idx, (metric, 'std')]
            sub, adv = fixed['subfind'].loc[idx, metric], fixed['color_sorted'].loc[idx, metric]
            z = (sub - mean) / std if std > 0 else np.nan
            print(f"  {key} {name:10s} snaps>={first_snap:<3d} {sub:.4f} / {adv:.4f} vs "
                  f"{mean:.4f} ± {std:.4f}, z = {z:+.1f}")

coef_cols = [f'coef_{col}' for col in MAH_COLS[:N_COEF_SNAPS]]
print("\nmah_mpeak coefficients on mpeak_mah0..9 (snaps >= 0): subfind, shuffle mean, color_sorted")
for key in mass_bins:
    sel = leak[(leak['bin'] == key) & (leak['set'] == 'mah_mpeak') & (leak['first_snap'] == 0)]
    for order in ['subfind', 'shuffle', 'color_sorted']:
        coefs = sel.loc[sel['order'] == order, coef_cols].mean()
        print(f"  {key} {order:12s} " + " ".join(f"{c:+.3f}" for c in coefs))

fig, axes = plt.subplots(len(METRICS), len(mass_bins), figsize=(16, 3 * len(METRICS)), sharex=True)
x = np.arange(len(variants))
for i, metric in enumerate(METRICS):
    for j, key in enumerate(mass_bins):
        ax = axes[i, j]
        idx = [(key, first_snap, name) for first_snap, name in variants]
        ax.errorbar(x, shuffle.loc[idx, (metric, 'mean')], yerr=shuffle.loc[idx, (metric, 'std')],
                    fmt='o', color='C0', capsize=3, label=f'shuffle (mean ± std, {N_SEEDS} seeds)')
        ax.plot(x, fixed['subfind'].loc[idx, metric], 's', color='C3', mfc='none', label='subfind order')
        ax.plot(x, fixed['color_sorted'].loc[idx, metric], 'x', color='k', label='sorted by g-i')
        if i == 0:
            low, high = mass_bins[key]
            ax.set_title(fr'SHAM $M_*$: ${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
        if j == 0:
            ax.set_ylabel(metric)
for ax in axes[-1]:
    ax.set_xticks(x)
    ax.set_xticklabels([f'{name}\nsnaps>={first_snap}' for first_snap, name in variants], fontsize=8)
axes[0, 0].legend(fontsize=8)
plt.tight_layout()
plt.show()

png_filenames = plot.save_named_figures({f'{OUT_NAME}_summary': fig}, OUT_DIR)
print(f'\nFigures saved as PNG files: {png_filenames}')
