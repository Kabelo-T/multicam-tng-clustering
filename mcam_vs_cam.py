import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr, ks_2samp
import stats
import cam
import plot
import paths
from constants import TNG_H, LBOX, HYDRO_STELLAR_MASS_MIN, MASS_BIN_EDGES
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables.two_point_clustering import tpcf_jackknife, tpcf
from multicam.multicam import MultiCAM, MultiCamSampling

# Hydro-only test of whether MultiCAM (1 feature: z12, 1 target: g-i), with and without
# scatter, reduces to the CAM implementation in cam.py inside the same stellar-mass bins.
# No DMO data is used: every halo property comes from the hydro catalog's own trees.
#
# Stellar mass is assigned in one of two ways (SHAM_STELLAR_MASS):
# - True: SHAM of hydro mpeak (peak total SubhaloMass) onto the hydro stellar mass
#   distribution, giving a SHAM stellar-mass bin that differs from the true one.
# - False: the true hydro stellar mass (passed to cam.cam as its own SHAM proxy, which
#   qt maps onto itself exactly), so the SHAM bin is the true bin.
#
# CAM: within each SHAM stellar-mass bin, hydro z12 is rank-matched onto the hydro g-i
# distribution of the true stellar-mass bin.
#
# MultiCAM: fit on the (z12, g-i) pairs of the true stellar-mass bin and applied to the
# z12 of the SHAM bin, so both methods predict colors for the same galaxies. With no
# scatter, MultiCAM ranks SHAM-bin z12 against the training (true-bin) z12, whereas CAM
# ranks it against itself, so any shift in the z12 distribution between the two bins
# shows up as a CAM - MultiCAM difference. With true stellar masses the two z12
# distributions are identical, so MultiCAM should reproduce CAM up to interpolation.

COLOR_CUT = 0.85
N_THREADS = 16
N_SAMPLES = 100
SAMPLE_PCTILES = (16, 84)
RUN_SAMPLING = False  # MultiCamSampling (scatter) mode; slow because of the N_SAMPLES tpcf draws
SHAM_STELLAR_MASS = True  # True: SHAM mpeak -> stellar mass; False: true hydro stellar mass (pure reduction test)

## Load data ###
hydro_full = pd.read_csv(paths.HYDRO_COLOR_TREE_CACHE, usecols=['SubhaloMass', 'StellarMass', 'x', 'y', 'z',
                                                                'logStellarMass', 'logSubhaloMass', 'g-i',
                                                                'a12', 'mpeak', 'is_sub'])
print("\nHydro Table")
print(len(hydro_full))
print("\nCache files read in!")

# minimum on Stellar Mass for hydro catalog
hydro_full = hydro_full[hydro_full['StellarMass'] >= HYDRO_STELLAR_MASS_MIN]
print("\nNew Hydro Table Length:")
print(len(hydro_full))

# Add z12 column
hydro_full['z12'] = (1 / hydro_full['a12']) - 1


### Run CAM and MultiCAM predictions for bins ###
mass_bins = {f'bin{i+1}': edges for i, edges in enumerate(MASS_BIN_EDGES)}

results = {}
diagnostics = []
mass_proxy = 'mpeak' if SHAM_STELLAR_MASS else 'logStellarMass'

for key, (low, high) in mass_bins.items():
    # CAM: mass proxy -> SHAM stellar mass, then z12 -> g-i in the SHAM bin
    color_bin, color_pred_cam, mask_gal, mask_sham = cam.cam(
        hydro_full[mass_proxy].to_numpy(),
        hydro_full['logStellarMass'].to_numpy(),
        hydro_full['z12'].to_numpy(),
        hydro_full['g-i'].to_numpy(),
        low, high)
    if not SHAM_STELLAR_MASS:
        assert np.array_equal(mask_gal, mask_sham)

    hydro_bin = hydro_full[mask_gal].copy()  # truth and MultiCAM training set (true stellar mass)
    cam_bin = hydro_full[mask_sham].copy()   # CAM and MultiCAM population (SHAM stellar mass)

    # MultiCAM fit on hydro (z12, g-i) pairs in the true stellar-mass bin,
    # applied to z12 in the SHAM stellar-mass bin
    x_train = hydro_bin[['z12']].to_numpy()
    y_train = hydro_bin[['g-i']].to_numpy()
    x_sham = cam_bin[['z12']].to_numpy()

    model = MultiCAM(1, 1)
    model.fit(x_train, y_train)
    color_pred_mcam = model.predict(x_sham)[:, 0]

    if RUN_SAMPLING:
        model_samp = MultiCamSampling(1, 1)
        model_samp.fit(x_train, y_train)
        color_pred_samp_draws = np.array([model_samp.sample(x_sham)[:, 0] for _ in range(N_SAMPLES)])
    else:
        color_pred_samp_draws = None

    cam_bin['color_pred_cam'] = color_pred_cam
    cam_bin['color_pred_mcam'] = color_pred_mcam
    hydro_bin['color'] = color_bin

    # Reduction diagnostics, halo by halo on the SHAM bin
    rho_train = spearmanr(x_train[:, 0], y_train[:, 0]).statistic
    rho_mcam = spearmanr(color_pred_cam, color_pred_mcam).statistic
    diff = color_pred_mcam - color_pred_cam
    # fraction of true-bin galaxies that SHAM also places in the bin
    overlap_frac = len(cam_bin.index.intersection(hydro_bin.index)) / len(hydro_bin)
    # galaxies in only one of the two bins: SHAM swaps these in/out of the true bin
    sham_only = cam_bin.loc[cam_bin.index.difference(hydro_bin.index)]
    hydro_only = hydro_bin.loc[hydro_bin.index.difference(cam_bin.index)]
    ks_z12 = ks_2samp(hydro_bin['z12'], cam_bin['z12']).statistic
    print(f"\n{key} [{low}, {high}): N_hydro = {len(hydro_bin)}, N_sham = {len(cam_bin)}, "
          f"SHAM/true bin overlap = {overlap_frac:.3f}")
    print(f"  median z12 true / SHAM bin (KS):              {hydro_bin['z12'].median():.3f} / "
          f"{cam_bin['z12'].median():.3f} ({ks_z12:.3f})")
    print(f"  median z12 true-only / SHAM-only:             {hydro_only['z12'].median():.3f} / "
          f"{sham_only['z12'].median():.3f}")
    print(f"  satellite fraction true / SHAM bin:           {hydro_bin['is_sub'].mean():.3f} / "
          f"{cam_bin['is_sub'].mean():.3f}")
    print(f"  Spearman(z12, g-i) in hydro training set:     {rho_train:.3f}")
    print(f"  Spearman(CAM, MultiCAM no scatter):           {rho_mcam:.4f}")
    print(f"  max |CAM - MultiCAM no scatter|:              {np.max(np.abs(diff)):.4f}")
    print(f"  median (MultiCAM - CAM):                      {np.median(diff):.4f}")
    diag = {'bin': key, 'logmstar_min': low, 'logmstar_max': high,
            'n_hydro': len(hydro_bin), 'n_sham': len(cam_bin),
            'overlap_frac': overlap_frac,
            'median_z12_hydro': hydro_bin['z12'].median(),
            'median_z12_sham': cam_bin['z12'].median(),
            'ks_z12': ks_z12,
            'median_z12_hydro_only': hydro_only['z12'].median(),
            'median_z12_sham_only': sham_only['z12'].median(),
            'sat_frac_hydro': hydro_bin['is_sub'].mean(),
            'sat_frac_sham': cam_bin['is_sub'].mean(),
            'spearman_train_z12_gi': rho_train,
            'spearman_cam_mcam': rho_mcam,
            'max_abs_diff_mcam_cam': np.max(np.abs(diff)),
            'median_diff_mcam_cam': np.median(diff),
            'red_frac_hydro': np.mean(color_bin > COLOR_CUT),
            'red_frac_cam': np.mean(color_pred_cam > COLOR_CUT),
            'red_frac_mcam': np.mean(color_pred_mcam > COLOR_CUT)}

    # Satellite fraction of each red/blue sample
    for name, pop, col in [('hydro', hydro_bin, 'color'),
                           ('cam', cam_bin, 'color_pred_cam'),
                           ('mcam', cam_bin, 'color_pred_mcam')]:
        red = pop[col] > COLOR_CUT
        diag[f'sat_frac_red_{name}'] = pop.loc[red, 'is_sub'].mean()
        diag[f'sat_frac_blue_{name}'] = pop.loc[~red, 'is_sub'].mean()
    print("  satellite fraction (red / blue):              " + ", ".join(
        f"{name} {diag[f'sat_frac_red_{name}']:.3f} / {diag[f'sat_frac_blue_{name}']:.3f}"
        for name in ['hydro', 'cam', 'mcam']))
    if RUN_SAMPLING:
        rho_samp = np.mean([spearmanr(color_pred_cam, d).statistic for d in color_pred_samp_draws])
        print(f"  mean Spearman(CAM, MultiCAM sampling draw):   {rho_samp:.3f}")
        print(f"  MultiCAM sampling residual scatter (gauss):   {np.sqrt(model_samp.sigma_bar[0, 0]):.3f}")
        diag['mean_spearman_cam_mcam_samp'] = rho_samp
        diag['red_frac_mcam_samp'] = np.mean(color_pred_samp_draws > COLOR_CUT)
    diagnostics.append(diag)

    results[key] = {'hydro': hydro_bin, 'cam': cam_bin,
                    'hydro_only': hydro_only, 'sham_only': sham_only,
                    'color_pred_samp_draws': color_pred_samp_draws}

print("\nCAM and MultiCAM predictions complete for all bins.")

suffix = '_hydro' + ('' if SHAM_STELLAR_MASS else '_true_mstar') + ('_with_sampling' if RUN_SAMPLING else '')
paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
diagnostics_file = paths.CACHE_DIR / f'mcam_vs_cam_diagnostics{suffix}.csv'
pd.DataFrame(diagnostics).to_csv(diagnostics_file, index=False)
print(f"\nSaved diagnostics to {diagnostics_file}")


### Histogram of color vs. predicted color ###
color_preds = [('CAM', {key: res['cam']['color_pred_cam'] for key, res in results.items()}, {'color': 'C1'}),
               ('MultiCAM', {key: res['cam']['color_pred_mcam'] for key, res in results.items()},
                {'color': 'C2', 'linestyle': '--'})]
if RUN_SAMPLING:
    color_preds.append(('MultiCAM Sampling (1 draw)',
                        {key: res['color_pred_samp_draws'][0] for key, res in results.items()},
                        {'color': 'C3', 'linestyle': ':'}))
fig0 = plot.color_hist_by_bin(mass_bins, {key: res['hydro']['color'] for key, res in results.items()},
                              color_preds, COLOR_CUT,
                              np.linspace(hydro_full['g-i'].min(), hydro_full['g-i'].max(), 40))
plt.show()


### Histogram of z12 in the true vs. SHAM stellar-mass bin ###
fig_z12, axes = plt.subplots(1, len(mass_bins), figsize=(18, 4))
for i, key in enumerate(mass_bins.keys()):
    res = results[key]
    ax = axes[i]
    z12_bins = np.linspace(0, res['hydro']['z12'].quantile(0.99), 40)
    ax.hist(res['hydro']['z12'], bins=z12_bins, density=True, alpha=0.4, color='gray', label='True bin')
    ax.hist(res['cam']['z12'], bins=z12_bins, density=True, histtype='step', linewidth=2, color='k', label='SHAM bin')
    # empty in true stellar-mass mode
    if len(res['hydro_only']):
        ax.hist(res['hydro_only']['z12'], bins=z12_bins, density=True, histtype='step', linewidth=2, linestyle='--', color='tab:red', label='True bin only')
        ax.hist(res['sham_only']['z12'], bins=z12_bins, density=True, histtype='step', linewidth=2, linestyle='--', color='tab:blue', label='SHAM bin only')
    low, high = mass_bins[key]
    ax.set_title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    ax.set_xlabel(r'$z_{1/2}$')
    if i == 0:
        ax.set_ylabel('Normalized frequency')
        ax.legend()

plt.tight_layout()
plt.show()


### CAM vs MultiCAM color, halo by halo ###
fig1, axes = plt.subplots(1, len(mass_bins), figsize=(18, 5))
for i, key in enumerate(mass_bins.keys()):
    res = results[key]
    ax = axes[i]
    if RUN_SAMPLING:
        ax.scatter(res['cam']['color_pred_cam'], res['color_pred_samp_draws'][0], s=1, alpha=0.3, color='gray', label='MultiCAM Sampling (1 draw)')
    ax.scatter(res['cam']['color_pred_cam'], res['cam']['color_pred_mcam'], s=1, color='k', label='MultiCAM')
    lims = [res['hydro']['color'].min(), res['hydro']['color'].max()]
    ax.plot(lims, lims, 'r--', linewidth=1)
    low, high = mass_bins[key]
    ax.set_title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    ax.set_xlabel(r'CAM $g - i$')
    if i == 0:
        ax.set_ylabel(r'MultiCAM $g - i$')
        ax.legend(markerscale=10)

plt.tight_layout()
plt.show()

### Define red/blue galaxies ###
for key, res in results.items():
    cam_pop = res['cam']
    hydro = res['hydro']

    res['red_cam'] = cam_pop[cam_pop['color_pred_cam'] > COLOR_CUT]
    res['blue_cam'] = cam_pop[cam_pop['color_pred_cam'] <= COLOR_CUT]
    res['red_mcam'] = cam_pop[cam_pop['color_pred_mcam'] > COLOR_CUT]
    res['blue_mcam'] = cam_pop[cam_pop['color_pred_mcam'] <= COLOR_CUT]
    res['red_hydro'] = hydro[hydro['color'] > COLOR_CUT]
    res['blue_hydro'] = hydro[hydro['color'] <= COLOR_CUT]

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
    'cam': [],
    'mcam': [],
    'mcam_samp': [],
}

for key, res in results.items():
    print(f"\nComputing TPCF for {key}.")

    for name in ['hydro', 'cam', 'mcam']:
        xi_red, cov_red = jackknife(res[f'red_{name}'])
        xi_blue, cov_blue = jackknife(res[f'blue_{name}'])
        tpcf_results[name].append(((xi_red, xi_blue), (cov_red, cov_blue)))

    if not RUN_SAMPLING:
        continue

    # MultiCAM Sampling: tpcf of each draw, median and percentile band across draws
    cam_pop = res['cam']
    xi_red_draws, xi_blue_draws = [], []
    for color_draw in res['color_pred_samp_draws']:
        red = color_draw > COLOR_CUT
        xi_red_draws.append(tpcf(xyz(cam_pop[red]), rbins, randoms=randoms, period=Lbox, num_threads=N_THREADS))
        xi_blue_draws.append(tpcf(xyz(cam_pop[~red]), rbins, randoms=randoms, period=Lbox, num_threads=N_THREADS))

    summary = []
    for draws in [np.array(xi_red_draws), np.array(xi_blue_draws)]:
        summary.append((np.median(draws, axis=0),
                        np.percentile(draws, SAMPLE_PCTILES[0], axis=0),
                        np.percentile(draws, SAMPLE_PCTILES[1], axis=0)))
    tpcf_results['mcam_samp'].append(tuple(summary))

print("\nTPCF computed for all bins.")


### Plot results: xi(r) and ratio to hydro truth ###
# CAM and MultiCAM points are offset slightly in r so their jackknife error bars don't overlap
r_cam, r_mcam = rbin_centers * 0.97, rbin_centers * 1.03
fig2, axes = plot.tpcf_ratio_grid(len(mass_bins))
for i in range(len(mass_bins)):
    for c, (color, truth_color, samp_color) in enumerate([('red', 'gray', 'magenta'), ('blue', 'k', 'cyan')]):
        label = color.capitalize()
        curves = [{'r': r_cam, 'xi': tpcf_results['cam'][i][0][c],
                   'err': stats.compute_tpcf_jackknife_errors(tpcf_results['cam'][i][1][c]),
                   'color': color, 'marker': 's', 'linestyle': '--', 'label': fr'Prediction {label}: CAM $\xi(r)$'},
                  {'r': r_mcam, 'xi': tpcf_results['mcam'][i][0][c],
                   'err': stats.compute_tpcf_jackknife_errors(tpcf_results['mcam'][i][1][c]),
                   'color': color, 'marker': 'x', 'linestyle': '-', 'label': fr'Prediction {label}: MultiCAM $\xi(r)$'}]
        if RUN_SAMPLING:
            xi_s, xi_s_lo, xi_s_hi = tpcf_results['mcam_samp'][i][c]
            curves.append({'median': xi_s, 'lo': xi_s_lo, 'hi': xi_s_hi, 'color': samp_color,
                           'label': fr'Prediction {label}: MultiCAM Sampling $\xi(r)$ (median of {N_SAMPLES})'})

        plot.tpcf_ratio_panel(axes[0, i], axes[1, i], rbin_centers, tpcf_results['hydro'][i][0][c],
                              stats.compute_tpcf_jackknife_errors(tpcf_results['hydro'][i][1][c]),
                              fr'Truth {label}: TNG100 Hydro $\xi(r)$', truth_color, curves)

plot.finish_tpcf_ratio_grid(fig2, axes, mass_bins)
plt.show()

figure_names = {
    f'mcam_vs_cam_color_hist{suffix}': fig0,
    f'mcam_vs_cam_z12_hist{suffix}': fig_z12,
    f'mcam_vs_cam_color_scatter{suffix}': fig1,
    f'mcam_vs_cam_red_blue_tpcf{suffix}': fig2,
}
png_filenames = plot.save_named_figures(figure_names)
print(f'\nFigures saved as PNG files: {png_filenames}')

with open(paths.CACHE_DIR / f'tpcf_results_mcam_vs_cam{suffix}.pkl', 'wb') as f:
    pickle.dump({'tpcf': tpcf_results, 'rbins': rbins, 'results': results}, f)
print("\nSaved TPCF, CAM, and MultiCAM results to pickle.")
