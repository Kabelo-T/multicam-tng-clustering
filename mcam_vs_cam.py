import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr
import stats
import cam
import paths
from constants import TNG_H, LBOX, HYDRO_STELLAR_MASS_MIN, MASS_BIN_EDGES
from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables.two_point_clustering import tpcf_jackknife, tpcf
from multicam.multicam import MultiCAM, MultiCamSampling

# Hydro-only test of whether MultiCAM (1 feature: z12, 1 target: g-i), with and without
# scatter, reduces to the CAM implementation in cam.py inside the same stellar-mass bins.
# No DMO data is used: every halo property comes from the hydro catalog's own trees.
#
# CAM: stellar mass is assigned by SHAM of hydro mpeak (peak total SubhaloMass) onto the
# hydro stellar mass distribution; within each SHAM stellar-mass bin, hydro z12 is then
# rank-matched onto the hydro g-i distribution of the true stellar-mass bin.
#
# MultiCAM: uses the same SHAM stellar-mass bins as CAM. It is fit on the (z12, g-i) pairs
# of the true stellar-mass bin and applied to the z12 of the SHAM bin, so both methods
# predict colors for the same galaxies. With no scatter, MultiCAM ranks SHAM-bin z12
# against the training (true-bin) z12, whereas CAM ranks it against itself, so any shift
# in the z12 distribution between the two bins shows up as a CAM - MultiCAM difference.

COLOR_CUT = 0.85
N_THREADS = 16
N_SAMPLES = 100
SAMPLE_PCTILES = (16, 84)
RUN_SAMPLING = False  # MultiCamSampling (scatter) mode; slow because of the N_SAMPLES tpcf draws

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

for key, (low, high) in mass_bins.items():
    # CAM: mpeak -> SHAM stellar mass, then z12 -> g-i in the SHAM bin
    color_bin, color_pred_cam, mask_gal, mask_sham = cam.cam(
        hydro_full['mpeak'].to_numpy(),
        hydro_full['logStellarMass'].to_numpy(),
        hydro_full['z12'].to_numpy(),
        hydro_full['g-i'].to_numpy(),
        low, high)

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
    print(f"\n{key} [{low}, {high}): N_hydro = {len(hydro_bin)}, N_sham = {len(cam_bin)}, "
          f"SHAM/true bin overlap = {overlap_frac:.3f}")
    print(f"  Spearman(z12, g-i) in hydro training set:     {rho_train:.3f}")
    print(f"  Spearman(CAM, MultiCAM no scatter):           {rho_mcam:.4f}")
    print(f"  max |CAM - MultiCAM no scatter|:              {np.max(np.abs(diff)):.4f}")
    print(f"  median (MultiCAM - CAM):                      {np.median(diff):.4f}")
    diag = {'bin': key, 'logmstar_min': low, 'logmstar_max': high,
            'n_hydro': len(hydro_bin), 'n_sham': len(cam_bin),
            'overlap_frac': overlap_frac,
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
                    'color_pred_samp_draws': color_pred_samp_draws}

print("\nCAM and MultiCAM predictions complete for all bins.")

suffix = '_hydro' + ('_with_sampling' if RUN_SAMPLING else '')
paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
diagnostics_file = paths.CACHE_DIR / f'mcam_vs_cam_diagnostics{suffix}.csv'
pd.DataFrame(diagnostics).to_csv(diagnostics_file, index=False)
print(f"\nSaved diagnostics to {diagnostics_file}")


### Histogram of color vs. predicted color ###
fig0 = plt.figure(figsize=(18, 4))

for i, key in enumerate(mass_bins.keys()):
    plt.subplot(1, len(mass_bins), i+1)

    res = results[key]
    plt.hist(res['hydro']['color'], bins=40, density=True, alpha=0.4, label='True color (Hydro)')
    plt.hist(res['cam']['color_pred_cam'], bins=40, density=True, histtype='step', linewidth=2, label='CAM')
    plt.hist(res['cam']['color_pred_mcam'], bins=40, density=True, histtype='step', linewidth=2, linestyle='--', label='MultiCAM')
    if RUN_SAMPLING:
        plt.hist(res['color_pred_samp_draws'][0], bins=40, density=True, histtype='step', linewidth=2, linestyle=':', label='MultiCAM Sampling (1 draw)')
    plt.axvline(COLOR_CUT, color='k', linestyle='--')
    low, high = mass_bins[key]
    plt.title(fr'${low} \leq \log(M_*/h^{{-1}}M_\odot) < {high}$')
    plt.xlabel(r'$g - i$ color')
    plt.ylabel('Normalized frequency')

    if i == 0:
        plt.legend()

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
fig2, axes = plt.subplots(2, len(mass_bins), figsize=(19, 9), sharex=True, sharey='row',
                          gridspec_kw={'height_ratios': [3, 1]})
for i, key in enumerate(mass_bins.keys()):
    (xi_red_hydro, xi_blue_hydro), (cov_red_hydro, cov_blue_hydro) = tpcf_results['hydro'][i]
    (xi_red_cam, xi_blue_cam), (cov_red_cam, cov_blue_cam) = tpcf_results['cam'][i]
    (xi_red_mcam, xi_blue_mcam), (cov_red_mcam, cov_blue_mcam) = tpcf_results['mcam'][i]
    samp_red, samp_blue = tpcf_results['mcam_samp'][i] if RUN_SAMPLING else (None, None)

    low, high = mass_bins[key]
    ax, rax = axes[0, i], axes[1, i]

    for color, xi_hydro, cov_hydro, xi_cam, cov_cam, xi_mcam, cov_mcam, samp in [
        ('red', xi_red_hydro, cov_red_hydro, xi_red_cam, cov_red_cam, xi_red_mcam, cov_red_mcam, samp_red),
        ('blue', xi_blue_hydro, cov_blue_hydro, xi_blue_cam, cov_blue_cam, xi_blue_mcam, cov_blue_mcam, samp_blue),
    ]:
        err = stats.compute_tpcf_jackknife_errors(cov_hydro)
        err_cam = stats.compute_tpcf_jackknife_errors(cov_cam)
        err_mcam = stats.compute_tpcf_jackknife_errors(cov_mcam)
        label = color.capitalize()
        mcam_color = 'k' if color == 'red' else 'gray'

        ax.plot(rbin_centers, xi_hydro, marker='o', color=color, label=fr'Truth {label}: TNG100 Hydro $\xi(r)$')
        ax.fill_between(rbin_centers, xi_hydro - err, xi_hydro + err, color=color, alpha=0.2)
        ax.errorbar(r_cam, xi_cam, yerr=err_cam, marker='s', color=color, linestyle='--', capsize=2, label=fr'Prediction {label}: CAM $\xi(r)$')
        ax.errorbar(r_mcam, xi_mcam, yerr=err_mcam, marker='x', color=mcam_color, linestyle=':', capsize=2, label=fr'Prediction {label}: MultiCAM $\xi(r)$')
        if samp is not None:
            xi_s, xi_s_lo, xi_s_hi = samp
            ax.plot(rbin_centers, xi_s, color='magenta' if color == 'red' else 'cyan', linestyle='-.', label=fr'Prediction {label}: MultiCAM Sampling $\xi(r)$ (median of {N_SAMPLES})')
            ax.fill_between(rbin_centers, xi_s_lo, xi_s_hi, color='magenta' if color == 'red' else 'cyan', alpha=0.2)

        rax.fill_between(rbin_centers, 1 - err / xi_hydro, 1 + err / xi_hydro, color=color, alpha=0.2)
        rax.errorbar(r_cam, xi_cam / xi_hydro, yerr=err_cam / xi_hydro, marker='s', color=color, linestyle='--', capsize=2)
        rax.errorbar(r_mcam, xi_mcam / xi_hydro, yerr=err_mcam / xi_hydro, marker='x', color=mcam_color, linestyle=':', capsize=2)
        if samp is not None:
            rax.plot(rbin_centers, xi_s / xi_hydro, color='magenta' if color == 'red' else 'cyan', linestyle='-.')

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
    f'mcam_vs_cam_color_hist{suffix}': fig0,
    f'mcam_vs_cam_color_scatter{suffix}': fig1,
    f'mcam_vs_cam_red_blue_tpcf{suffix}': fig2,
}
paths.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
png_filenames = []
for name, fig in figure_names.items():
    filename = paths.FIGURES_DIR / f'{name}.png'
    fig.savefig(filename, format='png', bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    png_filenames.append(str(filename))
print(f'\nFigures saved as PNG files: {png_filenames}')


### Save results with pickle ###
with open(paths.CACHE_DIR / f'tpcf_results_mcam_vs_cam{suffix}.pkl', 'wb') as f:
    pickle.dump({'tpcf': tpcf_results, 'rbins': rbins, 'results': results}, f)
print("\nSaved TPCF, CAM, and MultiCAM results to pickle.")
