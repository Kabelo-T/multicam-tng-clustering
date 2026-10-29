# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Astrophysics research code (thesis project) for the galaxy–halo connection using the IllustrisTNG100
cosmological simulation. The central question: given only dark-matter halo properties from a cheap
dark-matter-only (DMO) run, can we predict galaxy properties (stellar mass, `g-i` color) well enough to
reproduce a full hydrodynamic (baryonic) simulation — including its two-point clustering statistics?

There is no build system, package manifest, linter, or test suite — this is exploratory scientific
analysis code, not a deployable package. Work happens mostly in Jupyter notebooks that import the
top-level `.py` modules as a shared library.

## Running things

- No `requirements.txt`/`environment.yml` exists. Key third-party deps observed across the codebase:
  `numpy`, `pandas`, `astropy`, `h5py`, `scikit-learn`, `halotools`, `matplotlib`, `tqdm`.
- Raw data lives under `data/` (git-ignored, not present in the repo — must be supplied locally, e.g.
  TNG100 hydro/DMO HDF5 subhalo catalogs, merger tree `.npz` files, and derived CSV caches under
  `data/cache_files/`). Scripts/notebooks assume this layout and will fail without it.
- Pipeline scripts (`cam_pipeline.py`, `sham_pipeline.py`, `cache.py`) are written to be run top-to-bottom
  as scripts (`python sham_pipeline.py`), not imported — they read fixed cache paths, do the analysis, show
  plots, and write outputs. There are no CLI args; edit the constants at the top of the file to change
  inputs (e.g. `file_number`, `directory`, mass bin edges).
- No automated tests exist.

## Core pipeline (read this before touching multiple files)

1. **Ingest** (`data.py`): reads raw TNG100 HDF5 subhalo catalogs (`import_hydro`/`import_hydro_prop` for
   the baryonic run, `import_dmo` for the dark-matter-only run) and merger-tree `.npz` files
   (`import_tree_bar`/`import_tree_dmo`, containing per-halo mass accretion histories `mah`/`mpeak_mah`,
   `vpeak`, `a12` formation time, etc.), plus broadband photometry (`import_color`) used to build `g-i`.
   Units: masses converted from TNG code units to Msun/h; positions from kpc to Mpc.
2. **Cache** (`cache.py`): merges hydro + color + tree data and DMO + tree data into flat pandas
   DataFrames, expands per-snapshot MAH arrays into `mah{0..99}`/`mpeak_mah{0..99}` columns, and writes
   them to CSV under `data/cache_files/`. Downstream pipelines/notebooks read only these caches, not raw
   HDF5 — regenerate the cache if the underlying tree/catalog data changes.
3. **Sample matching** (`sample.py`): both catalogs are mass-cut and the DMO catalog is truncated to the
   same length as the hydro catalog (abundance matching assumption — same number density in both).
4. **Prediction methods**, all built on `qt()` (quantile transform: rank-match values in `x` onto the
   distribution of `y`), duplicated identically in `sham.py` and `cam.py`:
   - `sham.sham(proxy, observed)` — single-variable Subhalo Abundance Matching (e.g. `vpeak` → stellar mass).
   - `cam.cam(proxy, observed, cond_proxy, cond_observed, mass_min, mass_max)` — Conditional Abundance
     Matching: within a stellar-mass bin, rank-match a secondary halo property (e.g. `a12`/`z12`) onto a
     secondary galaxy property (e.g. `g-i` color).
   - **MultiCAM** (`analysis/multicam/`, helpers in `multicam_tng_utils.py`) generalizes CAM to condition on
     multiple halo-history features at once (full MAH, `a_n` values derived via `get_an_from_mpeak_mah`,
     `mpeak`, etc.) — this is the primary/current line of work (see recent commits and
     `MultiCAM Code Organization.pdf`).
5. **Validation via clustering** (`stats.py`, `plot.py`, `halotools.mock_observables.tpcf_jackknife`):
   predicted and true galaxy populations are split into mass/color bins, a two-point correlation function
   ξ(r) is computed with jackknife errors over a periodic box (`Lbox = 110.7 * TNG_H`), and predicted vs.
   true clustering is compared — this is the real test of whether a method works, beyond matching 1D
   distributions. Results are also compared against the external **Favole** mock/observational catalog
   (`analysis/favole/`).
6. Figures are saved via `plot.save_figures_as_png` into `figures/` (with `figures/old/` holding
   superseded versions); intermediate results (e.g. `tpcf_results`) are cached with `pickle`.

`TNG_H = 0.6774` (the TNG100 little-h) and TNG's `1e10 Msun/h` mass unit convention recur throughout — watch
for double-counting `h` factors when combining functions from different modules.

## Repository layout

- Top-level `.py` files: the shared library (`data`, `cache`, `sample`, `sham`, `cam`, `stats`, `plot`) and
  standalone pipeline scripts (`cam_pipeline.py`, `sham_pipeline.py`).
- `analysis/` — exploratory notebooks grouped by method (`cam/`, `sham/`, `multicam/`, `qt/`, `tpcf/`,
  `favole/` comparisons, `tutorials/`, `other/` for cache-building/debug notebooks).
- `thesis/` — polished notebooks producing final write-up figures (SHMR, MAH, corner plots, color-CAM
  comparisons); treat these as the "current best" versions of analyses that also exist in draft form under
  `analysis/`.
- `figures/` — output plots. All repo folders use lowercase names by convention.

## Working in this codebase

- `sham.py` and `cam.py` both define their own copy of `qt()` — if you fix a bug in one, check the other.
- Path handling is centralized in `paths.py` (repo-root constants for `data/`, `data/cache_files/`,
  `figures/`, etc.) — import from it rather than hardcoding path strings in `.py` files. Notebooks
  still use plain relative string literals (e.g. `'data/cache_files/...'`), which work because Jupyter
  is run with cwd = repo root.
- When adding a new analysis notebook, follow the existing convention: read pre-built CSV caches from
  `data/cache_files/`, not raw HDF5, unless you are specifically regenerating the cache.
