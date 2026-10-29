# multicam-tng-clustering

Predicting galaxy properties (stellar mass, `g-i` color) from dark-matter halo assembly histories in
IllustrisTNG100, and testing the predictions against the two-point clustering of the full hydrodynamic
simulation.

The methods build on Subhalo Abundance Matching (SHAM) and Conditional Abundance Matching (CAM), and
generalize them with **MultiCAM**, which conditions on multiple halo-history features at once
(full mass accretion history, formation times `a_n`, `mpeak`, ...). Predictions are validated by comparing
the predicted and true two-point correlation functions ξ(r), with jackknife errors computed via `halotools`.

## Layout

- Top-level `.py` files: shared library (`data`, `cache`, `sample`, `sham`, `cam`, `stats`, `plot`,
  `multicam_tng_utils`) and pipeline scripts (`cam_pipeline.py`, `sham_pipeline.py`, `mcam_*.py`).
- `tutorials/`: exploratory and tutorial notebooks.
- `thesis/`: notebooks producing the final thesis figures.
- `MultiCAM Code Organization.pdf`: overview of how the MultiCAM code is organized.

## Data

Simulation data is **not** included. The code expects TNG100 subhalo catalogs (hydro and DMO), merger-tree
files, and derived CSV caches under `data/` (see `paths.py`). TNG data is available from the
[IllustrisTNG project](https://www.tng-project.org/data/).

Key dependencies: `numpy`, `pandas`, `astropy`, `h5py`, `scikit-learn`, `halotools`, `matplotlib`, `tqdm`.

## Credits

This project grew out of earlier work by Haley ([@haleygip](https://github.com/haleygip)), whose
code forms the foundation of this repository.
