import numpy as np
from numpy.random import default_rng
from halotools.mock_observables import return_xyz_formatted_array


def generate_randoms(lbox, n_randoms, rng=None):
    """Uniform random points in a periodic box, e.g. for tpcf_jackknife.

    Parameters
    ----------
    lbox : float
        Box side length (usually Mpc/h).
    n_randoms : int
        Number of random points.
    rng : numpy.random.Generator, int or None, optional
        Generator or seed. None uses a fixed seed of 42 for reproducibility.

    Returns
    -------
    ndarray, shape (n_randoms, 3)
        xyz positions in [0, lbox).
    """
    if rng is None:
        _rng = default_rng(42)
    elif isinstance(rng, int):
        _rng = default_rng(rng)
    else:
        _rng = rng

    Xran = _rng.uniform(0, lbox, n_randoms)
    Yran = _rng.uniform(0, lbox, n_randoms)
    Zran = _rng.uniform(0, lbox, n_randoms)
    Nran = return_xyz_formatted_array(Xran, Yran, Zran)
    return Nran


def compute_tpcf_jackknife_errors(xi_cov):
    """1-sigma errors on xi(r) from a jackknife covariance matrix.

    Parameters
    ----------
    xi_cov : ndarray, shape (n_bins, n_bins)
        Covariance returned by halotools' tpcf_jackknife.

    Returns
    -------
    ndarray, shape (n_bins,)
        Square root of the diagonal.
    """
    return np.sqrt(np.diag(xi_cov))
