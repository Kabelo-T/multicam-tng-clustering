import numpy as np
import pandas as pd
import pytest
from constants import MASS_BIN_EDGES
from sham import sham
from cam import cam
from sample import mass_sample


@pytest.fixture
def cat():
    # Toy catalog: halo mass sets stellar mass (with scatter); a12 sets color.
    rng = np.random.default_rng(0)
    n = 2000
    log_mh = rng.uniform(10, 14, n)
    a12 = rng.uniform(0.2, 0.8, n)
    return pd.DataFrame({
        'log_mh': log_mh,
        # Rounded to 0.05 dex so some galaxies sit exactly on the bin edges.
        'log_mstar': np.round((0.8 * log_mh + rng.normal(0, 0.3, n) + 1) * 20)
                     / 20,
        'a12': a12,
        'gi': 0.5 + a12 + rng.normal(0, 0.1, n),
    })


def test_sham_most_massive_halo_gets_most_massive_galaxy(cat):
    # test if the abundances match
    pred = sham(cat.log_mh, cat.log_mstar)
    assert pred[cat.log_mh.argmax()] == cat.log_mstar.max()


@pytest.mark.parametrize('lo, hi', MASS_BIN_EDGES)
def test_mass_sample_edges(cat, lo, hi):
    # test that the mass sample is in the right mass bin
    t = mass_sample(cat, 'log_mstar', lo, hi)
    assert len(t) > 0
    assert (t.log_mstar >= lo).all() and (t.log_mstar < hi).all()
    assert len(t) == ((cat.log_mstar >= lo) & (cat.log_mstar < hi)).sum()


@pytest.mark.parametrize('lo, hi', MASS_BIN_EDGES)
def test_cam_bin(cat, lo, hi):
    obs_bin, pred, mask_gal, mask_halo = cam(
        cat.log_mh, cat.log_mstar, cat.a12, cat.gi, lo, hi)
    # Equal-length catalogs: SHAM is a permutation, so both bins hold the same count.
    assert mask_gal.sum() == mask_halo.sum() == len(pred) == len(obs_bin)
    # Within the bin, colors are the true bin colors, reassigned in a12 order.
    np.testing.assert_allclose(np.sort(pred), np.sort(obs_bin))
    a12_bin = cat.a12[mask_halo].values
    np.testing.assert_array_equal(np.argsort(pred), np.argsort(a12_bin))
