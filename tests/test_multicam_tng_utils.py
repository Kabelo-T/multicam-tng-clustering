import numpy as np
import pandas as pd
import pytest
from multicam_tng_utils import (
    convert_tng_mass, get_mpeak_from_mah, get_an_from_mpeak_mah,
    get_vvir, get_vmax_over_vvir, get_msmhmr)


def test_convert_tng_mass():
    # 1e10 Msun/h = 1e10/0.6774 Msun. Hard-coded: using TNG_H here would make
    # the test circular (a wrong TNG_H would change both sides).
    assert convert_tng_mass(1.0) == pytest.approx(10.169155)
    assert convert_tng_mass(0.0) == 0


def test_mpeak_is_running_max_normalized():
    mah = np.log10([[1, 4, 2, 8]])
    np.testing.assert_allclose(get_mpeak_from_mah(mah), [[1/8, 4/8, 4/8, 1]])


def test_an_linear_history():
    # If mpeak grows linearly with scale factor, a_n = n.
    scales = np.linspace(0, 1, 11)
    ns = np.array([0.25, 0.5, 0.75])
    an = get_an_from_mpeak_mah(scales[None, :], scales=scales, ns=ns)
    np.testing.assert_allclose(an, [ns])


def test_vvir_milky_way():
    # sqrt(G M / R), G = 4.3009e-6 kpc (km/s)^2 / Msun.
    expected = np.sqrt(4.3009e-6 * 1e12 / 200)
    assert get_vvir(200, 1e12) == pytest.approx(expected, rel=1e-3)


def test_msmhmr_no_scatter_gives_zero_deviation():
    mvir = np.linspace(11.01, 12.99, 200)
    dev, (m, b) = get_msmhmr(mvir - 2, mvir, mass_bin=(11, 13))
    np.testing.assert_allclose(dev, 0, atol=1e-10)
    assert m == pytest.approx(0, abs=1e-10) and b == pytest.approx(-2)



def test_vmax_over_vvir_is_one_at_vvir():
    vvir = np.sqrt(4.3009e-6 * 1e12 / 200)
    cat = pd.DataFrame({'Mvir': [12.0], 'Rvir': [200.0], 'Vmax_DM': [vvir]})
    assert get_vmax_over_vvir(cat)[0] == pytest.approx(1, rel=1e-3)
