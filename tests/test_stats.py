import numpy as np
from stats import generate_randoms, compute_tpcf_jackknife_errors


def test_randoms_in_box_and_reproducible():
    r = generate_randoms(10.0, 1000, rng=1)
    assert r.shape == (1000, 3)
    assert (r >= 0).all() and (r < 10).all()
    np.testing.assert_array_equal(r, generate_randoms(10.0, 1000, rng=1))
    np.testing.assert_array_equal(generate_randoms(10.0, 5),
                                  generate_randoms(10.0, 5))


def test_tpcf_errors_are_sqrt_of_diagonal():
    cov = np.array([[4.0, 1.0], [1.0, 9.0]])
    np.testing.assert_allclose(compute_tpcf_jackknife_errors(cov), [2, 3])
