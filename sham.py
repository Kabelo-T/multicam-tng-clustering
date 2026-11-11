from qt import qt


def sham(proxy, observed):
    """Subhalo abundance matching: rank-match a halo proxy onto galaxies.

    Parameters
    ----------
    proxy : array_like
        Halo property used for ranking (e.g. vpeak or Mpeak).
    observed : array_like
        Galaxy property to assign (e.g. log10 stellar mass).

    Returns
    -------
    ndarray
        Predicted galaxy property for each halo, same length as proxy.
    """
    return qt(proxy, observed)
