import numpy as np


def qt(x, y):
    """Quantile-transform x onto the distribution of y (Phil's QT).

    Each value in x is replaced by the value at the same quantile of y, so the
    output keeps the rank order of x but follows the distribution of y.

    Parameters
    ----------
    x : array_like
        Values to transform (e.g. halo vpeak). Needs at least 2 elements.
    y : array_like
        Target distribution (e.g. stellar mass). Needs at least 2 elements.

    Returns
    -------
    ndarray
        Transformed x, same length as x. Interpolates linearly in sorted y when
        len(x) != len(y).
    """

    # get ranks for each element of x
    n = len(x)
    rank = np.zeros(n)
    rank[np.argsort(x)] = np.arange(n)

    # rescale to size of y and convert to indices/remainders
    rank *= (len(y) - 1)/(len(x) - 1)
    idx = np.array(rank, dtype=int)
    remainder = rank - idx

    # reduce the indices that are too large and make up for it with the remainder
    too_big = idx == len(y) - 1
    idx[too_big], remainder[too_big] = len(y) - 2, 1.0

    # linearly interpolate
    y = np.sort(y)
    dy = y[1:] - y[:-1]
    return y[idx] + dy[idx]*remainder
