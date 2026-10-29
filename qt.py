import numpy as np


def qt(x, y):
    """Phil's QT Function
    - qt performs a quantile transformation from x -> y. (i.e. what values
    would the values in x have if they had the same ranks within the y
    distribution).
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
