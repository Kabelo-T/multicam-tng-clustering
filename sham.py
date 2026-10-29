import numpy as np
from sklearn.preprocessing import QuantileTransformer
from qt import qt
from constants import TNG_H


def sham(proxy, observed):
    """Updated SHAM with Phil's QT"""
    proxy_pred = qt(proxy, observed)
    return proxy_pred


def sham_sklearn_qt(proxy, observed):
    """Executes SHAM on two inputs proxy and observed
    Args:
        proxy: ex. model stellar mass (usually halo mass)
        observed: ex. observed stellar mass
    Return:
        Outputs a numpy array with proxy_pred
    Note:
        - proxy and observed must be numpy arrays
        - if inputs are both log10 then prediction will be log10
    """
    x = proxy
    x2 = observed
    
    # Reshape x and x2 for quantile transformer (expects a 2D input)
    x = x.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    
    # Fit the quantile transformer for x and get ranks
    qt_x = QuantileTransformer(n_quantiles=len(x), output_distribution='uniform', subsample=None)
    r_x = qt_x.fit_transform(x) 

    # Fit the quantile transformer for x2
    qt_x2 = QuantileTransformer(n_quantiles=len(x2), output_distribution='uniform',subsample=None)
    qt_x2.fit(x2)

    # Use inverse transform on qt_x2 to get proxy_pred
    proxy_pred = qt_x2.inverse_transform(r_x)  # Map the ranks of x into x2's space

    return proxy_pred.flatten()
    
    