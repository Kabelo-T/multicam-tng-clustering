import numpy as np
from qt import qt
from constants import TNG_H


def cam(proxy, observed, cond_proxy, cond_observed, mass_min, mass_max):
    """Updated CAM using Phil's QT
    Note: Returns mask_gal and mask_halo in order to find galaxies and haloes
    in defined mass bin, i.e. subhalo_data[mask_gal], dmo_data[mask_halo]. 
    """
    stellar_mass_pred = qt(proxy, observed)

    mask_gal = (observed >= mass_min) & (observed < mass_max)
    cond_observed_bin = cond_observed[mask_gal]

    mask_halo = (stellar_mass_pred >= mass_min) & (stellar_mass_pred < mass_max)
    cond_proxy_bin = cond_proxy[mask_halo]

    cond_proxy_pred = qt(cond_proxy_bin, cond_observed_bin)

    return cond_observed_bin, cond_proxy_pred, mask_gal, mask_halo
