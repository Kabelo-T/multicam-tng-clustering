from qt import qt


def cam(proxy, observed, cond_proxy, cond_observed, mass_min, mass_max):
    """Conditional abundance matching within one stellar-mass bin.

    SHAM assigns stellar masses from `proxy`; then, inside
    [mass_min, mass_max), the secondary halo property is rank-matched onto the
    secondary galaxy property.

    Parameters
    ----------
    proxy : array_like
        Halo property for the SHAM step (e.g. vpeak).
    observed : array_like
        Galaxy stellar masses, same units as mass_min/mass_max.
    cond_proxy : array_like
        Secondary halo property, aligned with proxy (e.g. a12).
    cond_observed : array_like
        Secondary galaxy property, aligned with observed (e.g. g-i color).
    mass_min, mass_max : float
        Stellar-mass bin, lower edge inclusive, upper edge exclusive.

    Returns
    -------
    cond_observed_bin : array_like
        True secondary galaxy property in the bin.
    cond_proxy_pred : ndarray
        Predicted secondary property for haloes in the bin.
    mask_gal : array_like of bool
        Galaxies in the bin, e.g. ``hydro[mask_gal]``.
    mask_halo : array_like of bool
        Haloes whose SHAM stellar mass is in the bin, e.g. ``dmo[mask_halo]``.
    """
    stellar_mass_pred = qt(proxy, observed)

    mask_gal = (observed >= mass_min) & (observed < mass_max)
    cond_observed_bin = cond_observed[mask_gal]

    mask_halo = (stellar_mass_pred >= mass_min) & (stellar_mass_pred < mass_max)
    cond_proxy_bin = cond_proxy[mask_halo]

    cond_proxy_pred = qt(cond_proxy_bin, cond_observed_bin)

    return cond_observed_bin, cond_proxy_pred, mask_gal, mask_halo
