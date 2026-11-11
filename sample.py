def mass_sample(table, mass, mass_min, mass_max):
    """Select rows with mass_min <= table[mass] < mass_max.

    Parameters
    ----------
    table : astropy.table.Table or pandas.DataFrame
        Catalog to cut.
    mass : str
        Column to cut on, e.g. 'StellarMass' or 'SubhaloMass'.
    mass_min, mass_max : float
        Bin edges, lower inclusive and upper exclusive, in the column's units.

    Returns
    -------
    astropy.table.Table or pandas.DataFrame
        Rows of table inside the bin, same type as table.
    """
    return table[(table[mass] >= mass_min) & (table[mass] < mass_max)]
