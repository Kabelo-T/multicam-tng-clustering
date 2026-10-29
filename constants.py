"""Shared physical/simulation constants for proj-2pt-surf-2026."""

TNG_H = 0.6774

# TNG100 box size, converted to Mpc/h.
LBOX = 110.7 * TNG_H

# Minimum stellar mass cut applied to the hydro catalog (Msun/h).
HYDRO_STELLAR_MASS_MIN = 1.4e8 * TNG_H

# log10(stellar mass) bin edges shared by the SHAM/CAM pipelines.
MASS_BIN_EDGES = [(8.75, 9.3), (9.3, 10.0), (10.0, 12.5)]
