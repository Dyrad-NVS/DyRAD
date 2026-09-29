"""Sensor constants shared across the package: RADIal and the Boreas Navtech radar."""

#: RADIal Doppler bin width in m/s per raw 256-point FFT bin (RADIal dataset, Rebut et al.,
#: 2022). The 16-bin DDMA-reduced axis therefore spans 16 * 0.1123 = 1.7968 m/s.
DOPPLER_BIN_MPS = 0.1123

#: Elevation slice of RADIal's CalibrationTable used for the RA processing (+1 deg).
EL_IDX = 5

#: RADIal maximum range (m); the range bin width is RADIAL_RANGE_MAX_M / n_samples.
RADIAL_RANGE_MAX_M = 103.0

#: Navtech CIR304-H (Boreas) range bin width (m).
NAVTECH_RANGE_RES_M = 0.0596

#: Navtech log-compressed counts per decade k: a stored count u is power 10^(u / k).
NAVTECH_COUNTS_PER_DECADE = 20.0

#: Navtech near-field blind zone (m) that `boreas convert` blanks:
#: int(2.5 / NAVTECH_RANGE_RES_M) = 41 range bins, the Boreas recipe's near_range_blank_bins.
NAVTECH_NEAR_RANGE_BLACKOUT_M = 2.5
