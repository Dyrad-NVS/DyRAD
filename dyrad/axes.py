"""Sensor bin axes and the signed Doppler wrap, shared by the trainer, preprocessing and
evaluation (pure numpy).

Range bin b sits at (b + range_bin_offset) * radar_far_range / num_range_bins metres: offset
0 on RADIal and the synthetic sensor (the FFT convention their range profiles, labels and
generator follow), 0.5 on Boreas (the Navtech bin centre). The azimuth
axis is `num_azimuth_bins` evenly spaced centres over +-radar_az_fov_deg / 2 and the
Doppler axis the config's linspace from radar_doppler_min_mps to radar_doppler_max_mps.
The three axis functions take plain numbers and compute in `dtype` (the trainer's bins
are float32, the scorers' float64); `sensor_axes` reads the numbers from a Config or a
config dict.
"""

from __future__ import annotations

import numpy as np

_REQUIRED = object()


def range_bins_m(
    num_range_bins,
    radar_far_range,
    range_bin_offset=0.0,
    *,
    crop_first=0,
    crop_last=0,
    dtype=np.float64,
) -> np.ndarray:
    """Range bin positions (m), `crop_first` / `crop_last` bins dropped at either end."""
    n = int(num_range_bins)
    dr = float(radar_far_range) / float(n)
    centres = (np.arange(n, dtype=dtype) + float(range_bin_offset)) * dr
    return centres[int(crop_first) : n - int(crop_last)]


def azimuth_bins_deg(
    num_azimuth_bins, radar_az_fov_deg, *, dtype=np.float64
) -> np.ndarray:
    """Azimuth bin centres (deg), evenly spaced over the field of view."""
    half = float(radar_az_fov_deg) / 2.0
    return np.linspace(-half, half, int(num_azimuth_bins), dtype=dtype)


def doppler_bins_mps(
    num_doppler_bins, radar_doppler_min_mps, radar_doppler_max_mps, *, dtype=np.float64
) -> np.ndarray:
    """Doppler bin velocities (m/s), the config's linspace (not ego-compensated)."""
    return np.linspace(
        float(radar_doppler_min_mps),
        float(radar_doppler_max_mps),
        int(num_doppler_bins),
        dtype=dtype,
    )


def _field(cfg, key, default=_REQUIRED):
    if isinstance(cfg, dict):
        return cfg[key] if default is _REQUIRED else cfg.get(key, default)
    return getattr(cfg, key) if default is _REQUIRED else getattr(cfg, key, default)


def sensor_axes(cfg, *, crop=True, doppler=True, dtype=np.float64) -> tuple:
    """(range_m, az_deg, dop_mps) of a run config (a `Config` or a config dict).

    `crop=False` keeps the full range axis (before range_crop_first/last);
    `doppler=False` returns None for the Doppler axis (Doppler-less sensors and
    callers that only need range / azimuth).
    """
    n_r = int(_field(cfg, "num_range_bins"))
    range_m = range_bins_m(
        n_r,
        _field(cfg, "radar_far_range"),
        _field(cfg, "range_bin_offset", 0.0),
        crop_first=int(_field(cfg, "range_crop_first", 0)) if crop else 0,
        crop_last=int(_field(cfg, "range_crop_last", 0)) if crop else 0,
        dtype=dtype,
    )
    az_deg = azimuth_bins_deg(
        _field(cfg, "num_azimuth_bins"), _field(cfg, "radar_az_fov_deg"), dtype=dtype
    )
    dop_mps = None
    if doppler:
        dop_mps = doppler_bins_mps(
            _field(cfg, "num_doppler_bins"),
            _field(cfg, "radar_doppler_min_mps"),
            _field(cfg, "radar_doppler_max_mps"),
            dtype=dtype,
        )
    return range_m, az_deg, dop_mps


def range_axis_identity(cfg) -> dict:
    """The numbers that fix the metre position of every range bin (see `range_bins_m`), as
    recorded in an init cloud's `.build.json`: a cloud back-projected on another axis sits at
    the wrong ranges for this config."""
    return {
        "num_range_bins": int(_field(cfg, "num_range_bins")),
        "radar_far_range": float(_field(cfg, "radar_far_range")),
        "range_bin_offset": float(_field(cfg, "range_bin_offset", 0.0)),
    }


def wrap_bins_signed(d: np.ndarray | float, period: float):
    """Wrap a Doppler difference (bins or m/s) into [-period/2, +period/2), in float64."""
    half = 0.5 * period
    return np.remainder(np.asarray(d, dtype=np.float64) + half, period) - half
