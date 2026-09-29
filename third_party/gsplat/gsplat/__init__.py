"""Radar CUDA rasterizer for DyRAD: a trimmed fork of gsplat (Apache-2.0).

Only the radar kernels remain: spherical projection of point reflectors, Doppler
radial velocity, and rasterization to a range-azimuth-Doppler tensor through the
fixed sensor PSF.
"""
from .cuda._wrapper import (
    projection_radar_3dgs_fused,
    compute_doppler_radial_velocity,
    rasterize_radar_rae,
)
from .version import __version__

__all__ = ["projection_radar_3dgs_fused", "compute_doppler_radial_velocity",
           "rasterize_radar_rae", "__version__"]
