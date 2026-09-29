# gsplat (radar fork)

A trimmed and modified fork of [gsplat](https://github.com/nerfstudio-project/gsplat)
(Apache-2.0, see `LICENSE`) that keeps only the CUDA kernels DyRAD needs:

| op | file | role |
|---|---|---|
| `projection_radar_3dgs_fused` | `csrc/ProjectionRadar3DGSFused*.cu` | project point reflectors to range / azimuth / elevation |
| `rasterize_radar_rae` | `csrc/RasterizeRadarRAE*.cu` | accumulate reflector power into the RAD tensor through the fixed sensor PSF |
| `compute_doppler_radial_velocity` | `cuda/_wrapper.py` (PyTorch) | radial velocity of each reflector for the Doppler axis |

The extension is built by `setup.py` (the build command is in the main README, Installation);
`cuda/_backend.py` only imports it. The sensor-PSF rasterization needs it and raises without it.
`cuda/_wrapper.py` also holds the pure-PyTorch rasterizer of the learned-extent and learned-PSF
ablations. The rasterizer kernels contain an isotropic non-PSF branch that DyRAD never
calls (the wrapper enters the kernel only in PSF mode).

The upstream camera rasterizers, strategies, compression and docs were removed; `glm` under
`csrc/third_party` is the header-only math library the kernels use.
