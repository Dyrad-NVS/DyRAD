"""Fixed sensor point-spread function (PSF) of the radar cube.

The measurement chain (windowed FFTs in range and Doppler, beamforming in azimuth)
spreads every point scatterer into a separable per-axis lobe. `SensorPSF`
models that lobe as a sinc^2 mainlobe times a Hann taper on each of the Doppler,
range and azimuth axes (paper Eq. 4), with a bandwidth `w` and support `k_eff` per
axis. In the default configuration the CUDA rasterizer applies the PSF itself and
reads the six scalars from `psf_params()`; on RADIal (`use_physical_dr_psf`,
`psf_az_sidelobes`) the kernel uses the Hamming-FFT Doppler/range lobes and the
measured azimuth response instead, and the six scalars only select the PSF mode.
`forward()` is the equivalent post-rasterization torch convolution, used by the
learned-PSF ablation (`psf_in_cuda: false`), where the bandwidths receive gradients. It
uses the same kernel as the CUDA sinc^2 x Hann path (peak 1, Doppler wrapped); the
ablation learns that parametric kernel on every axis, so `use_physical_dr_psf` does not
apply to it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_sinc_1d(
    x: torch.Tensor, w: torch.Tensor, dim: int, k: int, circular: bool
) -> torch.Tensor:
    """1D convolution with the odd kernel `w` (length k) along `dim` of a [B,D,R,A] tensor,
    zero-padded, or wrapped with `circular`."""
    ha = k // 2
    # Swap target dim to last position
    perm = [0, 1, 2, 3]
    perm[dim], perm[3] = perm[3], perm[dim]
    xp = x.permute(*perm)  # move target dim to position 3
    shape_p = list(xp.shape)
    xp_flat = xp.reshape(-1, 1, shape_p[3])  # [N, 1, L]

    if circular:
        xp_flat = F.pad(xp_flat, (ha, ha), mode="circular")
        pad = 0
    else:
        pad = ha

    y = F.conv1d(xp_flat, w.view(1, 1, k), padding=pad)
    shape_p[3] = y.shape[-1]
    y = y.reshape(shape_p).permute(*perm)
    return y


class SensorPSF(nn.Module):
    """
    3D separable Hann-windowed sinc PSF with two scalars per axis:

        kernel_i  =  sinc²(i / w)  ×  hann(i / k_eff),   peak 1 at i = 0 (as the CUDA PSF)

    where:
        w      = exp(log_w)     — bandwidth in bins (controls mainlobe shape)
        k_eff  = exp(log_k)     — effective half-width in bins (controls window extent)

    The convolution grid per axis is a fixed odd `max_k`; `k_eff` is clamped to
    max_k // 2 (and the torch kernel also clamps w and k_eff to >= 0.5).
    Larger k_eff → wider window (more support); smaller k_eff → window shrinks
    inward, approaching a delta.

    The smooth Hann window with fixed extent:
        hann(i, k_eff) = 0.5 * (1 + cos(π·i / k_eff)),  clamped ≥ 0
    This is naturally zero at |i| = k_eff and negative beyond (clamped away).

    Range and azimuth edges are zero-padded; the Doppler axis wraps.

    `log_w_*` are Parameters (learnable only with `learnable`); `log_k_*` and the
    blend strength are fixed buffers.
    """

    def __init__(
        self,
        *,
        max_kD: int,
        max_kR: int,
        max_kA: int,
        init_w_D: float,
        init_w_R: float,
        init_w_A: float,
        init_k_D: float,
        init_k_R: float,
        init_k_A: float,
        init_strength: float,  # blend: out = (1-s)*x + s*psf(x)
        learnable: bool = False,  # learn the bandwidths w_D/w_R/w_A (the supports
        # k_eff and blend strength stay fixed: support is
        # architecture, bandwidth is the PSF).
    ):
        super().__init__()
        for k, name in ((max_kD, "max_kD"), (max_kR, "max_kR"), (max_kA, "max_kA")):
            if k % 2 != 1:
                raise ValueError(f"{name} must be odd, got {k}")
        self.max_kD, self.max_kR, self.max_kA = int(max_kD), int(max_kR), int(max_kA)

        req = bool(learnable)
        self.log_w_D = nn.Parameter(torch.log(torch.tensor(float(init_w_D))), requires_grad=req)
        self.log_w_R = nn.Parameter(torch.log(torch.tensor(float(init_w_R))), requires_grad=req)
        self.log_w_A = nn.Parameter(torch.log(torch.tensor(float(init_w_A))), requires_grad=req)
        self.register_buffer("log_k_D", torch.log(torch.tensor(float(init_k_D))))
        self.register_buffer("log_k_R", torch.log(torch.tensor(float(init_k_R))))
        self.register_buffer("log_k_A", torch.log(torch.tensor(float(init_k_A))))
        # Blend strength: out = (1-s)*x + s*psf(x), stored as a logit so s is in (0, 1).
        s0 = float(init_strength)
        s0 = max(1e-6, min(1.0 - 1e-6, s0))
        self.register_buffer("logit_strength", torch.tensor(math.log(s0 / (1.0 - s0))))

    def _kernel(self, log_w: torch.Tensor, log_k: torch.Tensor, max_k: int) -> torch.Tensor:
        """Hann-windowed sinc², peak 1 at the centre tap."""
        x = torch.arange(max_k, device=log_w.device, dtype=torch.float32) - (max_k // 2)
        w = torch.exp(log_w).clamp(min=0.5)  # bandwidth
        k_eff = torch.exp(log_k).clamp(min=0.5, max=float(max_k // 2))  # half-width
        s = torch.sinc(x / w) ** 2  # mainlobe shape
        # Hard zero outside [-k_eff, k_eff], as the CUDA PSF: the Hann formula
        # 0.5*(1+cos(pi*x/k_eff)) is periodic and would rise again for |x| > k_eff.
        in_support = x.abs() <= k_eff
        cos_val = (0.5 * (1.0 + torch.cos(math.pi * x / k_eff))).clamp(min=0.0)
        hann = torch.where(in_support, cos_val, torch.zeros_like(x))
        return (s * hann).clamp(min=0.0)

    def psf_params(self) -> tuple:
        """Return (w_D, k_eff_D, w_R, k_eff_R, w_A, k_eff_A) as floats for CUDA kernel."""
        return (
            float(self.log_w_D.exp().item()),
            float(self.log_k_D.exp().clamp(max=float(self.max_kD // 2)).item()),
            float(self.log_w_R.exp().item()),
            float(self.log_k_R.exp().clamp(max=float(self.max_kR // 2)).item()),
            float(self.log_w_A.exp().item()),
            float(self.log_k_A.exp().clamp(max=float(self.max_kA // 2)).item()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,D,R,A]
        s = torch.sigmoid(self.logit_strength)  # blend in (0, 1)
        convolved = x
        if x.shape[1] > 1:  # no Doppler axis to spread along with a single bin (Boreas)
            convolved = _apply_sinc_1d(
                convolved,
                self._kernel(self.log_w_D, self.log_k_D, self.max_kD),
                dim=1,
                k=self.max_kD,
                circular=True,  # the Doppler axis is periodic (wrap)
            )
        convolved = _apply_sinc_1d(
            convolved,
            self._kernel(self.log_w_R, self.log_k_R, self.max_kR),
            dim=2,
            k=self.max_kR,
            circular=False,
        )
        convolved = _apply_sinc_1d(
            convolved,
            self._kernel(self.log_w_A, self.log_k_A, self.max_kA),
            dim=3,
            k=self.max_kA,
            circular=False,
        )
        return (1.0 - s) * x + s * convolved
