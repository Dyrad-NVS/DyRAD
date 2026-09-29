"""Python side of the radar ops: autograd wrappers around the CUDA kernels, plus the
pure-PyTorch pieces the renderer uses (the radial-velocity helper and the learned-extent
rasterizer of the learned-extent / learned-PSF ablations).

Modified from gsplat (Apache-2.0) for DyRAD.
"""
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

# Fixed values of kernel-binding arguments that no DyRAD path sets: the isotropic spread
# and Gaussian Doppler width of the kernel's non-PSF branch (never reached: DyRAD calls the
# kernel only in PSF mode), the 1/cos(az) azimuth-PSF broadening switch and the Doppler
# tent-delta switch.
_SPREAD_FACTOR = 3.0
_SIGMA_D_BINS = 0.0
_PSF_AZ_COS_BROADENING = False
_DOP_TENT_DELTA = False

# Window volume (elements) per chunk of the PyTorch learned-extent rasterizer.
_PT_RASTER_CHUNK_ELEMS = 48 * 1024 * 1024


def _make_lazy_cuda_func(name: str) -> Callable:
    """A callable that loads the CUDA extension on first use and calls `name` in it."""
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        if _C is None:
            raise RuntimeError(
                "gsplat CUDA extension not built; see the main README (Installation)."
            )

        func = getattr(_C, name, None)
        if func is None:
            raise AttributeError(
                f"Function '{name}' not found in CUDA extension. "
                f"Available functions: {dir(_C)}"
            )

        return func(*args, **kwargs)

    return call_cuda


def projection_radar_3dgs_fused_fwd(
    means: Tensor,           # [B, N, 3]
    quats: Tensor,           # [B, N, 4]
    scales: Tensor,          # [B, N, 3]
    viewmats: Tensor,        # [B, C, 4, 4]
    near_range: float,
    far_range: float,
    radius_clip: float,
    range_bins: Tensor,      # [num_range_bins]
    az_bins: Tensor,         # [num_az_bins]
    el_bins: Tensor,          # [num_el_bins]
    num_range_bins: int,
    num_az_bins: int,
    num_el_bins: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project 3D Gaussians to spherical (radar) coordinates.

    Returns:
        Tuple of (ranges, azimuths, elevations, range_bin_indices, 
                 az_bin_indices, el_bin_indices, gaussian_weights)
    """
    # Call with explicit argument order to match the C++ binding
    return _make_lazy_cuda_func("projection_radar_3dgs_fused_fwd")(
        means,
        quats,
        scales,
        viewmats,
        near_range,
        far_range,
        radius_clip,
        range_bins,
        az_bins,
        el_bins,
        num_range_bins,
        num_az_bins,
        num_el_bins,
    )


class _ProjectionRadar3DGSFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        viewmats: Tensor,
        near_range: float,
        far_range: float,
        radius_clip: float,
        range_bins: Tensor,
        az_bins: Tensor,
        el_bins: Tensor,
        num_range_bins: int,
        num_az_bins: int,
        num_el_bins: int,
    ):
        outs = projection_radar_3dgs_fused_fwd(
            means,
            quats,
            scales,
            viewmats,
            float(near_range),
            float(far_range),
            float(radius_clip),
            range_bins,
            az_bins,
            el_bins,
            int(num_range_bins),
            int(num_az_bins),
            int(num_el_bins),
        )

        (
            ranges,
            azimuths,
            elevations,
            range_bin_indices,
            az_bin_indices,
            el_bin_indices,
            gaussian_weights,
        ) = outs

        ctx.save_for_backward(
            means,
            quats,
            scales,
            viewmats,
            ranges,
            azimuths,
            elevations,
            gaussian_weights,
            range_bin_indices,
        )
        ctx.near_range = float(near_range)
        ctx.far_range = float(far_range)
        return outs

    @staticmethod
    def backward(
        ctx,
        v_ranges: Optional[Tensor],
        v_azimuths: Optional[Tensor],
        v_elevations: Optional[Tensor],
        v_range_bin_indices: Optional[Tensor],
        v_az_bin_indices: Optional[Tensor],
        v_el_bin_indices: Optional[Tensor],
        v_gaussian_weights: Optional[Tensor],
    ):
        (
            means,
            quats,
            scales,
            viewmats,
            ranges,
            azimuths,
            elevations,
            gaussian_weights,
            range_bin_indices,
        ) = ctx.saved_tensors

        def _zero_if_none(g: Optional[Tensor], like: Tensor) -> Tensor:
            if g is None:
                return torch.zeros_like(like)
            return g

        v_ranges = _zero_if_none(v_ranges, ranges).contiguous()
        v_azimuths = _zero_if_none(v_azimuths, azimuths).contiguous()
        v_elevations = _zero_if_none(v_elevations, elevations).contiguous()
        v_gaussian_weights = _zero_if_none(v_gaussian_weights, gaussian_weights).contiguous()

        v_means, v_quats, v_scales = _make_lazy_cuda_func("projection_radar_3dgs_fused_bwd")(
            means,
            quats,
            scales,
            viewmats,
            ctx.near_range,
            ctx.far_range,
            ranges,
            azimuths,
            elevations,
            gaussian_weights,
            range_bin_indices,
            v_ranges,
            v_azimuths,
            v_elevations,
            v_gaussian_weights,
        )

        return (
            v_means,
            v_quats,
            v_scales,
            None,  # viewmats
            None,  # near_range
            None,  # far_range
            None,  # radius_clip
            None,  # range_bins
            None,  # az_bins
            None,  # el_bins
            None,  # num_range_bins
            None,  # num_az_bins
            None,  # num_el_bins
        )


def projection_radar_3dgs_fused(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    viewmats: Tensor,
    near_range: float,
    far_range: float,
    radius_clip: float,
    range_bins: Tensor,
    az_bins: Tensor,
    el_bins: Tensor,
    num_range_bins: int,
    num_az_bins: int,
    num_el_bins: int,
):
    """Differentiable radar projection: the CUDA kernels (forward and backward), or the
    PyTorch reference `_projection_radar_3dgs_fused_pytorch` when the extension is not
    built. The fallback is silent because every render that uses the projection without the
    extension either raises in `rasterize_radar_rae` (sensor-PSF path) or runs entirely in
    PyTorch (learned-extent path)."""
    from ._backend import _C
    if _C is None:
        return _projection_radar_3dgs_fused_pytorch(
            means, quats, scales, viewmats,
            near_range, far_range, radius_clip,
            range_bins, az_bins, el_bins,
            num_range_bins, num_az_bins, num_el_bins,
        )
    return _ProjectionRadar3DGSFused.apply(
        means,
        quats,
        scales,
        viewmats,
        near_range,
        far_range,
        radius_clip,
        range_bins,
        az_bins,
        el_bins,
        num_range_bins,
        num_az_bins,
        num_el_bins,
    )


def rasterize_radar_rae_fwd(
    powers: Tensor,              # [B, C, N]
    range_bin_indices: Tensor,   # [B, C, N]
    az_bin_indices: Tensor,      # [B, C, N]
    el_bin_indices: Tensor,      # [B, C, N]
    doppler_bin_indices: Tensor, # [B, C, N]
    gaussian_weights: Tensor,    # [B, C, N]
    ranges: Tensor,              # [B, C, N]
    azimuths: Tensor,            # [B, C, N]
    range_bin_centers: Tensor,   # [num_range_bins]
    az_bin_centers: Tensor,      # [num_az_bins]
    num_doppler_bins: int,
    num_range_bins: int,
    num_az_bins: int,
    v_r_cont: Optional[Tensor],
    doppler_bin_spacing: float,
    psf_w_D: float,
    psf_k_eff_D: float,
    psf_w_R: float,
    psf_k_eff_R: float,
    psf_w_A: float,
    psf_k_eff_A: float,
    use_physical_dr_psf: bool,
    az_tent_delta: bool,
) -> Tensor:
    return _make_lazy_cuda_func("rasterize_radar_rae_fwd")(
        powers,
        range_bin_indices,
        az_bin_indices,
        el_bin_indices,
        doppler_bin_indices,
        gaussian_weights,
        ranges,
        azimuths,
        range_bin_centers,
        az_bin_centers,
        num_doppler_bins,
        num_range_bins,
        num_az_bins,
        _SPREAD_FACTOR,
        v_r_cont,
        _SIGMA_D_BINS,
        doppler_bin_spacing,
        psf_w_D,
        psf_k_eff_D,
        psf_w_R,
        psf_k_eff_R,
        psf_w_A,
        psf_k_eff_A,
        use_physical_dr_psf,
        _PSF_AZ_COS_BROADENING,
        az_tent_delta,
        _DOP_TENT_DELTA,
    )


class _RasterizeRadarRAE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        powers: Tensor,              # [B,C,N]
        range_bin_indices: Tensor,   # [B,C,N] int32
        az_bin_indices: Tensor,      # [B,C,N] int32
        el_bin_indices: Tensor,      # [B,C,N] int32 (forward only)
        doppler_bin_indices: Tensor, # [B,C,N] int32
        gaussian_sigmas_m: Tensor,   # [B,C,N] float (sigma in meters)
        ranges: Tensor,              # [B,C,N]
        azimuths: Tensor,            # [B,C,N]
        range_bin_centers: Tensor,   # [R]
        az_bin_centers: Tensor,      # [A]
        num_doppler_bins: int,
        num_range_bins: int,
        num_az_bins: int,
        v_r_cont: Optional[Tensor],  # [B,C,N] or None — differentiable Doppler via PSF
        doppler_bin_spacing: float,
        psf_w_D: float,
        psf_k_eff_D: float,
        psf_w_R: float,
        psf_k_eff_R: float,
        psf_w_A: float,
        psf_k_eff_A: float,
        use_physical_dr_psf: bool,
        az_tent_delta: bool,
    ) -> Tensor:
        _v_r_cont_save = v_r_cont if v_r_cont is not None else torch.empty(0, device=powers.device, dtype=powers.dtype)
        ctx.save_for_backward(
            powers,
            range_bin_indices,
            az_bin_indices,
            doppler_bin_indices,
            gaussian_sigmas_m,
            ranges,
            azimuths,
            range_bin_centers,
            az_bin_centers,
            _v_r_cont_save,
        )
        ctx.num_doppler_bins = int(num_doppler_bins)
        ctx.num_range_bins = int(num_range_bins)
        ctx.num_az_bins = int(num_az_bins)
        ctx.doppler_bin_spacing = float(doppler_bin_spacing)
        ctx.has_v_r_cont = (v_r_cont is not None)
        ctx.psf_w_D = float(psf_w_D)
        ctx.psf_k_eff_D = float(psf_k_eff_D)
        ctx.psf_w_R = float(psf_w_R)
        ctx.psf_k_eff_R = float(psf_k_eff_R)
        ctx.psf_w_A = float(psf_w_A)
        ctx.psf_k_eff_A = float(psf_k_eff_A)
        ctx.use_physical_dr_psf = bool(use_physical_dr_psf)
        ctx.az_tent_delta = bool(az_tent_delta)

        return rasterize_radar_rae_fwd(
            powers,
            range_bin_indices,
            az_bin_indices,
            el_bin_indices,
            doppler_bin_indices,
            gaussian_sigmas_m,
            ranges,
            azimuths,
            range_bin_centers,
            az_bin_centers,
            num_doppler_bins,
            num_range_bins,
            num_az_bins,
            v_r_cont,
            doppler_bin_spacing,
            psf_w_D,
            psf_k_eff_D,
            psf_w_R,
            psf_k_eff_R,
            psf_w_A,
            psf_k_eff_A,
            use_physical_dr_psf,
            az_tent_delta,
        )

    @staticmethod
    def backward(ctx, v_rad_tensor: Tensor):
        (
            powers,
            range_bin_indices,
            az_bin_indices,
            doppler_bin_indices,
            gaussian_sigmas_m,
            ranges,
            azimuths,
            range_bin_centers,
            az_bin_centers,
            v_r_cont_saved,
        ) = ctx.saved_tensors

        v_r_cont_bwd = v_r_cont_saved if ctx.has_v_r_cont else None

        v_powers, v_ranges, v_azimuths, v_sigmas, v_v_r_cont = _make_lazy_cuda_func("rasterize_radar_rae_bwd")(
            powers,
            range_bin_indices,
            az_bin_indices,
            doppler_bin_indices,
            gaussian_sigmas_m,
            ranges,
            azimuths,
            range_bin_centers,
            az_bin_centers,
            ctx.num_doppler_bins,
            ctx.num_range_bins,
            ctx.num_az_bins,
            _SPREAD_FACTOR,
            v_rad_tensor.contiguous(),
            v_r_cont_bwd,
            _SIGMA_D_BINS,
            ctx.doppler_bin_spacing,
            ctx.psf_w_D,
            ctx.psf_k_eff_D,
            ctx.psf_w_R,
            ctx.psf_k_eff_R,
            ctx.psf_w_A,
            ctx.psf_k_eff_A,
            ctx.use_physical_dr_psf,
            _PSF_AZ_COS_BROADENING,
            ctx.az_tent_delta,
            _DOP_TENT_DELTA,
        )

        return (
            v_powers,
            None,  # range_bin_indices
            None,  # az_bin_indices
            None,  # el_bin_indices
            None,  # doppler_bin_indices
            v_sigmas,
            v_ranges,
            v_azimuths,
            None,  # range_bin_centers
            None,  # az_bin_centers
            None,  # num_doppler_bins
            None,  # num_range_bins
            None,  # num_az_bins
            v_v_r_cont if ctx.has_v_r_cont else None,  # v_r_cont gradient
            None,  # doppler_bin_spacing
            None,  # psf_w_D
            None,  # psf_k_eff_D
            None,  # psf_w_R
            None,  # psf_k_eff_R
            None,  # psf_w_A
            None,  # psf_k_eff_A
            None,  # use_physical_dr_psf
            None,  # az_tent_delta
        )


def rasterize_radar_rae(
    powers: Tensor,
    range_bin_indices: Tensor,
    az_bin_indices: Tensor,
    el_bin_indices: Tensor,
    doppler_bin_indices: Tensor,
    gaussian_sigmas_m: Tensor,
    ranges: Tensor,
    azimuths: Tensor,
    range_bin_centers: Tensor,
    az_bin_centers: Tensor,
    num_doppler_bins: int,
    num_range_bins: int,
    num_az_bins: int,
    v_r_cont: Optional[Tensor] = None,      # [B, C, N] continuous radial velocity m/s
    doppler_bin_spacing: float = 0.06,       # m/s per Doppler bin
    sigma_r_m: Optional[Tensor] = None,     # [B, C, N] projected σ_r (learned-extent path)
    sigma_az_rad: Optional[Tensor] = None,  # [B, C, N] projected σ_az (learned-extent path)
    # 3D PSF params for CUDA path (psf_k_eff_D > 0 activates PSF mode)
    psf_w_D: float = 0.0,
    psf_k_eff_D: float = 0.0,
    psf_w_R: float = 0.0,
    psf_k_eff_R: float = 0.0,
    psf_w_A: float = 0.0,
    psf_k_eff_A: float = 0.0,
    # True = analytic Hamming-FFT PSF for Doppler and range; False = sinc-Hann for D & R
    use_physical_dr_psf: bool = False,
    # True = azimuth kernel is a unit-sum interpolating tent over the 2 nearest
    # bins (delta accumulation). The caller applies the measured beamformer
    # response over azimuth afterwards — exact full-response azimuth PSF
    # (sidelobes + pedestal) via a matmul instead of a full-axis support loop.
    az_tent_delta: bool = False,
) -> Tensor:
    """Differentiable radar rasterization into a [B, C, D, R, A] tensor.

    PSF mode (psf_k_eff_D > 0): the CUDA kernel applies the separable 3D sensor PSF;
    gradients flow from the RAD tensor through the PSF weights to v_r_cont, ranges and
    azimuths. It has no PyTorch counterpart, so it raises without the CUDA extension.
    Learned-extent mode (psf_k_eff_D == 0, with sigma_r_m and sigma_az_rad): the PyTorch
    rasterizer spreads each primitive by its projected covariance (the learned-extent
    and learned-PSF ablations); `gaussian_sigmas_m` is not used there.
    """
    if psf_k_eff_D > 0.0:
        from ._backend import _C

        if _C is None:
            raise RuntimeError(
                "rasterize_radar_rae: the sensor-PSF path (psf_k_eff_D > 0) needs the gsplat "
                "CUDA extension, which is not built; see the main README (Installation)."
            )
        return _RasterizeRadarRAE.apply(
            powers,
            range_bin_indices,
            az_bin_indices,
            el_bin_indices,
            doppler_bin_indices,
            gaussian_sigmas_m,
            ranges,
            azimuths,
            range_bin_centers,
            az_bin_centers,
            num_doppler_bins,
            num_range_bins,
            num_az_bins,
            v_r_cont,
            doppler_bin_spacing,
            psf_w_D,
            psf_k_eff_D,
            psf_w_R,
            psf_k_eff_R,
            psf_w_A,
            psf_k_eff_A,
            use_physical_dr_psf,
            az_tent_delta,
        )
    if sigma_r_m is None or sigma_az_rad is None:
        raise ValueError(
            "rasterize_radar_rae: without the sensor PSF (psf_k_eff_D == 0) both sigma_r_m "
            "and sigma_az_rad are required (learned-extent rasterizer)"
        )
    return _rasterize_radar_rae_fwd_pytorch(
        powers,
        range_bin_indices,
        az_bin_indices,
        doppler_bin_indices,
        ranges,
        azimuths,
        range_bin_centers,
        az_bin_centers,
        num_doppler_bins,
        num_range_bins,
        num_az_bins,
        sigma_r_m,
        sigma_az_rad,
        v_r_cont=v_r_cont,
        doppler_bin_spacing=doppler_bin_spacing,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pure-PyTorch ops: the projection reference used without the extension, the
# Doppler helpers, and the learned-extent rasterizer.
# ─────────────────────────────────────────────────────────────────────────────

def _projection_radar_3dgs_fused_pytorch(
    means: Tensor,      # [B, N, 3]
    quats: Tensor,      # [B, N, 4]
    scales: Tensor,     # [B, N, 3]  log-space
    viewmats: Tensor,   # [B, C, 4, 4]  w2c
    near_range: float,
    far_range: float,
    radius_clip: float,
    range_bins: Tensor,   # [R]
    az_bins: Tensor,      # [A]
    el_bins: Tensor,      # [E]
    num_r: int,
    num_az: int,
    num_el: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    PyTorch version of projection_radar_3dgs_fused (used when the extension is not built).

    Matches the CUDA kernel's conventions exactly:
      azimuth   = atan2(y, x)          (non-negated)
      elevation = -asin(z / r)         (negated)
      sigma_m   = sqrt(sum(exp(s)^2) / 3)  (iso-approx of projected covar trace)
    """
    B, N, _ = means.shape
    C = viewmats.shape[1]

    R = viewmats[:, :, :3, :3]   # [B, C, 3, 3]
    t = viewmats[:, :, :3, 3]    # [B, C, 3]

    # Transform means to radar frame: p[b,c,n,:] = R[b,c] @ mean[b,n] + t[b,c]
    means_bc = means[:, None, :, :].expand(B, C, N, 3)  # [B, C, N, 3]
    p = torch.einsum("bcij,bcnj->bcni", R, means_bc) + t[:, :, None, :]  # [B, C, N, 3]

    r   = p.norm(dim=-1).clamp(min=1e-8)                                  # [B, C, N]
    az  = torch.atan2(p[..., 1], p[..., 0])                               # [B, C, N]
    el  = -torch.asin((p[..., 2] / r).clamp(-1 + 1e-7, 1 - 1e-7))       # [B, C, N]

    az_max = az_bins.abs().max()
    el_max = el_bins.abs().max()
    valid  = (r >= near_range) & (r <= far_range) & (az.abs() <= az_max) & (el.abs() <= el_max)

    # Nearest bin indices (non-differentiable; integers used only for placement)
    with torch.no_grad():
        r_idx  = torch.argmin((r[..., None]  - range_bins).abs(), dim=-1).to(torch.int32)
        az_idx = torch.argmin((az[..., None] - az_bins).abs(),    dim=-1).to(torch.int32)
        el_idx = torch.argmin((el[..., None] - el_bins).abs(),    dim=-1).to(torch.int32)
        neg1   = torch.full_like(r_idx, -1)
        r_idx  = torch.where(valid, r_idx,  neg1)
        az_idx = torch.where(valid, az_idx, neg1)
        el_idx = torch.where(valid, el_idx, neg1)

    # sigma_m ≈ sqrt(trace(R_wc @ S^2 @ R_wc^T) / 3) = ||exp(scales)|| / sqrt(3)
    sigma_m = scales.exp().norm(dim=-1) / (3.0 ** 0.5)  # [B, N]
    sigma_m = sigma_m[:, None, :].expand(B, C, N).clamp(min=1e-3)  # [B, C, N]

    return r, az, el, r_idx, az_idx, el_idx, sigma_m


def compute_doppler_radial_velocity(
    means: Tensor,                        # [B, N, 3]
    radar_poses: Tensor,                  # [B, C, 4, 4]  w2c at t
    prev_radar_poses: Tensor,             # [B, C, 4, 4]  w2c at t-1
    dt: float,
    velocities: Optional[Tensor] = None,  # [B, N, 3] per-Gaussian world velocity
    doppler_sign: float = 1.0,            # -1 flips approaching/receding (RADIal convention)
) -> Tensor:                              # [B, C, N] float — differentiable radial velocity m/s
    """Continuous (differentiable) radial velocity of each reflector, in m/s.

    The ego-motion term is the change of the reflector's position between the two radar
    frames, (p_t - p_prev) / dt, projected on the line of sight at t. If `velocities`
    (per-reflector world-space velocity [B, N, 3]) is given, the radial component of each
    reflector's own motion is subtracted from it; gradients flow back to `velocities`.
    `doppler_sign=-1` negates the result (RADIal's Doppler FFT convention).
    """
    B, N, _ = means.shape
    C = radar_poses.shape[1]

    R_t = radar_poses[:, :, :3, :3]       # [B, C, 3, 3]
    t_t = radar_poses[:, :, :3, 3]        # [B, C, 3]
    R_p = prev_radar_poses[:, :, :3, :3]  # [B, C, 3, 3]
    t_p = prev_radar_poses[:, :, :3, 3]   # [B, C, 3]

    means_bc = means[:, None, :, :].expand(B, C, N, 3)  # [B, C, N, 3]

    p_t = torch.einsum("bcij,bcnj->bcni", R_t, means_bc) + t_t[:, :, None, :]  # [B,C,N,3]
    p_p = torch.einsum("bcij,bcnj->bcni", R_p, means_bc) + t_p[:, :, None, :]  # [B,C,N,3]

    r_t = p_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit_r = p_t / r_t                              # [B,C,N,3]
    v_r = ((p_t - p_p) / dt * unit_r).sum(dim=-1)  # [B,C,N] — ego contribution, m/s

    if velocities is not None:
        # velocities: [B, N, 3] world-space physical velocity; project into radar frame.
        # Sign: the ego term above is approaching-positive (with the trainer's
        # forward-shifted prev pose). A target moving along +r_hat (away from the
        # sensor) reduces the closing rate, so its radial component must be
        # subtracted: v_r = v_ego_radial - dot(v_target, r_hat).
        v_g_bc = velocities[:, None, :, :].expand(B, C, N, 3)  # [B,C,N,3]
        v_g_radar = torch.einsum("bcij,bcnj->bcni", R_t, v_g_bc)  # [B,C,N,3]
        v_r = v_r - (v_g_radar * unit_r).sum(dim=-1)  # subtract target's own radial velocity

    return doppler_sign * v_r  # [B, C, N], differentiable w.r.t. `velocities`


def _rasterize_radar_rae_fwd_pytorch(
    powers: Tensor,              # [B, C, N]
    r_idx: Tensor,               # [B, C, N] int32
    az_idx: Tensor,              # [B, C, N] int32
    d_idx: Tensor,               # [B, C, N] int32
    ranges_bc: Tensor,           # [B, C, N]  continuous
    azimuths_bc: Tensor,         # [B, C, N]  continuous
    range_bins: Tensor,          # [R]
    az_bins: Tensor,             # [A]
    num_d: int,
    num_r: int,
    num_az: int,
    sigma_r_m: Tensor,           # [B, C, N] projected σ_r in metres
    sigma_az_rad: Tensor,        # [B, C, N] projected σ_az in radians
    v_r_cont: Optional[Tensor] = None,  # [B, C, N] continuous radial velocity m/s (differentiable)
    doppler_bin_spacing: float = 0.06,  # m/s per Doppler bin
) -> Tensor:  # [B, C, D, R, A]
    """
    Learned-extent rasterizer (PyTorch): Gaussian range/azimuth spreading of each primitive.

    The projected Gaussian sigmas (from the Jacobian of the spherical projection) set the
    spread directly:
      sigma_r_bins  = clamp(sigma_r_m / dr, 0.1, 10)
      sigma_az_bins = clamp(sigma_az_rad / daz, 0.1, 10)

    When `v_r_cont` is provided (continuous radial velocity in m/s), bilinear
    Doppler placement is used: power is split between floor and ceil bins based
    on the fractional sub-bin position. Gradient flows through the fractional
    weight back to v_r_cont and hence to per-Gaussian velocities.

    Differentiable w.r.t. `powers`, `ranges_bc`, `azimuths_bc`, `sigma_r_m`,
    `sigma_az_rad`, and (when provided) `v_r_cont`.
    """
    B, C, N = powers.shape
    device  = powers.device

    dr  = (range_bins[1] - range_bins[0]).abs()   # scalar, metres/bin
    daz = (az_bins[1]    - az_bins[0]).abs()       # scalar, rad/bin

    out = torch.zeros(B, C, num_d * num_r * num_az, device=device, dtype=powers.dtype)

    for b in range(B):
        for c in range(C):
            ri  = r_idx[b, c]         # [N] int
            ai  = az_idx[b, c]        # [N] int
            di  = d_idx[b, c]         # [N] int
            p   = powers[b, c]        # [N]  (requires_grad possible)
            rng = ranges_bc[b, c]     # [N]  (requires_grad possible)
            azc = azimuths_bc[b, c]   # [N]  (requires_grad possible)

            valid = (ri >= 0) & (ri < num_r)
            vi = valid.nonzero(as_tuple=False).squeeze(1)
            if vi.numel() == 0:
                continue

            p_v   = p[vi]
            rng_v = rng[vi]
            az_v  = azc[vi]
            ri_v  = ri[vi].long()
            ai_v  = ai[vi].long()
            di_v  = di[vi].long().clamp(0, num_d - 1)
            V     = p_v.shape[0]

            # Spread in bins from the projected covariance; the lower clamp allows sub-bin spreads.
            sr = (sigma_r_m[b, c][vi]   / dr ).clamp(0.1, 10.0)  # [V]
            sa = (sigma_az_rad[b, c][vi] / daz).clamp(0.1, 10.0)  # [V]

            # Window half-size covers the full 3σ of the projected extent; the cap of
            # 32 equals 3σ at the σ-clamp ceiling of 10, so large Gaussians render as
            # smooth ellipses rather than truncated squares. The CUDA PSF path never
            # enters here.
            half_r = min(max(int(sr.detach().max().item() * 3) + 1, 1), 32)
            half_a = min(max(int(sa.detach().max().item() * 3) + 1, 1), 32)

            dr_offs = torch.arange(-half_r, half_r + 1, device=device, dtype=torch.long)  # [Hr]
            da_offs = torch.arange(-half_a, half_a + 1, device=device, dtype=torch.long)  # [Ha]
            Hr, Ha = dr_offs.shape[0], da_offs.shape[0]

            # ── Chunked over Gaussians ────────────────────────────────────────────────────
            # Every intermediate below is [V, Hr, Ha]; for ~100k Gaussians and a 65x65 window
            # this exceeds GPU memory. Process the valid Gaussians in chunks whose window
            # volume stays under _PT_RASTER_CHUNK_ELEMS, freeing each chunk's
            # intermediates with torch.utils.checkpoint (recomputed in backward). The result
            # is the same sum; only the float32 accumulation order differs. A single chunk
            # (small V) takes the un-checkpointed path.
            vrc_all = v_r_cont[b, c][vi] if v_r_cont is not None else None
            sr_full, sa_full = sr, sa
            v_chunk = max(1, _PT_RASTER_CHUNK_ELEMS // max(Hr * Ha, 1))
            n_flat = num_d * num_r * num_az

            def _chunk_contrib(p_c, rng_c, az_c, sr_c, sa_c, ri_c, ai_c, di_c, vrc_c):
                Vc = p_c.shape[0]
                r_grid = (ri_c[:, None] + dr_offs[None, :]).clamp(0, num_r  - 1)  # [Vc, Hr]
                a_grid = (ai_c[:, None] + da_offs[None, :]).clamp(0, num_az - 1)  # [Vc, Ha]
                r_centres  = range_bins[r_grid]
                az_centres = az_bins[a_grid]
                r_norm  = (r_centres  - rng_c[:, None]) / (dr  * sr_c[:, None] + 1e-8)
                az_norm = (az_centres - az_c[:, None])  / (daz * sa_c[:, None] + 1e-8)
                w_2d = torch.exp(-0.5 * r_norm ** 2)[:, :, None] * torch.exp(-0.5 * az_norm ** 2)[:, None, :]
                vals = p_c[:, None, None] * w_2d                        # [Vc, Hr, Ha]
                acc = torch.zeros(n_flat, device=device, dtype=powers.dtype)
                if vrc_c is not None:
                    # Bilinear Doppler placement: split power between floor and ceil bins.
                    # Gradient flows: loss -> out[floor/ceil] -> frac -> d_cont -> v_r -> velocities.
                    half_D = float(num_d) * 0.5  # v=0 at bin D/2, matches the CUDA kernel
                    d_cont = (vrc_c / float(doppler_bin_spacing) + half_D) % num_d   # differentiable a.e.
                    d_floor = d_cont.detach().floor().long().clamp(0, num_d - 1)
                    frac = (d_cont - d_floor.to(d_cont.dtype)).clamp(0.0, 1.0)
                    d_ceil = (d_floor + 1) % num_d
                    for d_bin, w_d in ((d_floor, 1.0 - frac), (d_ceil, frac)):
                        d_exp = d_bin[:, None, None].expand(Vc, Hr, Ha)
                        r_exp = r_grid[:, :, None].expand(Vc, Hr, Ha)
                        a_exp = a_grid[:, None, :].expand(Vc, Hr, Ha)
                        flat = (d_exp * num_r * num_az + r_exp * num_az + a_exp).reshape(-1)
                        acc = acc.scatter_add(0, flat, (vals * w_d[:, None, None]).reshape(-1))
                else:
                    d_exp = di_c[:, None, None].expand(Vc, Hr, Ha)
                    r_exp = r_grid[:, :, None].expand(Vc, Hr, Ha)
                    a_exp = a_grid[:, None, :].expand(Vc, Hr, Ha)
                    flat = (d_exp * num_r * num_az + r_exp * num_az + a_exp).reshape(-1)
                    acc = acc.scatter_add(0, flat, vals.reshape(-1))
                return acc

            if V <= v_chunk:
                out[b, c] = out[b, c] + _chunk_contrib(p_v, rng_v, az_v, sr_full, sa_full, ri_v, ai_v, di_v, vrc_all)
            else:
                from torch.utils.checkpoint import checkpoint as _ckpt
                acc_total = out[b, c]
                for s0 in range(0, V, v_chunk):
                    s1 = min(V, s0 + v_chunk)
                    args = (p_v[s0:s1], rng_v[s0:s1], az_v[s0:s1], sr_full[s0:s1], sa_full[s0:s1],
                            ri_v[s0:s1], ai_v[s0:s1], di_v[s0:s1],
                            vrc_all[s0:s1] if vrc_all is not None else None)
                    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in args):
                        acc_total = acc_total + _ckpt(_chunk_contrib, *args, use_reentrant=False)
                    else:
                        acc_total = acc_total + _chunk_contrib(*args)
                out[b, c] = acc_total

    return out.view(B, C, num_d, num_r, num_az)


__all__ = [
    "projection_radar_3dgs_fused",
    "compute_doppler_radial_velocity",
    "rasterize_radar_rae",
]
