"""Rendering (paper Sec. 3.2): reflectors are placed on the range/azimuth/Doppler grid from
the sensor pose, the object tracks and the ego velocity, and rasterized through the sensor PSF
into an incoherent power tensor that is then mapped to the sensor's measurement domain.
`radar_rasterization` is the per-frame forward model; `RenderMixin` adds the Runner's
time axis, ego-Doppler handling and the interpolation-consistency targets.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import (
    compute_doppler_radial_velocity,
    projection_radar_3dgs_fused,
    rasterize_radar_rae,
)
from torch import Tensor

from dyrad.poses import c2w_to_w2c
from dyrad.psf import SensorPSF

#: Gaussian blur (bins) of the interpolation-consistency pseudo-GT and of the render it
#: is compared with (L_int, paper Eq. 3).
INTERP_BLUR_SIGMA_BINS = 1.0


# -----------------------------
# SH / reflectivity
# -----------------------------
def _evaluate_sh_view_dependent(
    dirs: Tensor,  # [B, N, 3]
    sh_coeffs: Tensor,  # [N, K, 1]  (single intensity channel)
    sh_degree: int = 3,
) -> Tensor:
    """
    View-dependent reflectivity from SH coefficients.

    Radar needs only one intensity channel; sh_coeffs is stored as [N, K, 1].
    Uses _eval_sh_bases_fast directly so autograd differentiates through the
    coefficients without relying on any CUDA SH kernel (the CUDA path has no
    registered backward).

    Returns:
        reflectivity: [B, N] = exp(SH), range (0, ∞), for high-dynamic-range radar.
    """
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast

    B, N, _ = dirs.shape
    K = sh_coeffs.shape[1]
    num_bases = (sh_degree + 1) ** 2

    dirs_flat = F.normalize(dirs.reshape(-1, 3), p=2, dim=-1)  # [B*N, 3]

    # SH basis vectors (zeros for inactive higher degrees)
    bases = dirs_flat.new_zeros(B * N, K)  # [B*N, K]
    bases[:, :num_bases] = _eval_sh_bases_fast(num_bases, dirs_flat)

    # Channel 0 only — [N, K] expanded to [B*N, K]
    sh0 = sh_coeffs[..., 0]  # [N, K]
    sh0_b = sh0.unsqueeze(0).expand(B, -1, -1).reshape(B * N, K)  # [B*N, K]

    reflectivity = (bases * sh0_b).sum(dim=-1).reshape(B, N)  # [B, N]

    # Overflow guard only (exp(40) ~ 2e17; with sh_dc pinned to 0 the DC term is 1).
    return torch.exp(reflectivity.clamp(max=40.0))


def _projected_extent(
    quats_b: Tensor, scales_b: Tensor, R: Tensor, means_radar: Tensor
) -> Tuple[Tensor, Tensor]:
    """Per-reflector range and azimuth extent (sigma_r [m], sigma_az [rad]), each [B, 1, N].

    The torch rasterizer's Gaussian range/azimuth spreading (the learned-PSF and
    learned-extent ablations); the CUDA-PSF path does not use it. `R` is the w2c
    rotation [B, 3, 3] and `means_radar` the reflector positions in the radar frame.
    """
    # -----------------------------------------------------------------------
    # Projected Gaussian covariance (gsplat-style, spherical projection).
    #
    # Standard gsplat: Σ_2D = J @ Σ_3D @ J^T  (perspective Jacobian).
    # Radar equivalent: project to (range, azimuth) using spherical Jacobian.
    #
    # J_r   = p_r / |p_r|            — derivative of range w.r.t. world pos
    # J_az  = [-py/ρ², px/ρ², 0]    — derivative of azimuth w.r.t. world pos
    # where ρ = sqrt(px² + py²) in radar frame.
    #
    # σ²_r  = J_r^T @ Σ_r @ J_r     (metres²)
    # σ²_az = J_az^T @ Σ_r @ J_az   (radians²)
    # -----------------------------------------------------------------------
    B, n_vis = means_radar.shape[:2]
    # Convert quaternions [B, N_vis, 4] wxyz → rotation matrices [B, N_vis, 3, 3]
    q = quats_b  # [B, N_vis, 4]
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R_g = torch.stack(
        [
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy - qw * qz),
            2 * (qx * qz + qw * qy),
            2 * (qx * qy + qw * qz),
            1 - 2 * (qx * qx + qz * qz),
            2 * (qy * qz - qw * qx),
            2 * (qx * qz - qw * qy),
            2 * (qy * qz + qw * qx),
            1 - 2 * (qx * qx + qy * qy),
        ],
        dim=-1,
    ).reshape(B, n_vis, 3, 3)  # [B, N_vis, 3, 3]

    # Σ_3D = R_g @ diag(s²) @ R_g^T   (world frame)
    s = scales_b  # [B, N_vis, 3]
    Rs = R_g * (s**2).unsqueeze(-2)  # [B, N_vis, 3, 3]: R_g * s²
    Sigma_3D = Rs @ R_g.transpose(-1, -2)  # [B, N_vis, 3, 3]

    # Σ_r = R_w2c @ Σ_3D @ R_w2c^T   (radar frame)
    R_exp = R[:, None, :, :]  # [B, 1, 3, 3]
    Sigma_r = R_exp @ Sigma_3D @ R_exp.transpose(-1, -2)  # [B, N_vis, 3, 3]

    # Jacobian rows in radar frame
    p = means_radar  # [B, N_vis, 3]
    rng_cont = p.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, N_vis, 1]
    J_r = p / rng_cont  # [B, N_vis, 3] — unit radial vector
    px, py = p[..., 0], p[..., 1]  # [B, N_vis]
    rho = (px**2 + py**2).sqrt().clamp(min=1e-6)  # [B, N_vis]
    J_az = torch.stack([-py / rho**2, px / rho**2, torch.zeros_like(rho)], dim=-1)  # [B, N_vis, 3]

    # σ²_r  = J_r^T @ Σ_r @ J_r
    SJ_r = (Sigma_r @ J_r.unsqueeze(-1)).squeeze(-1)  # [B, N_vis, 3]
    sigma2_r = (J_r * SJ_r).sum(-1).clamp(min=1e-8)  # [B, N_vis] metres²

    # σ²_az = J_az^T @ Σ_r @ J_az
    SJ_az = (Sigma_r @ J_az.unsqueeze(-1)).squeeze(-1)  # [B, N_vis, 3]
    sigma2_az = (J_az * SJ_az).sum(-1).clamp(min=1e-8)  # [B, N_vis] radians²

    sigma_r_proj = sigma2_r.sqrt()  # [B, N_vis] metres
    sigma_az_proj = sigma2_az.sqrt()  # [B, N_vis] radians

    # Reshape for the rasterizer: [B, 1, N_vis].
    return sigma_r_proj[:, None, :], sigma_az_proj[:, None, :]


# -----------------------------
# Per-frame forward model
# -----------------------------
def radar_rasterization(
    means: Tensor,  # [N,3]
    quats: Tensor,  # [N,4]
    scales: Tensor,  # [N,3]
    opacities: Tensor,  # [N]
    sh_coeffs: Tensor,  # [N,K,1]
    c2w_t: Tensor,  # [B,4,4]
    c2w_prev: Tensor,  # [B,4,4]
    range_bins: Tensor,  # [R]
    az_bins: Tensor,  # [A]
    el_bins: Tensor,  # [E]
    doppler_bins: Tensor,  # [D]
    *,
    velocities: Tensor,  # [N, 3] per-reflector world-space velocity (m/s)
    dt: float,
    sh_degree: int = 3,
    near_range: float = 0.1,
    far_range: float = 200.0,
    psf_kernel: Optional[torch.nn.Module] = None,
    use_physical_dr_psf: bool = False,
    doppler_sign: float = 1.0,  # -1 flips approaching/receding (RADIal forward-FFT convention)
    az_response: Optional[Tensor] = None,  # [A,A] full measured beamformer power response
    psf_in_cuda: bool = True,  # False: apply the sinc-Hann PSF as a post-rasterization
    # torch conv (differentiable w.r.t. bandwidths, used
    # for the learned-PSF variant). True (default): in-CUDA PSF.
) -> Tensor:
    """Incoherent power sum of all reflectors on the RAD grid, [B, D, R, A] (linear, >= 0).

    The radial velocity of each reflector is the ego motion (c2w_prev -> c2w_t over dt)
    plus its own `velocities`; it places the reflector on the Doppler axis.

    az_response: when given, the azimuth PSF is the full measured beamformer
    power response (main lobe + shoulders + sidelobes + pedestal), not the
    analytic sinc-Hann. The rasterizer accumulates per-bin source power via a
    unit-sum interpolating tent (az_tent_delta CUDA mode) and the response is
    applied as `src_field @ az_response`, the power-domain analogue of the
    real chain's `|CalibMat @ mimo|` beamform. This is mathematically identical to
    evaluating the response as the per-reflector azimuth kernel (rasterization
    is linear in power) and exact; applying the response to an already-spread
    field would instead compose mainlobe⊛response and over-leak. Requires the
    CUDA-PSF path and power-domain intensities.
    """
    B = c2w_t.shape[0]

    # w2c (CUDA expects w2c)
    w2c = c2w_to_w2c(c2w_t)  # [B,4,4]
    w2c_prev = c2w_to_w2c(c2w_prev)  # [B,4,4]

    # CUDA boundary expects [B,1,4,4]
    w2c_cuda = w2c[:, None]  # [B,1,4,4]
    w2c_prev_cuda = w2c_prev[:, None]  # [B,1,4,4]

    # Broadcast reflectors across batch for CUDA wrappers
    means_b = means[None].expand(B, -1, -1).contiguous()  # [B,N,3]
    quats_b = quats[None].expand(B, -1, -1).contiguous()  # [B,N,4]
    scales_b = scales[None].expand(B, -1, -1).contiguous()  # [B,N,3]

    # ------------------------------------------------------------
    # Projection (CUDA)
    # ------------------------------------------------------------
    (
        ranges_bc,
        azimuths_bc,
        elevations_bc,
        range_bin_indices_bc,
        az_bin_indices_bc,
        el_bin_indices_bc,
        gaussian_weights_bc,
    ) = projection_radar_3dgs_fused(
        means_b,
        quats_b,
        scales_b,
        w2c_cuda,
        near_range,
        far_range,
        0.0,  # radius_clip: no projected-radius culling
        range_bins,
        az_bins,
        el_bins,
        len(range_bins),
        len(az_bins),
        len(el_bins),
    )

    # ----------------------------------------------------------------
    # Visibility culling: skip reflectors outside the radar FOV.
    # The CUDA projector sets range_bin_index = -1 for out-of-range reflectors,
    # but clamps az_bin to [0, num_az-1] for azimuth-out-of-FOV ones. A clamped
    # reflector would then be spread by the kernel from the edge inward, creating
    # spurious signal inside the FOV boundary. So the continuous azimuth is also
    # checked against [az_bins[0], az_bins[-1]]. A reflector is kept if it is
    # in-FOV (range and azimuth) in at least one batch frame.
    az_lo = float(az_bins[0].item())
    az_hi = float(az_bins[-1].item())
    in_az_fov = (azimuths_bc[:, 0] >= az_lo) & (azimuths_bc[:, 0] <= az_hi)  # [B,N]
    visible_mask = ((range_bin_indices_bc[:, 0] >= 0) & in_az_fov).any(dim=0)  # [N]
    n_vis = int(visible_mask.sum().item())
    if n_vis == 0:
        # No reflectors in FOV: an empty power field.
        return means.new_zeros(B, doppler_bins.shape[0], range_bins.shape[0], az_bins.shape[0])
    if not visible_mask.all():
        vis_idx = visible_mask.nonzero(as_tuple=True)[0]  # [N_vis]
        # Filter [B,1,N] projection outputs → [B,1,N_vis]
        ranges_bc = ranges_bc[:, :, vis_idx]
        azimuths_bc = azimuths_bc[:, :, vis_idx]
        range_bin_indices_bc = range_bin_indices_bc[:, :, vis_idx]
        az_bin_indices_bc = az_bin_indices_bc[:, :, vis_idx]
        el_bin_indices_bc = el_bin_indices_bc[:, :, vis_idx]
        gaussian_weights_bc = gaussian_weights_bc[:, :, vis_idx]
        # Filter world-frame means and per-reflector parameters
        means_b = means_b[:, vis_idx].contiguous()  # [B, N_vis, 3]
        quats_b = quats_b[:, vis_idx].contiguous()  # [B, N_vis, 4]
        scales_b = scales_b[:, vis_idx].contiguous()  # [B, N_vis, 3]
        sh_coeffs = sh_coeffs[vis_idx]  # [N_vis, K, 1]
        opacities = opacities[vis_idx]  # [N_vis]
        velocities = velocities[vis_idx]  # [N_vis, 3]

    # -------------------------
    # View dirs in radar frame
    # -------------------------
    R = w2c[:, :3, :3].contiguous().to(dtype=torch.float32)  # [B,3,3]
    t = w2c[:, :3, 3].contiguous().to(dtype=torch.float32)  # [B,3]
    means_bn = means_b  # [B,N_vis,3]
    means_radar = torch.bmm(means_bn, R.transpose(1, 2)) + t[:, None, :]  # [B,N_vis,3]
    view_dirs = means_radar / (means_radar.norm(dim=-1, keepdim=True) + 1e-8)

    # ------------------------------------------------------------
    # Reflectivity + RCS
    # ------------------------------------------------------------
    reflectivity = _evaluate_sh_view_dependent(
        view_dirs,
        sh_coeffs,
        sh_degree=sh_degree,
    )  # [B,N_vis]
    opac_bn = opacities.view(1, n_vis).expand(B, n_vis)  # [B,N_vis]
    rcs = reflectivity * opac_bn  # [B,N_vis]

    # ------------------------------------------------------------
    # Linear powers (positive)
    # ------------------------------------------------------------
    # No range-falloff term: the reflector's power is its RCS as seen from the sensor.
    # No offset is added here either: a constant on every reflector's power would create
    # a non-zero floor in all rendered bins, preventing weak signal bins from being
    # fitted and making the rendered background inconsistent with the GT.
    powers = rcs.clamp(min=1e-8)  # [B,N]

    # ------------------------------------------------------------
    # Continuous radial velocity (differentiable, for Doppler spreading)
    # ------------------------------------------------------------
    # Doppler bin spacing (for the continuous bin index and the rasterizer).
    dv_doppler = (
        float((doppler_bins[1] - doppler_bins[0]).abs().item()) if len(doppler_bins) > 1 else 0.06
    )

    # ── Doppler-less sensors (D == 1): marginalize, don't sample ──────────────
    # A single Doppler bin means the sensor has no Doppler axis (e.g. Boreas's
    # Navtech scanning radar), so every scatterer's power belongs in that one bin
    # regardless of its radial velocity. Sampling instead of marginalizing would
    # turn the renderer into a velocity filter: with D=1 the spacing above falls
    # back to 0.06 m/s, d_cont = v_r/0.06 lands tens of bins away, and the mod-D
    # wrap leaves a pseudo-random sub-bin offset per reflector whose Doppler PSF
    # weight then attenuates it, so only near-broadside scatterers would survive.
    #
    # Passing v_r_cont=None makes the CUDA kernel take d_cont = doppler_bin = 0,
    # half_kD = 0 and d_wt = psf(0) = 1.0 exactly → full power into the single bin,
    # which is the marginalized render.
    no_doppler_axis = len(doppler_bins) <= 1

    velocities_b = velocities[None].expand(B, -1, -1).contiguous()  # [B, N, 3]
    v_r_cont = compute_doppler_radial_velocity(
        means_b,
        w2c_cuda,
        w2c_prev_cuda,
        dt,
        velocities=velocities_b,
        doppler_sign=doppler_sign,
    )  # [B, 1, N]: differentiable w.r.t. velocities (the object tracks)

    # Integer Doppler bin of each reflector from its continuous index. detach(): the
    # integer centre only sets the iteration range; v_r_cont carries the gradient.
    d_cont = (v_r_cont[:, 0, :] - doppler_bins[0]) / dv_doppler  # [B, N] continuous bin index
    D_bins = len(doppler_bins)
    if no_doppler_axis:
        d_cont = torch.zeros_like(d_cont)  # marginalize: single bin, no velocity gating
    # Wrap with period = D bins, matching the CUDA kernel (RasterizeRadarRAE.cu
    # wraps d_cont with fmod(., D)). The circular aliasing period of the DDMA
    # output is D bin spacings, e.g. RADIal: 16 bins × 0.1123 m/s = 1.7968 m/s.
    # Clamping instead would pile out-of-range reflectors onto the edge bins.
    d_idx_vr = (d_cont.detach().round().long() % D_bins).to(torch.int32)
    doppler_bin_indices_bc = d_idx_vr[:, None, :]

    # ------------------------------------------------------------
    # Rasterize.
    # PSF mode (SensorPSF): analytic/sinc-Hann spreading applied inside the
    # CUDA kernel for all 3 dims; gradients flow from the RAD tensor → Doppler PSF
    # weight → v_r_cont → velocity. Post-rasterization sinc is skipped in this mode.
    # learned_extent mode (ablation): Gaussian R/A spreading of each primitive's
    # projected covariance, plus a post-rasterization PSF convolution when a PSF is
    # given (the learned-PSF ablation).
    range_bin_indices_bc = range_bin_indices_bc.clamp(0, len(range_bins) - 1)
    az_bin_indices_bc = az_bin_indices_bc.clamp(0, len(az_bins) - 1)

    use_cuda_psf = isinstance(psf_kernel, SensorPSF) and psf_in_cuda

    if use_cuda_psf:
        psf_w_D, psf_k_eff_D, psf_w_R, psf_k_eff_R, psf_w_A, psf_k_eff_A = psf_kernel.psf_params()
        rad_tensor_linear = rasterize_radar_rae(
            powers[:, None, :],
            range_bin_indices_bc.int(),
            az_bin_indices_bc.int(),
            el_bin_indices_bc.int(),
            doppler_bin_indices_bc.to(torch.int32),
            gaussian_weights_bc,
            ranges_bc,
            azimuths_bc,
            range_bins,
            az_bins,
            len(doppler_bins),
            len(range_bins),
            len(az_bins),
            # None when D==1 → kernel marginalizes into the single Doppler bin.
            v_r_cont=(None if no_doppler_axis else v_r_cont),
            doppler_bin_spacing=dv_doppler,
            psf_w_D=psf_w_D,
            psf_k_eff_D=psf_k_eff_D,
            psf_w_R=psf_w_R,
            psf_k_eff_R=psf_k_eff_R,
            psf_w_A=psf_w_A,
            psf_k_eff_A=psf_k_eff_A,
            use_physical_dr_psf=use_physical_dr_psf,
            # az_response needs per-bin SOURCE power in azimuth (see docstring).
            az_tent_delta=(az_response is not None),
        )
    else:
        sigma_r_bc, sigma_az_bc = _projected_extent(quats_b, scales_b, R, means_radar)
        rad_tensor_linear = rasterize_radar_rae(
            powers[:, None, :],
            range_bin_indices_bc.int(),
            az_bin_indices_bc.int(),
            el_bin_indices_bc.int(),
            doppler_bin_indices_bc.to(torch.int32),
            gaussian_weights_bc,
            ranges_bc,
            azimuths_bc,
            range_bins,
            az_bins,
            len(doppler_bins),
            len(range_bins),
            len(az_bins),
            # Not detached: sigma_r_m forces the PyTorch rasterizer, which supports
            # the photometric→velocity gradient (bilinear Doppler placement), the same
            # gradient the CUDA-PSF path provides.
            v_r_cont=(None if no_doppler_axis else v_r_cont),
            doppler_bin_spacing=dv_doppler,
            sigma_r_m=sigma_r_bc,
            sigma_az_rad=sigma_az_bc,
        )

    # Expected [B,1,D,R,A] -> drop camera dim
    if rad_tensor_linear.ndim == 5:
        rad_tensor_linear = rad_tensor_linear[:, 0]  # [B,D,R,A]

    # Azimuth beamform: apply the full measured response to the per-bin source
    # power the tent-delta rasterization produced (see docstring).
    if az_response is not None:
        if not use_cuda_psf:
            raise ValueError("az_response requires the CUDA PSF path (psf_kernel=SensorPSF)")
        rad_tensor_linear = rad_tensor_linear @ az_response.to(rad_tensor_linear.dtype)

    # Post-rasterization PSF convolution (learned-PSF ablation); the CUDA PSF path
    # handles the spreading in the kernel.
    if psf_kernel is not None and not use_cuda_psf:
        rad_tensor_linear = psf_kernel(rad_tensor_linear)

    return rad_tensor_linear  # [B, D, R, A] linear power, >= 0, no floor added


def _gaussian_blur_ra(img: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian blur over the last two dims (R, A) of [B, 1, R, A]."""
    radius = max(int(3.0 * sigma + 0.5), 1)
    x = torch.arange(-radius, radius + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    B, C, R, A = img.shape
    img = (
        F.conv1d(img.permute(0, 1, 3, 2).reshape(B * C * A, 1, R), k, padding=radius)
        .reshape(B, C, A, R)
        .permute(0, 1, 3, 2)
    )
    img = F.conv1d(img.reshape(B * C * R, 1, A), k, padding=radius).reshape(B, C, R, A)
    return img


def _yaw_quat(dtheta: Tensor, q: Tensor) -> Tensor:
    """Rotate wxyz quaternions `q` [N, 4] by a yaw `dtheta` [N] about z: r(dtheta) * q."""
    c, s = torch.cos(0.5 * dtheta), torch.sin(0.5 * dtheta)
    w, x, y, z = q.unbind(-1)
    return torch.stack([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w], dim=-1)


class RenderMixin:
    # -----------------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------------

    def ego_vel_sensor(self, frame_idx: int) -> Tensor:
        """The measured ego velocity (the sequence's ego_vel_npy) of one frame.

        Returns: [1, 2]  (x, y) m/s in the sensor frame.
        """
        idx = int(frame_idx)
        idx = max(0, min(idx, len(self.v_ego_smooth) - 1))
        return self.v_ego_smooth[idx].unsqueeze(0)  # [1, 2]

    def _interp_pose_targets(self) -> list:
        """Targets for the interpolated-pose consistency loss (built once).

        Pairs of temporally adjacent train frames (f1, f2) with f2 - f1 <= 2, each
        as (f1, f2, c2w_1, c2w_2, pseudo_gt_ra [1,1,R,A]), where pseudo_gt_ra is the
        blurred mean-over-D RA map of 0.5·(GT[f1] + GT[f2]) in the sensor's
        measurement domain on the normalised map
        (`_normalize_measurement(_to_measurement(·))`), the same as the photometric
        term. On a log sensor the two-frame mean is therefore geometric in linear
        power. At use time a random pose/time strictly between the pair is
        synthesised (unobserved-view regularisation). Built from the train split
        only.
        """
        if hasattr(self, "_interp_pose_targets_cache"):
            return self._interp_pose_targets_cache
        sigma = INTERP_BLUR_SIGMA_BINS
        train_ds = self.train_loader.dataset
        by_frame = {}
        for i in range(len(train_ds)):
            d = train_ds[i]
            by_frame[int(d["frame_idx"].item())] = i
        frames = sorted(by_frame.keys())
        targets = []
        for f1, f2 in zip(frames, frames[1:]):
            if f2 - f1 > 2:
                continue
            gts, poses = [], []
            for f in (f1, f2):
                d = train_ds[by_frame[f]]
                gt = d["rad_tensor"].to(self.device)
                # Same measurement domain and normalisation as the photometric term.
                gts.append(self._normalize_measurement(self._to_measurement(gt)))
                poses.append(d["radarpose"].unsqueeze(0).to(self.device))
            gt_mid = 0.5 * (gts[0] + gts[1])  # [D, R, A]
            n_d = gt_mid.shape[0]
            gm_ra = _gaussian_blur_ra((gt_mid.sum(dim=0) / n_d)[None, None], sigma)  # [1,1,R,A]
            targets.append((f1, f2, poses[0], poses[1], gm_ra))
        self._interp_pose_targets_cache = targets
        if targets:
            print(
                f"[InterpPoseConsist] {len(targets)} train-frame pairs "
                f"(spans of {sorted(set(f2 - f1 for f1, f2, *_ in targets))} "
                f"frames), blur σ={sigma}"
            )
        return targets

    @staticmethod
    def _lerp_pose(c2w_a: Tensor, c2w_b: Tensor, alpha: float) -> Tensor:
        """Pose lerp with SVD re-orthonormalized rotation. Valid for the small
        inter-frame rotations here; alpha outside [0,1] extrapolates."""
        m = (1.0 - alpha) * c2w_a + alpha * c2w_b
        U, _, Vt = torch.linalg.svd(m[:, :3, :3])
        m = m.clone()
        m[:, :3, :3] = U @ Vt
        return m

    def _az_response_table(self) -> Optional[Tensor]:
        """[A_src, A_cell] measured beamformer power response, or None.

        None when `psf_az_sidelobes` is off. Passed to
        `radar_rasterization(az_response=)`, where it is the azimuth PSF: the
        rasterizer accumulates per-bin source power (tent-delta mode) and the
        renderer applies `src_field @ response`, the power-domain analogue of
        `|CalibMat @ mimo|`. Main lobe, sidelobes and pedestal all come from the
        measured response.

        Built once from the RADIal CalibrationTable via DemuxOperator (the response
        of the same `rd_to_rad` operation that produced the GT tensors),
        peak-normalised per source row and evaluated on the trainer azimuth grid.
        Power-domain rendering only: the response composes linearly in power.
        """
        if not self.cfg.psf_az_sidelobes:
            return None
        T = getattr(self, "_az_response_cache", None)
        if T is not None:
            return T
        from dyrad.radial_calib import DemuxOperator

        op = DemuxOperator(device=str(self.device))
        az_deg = torch.rad2deg(self.az_bins.detach().cpu().double())
        tab = op.az_bins_deg.cpu()
        idx = torch.searchsorted(tab, az_deg).clamp(1, len(tab) - 1)
        left = (az_deg - tab[idx - 1]).abs()
        right = (tab[idx] - az_deg).abs()
        src = torch.where(left < right, idx - 1, idx)  # [A] table rows
        CH = op.calib_mat.cpu() * op.hamming.cpu()[None, :]  # [751,192]
        steer = op.calib_mat.cpu()[src]
        steer = steer / steer.abs().clamp(min=1e-12)  # unit-modulus phases
        resp = (steer.conj() @ CH.T.to(steer.dtype)).abs()  # [A_src, 751] amplitude
        resp = resp[:, src]  # → trainer grid [A,A]
        resp = resp / resp.amax(dim=1, keepdim=True).clamp(min=1e-12)
        T = (resp**2).to(torch.float32).to(self.device)  # power response
        self._az_response_cache = T
        A = len(self.az_bins)
        row = T.sum(dim=1)
        print(
            f"[AzResponse] full beamformer power response [{A},{A}] — "
            f"row power {row.min():.1f}..{row.max():.1f} "
            f"(boresight {row[A // 2]:.1f}); azimuth PSF = measured response"
        )
        return T

    def _raster_range_bounds(self) -> Dict:
        """kwargs for radar_rasterization under cull_outside_range_window.

        Empty dict (defaults: near 0.1 / far 200 + CUDA edge-row clamp) unless
        the flag is on; then cull at the cropped window edges ± half a bin."""
        if not self.cfg.cull_outside_range_window:
            return {}
        rb = self.range_bins
        dr = float(rb[1] - rb[0]) if len(rb) > 1 else 1.0
        return {
            "near_range": max(0.1, float(rb[0]) - 0.5 * dr),
            "far_range": float(rb[-1]) + 0.5 * dr,
        }

    def render_all(
        self,
        c2w: Tensor,
        *,
        sh_degree: int,
        t_frame: float,
        v_ego_sensor: Optional[Tensor] = None,  # [B, 2]; if None, looked up at t_frame
        detach_tracks: bool = False,  # render with the current object tracks but send them no gradient
    ) -> Tuple[Tensor, Dict]:
        """Render all reflectors at time t_frame from pose c2w.

        Returns:
            pred_log:  normalised log10 RAD [B, D, R, A] (visualisation and
                       densification view)
            meta:      meta["pred_lin"] is Ŷ as the sensor's native linear quantity
                       (floor included); meta["pred_meas"] is Ŷ in the sensor's
                       measurement domain, the tensor loss and eval use.
        """
        z = self.params["means"]  # [N, 3] learnable positions
        n_lbl = self._n_dynamic

        if n_lbl > 0:
            # Move only the dynamic reflectors (the first n_lbl); static ones get
            # dx = 0, v = 0.
            z_enc_dyn = z[:n_lbl].detach()  # track input decoupled from means grad
            obj_dyn = self.obj_idx[:n_lbl] if self.obj_idx is not None else None
            dx_dyn = self.tracks(z_enc_dyn, t_frame, obj_dyn)  # [n_lbl, 3]
            # Warn once if the displacement clamp binds: a clamped object renders
            # short of its true position.
            _xs = float(self.cfg.track_max_displacement_m)
            if not hasattr(self, "_dx_clamp_warned") and bool((dx_dyn.detach().abs() > _xs).any()):
                self._dx_clamp_warned = True
                print(
                    f"[WARN] track displacement exceeds track_max_displacement_m={_xs:.0f} m "
                    f"(max |dx|={float(dx_dyn.detach().abs().max()):.0f} m) — "
                    f"object will render at the WRONG position. Raise track_max_displacement_m."
                )
            dx_dyn = dx_dyn.clamp(-_xs, _xs)
            v_dyn = self.tracks.velocity(z_enc_dyn, t_frame, obj_idx=obj_dyn)
            v_dyn = v_dyn.nan_to_num(0.0).clamp(-500.0, 500.0)
            if detach_tracks:
                dx_dyn = dx_dyn.detach()
                v_dyn = v_dyn.detach()
            dx = torch.zeros_like(z)
            v_t = torch.zeros_like(z)
            dx[:n_lbl] = dx_dyn
            v_t[:n_lbl] = v_dyn
        else:
            dx = torch.zeros_like(z)
            v_t = torch.zeros_like(z)

        means_t = z + dx  # [N, 3]

        quats = F.normalize(self.params["quats"], dim=-1)
        if self.cfg.train_quats and n_lbl > 0 and self.obj_idx is not None:
            # Anisotropic reflectors (learned-extent ablation): a dynamic reflector's
            # orientation turns with its object, like its position in tracks.forward.
            dtheta = self.tracks.heading(t_frame, self.obj_idx[:n_lbl])
            if detach_tracks:
                dtheta = dtheta.detach()
            quats = torch.cat([_yaw_quat(dtheta, quats[:n_lbl]), quats[n_lbl:]], dim=0)
        scales = torch.exp(self.params["scales"])
        opacs = self._get_opacs()
        sh_coeffs = self.params["sh_coeffs"]

        # Ego motion enters through a synthetic previous pose built from the measured
        # ego velocity (see _c2w_prev_can).
        c2w_prev = self._c2w_prev_can(
            c2w, t_frame, v_sen=v_ego_sensor[0] if v_ego_sensor is not None else None
        )

        # b_s, the background pedestal, is derived from the data, not tuned. Under the
        # ceiling normalization it is the norm.json sidecar's `floor_u` (written into
        # self.noise_floor in __init__): for a linear-mode sidecar (RADIal) the p1 of
        # log10(GT/hi); for a counts-mode sidecar (Boreas) 1/hi, the power at zero
        # counts. No config multiplier is exposed.
        raster_floor = self.noise_floor.item()

        pow_field = radar_rasterization(
            means_t,
            quats,
            scales,
            opacs,
            sh_coeffs,
            c2w_t=c2w,
            c2w_prev=c2w_prev,
            range_bins=self.range_bins,
            az_bins=self.az_bins,
            el_bins=self.el_bins,
            doppler_bins=self.doppler_bins,
            sh_degree=sh_degree,
            psf_kernel=self.psf,
            velocities=v_t,
            dt=self.cfg.dt,
            use_physical_dr_psf=self.cfg.use_physical_dr_psf,
            doppler_sign=float(self.cfg.doppler_sign),
            az_response=self._az_response_table(),
            psf_in_cuda=self.cfg.psf_in_cuda,
            **self._raster_range_bounds(),
        ).clamp(min=0.0)  # [B,D,R,A] POWER

        # ── Incoherent power sum -> the sensor's native linear quantity ────────────
        # The rasterizer output is an incoherent power sum, so every formation operator
        # (including the measured azimuth response, applied inside the rasterizer) acts
        # linearly on power. Then match the GT's native quantity: an amplitude sensor
        # (`_root_render`) takes one sqrt at output, a power sensor keeps power; the
        # noise-floor pedestal is added in that domain.
        # Sensor near-range blanking (Config.near_range_blank_bins): no scattered power
        # reaches Ŷ there, so it renders as the pedestal alone, as the GT does.
        _nb = int(self.cfg.near_range_blank_bins) - int(self.cfg.range_crop_first)
        if _nb > 0:
            _keep = torch.ones(
                pow_field.shape[-2], 1, device=pow_field.device, dtype=pow_field.dtype
            )
            _keep[:_nb] = 0.0
            pow_field = pow_field * _keep
        _sps = float(self.cfg.sensor_power_scale)
        if _sps != 1.0:  # sensor transfer only
            pow_field = pow_field * _sps
        # clamp(min=1e-12) avoids an infinite sqrt gradient at 0
        field = torch.sqrt(pow_field.clamp(min=1e-12)) if self._root_render else pow_field
        pred_lin = field + raster_floor
        # Normalised log view on the shared map N (visualisation, densification residual).
        pred_log = self._norm_log_map(pred_lin)
        # ── Ŷ in the sensor's measurement domain (Config.measurement_domain) ─────────
        # The tensor the loss, eval and persisted renders consume; the loss never reads
        # `pred_log`. `pred_lin` is kept for metrics defined on linear values (CFAR,
        # Doppler pooling, Pearson).
        meta = {"pred_lin": pred_lin, "pred_meas": self._to_measurement(pred_lin)}
        return pred_log, meta

    def _t_frame(self, frame_idx: Tensor) -> float:
        return self.t_of_frame(
            int(frame_idx.item()) if hasattr(frame_idx, "item") else int(frame_idx)
        )

    # ── Time axis ─────────────────────────────────────────────────────────────
    # `t = slot * dt` for sensors sampled on a fixed frame period (RADIal, synthetic), with
    # slot from <radar_poses_dir>/frame_slots.npy when the preprocessing wrote one (RADIal:
    # one period per frame, two across a dropped frame; its recorded stamps stall, see
    # preprocessing/radial/poses.py) and slot = frame otherwise. With
    # `use_frame_timestamps` (Boreas) the axis is the recorded timestamp relative to
    # frame 0, the times the poses were sampled at. All frame<->time conversions go
    # through `t_of_frame` / `_frame_of_t`.
    def _build_frame_times(self, n_frames: int) -> None:
        cfg = self.cfg
        t = np.arange(n_frames, dtype=np.float64) * float(cfg.dt)
        src = "frame*dt"
        _slots = Path(cfg.radar_poses_dir) / "frame_slots.npy"
        if not cfg.use_frame_timestamps and _slots.exists():
            slots = np.load(_slots)
            if len(slots) < n_frames:
                raise ValueError(f"{_slots} has {len(slots)} slots for {n_frames} frames")
            t = slots[:n_frames].astype(np.float64) * float(cfg.dt)
            src = f"slot*dt ({_slots})"
        if cfg.use_frame_timestamps:
            _p = Path(cfg.radar_poses_dir) / "timestamps_us.npy"
            if not _p.exists():
                raise FileNotFoundError(f"use_frame_timestamps is set but {_p} is missing")
            _ts = np.load(_p).astype(np.float64) * 1e-6
            if len(_ts) < n_frames:
                raise ValueError(f"{_p} has {len(_ts)} stamps for {n_frames} frames")
            t = _ts[:n_frames] - _ts[0]
            src = str(_p)
        self._frame_t = t
        fs = int(cfg.frame_start)
        fe = int(cfg.frame_end) if int(cfg.frame_end) >= 0 else n_frames  # -1 = all frames
        d = np.diff(t[fs : max(fs + 1, min(fe, n_frames))])
        n_irr = int(np.sum(np.abs(d - float(cfg.dt)) > 0.02)) if len(d) else 0
        print(
            f"[Time] axis from {src}: {n_frames} frames, window {fs}-{fe}: "
            f"{n_irr} irregular interval(s)" + (f" (max {d.max():.2f} s)" if len(d) else "")
        )

    def t_of_frame(self, fi: int) -> float:
        ft = self._frame_t
        return float(ft[min(max(int(fi), 0), len(ft) - 1)])

    def _frame_of_t(self, t: float) -> int:
        return int(np.argmin(np.abs(self._frame_t - float(t))))

    def _get_opacs(self) -> Tensor:
        """Per-reflector opacity sigmoid(opacities): the per-reflector brightness (sh_dc is pinned)."""
        return torch.sigmoid(self.params["opacities"])

    @torch.no_grad()
    def _c2w_prev_can(
        self, c2w: Tensor, t_frame: float, v_sen: Optional[Tensor] = None
    ) -> Tensor:
        """Synthetic prev pose from the measured ego velocity.

        c2w_prev = c2w with the position shifted forward by R_c2w @ [v_x, v_y, 0] * dt
        (the future pose), so compute_doppler_radial_velocity returns approaching-positive
        v_r, matching the GT convention. Consecutive dataset poses are not used for the
        Doppler: interpolated GPS poses give per-frame speed errors of several m/s, many
        Doppler bins.

        v_sen: [2] sensor-frame ego velocity; None looks it up at the frame nearest
        t_frame. Reads batch element 0 only (the loaders use batch size 1).
        """
        if v_sen is None:
            fi_c = min(self._frame_of_t(t_frame), len(self.v_ego_smooth) - 1)
            v_sen = self.v_ego_smooth[fi_c]  # [2] sensor frame
        R_c2w = c2w.reshape(-1, 4, 4)[0, :3, :3]
        v_world = R_c2w @ torch.cat([v_sen, v_sen.new_zeros(1)], dim=0)  # [3]
        c2w_prev = c2w.detach().clone()
        c2w_prev.reshape(-1, 4, 4)[0, :3, 3] += v_world * self.cfg.dt  # + not -
        return c2w_prev

    def _compute_doppler_gate(self, c2w: Tensor, c2w_prev: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (static_mask, dynamic_mask) of shape [D, R_crop, A].

        For each azimuth bin, compute the Doppler bin where a stationary object
        would appear given the ego displacement, then mark ±sigma_static_bins
        around it as the static band. dynamic_mask = ~static_mask. A sensor with a
        single Doppler bin (Boreas) has no Doppler to separate by: every cell is static.
        """
        D = len(self.doppler_bins)
        R = len(self.range_bins)
        A = len(self.az_bins)
        if D == 1:
            static_mask = torch.ones(1, R, A, dtype=torch.bool, device=self.device)
            return static_mask, ~static_mask

        r_test = float(self.range_bins[R // 2].item())
        az = self.az_bins  # [A] radians

        # Dummy stationary world points at each azimuth
        p_cam = torch.stack(
            [
                torch.cos(az) * r_test,
                torch.sin(az) * r_test,
                torch.zeros(A, device=self.device),
            ],
            dim=1,
        )  # [A, 3]
        p_world = p_cam @ c2w[0, :3, :3].T + c2w[0, :3, 3]  # [A, 3]

        w2c_t = c2w_to_w2c(c2w)
        w2c_p = c2w_to_w2c(c2w_prev)
        v_r = compute_doppler_radial_velocity(
            p_world.unsqueeze(0),
            w2c_t.unsqueeze(1),
            w2c_p.unsqueeze(1),
            dt=self.cfg.dt,
            velocities=None,
            doppler_sign=float(self.cfg.doppler_sign),
        )[0, 0, :]  # [A]

        dv = (self.doppler_bins[-1] - self.doppler_bins[0]) / (D - 1)
        d_static = (v_r - self.doppler_bins[0]) / dv  # [A] continuous bin index

        d_idx = torch.arange(D, dtype=torch.float32, device=self.device)
        d_delta = (d_idx.unsqueeze(1) - d_static.unsqueeze(0)).abs()  # [D, A]
        if float(self.cfg.doppler_wrap_period_mps) > 0.0:
            # Folded axis: static bin index is only defined mod D; use circular distance.
            d_delta = (d_idx.unsqueeze(1) - torch.remainder(d_static, D).unsqueeze(0)).abs()
            d_delta = torch.minimum(d_delta, D - d_delta)
        static_da = d_delta <= self.cfg.sigma_static_bins  # [D, A]
        static_mask = static_da.unsqueeze(1).expand(D, R, A).contiguous()
        dynamic_mask = ~static_mask
        return static_mask, dynamic_mask
