"""Run configuration: the `Config` dataclass, yaml loading with `base:` inheritance
(`load_config`), the holdout rule (`is_held_out`) and the measurement-domain check
(`resolve_measurement_domain`).

Pure yaml/os, safe to import from torch-free tools (scoring, preprocessing).

A config may declare
    base: recipes/radial.yaml           # string, or
    base: [recipes/radial.yaml, sequences/radial_31_22.yaml]
with paths relative to the declaring file's directory (absolute paths also work).
Bases are resolved recursively, merged in list order (later bases override
earlier ones), and the declaring file's own keys override everything.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


def is_held_out(frame_idx: int, test_every: int, test_offset: int = 0) -> bool:
    """The holdout rule shared by the trainer, the cloud builder and the scorers: with
    `test_every` > 0, every test_every-th global frame counted from `test_offset` is held
    out for evaluation; `test_every` 0 holds out nothing (off-path fits, the synthetic
    benchmark)."""
    every = int(test_every)
    return every > 0 and (int(frame_idx) - int(test_offset)) % every == 0


def held_out_frames(cfg) -> list:
    """The held-out frames of the config's window [frame_start, frame_end) (`is_held_out`)."""
    return [
        f
        for f in range(int(cfg.frame_start), int(cfg.frame_end))
        if is_held_out(f, cfg.test_every, cfg.test_offset)
    ]


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    bases = raw.pop("base", None)
    if bases is None:
        return raw
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        b_path = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
        merged.update(load_yaml_config(b_path))
    merged.update(raw)
    return merged


@dataclass
class Config:
    # ── data ────────────────────────────────────────────────────────────────
    # The sequence stub (configs/sequences/*.yaml) sets the data paths.
    # Per-frame RAD tensors, rad_<frame>.npy.
    rad_tensors_dir: str = ""
    # radar_poses.npy + poses_metadata.json (+ timestamps_us.npy).
    radar_poses_dir: str = ""
    result_dir: str = "results/default"
    # radial | boreas | synthetic (set by the sequence stub).
    dataset: str = ""
    # The sequence directory (holds norm.json and the init cloud).
    seq_dir: str = ""
    # Voxel size of the init cloud (names the cloud file).
    init_cloud_voxel_m: float = 0.5

    # ── initialization source ────────────────────────────────────────────────
    # A [N,3] world-frame radar pseudo-LiDAR cloud built from the RA tensors seeds the
    # static reflectors; per-frame labels seed the dynamic objects (paper Sec. 3.3, B.1).
    # [N,3] world positions of the static cloud.
    init_cloud_path: str = ""
    # Sensor-frame ego velocity [N,2] float32, one row per pose, written by the
    # preprocessing (required).
    ego_vel_npy: str = ""
    # NMS suppression radius (bins) of the densification residual peaks.
    rad_peak_nms_radius: int = 5

    # ── radar dims ───────────────────────────────────────────────────────────
    # Every recipe sets the sensor grid (num_*_bins, crops, radar_far_range,
    # radar_az_fov_deg, the Doppler extent); the defaults here are placeholders.
    num_range_bins: int = 256
    num_azimuth_bins: int = 107
    num_doppler_bins: int = 64
    range_crop_first: int = 10
    range_crop_last: int = 10
    # Sensor near-range blanking, in raw range bins (before `range_crop_first`). The
    # render's scattered power is zeroed on the first `near_range_blank_bins` bins, so
    # Ŷ there is the bare pedestal, matching the preprocessed GT, which zeroes the same
    # bins (Boreas: 41 = int(2.5 m / 0.0596 m)). 0 = no blanking.
    near_range_blank_bins: int = 0
    radar_far_range: float = 118.0
    # Sensor-transfer gain on the summed render power, used when re-rendering one
    # sensor's scene through another sensor's axes. 1.0 = identity; not a training
    # brightness knob.
    sensor_power_scale: float = 1.0
    dt: float = 0.1
    # Sensor grid (see dyrad.axes). Elevation is a single bin at 0 deg.
    radar_az_fov_deg: float = 107.0
    # Doppler axis extent, m/s (set by every recipe).
    radar_doppler_min_mps: Optional[float] = None
    radar_doppler_max_mps: Optional[float] = None
    # Range bin b <-> (b + range_bin_offset) * dr. 0.0 is the FFT convention of an FMCW
    # range profile (a target at range R in bin R/dr: RADIal, the synthetic sensor); the
    # Navtech polar images of Boreas put bin b at its centre, (b + 0.5) * dr.
    range_bin_offset: float = 0.0

    # Initial reflected-power alpha of every reflector (paper's alpha).
    init_alpha: float = 0.5
    init_scale: float = 0.3
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    # Cull reflectors outside [range_bins[0], range_bins[-1]] at projection instead of
    # clamping them onto the first/last range row (which creates bright edge rows).
    # Matters when the cloud extends well beyond the range window.
    cull_outside_range_window: bool = False

    # ── learning rates ───────────────────────────────────────────────────────
    means_lr: float = 1e-4  # reflector positions
    sh_coeffs_lr: float = 5e-3
    opacities_lr: float = 5e-2

    # ── representation ablation ──────────────────────────────────────────────
    # Defaults are the method: point reflectors + fixed physical PSF (scales/quats
    # frozen, sinc bandwidths fixed). The ablations relax one axis at a time.
    # Learnable per-reflector extent (Gaussian primitives, learned-extent ablation).
    train_scales: bool = False
    train_quats: bool = False  # learnable orientation
    scales_lr: float = 0.0  # e.g. 5e-3 when train_scales
    quats_lr: float = 0.0  # e.g. 1e-3 when train_quats
    # > 0: clamp exp(scales) to [0.02 m, this] after each step.
    scales_clamp_max_m: float = 0.0
    psf_learned: bool = False  # true = learnable PSF bandwidths (ablation)
    psf_lr: float = 1e-3
    psf_in_cuda: bool = True  # false = PSF as a post-raster torch convolution

    # ── object tracks (RigidObjectTrack) ─────────────────────────────────────
    deform_lr: float = 1e-3  # track-point learning rate (paper Table 3 "Motion LR")

    # Bound (m) on each coordinate of an object reflector's track offset: the render
    # clamps tracks(means) to [-this, this], and because `means` are object-frame
    # coordinates that offset is the object's world position. Must exceed the largest
    # |world coordinate| an object reaches in the window, or its motion is truncated.
    track_max_displacement_m: float = 50.0

    # ── object labels ────────────────────────────────────────────────────────
    # Per-frame object annotations separate the dynamic objects from the static
    # background at initialization; each dynamic reflector keeps its object index.
    object_label_path: str = ""  # path to the label CSV (labels_CVPR.csv)
    # The label CSV's `dataset` value for this sequence (required when labels are used).
    seq_name: str = ""
    frame_start: int = 0  # first frame (global index, inclusive)
    frame_end: int = -1  # last frame (global index, exclusive); -1 = all
    # Global frame indices to exclude from train/val.
    bad_frame_ids: list = field(default_factory=list)
    # Rigid-track safety segment: extend each object track by this many frames at
    # both ends with a learnable, linearly-extrapolated anchor, so an object keeps
    # moving for a bounded window past its last control point, then flat-clamps.
    # 0 = off (flat beyond the last control point).
    track_safety_frames: int = 0
    # Least-squares velocity window, in segments either side of the query segment.
    # 1 = adjacent-control-point difference; k > 1 fits the slope over 2k points
    # (noise ~ sqrt(12/(N(N^2-1))), unbiased under constant acceleration). See _interp_vel.
    track_vel_fit_span: int = 1
    # Time axis from <radar_poses_dir>/timestamps_us.npy instead of slot*dt (see
    # Runner._build_frame_times).
    use_frame_timestamps: bool = False
    # > 0: re-densify the track with chord-initialised free control points every N
    # frames, so capacity is independent of annotation density. 0 = off.
    track_ctrl_interp_every: int = 0
    # ── rigid-body yaw ───────────────────────────────────────────────────────
    # Each object's reflectors rotate with its heading. Heading needs no annotation
    # (RADIal labels have no box yaw): it is the direction of the control polyline, so it
    # is differentiable in the control points. Positions rotate, and so do the
    # orientations (`quats`) when they are trained (the learned-extent ablation).
    # Objects whose annotation polyline travels less than this are pure translators:
    # their heading would be noise. Decided once at init from the annotation polyline.
    track_rotate_min_travel_m: float = 5.0
    # Meaning of `means` for a dynamic reflector i on object j:
    #     x_i(t) = R_j(t) · means_i + P_j(t)          means_i in object coordinates
    # and for a static reflector x_i = means_i. P_j is the track polyline and R_j the
    # heading derived from its tangent (relative to the seed heading). The object frame
    # is 2-D (xy); z stays a world height.
    # Dynamic objects are seeded from their own GT returns (paper B.1). For each labelled
    # train-frame detection, take the object's Doppler slice (argmax over D at the label
    # cell; the RA map when D = 1) and seed from the bright R/A cells inside the object's
    # footprint, motion-compensated (desmeared) to the object's canonical time t0.
    dyn_seed_peak_frac: float = 0.30  # keep cells > this fraction of the window peak
    dyn_seed_voxel_m: float = 0.5  # voxel-dedup the desmeared seed cloud (m)
    # ── RA partition (ra_partition.py) ───────────────────────────────────────
    # Split each RA frame into free space + one footprint per annotated object, so the
    # static and dynamic halves are complementary by construction.
    #   static_from_free_space: declare that init_cloud_path was built with
    #     `build_init_cloud --partition` (objects already removed per frame).
    # The static-only baseline must leave this off and use an unpartitioned cloud: it
    # assumes no annotations, so the whole frame is free space.
    static_from_free_space: bool = False
    # Median bbox-derived width over the RADIal labels.
    car_width_default_m: float = 1.94
    # ~4.6 m long / ~1.9 m wide; length lies along range.
    car_len_over_width: float = 2.3
    car_margin_m: float = 0.0  # grow the footprint on every side, in metres
    # Dilate the footprint by the PSF, in bins, as multiples of its half-FWHM. An object's
    # energy is geometry convolved with the beam, and the beam is an angle, so the
    # undilated box clips far objects' returns. It also carves the static free mask, so a
    # run and its *_free.npy cloud must share this value (checked in
    # _init_from_radar_cloud).
    car_psf_margin: float = 0.0

    # Static-only baseline: every radar-cloud point becomes a static reflector (no
    # dynamic objects, no track); labels stay loaded for eval.
    static_only: bool = False
    # Persist renders_npy/ (gt+pred RA/RAD npy, large). Off: regenerate from
    # checkpoints on demand.
    save_renders_npy: bool = False
    # Camera calibration (camera_calib.npy with intrinsic/extrinsic): gives the focal length
    # that turns a label's bbox width into an object size (the object footprint).
    # Empty = no camera (synthetic data): the default car footprint is used.
    camera_calib_path: str = ""
    # Greedy label tracker: max world-frame jump per frame (m).
    label_assoc_threshold_m: float = 8.0
    # Greedy label tracker: detections more than this many frames apart start a new
    # track. Small values fragment intermittently detected vehicles; the CV-prediction
    # gate and the camera veto guard identity.
    label_track_max_gap: int = 3
    # Annotation-quality gates for control points / Doppler targets. RADIal labels carry
    # a per-detection "strong"/"weak"/"incomplete" flag; "weak" admits strong+weak
    # detections (denser track), "strong" admits strong only.
    label_track_min_quality: str = "strong"  # min quality for a track control point
    # Drop a control point if its implied speed exceeds this (m/s).
    label_track_swap_thresh_mps: float = 50.0

    # Freeze sh_coeffs gradient for dynamic reflectors for this many steps, so
    # the photometric loss cannot dim them before their motion converges. 0 = off.
    sh_freeze_dynamic_steps: int = 0

    # ── sensor PSF (paper Sec. 3.2 / A.1) ────────────────────────────────────
    # render_mode: "psf" renders point reflectors through the fixed sensor PSF;
    # "learned_extent" is the ablation with Gaussian primitives of learned extent.
    render_mode: str = "psf"
    psf_k_D: int = 4
    psf_max_k_D: int = 41
    psf_w_D: float = 1.5
    psf_k_R: int = 6
    psf_max_k_R: int = 41
    psf_w_R: float = 1.5
    psf_k_A: int = 14
    psf_max_k_A: int = 61
    psf_w_A: float = 4.0
    psf_strength_init: float = 1.0

    # ── Doppler sign convention ──────────────────────────────────────────────
    # +1.0: approaching targets have positive Doppler (the forward model's convention).
    # RADIal's stored tensors, and the synthetic generator's, which follows them, are
    # receding-positive, so their recipes use -1.0 (Boreas, with one Doppler bin, is
    # unaffected). Applied in the Python radial-velocity computation.
    doppler_sign: float = 1.0

    # ── loss and normalization ───────────────────────────────────────────────
    # The GT power is divided by the sequence's ceiling `hi` at load (data max -> 1), the
    # SH DC term is pinned to 0, and one shared normalised-log map N (floor at 0, ceiling
    # at 1) serves loss, eval and viz. Its parameters (floor_u, norm_lo, norm_range) are
    # read from the sequence's norm.json.
    # Log10 percentiles of the train GT that anchor norm.json's log map
    # (dyrad.preprocessing.normalize_dataset): in linear mode (RADIal, synthetic)
    # norm_percentile_lo sets the render pedestal floor_u = 10**p_lo, and
    # norm_percentile_hi the top of the secondary log view. The Runner also prints
    # both percentiles of the training frames as an init diagnostic.
    norm_percentile_lo: float = 1.0
    norm_percentile_hi: float = 99.5
    # Global linear-normalisation range (overrides the per-scene robust percentiles when
    # > 0). A shared (lo, hi) makes the [0,1] map identical across sequences and
    # independent of a scene's outlier peaks. 0 = per-scene robust range.
    lin_lo_global: float = 0.0  # RADIal: 1833.0 (~p0.01 of the RAD cube)
    lin_hi_global: float = 0.0  # RADIal: 5.9e6 (p99.99 of the RAD cube)
    # Per-scene hi when lin_hi_global is 0: the lin_hi_pct percentile of the train GT,
    # written to norm.json by dyrad.preprocessing.normalize_dataset (the linear lo is
    # always lin_lo_global).
    lin_hi_pct: float = 99.9
    # Clip on the linear-normalised values, in units of hi. Applied identically in
    # loss, eval and viz. 1.0 = clip to [0, 1]; <= 0 disables the clip.
    norm_clip_ceiling: float = 1.0
    # The photometric term is the plain MSE over every RAD bin, L_rec = mean((Y_hat - Y)^2).
    # Squared error weights each cell by its own error magnitude, so bright cells keep
    # gradient weight without a hand-set threshold.
    # Weight w_D on the Doppler-shape half of the L2 photometric term.
    #
    # Each (R,A) cell's Doppler profile is split into its D-mean (DC) and its zero-mean
    # shape. The split is exact and orthogonal, because the shape sums to zero over D:
    #     mean_D (dDC + dSHAPE_d)^2  ==  dDC^2 + mean_D dSHAPE_d^2
    # so w_D = 1 is the plain full-RAD L2 and w_D = 0 leaves a pure mean-over-D RA loss.
    # w_D is therefore the Doppler ablation axis.
    # Must be 1.0 when num_doppler_bins = 1 (the Runner raises otherwise).
    doppler_axis_weight: float = 1.0
    # ── Measurement domain: the units of Ŷ and Y, fixed per sensor ──────────────
    # The renderer accumulates incoherent power Ψ (powers add; amplitudes and logs do
    # not). The model output is defined in the sensor's own reporting units, Ŷ = g(Ψ):
    #     amplitude      g(Ψ) = sqrt(Ψ) + b_s            RADIal |beamform|, synthetic
    #     power          g(Ψ) = Ψ + b_s
    #     log_power      g(Ψ) = log10(Ψ + b_s)           Boreas/Navtech (20·log10 -> counts)
    #     log_amplitude  g(Ψ) = log10(sqrt(Ψ) + b_s)
    # GT is never transformed; the render is mapped to meet it. Loss, eval and the
    # persisted renders all consume this one Ŷ (meta["pred_meas"]), so the photometric
    # term is mean((Ŷ_N - Y_N)²) for every sensor. Validated by
    # `resolve_measurement_domain` below.
    measurement_domain: str = "amplitude"
    # Apply the measured azimuth beamformer response (sidelobes and pedestal) that the
    # sinc-Hann main lobe cannot express (see Runner._az_response_table).
    psf_az_sidelobes: bool = False
    # lambda_int: render at a pose/time midway between two train frames and penalise the
    # deviation from the neighbour renders (paper Sec. 3.3). The object tracks are
    # detached in that render: its pseudo-GT (a two-frame mean) shows moving objects
    # smeared across two positions, which is not a valid trajectory target.
    interp_consist_weight: float = 0.0
    # Half-width (Doppler bins) of the static band around the ego Doppler. Used for the
    # densification static mask and the dyn_l1 metric; 0 = disabled.
    sigma_static_bins: float = 2.0

    # ── noise floor ──────────────────────────────────────────────────────────
    # norm.json noise floor nf = k · median(Doppler-mean RA of the train GT), written by
    # dyrad.preprocessing.normalize_dataset. nf feeds only the sensor-transfer gain
    # (evaluation/sensor_transfer.py); the trainer's pedestal is norm.json's floor_u.
    noise_floor_median_k: float = 0.0

    # ── folded (DDMA) Doppler axis ───────────────────────────────────────────
    # doppler_wrap_period_mps > 0 makes velocity comparisons circular: a physical v_r
    # lands on the axis only modulo the period (RADIal: 16 bins x 0.1123 m/s = 1.7968
    # m/s). 0 = linear axis.
    # doppler_roll_bins rolls the GT tensor along D at load: GT bin 0 = DC, while the
    # renderer's axis puts v=0 at bin D/2 (use D/2).
    doppler_wrap_period_mps: float = 0.0
    doppler_roll_bins: int = 0

    # Allow rad_tensors_dir to hold only the training window of a recording (the coarse
    # sensor-configuration sequences); see RadarParser.
    allow_missing_rad: bool = False

    # True: analytic Hamming-FFT range + Doppler PSF (RADIal); False: sinc-Hann.
    use_physical_dr_psf: bool = False

    # ── densification (residual-peak seeding of static reflectors, paper B.3) ─
    densify_start_step: int = 500
    densify_stop_step: int = 15000
    densify_every: int = 100
    # Residual-peak threshold (normalised log units) for seeding.
    densify_residual_thresh: float = 0.05
    densify_max_reflectors: int = 0  # total reflector cap (0 = unlimited)
    # Residual-peak densification: seed new reflectors at back-projected residual peaks,
    # filling under-predicted cells rather than duplicating already-bright ones.
    # Max new reflectors seeded per grow (top-K by residual).
    densify_max_per_grow: int = 1000
    # Skip a seed if an existing reflector is within this XY radius (m).
    densify_dedup_radius_m: float = 1.0

    # ── render saves ─────────────────────────────────────────────────────────
    # Global frame index to visualise when render_frame_idxs is empty.
    render_frame_idx: int = 0
    # Global frame indices to visualise; if non-empty, overrides render_frame_idx.
    render_frame_idxs: List[int] = field(default_factory=list)
    # Render the val (holdout) frames, all in-window; falls back to render_frame_idxs
    # when there is no val split.
    render_val_frames: bool = True
    render_save_steps: List[int] = field(
        default_factory=lambda: [0, 1000, 2000, 5000, 10000, 15000]
    )

    # ── logging ───────────────────────────────────────────────────────────────
    # log_every: write a row to train_log.csv every N steps (0 = disable).
    # Loss curves are auto-plotted at end of training when log_every > 0.
    log_every: int = 100

    # ── training schedule ────────────────────────────────────────────────────
    max_steps: int = 15000
    seed: int = 0  # >0: seed python/numpy/torch (deterministic cudnn)
    # Holdout: every test_every-th global frame from test_offset is a validation frame
    # (see is_held_out); test_every 0 = no holdout.
    test_every: int = 0
    test_offset: int = 0
    save_steps: List[int] = field(
        default_factory=lambda: [1000, 2000, 5000, 10000, 15000]
    )


MEASUREMENT_DOMAINS = ("amplitude", "power", "log_power", "log_amplitude")


def resolve_measurement_domain(cfg) -> str:
    """The sensor's reporting domain of a resolved config (Config.measurement_domain),
    normalised to lower case; raises ValueError when it is not one of MEASUREMENT_DOMAINS."""
    explicit = str(cfg.measurement_domain or "").strip().lower()
    if explicit not in MEASUREMENT_DOMAINS:
        raise ValueError(
            f"measurement_domain={explicit!r}; expected one of {MEASUREMENT_DOMAINS}"
        )
    return explicit


def load_config(path: str) -> Config:
    """Load a run config (resolving `base:` inheritance) into a Config; unknown keys are errors."""
    raw = load_yaml_config(path)
    cfg = Config()
    unknown = [k for k in raw if not hasattr(cfg, k)]
    if unknown:
        raise ValueError(f"{path}: unknown config key(s): {', '.join(sorted(unknown))}")
    for k, v in raw.items():
        setattr(cfg, k, v)
    return cfg
