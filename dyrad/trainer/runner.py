"""The Runner: fits point reflectors and object tracks to recorded RAD tensors (`python -m dyrad.train`).

Scene model (paper Sec. 3.1): static background reflectors keep their world position;
dynamic reflectors belong to an annotated object and follow its learned rigid track
(``RigidObjectTrack``), x_i(t) = p_j(t) + R_j(t) mu_i. Each reflector's radial velocity
(ego motion plus object motion) places it on the Doppler axis, so the rendered Doppler
supervises the tracks. Rendering goes through the fixed sensor PSF (paper Sec. 3.2).

Loss (paper Sec. 3.3): L = L_rec + lambda_int L_int, the mean squared error over every RAD
bin in the sensor's measurement domain plus the interpolation-consistency term between
adjacent training poses. The Runner is assembled from the mixins in initialization,
rendering, measurement, densification, evaluate, visualize and checkpoint.
"""

import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dyrad.config import Config, resolve_measurement_domain
from dyrad.data import RadarDataset, RadarParser
from dyrad.poses import load_ego_velocity
from dyrad.loss import reconstruction_loss
from dyrad.psf import SensorPSF
from dyrad.trainer.checkpoint import CheckpointMixin, _write_provenance
from dyrad.trainer.densification import DensifyMixin
from dyrad.trainer.evaluate import EvalMixin
from dyrad.trainer.initialization import InitMixin, create_reflectors, make_render_bins
from dyrad.trainer.measurement import MeasurementMixin
from dyrad.trainer.rendering import INTERP_BLUR_SIGMA_BINS, RenderMixin, _gaussian_blur_ra
from dyrad.trainer.visualize import VizMixin, _plot_train_log, _write_train_log

# The log10 percentiles printed at init are taken over at most this many training frames.
_NORM_PRINT_FRAMES = 20
# torch.quantile accepts at most 2^24 elements.
_QUANTILE_MAX_ELEMS = 1 << 24
# Lower bound of the learned extent when scales are trained, m.
_SCALES_CLAMP_MIN_M = 0.02


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) and make cuDNN deterministic."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _grad_norm(param: "torch.Tensor") -> float:
    """L2 norm of param.grad, or 0 if no grad."""
    return param.grad.norm().item() if param.grad is not None else 0.0


def _grad_norm_iter(params) -> float:
    """L2 norm pooled across all parameters in an iterator."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total**0.5


class Runner(
    InitMixin, RenderMixin, MeasurementMixin, DensifyMixin, EvalMixin, VizMixin, CheckpointMixin
):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if int(cfg.seed) > 0:
            set_random_seed(int(cfg.seed))
            print(f"[Seed] {int(cfg.seed)}")

        self._setup_bins()

        # ── Map the render to the GT domain (never the GT to the render) ─────────
        # The forward model accumulates incoherent power. GT stays in its native domain
        # and the render is mapped to match (Config.measurement_domain): sqrt for an
        # amplitude sensor (`_root_render`), log10 for a log-compressed one, applied
        # once at render output so loss/eval/persisted renders share the same Ŷ.
        self._meas_domain = resolve_measurement_domain(cfg)
        self._root_render = self._meas_domain in ("amplitude", "log_amplitude")
        print(
            f"[MeasurementDomain] {self._meas_domain}: Ŷ = "
            f"{self._MEAS_G[self._meas_domain]}  (GT kept native; render mapped to it)"
        )

        parser, train_ds = self._setup_data()
        self._setup_normalization(train_ds)

        # ── seed positions (also sets the label bookkeeping) ─────────────────
        seed_pts = self._init_from_radar_cloud(parser)

        # ── Reflector params + optimisers (z_i = means, initialised from seed) ─
        self.params, self.optimizers = create_reflectors(
            seed_pts,
            init_alpha=cfg.init_alpha,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            sh_coeffs_lr=cfg.sh_coeffs_lr,
            opacities_lr=cfg.opacities_lr,
            sh_degree=cfg.sh_degree,
            device=str(self.device),
            # Point reflectors with a fixed hardware PSF: scales/quats are frozen by
            # default; train_scales/train_quats exist for the representation ablation.
            train_scales=bool(cfg.train_scales),
            train_quats=bool(cfg.train_quats),
            scales_lr=cfg.scales_lr,
            quats_lr=cfg.quats_lr,
        )
        print(f"[Init] {len(self.params['means'])} reflectors seeded from the radar cloud")

        self._setup_objects()

        Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)

        self._setup_psf()

        self._metrics: List[dict] = []
        self._train_log: List[dict] = []

        # sh_dc is pinned to 0: in ceiling units a full-opacity reflector then emits one
        # ceiling unit, and opacity is the per-reflector brightness.
        with torch.no_grad():
            self.params["sh_coeffs"][:, 0] = 0.0

        self._dens_cand: Optional[dict] = None  # residual-peak seed candidates

        # doppler_axis_weight reweights the Doppler-shape half of the plain L2; invalid
        # combinations are rejected rather than silently ignored.
        _wD_cfg = float(cfg.doppler_axis_weight)
        if _wD_cfg != 1.0:
            if _wD_cfg < 0.0:
                raise ValueError(f"doppler_axis_weight must be >= 0, got {_wD_cfg}")
            if int(cfg.num_doppler_bins) == 1:
                raise ValueError(
                    "doppler_axis_weight needs num_doppler_bins > 1 "
                    "(a Doppler-free sensor has no shape component)"
                )
            print(
                f"[Loss] doppler_axis_weight={_wD_cfg}: Doppler-SHAPE half of the L2 "
                f"reweighted (w_D=1 is the plain L2, w_D=0 a mean-over-D RA loss)"
            )

    # -----------------------------------------------------------------------
    # Setup steps of __init__, in call order
    # -----------------------------------------------------------------------

    def _setup_bins(self) -> None:
        """Render-grid bin centres (range, azimuth, elevation, Doppler) on the device."""
        cfg = self.cfg
        if cfg.radar_doppler_min_mps is None or cfg.radar_doppler_max_mps is None:
            raise ValueError("radar_doppler_min_mps / radar_doppler_max_mps must be set")
        r, az, el, dop = make_render_bins(cfg)
        self.range_bins = torch.from_numpy(r).float().to(self.device)
        self.az_bins = torch.from_numpy(az).float().to(self.device)
        self.el_bins = torch.from_numpy(el).float().to(self.device)
        self.doppler_bins = torch.from_numpy(dop).float().to(self.device)

    def _setup_data(self):
        """Parser, time axis, train/val loaders and the sensor-frame ego velocity.

        Returns (parser, train_ds).
        """
        cfg = self.cfg
        parser = RadarParser(
            rad_tensors_dir=cfg.rad_tensors_dir,
            poses_dir=cfg.radar_poses_dir,
            test_every=cfg.test_every,
            test_offset=cfg.test_offset,
            num_doppler_bins=cfg.num_doppler_bins,
            num_range_bins=cfg.num_range_bins,
            num_azimuth_bins=cfg.num_azimuth_bins,
            range_crop_first=cfg.range_crop_first,
            range_crop_last=cfg.range_crop_last,
            doppler_roll_bins=cfg.doppler_roll_bins,
            allow_missing_rad=bool(cfg.allow_missing_rad),
        )
        # The time axis, before anything reads a frame time.
        self._build_frame_times(len(parser.poses))
        # The datasets resolve the sequence's norm.json and scale GT power by 1/hi
        # (ceiling units).
        train_ds = RadarDataset(
            parser,
            split="train",
            frame_start=cfg.frame_start,
            frame_end=cfg.frame_end,
            bad_frame_ids=cfg.bad_frame_ids,
        )
        val_ds = RadarDataset(
            parser,
            split="val",
            frame_start=cfg.frame_start,
            frame_end=cfg.frame_end,
            bad_frame_ids=cfg.bad_frame_ids,
        )
        self.train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        # Ego velocity: self.v_ego_smooth[frame_idx] -> [2] x, y in the sensor frame (m/s),
        # precomputed by the preprocessing (ego_vel_npy).
        if not cfg.ego_vel_npy or not Path(cfg.ego_vel_npy).exists():
            raise FileNotFoundError(
                f"ego_vel_npy {cfg.ego_vel_npy!r} not found; the preprocessing writes it "
                f"next to the sequence's rad_tensors"
            )
        _v_sensor = load_ego_velocity(cfg.ego_vel_npy, len(parser.poses))
        print(f"[Ego] Loaded precomputed ego vel from {cfg.ego_vel_npy}: {_v_sensor.shape}")
        self.v_ego_smooth = torch.from_numpy(_v_sensor).to(self.device)  # [N, 2]
        return parser, train_ds

    def _setup_normalization(self, train_ds) -> None:
        """Ceiling normalization, noise pedestal and the two shared normalization maps.

        GT power is divided by the per-sequence scale `hi` (data max -> 1), the only
        per-sensor magnitude besides the floor. That is a linear rescaling (it commutes
        with the incoherent power sum and the PSF), matched on the render side by pinning
        sh_dc = 0 (each reflector emits at most opacity). The sequence's norm.json
        (resolved by the dataset) then gives:
          - floor_u, the noise pedestal b_s in ceiling units (self.noise_floor, also the
            rasterizer floor);
          - (lin_lo, lin_hi), the linear normalization of linear sensors (`lin_range`);
          - (norm_lo, norm_range), the normalized log-power map N used by the log-sensor
            loss, eval and viz (background -> N = 0, ceiling -> 1).
        """
        cfg = self.cfg
        _norm = train_ds._norm
        self.gt_hi_power = float(_norm.hi)
        _floor_u = float(_norm.params["floor_u"])
        _lo_log = float(_norm.params["norm_lo"])
        _hi_log = _lo_log + float(_norm.params["norm_range"])
        print(f"[Norm] N map from {_norm.source} (mode={_norm.mode})")
        self.noise_floor = torch.tensor(_floor_u, device=self.device, dtype=torch.float32)
        self.norm_floor_u = _floor_u
        self.norm_lo_log = _lo_log
        self.norm_hi_log = _hi_log
        if _floor_u <= 1e-8:
            print(
                f"[Norm] WARN floor_u={_floor_u:.3e} ≤ render powers.clamp"
                f"(min=1e-8) — dim reflectors may hit the clamp"
            )
        print(
            f"[Norm] hi={self.gt_hi_power:.4g} → gt_power_scale=1/hi; "
            f"floor_u={_floor_u:.4e}; N log-range=[{_lo_log:.3f}, {_hi_log:.3f}]"
        )

        # Printed diagnostic: log10 percentiles (norm_percentile_lo/hi) of the raw training
        # GT over a few frames. Nothing reads lo/hi; the normalization comes from norm.json.
        # Iterating the shuffled loader draws from torch's global RNG, so a fixed seed's
        # training-frame order includes these draws.
        log_vals = []
        for i, data in enumerate(self.train_loader):
            if i >= _NORM_PRINT_FRAMES:
                break
            raw = data["rad_tensor"].float()
            lv = torch.log10(raw.clamp(min=1e-10)).flatten()
            log_vals.append(lv)
        all_log = torch.cat(log_vals)
        if len(all_log) > _QUANTILE_MAX_ELEMS:
            idx = torch.randperm(len(all_log))[:_QUANTILE_MAX_ELEMS]
            all_log = all_log[idx]
        lo = float(torch.quantile(all_log, cfg.norm_percentile_lo / 100.0).item())
        hi = float(torch.quantile(all_log, cfg.norm_percentile_hi / 100.0).item())
        print(f"[Norm] log10 [{lo:.3f}, {hi:.3f}]  range={hi - lo:.3f}  linear_scaler={10**hi:.3e}")

        # Linear-space normalization: the norm.json (lo, hi) used by loss, eval and viz
        # via lin_range(); calling it here populates its cache so all consumers
        # use identical numbers.
        self._lin_range_cache = None
        lin_lo, lin_hi = self.lin_range()
        self.lin_lo = torch.tensor(lin_lo, device=self.device, dtype=torch.float32)
        self.lin_span = torch.tensor(
            max(lin_hi - lin_lo, 1e-6), device=self.device, dtype=torch.float32
        )
        # Shared clip for loss, eval and viz (1.0 = clip to [0, 1]; <= 0 = no clip).
        self.lin_clip_ceiling = float(cfg.norm_clip_ceiling)
        print(
            f"[Norm] linear (lo, hi)=({lin_lo:.4g}, {lin_hi:.4g})  "
            f"span={lin_hi - lin_lo:.4g}  "
            f"clip_ceiling={self.lin_clip_ceiling if self.lin_clip_ceiling > 0 else 'none'}"
            f"  (shared by loss/eval/viz)"
        )
        # Shared map N: (log10(x) - norm_lo) / norm_range, noise floor at 0, ceiling at 1.
        self.norm_lo = torch.tensor(self.norm_lo_log, device=self.device, dtype=torch.float32)
        self.norm_range = torch.tensor(
            max(self.norm_hi_log - self.norm_lo_log, 1e-6),
            device=self.device,
            dtype=torch.float32,
        )
        print(
            f"[Norm] log normalization range = shared N (floor@0, ceil@1): "
            f"norm_lo={self.norm_lo_log:.3f} "
            f"norm_range={self.norm_hi_log - self.norm_lo_log:.3f} "
            f"(floor_u={self.norm_floor_u:.4e})"
        )

    def _setup_objects(self) -> None:
        """Object index of every reflector, the rigid object tracks and their optimizer.

        Dynamic reflectors are the first _n_dynamic, tagged 1-indexed by object (one id
        per vehicle); static reflectors get 0. static_only and label-free runs have no
        object assignments (obj_idx None: everything static). Each object's rigid track
        is seeded from its annotation positions (`_track_targets`, (t_sec, world_pos[3],
        obj_id)) plus its p0 anchor; the Doppler refines the velocity.
        """
        cfg = self.cfg
        N = len(self.params["means"])
        self.obj_idx = None
        n_lbl = self._n_dynamic
        if n_lbl > 0:
            obj_idx_np = np.zeros(N, dtype=np.int64)
            obj_idx_np[:n_lbl] = self._dynamic_obj_ids.astype(np.int64) + 1
            self.obj_idx = torch.from_numpy(obj_idx_np).to(self.device)
            print(
                f"[Init:labels] {self._n_objects} object(s), "
                f"{n_lbl} dynamic reflectors assigned (1-indexed)"
            )
            print(
                f"[Init] {n_lbl}/{N} dynamic reflectors follow object tracks; "
                f"background {N - n_lbl} are static"
            )
        elif cfg.object_label_path and not cfg.static_only:
            # Labels exist but none map into the frame range: everything is static.
            print(
                "[Init:labels] labels present but no dynamic reflectors in this "
                "window — all reflectors static"
            )

        self.tracks = self._make_rigid_track(self._track_targets, self._label_seed_anchors)
        param_groups = [{"params": list(self.tracks.parameters()), "lr": cfg.deform_lr}]
        self.track_optimizer = torch.optim.Adam(param_groups)
        n_trk = sum(p.numel() for p in self.tracks.parameters())
        print(
            f"[Init:tracks] {len(self.tracks._ridx)} object track(s), "
            f"{n_trk} control-point params | seeded from "
            f"{len(self._track_targets)} annotation positions"
        )

    def _setup_psf(self) -> None:
        """The sensor PSF (None for the learned-extent ablation) and its optimizer.

        Learned PSF bandwidths (representation ablation): gradients only flow through the
        post-rasterisation conv path (psf_in_cuda: false); the CUDA PSF kernel receives
        the bandwidths as plain floats.
        """
        cfg = self.cfg
        if cfg.render_mode == "learned_extent":
            self.psf = None  # Gaussian primitives with learned extent (ablation)
        elif cfg.render_mode == "psf":
            self.psf = SensorPSF(
                max_kD=cfg.psf_max_k_D,
                max_kR=cfg.psf_max_k_R,
                max_kA=cfg.psf_max_k_A,
                init_k_D=cfg.psf_k_D,
                init_k_R=cfg.psf_k_R,
                init_k_A=cfg.psf_k_A,
                init_w_D=cfg.psf_w_D,
                init_w_R=cfg.psf_w_R,
                init_w_A=cfg.psf_w_A,
                init_strength=cfg.psf_strength_init,
                learnable=cfg.psf_learned,
            ).to(self.device)
        else:
            raise ValueError(f"render_mode={cfg.render_mode!r}; expected 'psf' or 'learned_extent'")
        self.psf_optimizer = None
        if self.psf is not None and cfg.psf_learned:
            if cfg.psf_in_cuda:
                raise ValueError(
                    "psf_learned: true needs psf_in_cuda: false; the CUDA PSF path passes "
                    "the bandwidths as floats, so no gradient would reach them"
                )
            _psf_learn = [p for p in self.psf.parameters() if p.requires_grad]
            if _psf_learn:
                self.psf_optimizer = torch.optim.Adam(_psf_learn, lr=cfg.psf_lr)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    @staticmethod
    def _infinite(loader):
        while True:
            yield from loader

    def train(self) -> None:
        cfg = self.cfg
        _write_provenance(cfg)
        # Runtime anchor for `_write_run_stats`. The CUDA peak-memory counter is not
        # reset here, so the reported peak covers the whole process (init + training).
        self._t_train_start = time.time()
        self._n_nonfinite_steps = 0

        loader = self._infinite(self.train_loader)
        pbar = tqdm(range(cfg.max_steps), desc="DyRAD")

        # Step-0 render, before any training; an empty render_save_steps disables it.
        if cfg.render_save_steps:
            self._save_render(0)

        for step in pbar:
            data = next(loader)
            c2w = data["radarpose"].to(self.device)
            gt_linear = data["rad_tensor"].to(self.device)
            # ── Y_t in the measurement domain, normalised by the shared map ─────────
            # (Config.measurement_domain). Same helper as eval and the persisted renders.
            gt_N = self._normalize_measurement(self._to_measurement(gt_linear))
            # `gt_log` is not gt_N: it is the normalised log map of the GT, the partner of
            # the `pred_log` view from `render_all`. The densification residual
            # (gt_log - pred_log).clamp(min=0) is defined on this pair on every sensor,
            # in normalised log units (`densify_residual_thresh`); mixing it with gt_N would
            # compare different units.
            gt_log = self._norm_log_map(gt_linear)
            t_frame = self._t_frame(data["frame_idx"][0])
            sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            for opt in self.optimizers.values():
                opt.zero_grad()
            self.track_optimizer.zero_grad()
            if self.psf_optimizer is not None:
                self.psf_optimizer.zero_grad()

            # Ego velocity for this frame (sensor frame, precomputed at init).
            _frame_idx = int(data["frame_idx"][0].item())
            v_ego_sensor_sm = self.ego_vel_sensor(_frame_idx)  # [1, 2]

            # Forward render of the training frame.
            pred_log, _raster_meta = self.render_all(
                c2w, sh_degree=sh_degree, t_frame=t_frame, v_ego_sensor=v_ego_sensor_sm
            )

            # ── Photometric term ──────────────────────────────────────────────
            # Ŷ_N and Y_N are the same quantity on every sensor: the render mapped to the
            # sensor's measurement domain at output (meta["pred_meas"]) and the GT in its
            # native domain, both through one normalisation (_normalize_measurement).
            #     L_rec = mean over every RAD bin of (Ŷ_N - Y_N)²
            # optionally with the Doppler-shape half reweighted by doppler_axis_weight
            # (w_D). The DC/shape split is exact (the shape is zero-mean over D), so
            # w_D == 1 is the plain L2 and w_D == 0 is the mean-over-Doppler RA loss. A
            # length-1 Doppler axis has no shape component.
            pred_N = self._normalize_measurement(_raster_meta["pred_meas"])
            loss = reconstruction_loss(pred_N, gt_N, float(cfg.doppler_axis_weight))

            # ── Interpolated-pose consistency: L = L_rec + lambda_int L_int ─────────
            _ic_w = float(cfg.interp_consist_weight)
            _ic_loss = self._interp_consistency_loss(sh_degree) if _ic_w > 0.0 else None
            if _ic_loss is None:
                _ic_loss = loss.new_tensor(0.0)
            else:
                loss = loss + _ic_w * _ic_loss

            dx_reg, v_reg = self._track_diagnostics(t_frame)

            if not torch.isfinite(loss):
                self._n_nonfinite_steps += 1
                pbar.set_postfix({"loss": "NaN"})
                continue

            loss.backward()

            # Gradient norms before clipping (for diagnosis)
            gn_sh = _grad_norm(self.params["sh_coeffs"])
            gn_track = _grad_norm_iter(self.tracks.parameters())
            gn_means = _grad_norm(self.params["means"])

            # Densification: accumulate residual-peak seed candidates, starting one
            # densify_every window before the first grow so the buffer covers exactly
            # one inter-grow window.
            if (
                step >= cfg.densify_start_step - cfg.densify_every
                and step < cfg.densify_stop_step
            ):
                self._densify_accum(pred_log.detach(), gt_log, gt_linear, c2w=c2w, t_frame=t_frame)

            self._mask_and_clip_grads(step)
            self._optimizer_step()

            postfix = self._log_step(
                step, loss, pred_N, gt_N, c2w, t_frame, _ic_loss, dx_reg, v_reg,
                gn_sh, gn_track, gn_means,
            )
            pbar.set_postfix(postfix)

            # Densification: seed new reflectors at residual peaks (coverage gaps).
            # _densify_grow consumes + clears the candidate buffer itself.
            if (
                step >= cfg.densify_start_step
                and step < cfg.densify_stop_step
                and (step + 1) % cfg.densify_every == 0
            ):
                self._densify_grow(step + 1)

            if (step + 1) in cfg.render_save_steps:
                self._save_render(step + 1)

            if (step + 1) in cfg.save_steps:
                self._save_checkpoint(step + 1)
                self._evaluate(step + 1)

        # End of the optimisation loop (`train_loop_seconds`). `wall_time_seconds` also
        # includes the final checkpoint and train-log plot below.
        self._t_train_loop_end = time.time()

        self._save_checkpoint("final")
        if self._train_log:
            _write_train_log(self._train_log, Path(cfg.result_dir) / "train_log.csv")
            _plot_train_log(self._train_log, Path(cfg.result_dir) / "train_loss_curves.png")
        self._write_run_stats()
        print("Training complete.")

    # -----------------------------------------------------------------------
    # Steps of one training iteration, in call order
    # -----------------------------------------------------------------------

    def _interp_consistency_loss(self, sh_degree: int):
        """L_int at a random pose between two adjacent training frames, or None without pairs.

        Unobserved-view regularisation: render at a random pose/time strictly between two
        adjacent train frames and supervise with the blurred mean of their GT RA maps.
        Built from the train split only. The blur is needed because the two-frame mean is
        only a valid target for the speckle-averaged reflectivity, not per bin.
        """
        _vp = self._interp_pose_targets()
        if not _vp:
            return None
        _f1, _f2, _c2w_1, _c2w_2, _gm_ra = _vp[int(torch.randint(len(_vp), (1,)).item())]
        # alpha in [0.25, 0.75]: stay near the pair's midpoint where the averaged
        # pseudo-GT is most valid.
        _alpha = 0.25 + 0.5 * float(torch.rand(()).item())
        _c2w_i = self._lerp_pose(_c2w_1, _c2w_2, _alpha)
        _t_i = self.t_of_frame(int(_f1)) + _alpha * (
            self.t_of_frame(int(_f2)) - self.t_of_frame(int(_f1))
        )
        # detach_tracks: the pseudo-GT is the mean of two frames, so moving objects appear
        # smeared across both positions; the object tracks are detached and this term
        # trains only the reflector parameters.
        _, _ic_meta = self.render_all(
            _c2w_i,
            sh_degree=sh_degree,
            t_frame=_t_i,
            v_ego_sensor=self.ego_vel_sensor(int(round(_f1 + _alpha * (_f2 - _f1)))),
            detach_tracks=True,
        )
        # Ŷ in the measurement domain, same normalisation as the photometric term
        _pred_i01 = self._normalize_measurement(_ic_meta["pred_meas"])
        _n_d = _pred_i01.shape[1]
        _pm_ra = _gaussian_blur_ra(
            _pred_i01.sum(dim=1, keepdim=True) / _n_d, INTERP_BLUR_SIGMA_BINS
        )
        return (_pm_ra - _gm_ra).abs().mean()

    @torch.no_grad()
    def _track_diagnostics(self, t_frame):
        """(displacement, velocity) of the dynamic reflectors under the current tracks (logged)."""
        _n_lbl_reg = self._n_dynamic
        _z_all = self.params["means"].detach()
        if _n_lbl_reg > 0:
            _z_reg = _z_all[:_n_lbl_reg]
            _obj_reg = self.obj_idx[:_n_lbl_reg] if self.obj_idx is not None else None
        else:
            _z_reg, _obj_reg = _z_all, self.obj_idx
        dx_reg = self.tracks(_z_reg, t_frame, _obj_reg)
        v_reg = self.tracks.velocity(_z_reg, t_frame, obj_idx=_obj_reg)
        return dx_reg, v_reg

    def _mask_and_clip_grads(self, step: int) -> None:
        """Zero the frozen gradient entries, sanitise the track gradients, clip."""
        cfg = self.cfg
        # Optional freeze for dynamic reflectors: sh_coeffs (sh_freeze_dynamic_steps),
        # so the photometric loss cannot dim them before the trajectory has settled.
        _n_lbl_sh = self._n_dynamic
        if _n_lbl_sh > 0:
            _sh_freeze_steps = int(cfg.sh_freeze_dynamic_steps)
            if _sh_freeze_steps > 0 and step < _sh_freeze_steps:
                if self.params["sh_coeffs"].grad is not None:
                    self.params["sh_coeffs"].grad[:_n_lbl_sh] = 0.0

        # Freeze sh_dc (0th SH coefficient) to pin the base magnitude: opacity
        # carries the learnable magnitude and higher SH coefficients carry
        # view dependence, removing the sh_dc/opacity redundancy.
        if self.params["sh_coeffs"].grad is not None:
            self.params["sh_coeffs"].grad[:, 0] = 0.0

        # Freeze z of means for all reflectors: elevation is unsupervised
        # (a single elevation bin), so means[:,2] would only drift on noise.
        if self.params["means"].grad is not None:
            self.params["means"].grad[:, 2] = 0.0

        # Replace inf/nan gradients with 0 so clip_grad_norm_ does not zero the
        # whole track model. The clip at max_norm 1.0 is a safety guard.
        for p in self.tracks.parameters():
            if p.grad is not None:
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        torch.nn.utils.clip_grad_norm_(list(self.params.values()), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.tracks.parameters(), max_norm=1.0)

    def _optimizer_step(self) -> None:
        """Step the reflector, track and PSF optimizers."""
        cfg = self.cfg
        for opt in self.optimizers.values():
            opt.step()
        # Bound the learnable extent when scales are trained (the rasterizer
        # clamps rendered sigma to [0.1, 10] bins, so unbounded primitives saturate
        # wide and smear).
        if cfg.train_scales and cfg.scales_clamp_max_m > 0:
            with torch.no_grad():
                self.params["scales"].data.clamp_(
                    min=math.log(_SCALES_CLAMP_MIN_M), max=math.log(cfg.scales_clamp_max_m)
                )
        self.track_optimizer.step()
        if self.psf_optimizer is not None:
            self.psf_optimizer.step()

    @torch.no_grad()
    def _log_step(
        self, step, loss, pred_N, gt_N, c2w, t_frame, ic_loss, dx_reg, v_reg,
        gn_sh, gn_track, gn_means,
    ) -> dict:
        """Step diagnostics: appends a train_log.csv row every `log_every` steps and
        returns the progress-bar postfix."""
        cfg = self.cfg
        rad_l1 = (pred_N - gt_N).abs().mean().item()
        vel_mag = v_reg.norm(dim=-1).mean().item()
        disp_mag = dx_reg.norm(dim=-1).mean().item()
        if cfg.sigma_static_bins > 0:
            _, dyn_mask = self._compute_doppler_gate(c2w, self._c2w_prev_can(c2w, t_frame))
            dyn_l1 = (
                (pred_N - gt_N)[dyn_mask.unsqueeze(0)].abs().mean().item()
                if dyn_mask.any()
                else float("nan")
            )
        else:
            dyn_l1 = float("nan")

        postfix = {
            "loss": f"{loss.item():.4f}",
            "rad_l1": f"{rad_l1:.3f}",
            "dyn_l1": f"{dyn_l1:.3f}",
            "interp": f"{ic_loss.item():.4f}",
            "v_mean": f"{vel_mag:.3f}",
            "dx_mean": f"{disp_mag:.3f}",
            "gn_sh": f"{gn_sh:.3f}",
            "gn_track": f"{gn_track:.3f}",
            "gn_z": f"{gn_means:.3f}",
            "N": len(self.params["means"]),
        }

        if cfg.log_every > 0 and (step + 1) % cfg.log_every == 0:
            self._train_log.append(
                {
                    "step": step + 1,
                    "loss": loss.item(),
                    "rad_l1": rad_l1,
                    "dyn_l1": dyn_l1,
                    "interp": ic_loss.item(),
                    "v_mean": vel_mag,
                    "dx_mean": disp_mag,
                    "n_pts": len(self.params["means"]),
                    "gn_track": gn_track,
                }
            )
            _write_train_log(self._train_log, Path(cfg.result_dir) / "train_log.csv")
        return postfix
