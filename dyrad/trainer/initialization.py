"""Scene initialization (paper B.1): the reflector parameters and optimizers, the labels,
the object tracks and the static/dynamic seed points.

`create_reflectors` turns the seed point cloud into the reflector ParameterDict (means,
quats, scales, opacities, SH reflectivity coefficients) together with one Adam optimizer
per trainable parameter group. `InitMixin` loads the labels, links detections into
tracks and seeds the static background and the dynamic objects.
"""

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from dyrad import ra_partition
from dyrad.config import held_out_frames, is_held_out
from dyrad.axes import range_axis_identity, sensor_axes
from dyrad.data import RadarParser
from dyrad.labels import load_index_remap, remap_label_index
from dyrad.tracks import RigidObjectTrack
from dyrad.paths import ROOT


def create_reflectors(
    points: np.ndarray,  # [N, 3]
    *,
    init_alpha: float,
    init_scale: float,
    means_lr: float,
    scales_lr: float,
    opacities_lr: float,
    quats_lr: float,
    sh_coeffs_lr: float,
    sh_degree: int,
    device: str,
    train_quats: bool,
    train_scales: bool,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Reflector parameters and their optimizers; means, opacities and SH are always trained.

    SH coefficients start at zero; the DC term stays pinned at 0 (the Runner zeroes its
    gradient), so opacity is the per-reflector brightness.
    """
    N = len(points)
    points_tensor = torch.from_numpy(points).float().to(device)

    # Isotropic init scales reduce directional smearing artifacts in RAD projections.
    base = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32).view(1, 3)
    scales = base.repeat(N, 1) * float(init_scale)
    scales = torch.log(scales.clamp(min=1e-6))

    # `init_alpha` is a probability: it is stored as a logit below, so every reflector
    # starts at an effective opacity of exactly init_alpha. Densification in the trainer
    # initialises new rows the same way.
    opacities = torch.full((N,), init_alpha, device=device)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(N, 1)

    num_sh_coeffs = (sh_degree + 1) ** 2
    sh_coeffs_full = torch.zeros(N, num_sh_coeffs, 1, device=device)

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points_tensor, requires_grad=True),
            "quats": torch.nn.Parameter(quats, requires_grad=bool(train_quats)),
            "scales": torch.nn.Parameter(scales, requires_grad=bool(train_scales)),
            "opacities": torch.nn.Parameter(
                torch.logit(opacities.clamp(1e-6, 1 - 1e-6)), requires_grad=True
            ),
            "sh_coeffs": torch.nn.Parameter(sh_coeffs_full, requires_grad=True),
        }
    )

    optimizers: Dict[str, torch.optim.Optimizer] = {}
    optimizers["means"] = torch.optim.Adam([params["means"]], lr=means_lr)
    if train_quats:
        optimizers["quats"] = torch.optim.Adam([params["quats"]], lr=quats_lr)
    if train_scales:
        optimizers["scales"] = torch.optim.Adam([params["scales"]], lr=scales_lr)
    optimizers["opacities"] = torch.optim.Adam([params["opacities"]], lr=opacities_lr)
    optimizers["sh_coeffs"] = torch.optim.Adam([params["sh_coeffs"]], lr=sh_coeffs_lr)
    return params, optimizers


def make_render_bins(cfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render-grid bin centres of the configured sensor.

    (range m, azimuth rad, elevation rad, Doppler m/s), float32: the cropped range and
    the azimuth and Doppler axes of `dyrad.axes.sensor_axes`, plus a single elevation
    bin at 0 (road plane). Doppler is not ego-compensated; ego motion enters the render
    through the poses.
    """
    # Range (cropped), azimuth in degrees and Doppler, all float32 (dyrad.axes).
    range_centers, az_deg, doppler_centers = sensor_axes(cfg, dtype=np.float32)
    az_centers = np.deg2rad(az_deg)

    # Elevation — road plane only
    el_centers = np.zeros(1, dtype=np.float32)

    return range_centers, az_centers, el_centers, doppler_centers


# ── Greedy label tracker (RADIal CSVs carry no vehicle_id) ─────────────────────────
# Camera veto: when both detections carry a camera bbox, a match is rejected if the bbox
# centre moves more than CAM_GATE_PX pixels per frame or the per-frame width ratio leaves
# the open interval CAM_WIDTH_RATIO.
CAM_GATE_PX = 60.0
CAM_WIDTH_RATIO = (0.5, 2.0)

# ── Dynamic seeding ────────────────────────────────────────────────────────────────
# A detection seeds its object only when the brightest Doppler bin at the label cell is
# above DOPPLER_PEAK_OVER_MEDIAN times that cell's median over Doppler (a clear return).
DOPPLER_PEAK_OVER_MEDIAN = 3.0
# Post-condition on an object's seeds: |xy| in the object frame stays below this (a car's
# footprint plus the PSF dilation, with slack); larger means a world-frame mix-up.
OBJECT_SEED_BOUND_M = 20.0
# An object whose returns yield no seed gets FALLBACK_SEED_COUNT points jittered by
# FALLBACK_SEED_JITTER_M around its origin, from a generator seeded with FALLBACK_SEED_RNG.
FALLBACK_SEED_COUNT = 5
FALLBACK_SEED_JITTER_M = 0.5
FALLBACK_SEED_RNG = 42

# Annotation quality, best first (`label_track_min_quality`); unknown values rank last.
_QUALITY_RANK = {"strong": 0, "weak": 1, "incomplete": 2}


def _split_key(test_every: int, test_offset: int) -> tuple:
    """(period, offset) of a holdout split; the offset is irrelevant without holdout."""
    return (test_every, test_offset) if test_every > 0 else (0, 0)


def _parse_label_rows(df, remap, frame_start: int, frame_end: int, has_vid: bool) -> dict:
    """Label rows inside [frame_start, frame_end) -> {frame_idx: [entry_dict]}.

    Empty-frame rows (radar_R_m <= 0, no detection) are skipped, as in
    `dyrad.labels.read_detections`."""
    by_frame: dict = {}
    for _, row in df.iterrows():
        fidx = remap_label_index(int(row["index"]), remap)
        if fidx < 0 or fidx < frame_start or (frame_end >= 0 and fidx >= frame_end):
            continue
        R_m = float(row["radar_R_m"])
        if R_m <= 0:
            continue
        entry = {
            "R_m": R_m,
            "A_deg": float(row["radar_A_deg"]),
            "annotation": str(row["Annotation"]) if "Annotation" in row.index else "strong",
            "radar_D_raw": int(float(row["radar_D_mps"])) if "radar_D_mps" in row.index else -1,
        }
        for px_col in ("x1_pix", "y1_pix", "x2_pix", "y2_pix"):
            if px_col in row.index:
                entry[px_col] = int(float(row[px_col]))
        # Boreas: the real box, already resolved to metres along/across range by
        # boreas_objects_to_labels.py. Absent on RADIal, which uses the camera bbox
        # (see ra_partition.car_extent_m).
        # The column can be present but empty, which pandas reads as NaN rather
        # than raising, so require a finite value.
        for m_col in ("box_half_cross_m", "box_half_range_m"):
            if m_col in row.index:
                try:
                    _v = float(row[m_col])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(_v):
                    entry[m_col.replace("box_", "")] = _v
        if has_vid:
            entry["vehicle_id"] = str(row["vehicle_id"])
        by_frame.setdefault(fidx, []).append(entry)
    return by_frame


def _bbox_of(e):
    """(centre x, centre y, width) of a detection's camera bbox in pixels, or None."""
    if all(k in e for k in ("x1_pix", "y1_pix", "x2_pix", "y2_pix")):
        w = float(e["x2_pix"] - e["x1_pix"])
        if w > 0:
            return (
                0.5 * (e["x1_pix"] + e["x2_pix"]),
                0.5 * (e["y1_pix"] + e["y2_pix"]),
                w,
            )
    return None


def _associate_detections(by_frame: dict, frames, poses, thresh: float, max_gap: int) -> int:
    """Greedy world-frame tracker: sets `vehicle_id` = "track_<n>" on every detection of
    `frames` and returns the number of tracks.

    A detection joins the closest active track (XY distance in the world frame; the radar
    has no elevation, and z = 0 rotates to spurious z under pose pitch/roll). A track is
    active while its last detection is at most `max_gap` frames back. Once a track has two
    detections, the gate is the error to its constant-velocity prediction (`thresh`,
    independent of speed and gap); a single-detection track allows `thresh` per gap frame.
    The camera veto (CAM_GATE_PX, CAM_WIDTH_RATIO) applies when both sides carry a bbox.
    """
    # track_id -> (last_frame_idx, last_world_pos [3], vel_xy [2] | None,
    #              last_bbox (cx, cy, w) | None)
    tracks: dict = {}
    n_tracks = 0
    for fidx in frames:
        for entry in by_frame[fidx]:
            # sensor position -> world frame. Uses radar (R, A): laser_X/Y are in the
            # LiDAR frame (y forward), unlike the projector's x-forward frame.
            az_r = np.radians(entry["A_deg"])
            R_r = entry["R_m"]
            p_sen = np.array([R_r * np.cos(az_r), R_r * np.sin(az_r), 0.0])
            if fidx >= len(poses):
                raise ValueError(
                    f"label frame {fidx} is beyond the pose table ({len(poses)} poses)"
                )
            c2w = poses[fidx].astype(np.float64)
            p_w = c2w[:3, :3] @ p_sen + c2w[:3, 3]

            bbox = _bbox_of(entry)
            best_tid, best_d = None, None
            for tid, (lf, lp, lv, lb) in tracks.items():
                gap = fidx - lf
                # gap <= 0: a track already matched this frame is a distinct
                # object; never merge (also avoids dividing by gap below).
                if gap <= 0 or gap > max_gap:
                    continue
                if lv is not None:
                    pred = lp[:2] + lv * gap
                    d = float(np.linalg.norm(p_w[:2] - pred))
                    gate = thresh
                else:
                    d = float(np.linalg.norm(p_w[:2] - lp[:2]))
                    gate = thresh * gap
                if d >= gate:
                    continue
                if bbox is not None and lb is not None:
                    d_px = math.hypot(bbox[0] - lb[0], bbox[1] - lb[1]) / gap
                    wr = (bbox[2] / lb[2]) ** (1.0 / gap)
                    if d_px > CAM_GATE_PX or not (CAM_WIDTH_RATIO[0] < wr < CAM_WIDTH_RATIO[1]):
                        continue
                if best_d is None or d < best_d:
                    best_d, best_tid = d, tid

            if best_tid is None:
                best_tid = n_tracks
                n_tracks += 1
                tracks[best_tid] = (fidx, p_w, None, bbox)
            else:
                lf, lp, _, _ = tracks[best_tid]
                vel = (p_w[:2] - lp[:2]) / max(fidx - lf, 1)  # m/frame
                tracks[best_tid] = (fidx, p_w, vel, bbox)
            entry["vehicle_id"] = f"track_{best_tid}"
    return n_tracks


def _emit_pass(track_targets, sg_dets, t_0, p_0, obj_id, max_rank, swap_speed) -> int:
    """One pass over an object's detections: skip track swaps (an XY speed from the last
    kept point above `swap_speed`) and append (t, p, obj_id) for the detections whose
    annotation quality rank is <= max_rank (None: any). Returns the number appended."""
    n_emitted = 0
    _prev_t, _prev_p = float(t_0), p_0.copy()
    for _, t_k, p_k, ann_k, _, _ in sg_dets:
        _dt_step = float(t_k) - _prev_t
        if _dt_step > 0:
            # XY-only distance: Z in world frame is unreliable for GPS poses.
            _speed = (
                float(np.linalg.norm(p_k.astype(np.float64)[:2] - _prev_p.astype(np.float64)[:2]))
                / _dt_step
            )
            if _speed > swap_speed:
                continue  # track swap: keep tracking from the last good point
        _prev_t, _prev_p = float(t_k), p_k.copy()
        if max_rank is None or _QUALITY_RANK.get(ann_k, 99) <= max_rank:
            track_targets.append((float(t_k), p_k.copy(), obj_id))
            n_emitted += 1
    return n_emitted


class InitMixin:
    def _load_labels_by_frame(self, seq_name: str, poses) -> dict:
        """Load the object-label CSV -> {frame_idx: [entry_dict]}.

        The 'index' column, remapped to the local frame (dyrad.labels), is the frame
        index. When the CSV has no 'vehicle_id' column (RADIal), the greedy tracker
        (`_associate_detections`) links the training-frame detections into tracks;
        held-out detections get no vehicle_id.

        poses: [N_frames, 4, 4] sensor-to-world, used by the tracker.
        """
        if not self.cfg.object_label_path:
            return {}
        import pandas as pd

        df = pd.read_csv(self.cfg.object_label_path)
        if "dataset" in df.columns:
            df = df[df["dataset"] == seq_name]
        if df.empty:
            print(f"[Init:labels] No labels found for seq '{seq_name}'")
            return {}

        # 'index' is the authors' frame-table position; remap it to the local frame.
        remap = load_index_remap(Path(self.cfg.rad_tensors_dir).parent)
        if remap is None and seq_name.startswith("RECORD@"):
            raise FileNotFoundError(
                f"label_index_remap.npy not found next to {self.cfg.rad_tensors_dir}; "
                "without it the raw label 'index' is off by the recorder's frame shift"
            )

        has_vid = "vehicle_id" in df.columns
        by_frame = _parse_label_rows(
            df, remap, int(self.cfg.frame_start), int(self.cfg.frame_end), has_vid
        )

        if not has_vid:
            thresh = float(self.cfg.label_assoc_threshold_m)
            max_gap = int(self.cfg.label_track_max_gap)
            # Only training frames are associated (paper B.1: the initialization uses
            # training-frame boxes only).
            train_frames = [
                f
                for f in sorted(by_frame)
                if not is_held_out(f, self.cfg.test_every, self.cfg.test_offset)
            ]
            n_tracks = _associate_detections(by_frame, train_frames, poses, thresh, max_gap)
            print(
                f"[Init:labels] greedy tracker: {n_tracks} track(s)  "
                f"thresh={thresh:.1f} m (CV-prediction gate)  max_gap={max_gap} frames"
            )

        n_det = sum(len(v) for v in by_frame.values())
        print(
            f"[Init:labels] {n_det} detections across {len(by_frame)} frames for seq '{seq_name}'"
        )
        return by_frame

    def _make_rigid_track(self, targets, seed_anchors):
        """Build a `RigidObjectTrack` from the config and labels alone.

        Held-out frames get no control point (a knot on a frame no loss renders would
        be unsupervised).
        """
        cfg = self.cfg
        return RigidObjectTrack(
            targets=targets,
            device=self.device,
            seed_anchors=seed_anchors,
            safety_frames=int(cfg.track_safety_frames),
            dt=float(cfg.dt),
            ctrl_interp_every=int(cfg.track_ctrl_interp_every),
            vel_fit_span=int(cfg.track_vel_fit_span),
            frame_times=self._frame_t,
            skip_knot_frames=held_out_frames(cfg),
            rotate_min_travel_m=float(cfg.track_rotate_min_travel_m),
        )

    def _seed_dynamic_from_returns(
        self,
        seg_dets,
        p0_world,
        parser,
        range_centers,
        az_deg,
        cube_cache,
        bbox_w=None,
        box_m=None,
    ):
        """Seed one object's reflectors from its Doppler-gated GT returns.

        For each labelled detection, take the object's Doppler slice (argmax over D at
        the label cell), keep bright R/A cells around the label, map them to world,
        and motion-compensate (desmear) to the canonical time t0 by shifting with
        (p0 - p_k). Aggregates across frames and voxel-dedups. Returns xy in the
        object's frame (origin p0) with z the world height of p0, [M, 3].
        The detections are training frames only (the caller drops held-out frames).

        cube_cache: {frame_idx: cube} to avoid reloading shared frames.
        """
        cfg = self.cfg
        _fx = ra_partition.load_fx(cfg, ROOT)
        # PSF dilation, in bins: an object's energy is its geometry convolved with the
        # beam, and the beam is an angle (see ra_partition).
        _pad_r, _pad_a = ra_partition.psf_margin_bins(cfg)
        bbox_w = bbox_w or {}
        box_m = box_m or {}
        frac = float(cfg.dyn_seed_peak_frac)
        nR, nA = len(range_centers), len(az_deg)
        seeds = []
        for fi, t_k, p_k, ann, R_m, A_deg in seg_dets:
            cube = cube_cache.get(fi)
            if cube is None:
                cube = np.load(parser.rad_tensor_files[fi]).astype(np.float32)  # [D,R,A]
                cube_cache[fi] = cube
            r0 = int(np.clip(np.argmin(np.abs(range_centers - R_m)), 0, nR - 1))
            a0 = int(np.clip(np.argmin(np.abs(az_deg - A_deg)), 0, nA - 1))
            # object Doppler bin = brightest Doppler at the label cell; it must stand
            # above the cell's own median over D (else no clear return: skip).
            # Without a Doppler axis (Boreas, D = 1) the RA map itself is thresholded.
            d0 = 0
            if cube.shape[0] > 1:
                col = cube[:, r0, a0]
                d0 = int(np.argmax(col))
                if col[d0] <= DOPPLER_PEAK_OVER_MEDIAN * (np.median(col) + 1e-30):
                    continue
            # this object's own footprint, metres -> bins (ra_partition)
            _bm = box_m.get(int(fi)) or (None, None)
            _hc, _hr = ra_partition.car_extent_m(
                bbox_w.get(int(fi)), R_m, _fx, cfg, _bm[0], _bm[1]
            )
            r_lo, r_hi, a_lo, a_hi = ra_partition.footprint_bins(
                R_m, A_deg, _hc, _hr, range_centers, az_deg, _pad_r, _pad_a
            )
            sub = cube[d0, r_lo:r_hi, a_lo:a_hi]  # object Doppler slice
            peak = float(sub.max())
            if peak <= 0:
                continue
            rr, aa = np.where(sub > frac * peak)
            if len(rr) == 0:
                continue
            rb = r_lo + rr
            ab = a_lo + aa
            Rm = range_centers[rb]
            Az = np.deg2rad(az_deg[ab])
            x = Rm * np.cos(Az)
            y = Rm * np.sin(Az)
            p_cam = np.stack([x, y, np.zeros_like(x)], axis=1)  # [k, 3] sensor
            c2w = parser.poses[fi].astype(np.float64)
            p_world = (c2w[:3, :3] @ p_cam.T).T + c2w[:3, 3]  # [k, 3] world
            # desmear: bring this frame's object-relative structure to the t0 anchor
            p_world += p0_world.astype(np.float64) - p_k.astype(np.float64)
            seeds.append(p_world.astype(np.float32))
        if not seeds:
            return np.zeros((0, 3), np.float32)
        pts = np.concatenate(seeds, axis=0)
        # voxel-dedup so overlapping per-frame returns collapse to one cluster
        vx = float(cfg.dyn_seed_voxel_m)
        keep = np.unique(np.round(pts[:, :2] / vx).astype(np.int64), axis=0, return_index=True)[1]
        _n_raw = len(pts)
        pts = pts[keep]
        print(
            f"    [Init] {_n_raw} raw returns over {len(seeds)} frame(s) -> "
            f"{len(pts)} after {vx:.2f} m dedup "
            f"({100.0 * len(pts) / max(_n_raw, 1):.1f}% survive)"
        )
        pts[:, 2] = float(p0_world[2])  # planar z = anchor z
        # Re-express the cloud in the object's frame. Origin = p0, the point the desmear
        # above brought every frame to.
        pts[:, :2] -= p0_world.astype(np.float32)[None, :2]
        return pts

    def _set_label_bookkeeping(
        self, labels_by_frame, track_targets, n_dynamic, dynamic_obj_ids, n_objects, anchors
    ) -> None:
        """The label state every `_init_from_radar_cloud` path leaves on the Runner."""
        self._labels_by_frame = labels_by_frame
        self._track_targets = track_targets
        self._n_dynamic = n_dynamic
        self._dynamic_obj_ids = dynamic_obj_ids
        self._n_objects = n_objects
        self._label_seed_anchors = anchors

    def _init_from_radar_cloud(self, parser: RadarParser) -> np.ndarray:
        """Seed reflectors from the radar pseudo-LiDAR cloud plus annotated objects.

        Static reflectors come from the free-space-partitioned cloud (the objects are
        already carved out, `static_from_free_space`). Dynamic reflectors are seeded
        per object from its own returns inside its annotated regions, first in the
        returned array.

        Returns pts [N,3]. Sets the label bookkeeping attributes
        (`_set_label_bookkeeping`).
        """
        cfg = self.cfg
        pts_cloud = self._load_init_cloud()
        labels_by_frame = self._load_labels_by_frame(cfg.seq_name, poses=parser.poses)

        # Static-only baseline: every cloud point becomes a static reflector (moving
        # objects are not separated). Labels stay loaded so it is evaluated against the
        # same labels as the dynamic model.
        if cfg.static_only:
            print(
                f"[Init] static_only: ALL {len(pts_cloud)} cloud points "
                f"STATIC (no dynamic/static separation; car ghosts as static clutter). "
                f"Labels kept for eval."
            )
            self._set_label_bookkeeping(labels_by_frame, [], 0, None, 0, [])
            return pts_cloud

        if not labels_by_frame:
            print("[Init] No labels — all points treated as static")
            self._set_label_bookkeeping(labels_by_frame, [], 0, None, 0, [])
            return pts_cloud

        if not cfg.static_from_free_space:
            raise ValueError(
                "labelled objects need static_from_free_space: true and a cloud built with "
                "--partition (the objects carved out of the static cloud); only the "
                "static-only variant (static_only: true) runs on an unpartitioned cloud"
            )

        vid_dets, vid_bbox_w, vid_box_m = self._collect_detections(labels_by_frame, parser)
        objects, track_targets, anchors = self._track_targets_and_anchors(vid_dets)
        label_pts, label_obj_ids = self._seed_objects(objects, parser, vid_bbox_w, vid_box_m)
        n_label = len(label_pts)

        # Static pool: the partitioned cloud. The cloud was built with --partition: each
        # object was already removed from the frame it was in, so there is no trajectory
        # trail to exclude.
        self._check_partition_sidecar()
        static_pts = pts_cloud

        lp = (
            np.stack(label_pts, axis=0).astype(np.float32)
            if label_pts
            else np.zeros((0, 3), np.float32)
        )
        pts_out = np.concatenate([lp, static_pts], axis=0)

        print(f"[Init] {n_label} cluster + {len(static_pts)} static = {len(pts_out)} total")
        print(f"[Init] {len(track_targets)} track targets across {len(objects)} objects")

        self._set_label_bookkeeping(
            labels_by_frame,
            track_targets,
            n_label,
            np.array(label_obj_ids, np.int32) if label_obj_ids else None,
            len(objects),
            anchors,
        )
        return pts_out

    def _load_init_cloud(self) -> np.ndarray:
        """The static seed cloud `init_cloud_path` [N, 3], after checking its holdout split."""
        cfg = self.cfg
        path = Path(cfg.init_cloud_path)
        if not path.exists():
            raise FileNotFoundError(
                f"init_cloud_path not found: {path}\n"
                f"Build the radar pseudo-cloud with "
                f"python -m dyrad.preprocessing.build_init_cloud --config <cfg>"
            )
        pts_cloud = np.load(path).astype(np.float32)[:, :3]  # [N, 3] xyz (drop extra columns)
        self._check_cloud_split(path)
        print(f"[Init] {len(pts_cloud)} seed pts from {path.name} (radar point cloud)")
        return pts_cloud

    def _check_cloud_split(self, path: Path) -> None:
        """The cloud's holdout and range axis must match the run's.

        The builder skips the held-out frames when it back-projects bright bins and records
        the split and the range axis in `<cloud>.build.json`. A cloud built for another split
        would put this run's validation returns into the init, and one back-projected on
        another range axis (e.g. another range_bin_offset) would seed every reflector at the
        wrong range, so both are refused.
        """
        cfg = self.cfg
        _test_every, _test_offset = int(cfg.test_every), int(cfg.test_offset)
        _bj = Path(str(path) + ".build.json")
        if _bj.is_file():
            _b = json.loads(_bj.read_text())
            _bk, _bo = int(_b.get("test_every", 0)), int(_b.get("test_offset", 0) or 0)
            if _split_key(_bk, _bo) != _split_key(_test_every, _test_offset):
                raise RuntimeError(
                    f"init_cloud_path {path.name} was built for test_every/offset "
                    f"{_bk}/{_bo} (per {_bj.name}) but this run uses "
                    f"{_test_every}/{_test_offset} -- the cloud would leak validation frames "
                    f"into the init. Rebuild it for this split "
                    f"(python -m dyrad.preprocessing.build_init_cloud --config <this config>)."
                )
            _axis, _want = _b.get("range_axis"), range_axis_identity(cfg)
            if _axis != _want:
                raise RuntimeError(
                    f"init_cloud_path {path.name} was back-projected on the range axis "
                    f"{_axis} (per {_bj.name}) but this run uses {_want}: every seed point "
                    f"would sit at the wrong range. Rebuild the cloud with this config "
                    f"(python -m dyrad.preprocessing.build_init_cloud --config <this config>)."
                )
        else:
            print(f"[Init] WARNING: {_bj.name} missing; the cloud's holdout split is unchecked")

    def _collect_detections(self, labels_by_frame: dict, parser: RadarParser):
        """Training-frame detections grouped by vehicle, in frame order.

        Returns (vid_dets, vid_bbox_w, vid_box_m):
          vid_dets   vehicle_id -> [(frame_idx, t_sec, p_world, annotation, R_m, A_deg)]
          vid_bbox_w vehicle_id -> {frame_idx: camera bbox width px} (object size w·R/fx)
          vid_box_m  vehicle_id -> {frame_idx: (half_cross_m, half_range_m)} (Boreas box)
        The initialization sees training-frame boxes only (paper B.1).
        """
        cfg = self.cfg
        _test_every, _test_offset = int(cfg.test_every), int(cfg.test_offset)
        vid_dets: dict = {}
        vid_bbox_w: dict = {}
        vid_box_m: dict = {}
        for frame_idx, lbl_list in sorted(labels_by_frame.items()):
            if is_held_out(frame_idx, _test_every, _test_offset):
                continue
            t_sec = self.t_of_frame(frame_idx)
            c2w = parser.poses[frame_idx].astype(np.float64)
            for lbl in lbl_list:
                vid = lbl["vehicle_id"]
                # Use radar (R, A), not laser_X/Y: laser_X/Y are in the LiDAR frame
                # (y forward) while the projector uses atan2(y, x) with x forward.
                az_r = np.radians(lbl["A_deg"])
                p_cam = np.array(
                    [lbl["R_m"] * np.cos(az_r), lbl["R_m"] * np.sin(az_r), 0.0],
                    np.float64,
                )
                p_world = (c2w[:3, :3] @ p_cam + c2w[:3, 3]).astype(np.float32)
                ann = lbl["annotation"]
                # Also store sensor-frame (R, A) for the GT Doppler lookup
                lbl_R = float(lbl["R_m"])
                lbl_A = float(lbl["A_deg"])
                vid_dets.setdefault(vid, []).append((frame_idx, t_sec, p_world, ann, lbl_R, lbl_A))
                if all(k in lbl for k in ("x1_pix", "x2_pix")):
                    _w_px = float(lbl["x2_pix"] - lbl["x1_pix"])
                    if _w_px > 0:
                        vid_bbox_w.setdefault(vid, {})[frame_idx] = _w_px
                # Boreas labels carry the real box, already resolved to metres along and
                # across range. Preferred over the bbox path (ra_partition.car_extent_m).
                if "half_cross_m" in lbl and "half_range_m" in lbl:
                    vid_box_m.setdefault(vid, {})[frame_idx] = (
                        float(lbl["half_cross_m"]),
                        float(lbl["half_range_m"]),
                    )
        return vid_dets, vid_bbox_w, vid_box_m

    def _track_targets_and_anchors(self, vid_dets: dict):
        """Objects, their track targets and their seed anchors.

        Returns (objects, track_targets, anchors): objects [{obj_id, vid, dets,
        seed_world, t0}], track_targets [(t_sec, world_pos[3], obj_id)] and anchors
        [(t0, p0, obj_id)]. p0, the object's first training-frame annotation, is the
        object frame's origin. No initial velocity is estimated: a finite difference of
        two noisy label positions is far noisier than the true velocity, which the track
        provides.
        """
        cfg = self.cfg
        objects: list = []
        track_targets: list = []
        min_rank = _QUALITY_RANK[cfg.label_track_min_quality]
        # Track-swap speed threshold: physical max is ~30 m/s for highway vehicles, and a
        # label swap causes an apparent speed far above 50 m/s in one frame step.
        swap_speed = float(cfg.label_track_swap_thresh_mps)

        for vid, dets in vid_dets.items():
            dets.sort(key=lambda x: x[0])
            obj_id = len(objects)
            _, t_0, p_0, _, _, _ = dets[0]
            objects.append(
                {"obj_id": obj_id, "vid": vid, "dets": dets, "seed_world": p_0, "t0": float(t_0)}
            )
            # dets[0] is the track-swap reference. Targets of the requested quality; any
            # quality when none reaches it.
            if _emit_pass(track_targets, dets[1:], t_0, p_0, obj_id, min_rank, swap_speed) == 0:
                _emit_pass(track_targets, dets[1:], t_0, p_0, obj_id, None, swap_speed)

        # The object frame's origin must be a control point. A dynamic `means` is measured
        # from p0 (`seed_world`), a radar detection (`radar_R_m`/`radar_A_deg`). The render
        # is x = R_j(t)·means + P_j(t), so the cloud sits where the radar saw it only if
        # P_j(t0) = p0 at initialisation. The targets above start at dets[1], so p0 enters
        # as a seed anchor.
        anchors = [
            (float(o["t0"]), np.asarray(o["seed_world"], dtype=np.float32), int(o["obj_id"]))
            for o in objects
        ]
        print(
            f"[Init:tracks] p0 (= first training-frame annotation) pinned as a control "
            f"point for {len(objects)} object(s)"
        )
        return objects, track_targets, anchors

    def _seed_objects(self, objects: list, parser: RadarParser, vid_bbox_w: dict, vid_box_m: dict):
        """Per-object seeding from the object's own returns (paper B.1).

        Returns (label_pts, label_obj_ids): one [3] object-frame point and its obj_id per
        dynamic reflector, objects in order.
        """
        cfg = self.cfg
        rng_cl = np.random.default_rng(FALLBACK_SEED_RNG)
        label_pts_cl: list = []
        label_obj_ids_cl: list = []
        # Uncropped float64 range / azimuth(deg) axes of the full cube.
        _range_centers_full, _az_deg_full, _ = sensor_axes(cfg, crop=False, doppler=False)
        _cube_cache: dict = {}

        for o in objects:
            oid = o["obj_id"]
            center_k = o["seed_world"]  # [3] p0, first train-frame world position
            t_0_k = float(o["t0"])
            vid = o["vid"]
            _n0_obj = len(label_pts_cl)

            # The object's GT returns inside its region across all train frames (its
            # Doppler slice on RADIal, the RA map on Boreas), desmeared to t0 and
            # expressed in the object frame (origin p0).
            _seed_pts = self._seed_dynamic_from_returns(
                o["dets"],
                np.asarray(center_k, np.float32),
                parser,
                _range_centers_full,
                _az_deg_full,
                _cube_cache,
                bbox_w=vid_bbox_w.get(vid, {}),
                box_m=vid_box_m.get(vid, {}),
            )
            if len(_seed_pts) == 0:
                # No clear return: a few jittered points around the object-frame origin.
                _c_fb = np.array([0.0, 0.0, float(center_k[2])], np.float32)
                _seed_pts = _c_fb[None, :] + rng_cl.normal(
                    0, FALLBACK_SEED_JITTER_M, (FALLBACK_SEED_COUNT, 3)
                ).astype(np.float32)
                _seed_pts[:, 2] = float(center_k[2])
            for _pt in _seed_pts:
                label_pts_cl.append(_pt.astype(np.float32))
                label_obj_ids_cl.append(oid)
            print(
                f"  [Init] obj{oid} ({vid}): {len(_seed_pts)} seeds from the returns of "
                f"{len(o['dets'])} frames  t_0={t_0_k:.1f}s"
            )

            # Post-condition: under the object frame `means` is an on-object offset.
            # Trips only on a world-vs-object-frame mix-up.
            if len(label_pts_cl) > _n0_obj:
                _seg = np.asarray(label_pts_cl[_n0_obj:], np.float32)
                _mx = float(np.abs(_seg[:, :2]).max())
                if _mx > OBJECT_SEED_BOUND_M:
                    raise RuntimeError(
                        f"obj{oid}'s seeds reach "
                        f"|xy| = {_mx:.2f} m > {OBJECT_SEED_BOUND_M:.0f} m. "
                        f"`means` looks like a WORLD position, not an "
                        f"on-car offset -- the object would render at ~2x its true "
                        f"position. A seeding branch is not applying the object "
                        f"frame."
                    )
        return label_pts_cl, label_obj_ids_cl

    def _check_partition_sidecar(self) -> None:
        """The static and dynamic halves must be carved with the same footprint.

        The --partition builder records the footprint in `<cloud>.partition.json`; every
        key of `ra_partition.identity(cfg)` must be present there and equal.
        """
        cfg = self.cfg
        _side = Path(str(cfg.init_cloud_path) + ".partition.json")
        if not _side.is_file():
            raise ValueError(
                f"static_from_free_space is on but {_side.name} is missing, so "
                f"{Path(cfg.init_cloud_path).name} was not built with --partition. "
                "Rebuild it with --partition."
            )
        _want = ra_partition.identity(cfg)
        _got = json.loads(_side.read_text())
        _bad = {
            k: (v, _got.get(k))
            for k, v in _want.items()
            if k not in _got or abs(float(_got[k]) - v) > 1e-9
        }
        if _bad:
            raise ValueError(
                f"init cloud {_side.stem} was carved with a different "
                f"footprint than this config asks for: "
                + ", ".join(f"{k}: cloud={g} config={w}" for k, (w, g) in _bad.items())
                + ". Rebuild it with --partition using THIS config, or the "
                "static and dynamic halves will not be complementary."
            )
