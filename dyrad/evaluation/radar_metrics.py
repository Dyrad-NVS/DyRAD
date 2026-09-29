"""Radar evaluation metric library (pure numpy; no torch/gsplat imports).

The reconstruction, Doppler and selection metrics the paper reports (Sec. 4.1,
Appendix C.1); the detection metrics are in `pointcloud_metrics`:

  Reconstruction : PSNR, SSIM and Pearson correlation on RA, RD and RAD, over the
                   full measurement and over the object region (`*_obj`), plus
                   LPIPS on the full RA and RAD measurements, all on the linear
                   normalization below. For a log-domain sensor (Boreas) the same
                   RA metrics are also emitted in the normalized-log map N (`*_N`).
  Doppler        : label_doppler_metrics -- wrap-aware Doppler peak error at the
                   annotated vehicles, in bins, with the ego-static band removed
                   (`lbl_dop_peak_mae_bins_dyn`).
  Selection      : peak_detection_metrics (NMS peak F1), the criterion of the
                   hyperparameter selection in Appendix B.2, and the scale
                   diagnostic `alpha`.

Conventions, stated once for the metric modules:
  * Inputs are numpy arrays in the sensor's linear domain (not log). For RADIal
    that is the beamformed linear amplitude |.| the preprocessing writes; the
    detector thresholds amplitude, see `pointcloud_metrics`.
  * RAD tensors are [D, R, A]. RA and RD images are their means over Doppler and
    over azimuth (`ra_project`, `rd_project`). Pred and GT share the renderer's
    Doppler axis.
  * Linear normalization: clip((x - lo) / (hi - lo), 0, lin_clip_ceiling) on one
    fixed range per view (`_lin_lohi`, `_norm_lin_nc`): the cube's `lin_norm`
    for RAD, and per-marginal ranges for RA and RD when configured.
  * The N map: `norm_log01`.

Default peak parameters:
  nms_radius=3   -- 7x7-bin NMS window, about one PSF mainlobe
  noise_factor=4 -- peak threshold = 4x the positive-bin median of the GT
  tol_r=3        -- 0.6 m ~ 3x range resolution
  tol_a=5        -- 1.0 deg ~ physical azimuth resolution (grid is 5x oversampled)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as _ssim

from dyrad.axes import wrap_bins_signed

_EPS = 1e-10


@dataclass
class MetricParams:
    """Every constant of the evaluation protocol. score_renders serializes it into
    metrics_extended.json ("params") for provenance."""

    nms_radius: int = 3  # NMS half-window (bins, both axes)
    noise_factor: float = 4.0  # peak threshold = noise_factor * median(positive GT)
    tol_r: int = 3  # peak match tolerance, range bins
    tol_a: int = 5  # peak match tolerance, azimuth bins
    ssim_win: int = 7  # SSIM window (odd)
    sig_quantile: float = 0.99  # signal bins (GT > this quantile) for the scale alpha
    # lin_clip_ceiling: clip on linear-normalized values (units of hi == 1.0),
    # applied identically in loss/eval/viz. 1.0 = clip to [0, 1] before PSNR/SSIM;
    # <= 0 disables clipping. Must match the trainer's norm_clip_ceiling so the
    # loss and the metric see the same values.
    lin_clip_ceiling: float = 1.0
    # marginal_lin_norm: {view: (lo, hi)} per-marginal normalization ranges in the
    # units of the scored arrays (see `_lin_lohi`); None -> `lin_norm` for every view.
    marginal_lin_norm: Optional[dict] = None
    # n_norm: (floor_u, lo_log, hi_log) of the N map (`norm_log01`), from the
    # sequence's norm.json. When set, the RA metrics are also emitted in N (`*_N`).
    n_norm: Optional[tuple] = None
    doppler_wrap_mps: float = 1.7968  # 16 x 0.1123 m/s DDMA period; score_renders overrides from the run config
    lbl_win_r: int = 2  # label-cell pooling half-window (range bins)
    lbl_win_a: int = 4  # label-cell pooling half-window (azimuth bins)
    # Half-width (bins, circular) of the ego-static Doppler band removed from both
    # label profiles before the Doppler peak error (`lbl_dop_peak_mae_bins_dyn`).
    lbl_static_excl_bins: float = 1.5
    # -- object region of the reconstruction metrics (`*_obj`, region_masks) --
    # A fixed box around each label centroid; not the per-object footprint of the
    # foreground point-cloud boxes (pointcloud_metrics.radargen_box_metrics).
    obj_half_r_m: float = 2.5  # object-box half-extent, range (metres)
    obj_half_a_deg: float = 2.5  # object-box half-extent, azimuth (degrees)
    # -- radar point cloud (pointcloud_metrics.radar_point_cloud) --
    # RADIal SignalProcessing/rpl.py method='PC': CA-CFAR on the Range-Doppler map
    # (thresholded on the azimuth-summed amplitude, not power), one azimuth per surviving RD
    # peak. RADIal params: est-window +-9 / guard +-3 in (range, Doppler);
    # threshold 2 dB.
    pc_cfar_win: tuple = (9, 9, 3, 3)  # (win_R, win_D, guard_R, guard_D) half-widths on [R, D]
    pc_cfar_threshold_db: float = 2.0  # CA-CFAR threshold above local noise (dB)
    # Object-level vehicle detection recall: a labelled vehicle is detected if a
    # point lies within pc_obj_gate_m (xy, metres).
    pc_obj_gate_m: float = 2.0  # vehicle-detection association gate (m)
    # -- RadarGen point-cloud protocol (Borreda et al., arXiv 2512.17897;
    # github.com/tomerborreda/RadarGen/tree/master/evaluation) --
    # delta_loc: IoU@1m / DA location gate. Their `pcl_f1_iou_np` takes np.sqrt() of
    # distances that are already Euclidean, so it gates at d < tau**2; that equals
    # d < tau only at tau = 1.
    pc_match_tau_m: float = 1.0
    # RadarGen's delta_Doppler = 2.5 m/s exceeds the whole unambiguous span of this
    # sensor: the DDMA axis wraps with period 16 x 0.1123 = 1.7968 m/s (max
    # wrap-aware separation 0.8984; axis -0.8984 .. +0.7861, DC at bin D/2), so that
    # gate would admit every velocity. We use 10% of the period (~1.6 bins, admits
    # ~20% of uniformly random Doppler) and compare wrap-aware.
    pc_da_dop_gate_mps: float = 0.1797  # 10% of the 1.7968 m/s DDMA period
    # RadarGen's delta_rcs = 8 dBsm assumes calibrated RCS. With uncalibrated power
    # the same 8 dB is applied to the pred/GT ratio, which is scale-invariant.
    pc_da_pow_gate_db: float = 8.0
    # MMD: multi-scale RBF, K kernels spaced by mul_factor around the mean pairwise
    # squared distance (RadarGen's bandwidth heuristic). Clouds are subsampled to
    # pc_mmd_max_pts with a fixed seed, since the kernel matrix is O(n^2).
    pc_mmd_kernels: int = 5
    pc_mmd_mul: float = 2.0
    pc_mmd_max_pts: int = 2000
    pc_mmd_seed: int = 0
    # CD-Full's power axis: dB relative to the GT cloud's median (method-independent,
    # so a shared normalization range), clipped to this range and mapped to [0, 1] -- the analogue
    # of RadarGen's fixed (rcs_min, rcs_max) on calibrated dBsm.
    pc_pow_range_db: tuple = (-20.0, 40.0)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def optimal_scale(
    pred: np.ndarray, gt: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """Optimal L2 scale: alpha = <pred, gt> / <pred, pred> (over mask)."""
    if mask is not None:
        pred, gt = pred[mask], gt[mask]
    denom = float(np.dot(pred.ravel(), pred.ravel()))
    if denom < _EPS:
        return 1.0
    return float(np.dot(pred.ravel(), gt.ravel())) / denom


def scale_alpha(
    pred_ra: np.ndarray, gt_ra: np.ndarray, p: MetricParams = MetricParams()
) -> dict:
    """Global scale of pred against GT over the GT signal bins (top 1 - sig_quantile).

    Not a quality metric: a domain blunder detector. A healthy render sits at O(1);
    a prediction saved in a different domain than the GT (raw power vs ceiling
    units) drives it orders of magnitude out (score_renders.domain_violations)."""
    sig_lin = gt_ra > np.quantile(gt_ra, p.sig_quantile)
    return {"alpha": optimal_scale(pred_ra, gt_ra, mask=sig_lin)}


# ---------------------------------------------------------------------------
# 1. Peak detection (the selection criterion of Appendix B.2)
# ---------------------------------------------------------------------------


def nms_peaks(img: np.ndarray, nms_radius: int, threshold: float) -> np.ndarray:
    """NMS local maxima above an absolute threshold.

    mode="nearest" avoids border artifacts. Returns [N, 3] float array: (r, a, amplitude).
    """
    size = 2 * int(nms_radius) + 1
    local_max = maximum_filter(img, size=size, mode="nearest")
    mask = (img == local_max) & (img > threshold)
    r, a = np.where(mask)
    return (
        np.stack(
            [r.astype(np.float64), a.astype(np.float64), img[r, a].astype(np.float64)],
            axis=1,
        )
        if len(r)
        else np.zeros((0, 3), dtype=np.float64)
    )


def match_peaks(
    pred_ra: np.ndarray, gt_ra: np.ndarray, p: MetricParams = MetricParams()
):
    """Extract + match NMS peaks. Returns (gt_pk, pd_pk, matches).

    matches is a list of (pred_idx, gt_idx) pairs within the elliptic gate
    (Δr/tol_r)² + (Δa/tol_a)² ≤ 1, solved by Hungarian assignment. The
    detection threshold is derived from GT only and applied identically to
    pred: an absolute scale on pred is part of the task (global calibration
    is reported separately as `alpha` by scale_alpha).
    """
    positive = gt_ra[gt_ra > 0]
    if positive.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    threshold = p.noise_factor * float(np.median(positive))
    gt_pk = nms_peaks(gt_ra, p.nms_radius, threshold)
    pd_pk = nms_peaks(np.clip(pred_ra, 0, None), p.nms_radius, threshold)
    return gt_pk, pd_pk, match_point_sets(pd_pk, gt_pk, p)


def match_point_sets(
    pd_pk: np.ndarray, gt_pk: np.ndarray, p: MetricParams = MetricParams()
):
    """Hungarian match two [N, >=2] bin-space point sets (cols 0=r_bin, 1=a_bin)
    within the elliptic gate (Δr/tol_r)² + (Δa/tol_a)² ≤ 1.

    Returns a list of (pred_idx, gt_idx) pairs."""
    if len(pd_pk) == 0 or len(gt_pk) == 0:
        return []
    dr = (pd_pk[:, None, 0] - gt_pk[None, :, 0]) / p.tol_r
    da = (pd_pk[:, None, 1] - gt_pk[None, :, 1]) / p.tol_a
    cost = dr**2 + da**2
    cost_gated = np.where(cost <= 1.0, cost, 1e6)
    ri, ci = linear_sum_assignment(cost_gated)
    return [(int(i), int(j)) for i, j in zip(ri, ci) if cost[i, j] <= 1.0]


def peak_detection_metrics(
    pred_ra: np.ndarray, gt_ra: np.ndarray, p: MetricParams = MetricParams()
) -> dict:
    """Match NMS peaks of pred vs GT within (tol_r, tol_a) → precision / recall / F1."""
    if not (gt_ra > 0).any():
        return {
            "peak_precision": float("nan"),
            "peak_recall": float("nan"),
            "peak_f1": float("nan"),
            "n_gt_peaks": 0,
            "n_pred_peaks": 0,
        }
    gt_pk, pd_pk, matches = match_peaks(pred_ra, gt_ra, p)
    n_gt, n_pd = len(gt_pk), len(pd_pk)
    tp = len(matches)

    precision = tp / n_pd if n_pd else float("nan")
    recall = tp / n_gt if n_gt else float("nan")
    if not (math.isnan(precision) or math.isnan(recall)) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        # A detector that predicts nothing against real GT peaks has recall 0 and F1 0
        # (not NaN, which pooling would silently drop). Only absent GT leaves F1 undefined.
        f1 = 0.0 if n_gt else float("nan")
    return {
        "n_gt_peaks": n_gt,
        "n_pred_peaks": n_pd,
        "peak_precision": precision,
        "peak_recall": recall,
        "peak_f1": f1,
    }


# ---------------------------------------------------------------------------
# 2. Doppler at the annotated vehicles
# ---------------------------------------------------------------------------


def _static_band(
    doppler_bins_mps: np.ndarray, A_deg: float, v_ego_xy: np.ndarray, p: MetricParams
) -> np.ndarray:
    """Boolean [D]: the Doppler bins within ±p.lbl_static_excl_bins (circular) of
    the ego-static velocity at azimuth A_deg.

    The static velocity -(v_x cos a + v_y sin a) is the receding-positive
    convention of the renderer's Doppler axis (`doppler_sign: -1` in the RADIal
    and synthetic recipes)."""
    D = len(doppler_bins_mps)
    dv = float(abs(doppler_bins_mps[1] - doppler_bins_mps[0]))
    half = D * dv / 2.0
    a = math.radians(float(A_deg))
    v_stat = -(float(v_ego_xy[0]) * math.cos(a) + float(v_ego_xy[1]) * math.sin(a))
    v_stat = (v_stat + half) % (2.0 * half) - half
    dist = np.abs(doppler_bins_mps - v_stat) / dv
    dist = np.minimum(dist, D - dist)
    return dist <= p.lbl_static_excl_bins


def label_doppler_metrics(
    pred_rad: np.ndarray,
    gt_rad: np.ndarray,
    frame_labels: Sequence[dict],
    range_bins_m: np.ndarray,
    az_bins_deg: np.ndarray,
    p: MetricParams = MetricParams(),
    *,
    doppler_bins_mps: np.ndarray,
    v_ego_xy: np.ndarray,
) -> dict:
    """Wrap-aware Doppler argmax-peak error at label (R, A) cells: the paper's
    Doppler MAE, `lbl_dop_peak_mae_bins_dyn` (bins).

    frame_labels: list of dicts with "R_m" and "A_deg" (trainer label format).
    Power is pooled over a (2·lbl_win_r+1) × (2·lbl_win_a+1) window around the
    label cell (labels are quantized; azimuth window is wider because the grid
    is 5× oversampled). A label is skipped when the largest GT cube cell in its
    window is at most noise_factor × the median positive value of the GT RA map
    (n_labels_valid counts the rest).

    The ego-static Doppler band (`_static_band`) is zeroed in both profiles first,
    so the window's static background does not stand in for the object's
    Doppler. A label whose GT or predicted profile is all zero outside the static
    band is not scored (n_labels_valid_dyn counts the scored ones).

    `lbl_dop_peak_mae_bins_dyn` is the mean over this frame's scored labels;
    `aggregate` then averages frames with equal weight (a frame-mean of per-frame
    means), while `n_labels_valid_dyn` is summed.
    """
    out = {"n_labels_valid": 0}
    if not frame_labels:
        return out
    D, R, A = gt_rad.shape
    gt_ra = ra_project(gt_rad)
    positive = gt_ra[gt_ra > 0]
    if positive.size == 0:
        return out
    gate_thr = p.noise_factor * float(np.median(positive))

    n_valid = 0
    peak_dyn = []
    for lbl in frame_labels:
        ri = int(np.argmin(np.abs(range_bins_m - float(lbl["R_m"]))))
        ai = int(np.argmin(np.abs(az_bins_deg - float(lbl["A_deg"]))))
        rs = slice(max(ri - p.lbl_win_r, 0), min(ri + p.lbl_win_r + 1, R))
        as_ = slice(max(ai - p.lbl_win_a, 0), min(ai + p.lbl_win_a + 1, A))
        gt_win = gt_rad[:, rs, as_]
        if float(gt_win.max()) <= gate_thr:
            continue
        n_valid += 1
        gt_d = gt_win.reshape(D, -1).sum(axis=1)
        pd_d = np.clip(pred_rad[:, rs, as_], 0, None).reshape(D, -1).sum(axis=1)
        stat = _static_band(doppler_bins_mps, lbl["A_deg"], v_ego_xy, p)
        gt_d[stat] = 0.0
        pd_d[stat] = 0.0
        if gt_d.sum() > 0 and pd_d.sum() > 0:
            err = wrap_bins_signed(float(np.argmax(pd_d)) - float(np.argmax(gt_d)), D)
            peak_dyn.append(abs(float(err)))

    out["n_labels_valid"] = n_valid
    if peak_dyn:
        out["lbl_dop_peak_mae_bins_dyn"] = float(np.mean(peak_dyn))
        out["n_labels_valid_dyn"] = len(peak_dyn)
    return out


# ---------------------------------------------------------------------------
# 3. Normalization shared by the reconstruction metrics
# ---------------------------------------------------------------------------
# PSNR/SSIM/Pearson/LPIPS are computed on the linear measurement normalized to
# [0, 1] by one global robust range lin_norm = (lo, hi):
# clip((x - lo) / (hi - lo), 0, 1). (lo, hi) is the sequence's norm.json range,
# itself robust percentiles over the train-split GT (robust_lin_range, run by
# preprocessing/normalize_dataset.py). Divide-by-max normalization would let one
# bright reflector set the scale and push most bins near zero, making MSE
# floor-dominated.


def robust_lin_range(
    arrays, lo_pct: float = 0.01, hi_pct: float = 99.9, per_frame_samples: int = 400_000
) -> tuple:
    """Global (lo, hi) linear-normalization range over an iterable of GT arrays.

    lo is the smallest per-array lo_pct percentile (lo_pct <= 0: the smallest
    array minimum); hi is the hi_pct percentile of all arrays pooled, over a
    bounded per-array subsample so it scales to many large [D,R,A] tensors
    without holding them all in memory. Clipping the upper tail keeps a single
    bright reflector from crushing the scale. Pass the same (lo, hi) for every
    frame. Returns (0.0, 1.0) for an empty iterable.
    """
    pooled = []
    lo = np.inf
    for a in arrays:
        a = np.asarray(a, dtype=np.float64).ravel()
        if a.size == 0:
            continue
        lo = min(lo, float(a.min()) if lo_pct <= 0 else float(np.percentile(a, lo_pct)))
        if a.size > per_frame_samples:
            a = a[np.linspace(0, a.size - 1, per_frame_samples).astype(np.int64)]
        pooled.append(a)
    if not pooled:
        return 0.0, 1.0
    hi = float(np.percentile(np.concatenate(pooled), hi_pct))
    lo = float(lo if np.isfinite(lo) else 0.0)
    return lo, max(hi, lo + _EPS)


def _lin_lohi(lin_norm, view: str, p: MetricParams) -> tuple:
    """Resolve a linear-normalization (lo, span) for clip((x - lo) / span, 0, 1).

    lin_norm = (lo, hi) is the RAD cube's robust train-GT range. RA and RD maps
    are means over the cube and sit in a different dynamic-range slice: on the
    cube's range they would occupy only part of [0, 1], inflating PSNR and
    mis-scaling SSIM's C1/C2. So a marginal view uses its own range
    `p.marginal_lin_norm[view]` when the caller supplied one (derivation:
    dyrad.norm.MARGINAL_LIN_RANGE). Without one, every view uses `lin_norm`
    (e.g. D == 1, where RA is the cube).

    The marginal ranges must be in the units of the arrays being scored
    (`score_renders._prepare` scales them alongside `lin_norm`). A raw-power
    range on ceiling-unit arrays maps everything to 0 and yields the degenerate
    PSNR=100 / SSIM=1.0 / rho=0.
    """
    if view != "rad" and p.marginal_lin_norm:
        rng = p.marginal_lin_norm.get(view)
        if rng is not None:
            lo, hi = float(rng[0]), float(rng[1])
            return lo, max(hi - lo, _EPS)
    lo, hi = float(lin_norm[0]), float(lin_norm[1])
    return lo, max(hi - lo, _EPS)


def _norm_lin_nc(gt, pred, lo, span, ceiling=None):
    """Shared linear normalization (loss/eval/viz): g=(gt-lo)/span, q=(pred-lo)/span.

    ceiling (units of hi == 1.0): clip both g and q to [0, ceiling]; PSNR peak /
    SSIM data_range are then `ceiling`. Must match the trainer's
    norm_clip_ceiling so loss == metric. None / <= 0 -> no clipping (only pred is
    floored at 0 before normalization; values below lo go negative).
    Returns (g, q, peak=ceiling or 1.0)."""
    g = (np.asarray(gt, np.float64) - lo) / span
    q = (np.clip(pred, 0, None).astype(np.float64) - lo) / span
    if ceiling is not None and ceiling > 0:
        g = np.clip(g, 0.0, ceiling)
        q = np.clip(q, 0.0, ceiling)
        return g, q, float(ceiling)
    return g, q, 1.0


def _lin_pair(pred, gt, lin_norm, view: str, p: MetricParams) -> tuple:
    """(g, q, data_range): gt and pred on the view's linear normalization."""
    return _norm_lin_nc(gt, pred, *_lin_lohi(lin_norm, view, p), p.lin_clip_ceiling)


def norm_log01(x, floor_u: float, lo_log: float, hi_log: float):
    """Normalized log-power map N(x) = clip((log10(max(x, floor_u)) - lo_log) /
    (hi_log - lo_log), 0, 1), for arrays or scalars in ceiling units.

    (floor_u, lo_log, hi_log) come from the sequence's norm.json (`dyrad.norm`):
    floor_u = floor/hi, lo_log = log10(floor_u), hi_log = log10(1 + floor_u), so
    background maps to 0 and the ceiling to 1. The trainer's loss holds a torch
    mirror that adds floor_u to pred (a smooth gradient near the floor); the
    metrics clamp both sides at floor_u instead, because the additive form would
    make the map asymmetric (a near-floor pred reads log10(2*floor_u) while the
    same GT value reads 0) and cap PSNR_N(x, x) at ~22 dB. The detection, Doppler
    and linear metrics do not use this map."""
    x = np.asarray(x, np.float64)
    v = np.maximum(x, floor_u)
    n = (np.log10(v) - lo_log) / (hi_log - lo_log)
    return np.clip(n, 0.0, 1.0)


def _n_pair(pred, gt, n_norm: tuple) -> tuple:
    """(g, q): gt and pred in the N map, both clamped at the floor."""
    floor_u, lo_log, hi_log = float(n_norm[0]), float(n_norm[1]), float(n_norm[2])
    g = norm_log01(gt, floor_u, lo_log, hi_log).astype(np.float64)
    q = norm_log01(pred, floor_u, lo_log, hi_log).astype(np.float64)
    return g, q


# ---------------------------------------------------------------------------
# 4. PSNR / SSIM / Pearson on normalized arrays
# ---------------------------------------------------------------------------


def _psnr_ssim(g, q, dr: float, p: MetricParams, mask=None) -> tuple:
    """PSNR and SSIM of two normalized arrays with peak / data_range `dr`.

    Whole array (mask None): PSNR on the full MSE, SSIM as skimage's mean.
    Region (boolean mask of the same shape): PSNR over the mask cells and the
    full-image SSIM map averaged over the mask (SSIM needs a 2-D neighbourhood);
    NaN for an empty mask. The SSIM window is `p.ssim_win`, shrunk to the
    largest odd size the array allows; SSIM is NaN below 3."""
    if mask is not None and not mask.any():
        return float("nan"), float("nan")
    diff2 = (g - q) ** 2
    mse = float(diff2.mean() if mask is None else diff2[mask].mean())
    psnr = 10.0 * math.log10(dr * dr / (mse + _EPS))
    md = min(g.shape)
    win = min(p.ssim_win, md if md % 2 == 1 else md - 1)
    if win < 3:
        return psnr, float("nan")
    if mask is None:
        return psnr, float(_ssim(g, q, data_range=dr, win_size=win))
    _, ssim_map = _ssim(g, q, data_range=dr, win_size=win, full=True)
    return psnr, float(ssim_map[mask].mean())


def _select(g, q, mask):
    """(g, q) restricted to the mask, or None for an empty mask."""
    if mask is None:
        return g, q
    if not mask.any():
        return None
    return g[mask], q[mask]


def _corr_lin(pred, gt, lin_norm, p: MetricParams, view: str, mask=None) -> float:
    """Pearson rho in the capped normalized linear space, the space PSNR/SSIM are
    scored in (`_lin_pair`).

    Pearson is invariant to the affine part of the normalization range, so what
    matters are the nonlinear steps: `clip(pred, 0, None)` and the clip of both
    sides to `[0, lin_clip_ceiling]`. Without them rho is dominated by the bright
    speckle tail the cap exists to bound. The order is project, then cap, as for
    PSNR/SSIM, so e.g. `rd_corr` and `rd_psnr_lin` are statistics of the same
    array.

    mask: optional boolean of pred/gt shape (region metrics). Returns NaN for an
    empty window, and 0.0 when either side is constant over the window (a constant
    prediction has no explanatory power; NaN would drop the frame from the mean).
    """
    g, q, _ = _lin_pair(pred, gt, lin_norm, view, p)
    sel = _select(g, q, mask)
    if sel is None or sel[0].size == 0:
        return float("nan")
    g, q = sel
    g = g.ravel() - g.mean()
    q = q.ravel() - q.mean()
    den = math.sqrt(float((g * g).sum()) * float((q * q).sum()))
    return float((g * q).sum() / den) if den > 0 else 0.0


def _corr_N(pred: np.ndarray, gt: np.ndarray, n_norm: tuple, mask=None) -> float:
    """Pearson correlation in the N map, robust to the speckle tail that dominates
    raw-linear Pearson. NaN for an empty mask; a constant side (e.g. an all-black
    prediction) scores 0.0, as in `_corr_lin`."""
    sel = _select(*_n_pair(pred, gt, n_norm), mask)
    if sel is None:
        return float("nan")
    g, q = sel[0].ravel(), sel[1].ravel()
    if g.std() < 1e-12 or q.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(g, q)[0, 1])


# ---------------------------------------------------------------------------
# 5. LPIPS
# ---------------------------------------------------------------------------

_LPIPS_MODEL = None


def _get_lpips_model():
    """The AlexNet LPIPS model, loaded once (on the GPU when there is one).

    torch and lpips are imported here, so the module stays importable without
    them as long as LPIPS is not requested. lpips is a hard dependency of the
    scorer: a failed load raises instead of reporting NaN columns."""
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips
        import torch

        m = lpips.LPIPS(net="alex", verbose=False).eval()
        _LPIPS_MODEL = m.cuda() if torch.cuda.is_available() else m  # AlexNet is ~9 MB
    return _LPIPS_MODEL


def _lpips_distance(gt01: np.ndarray, pred01: np.ndarray) -> float:
    """LPIPS (AlexNet) on two [H,W] images already in [0,1].

    NaN when an axis is below 32 bins (AlexNet downsamples ~16x, so a thinner
    axis, e.g. the 16-bin Doppler axis of an RD map, collapses to size 0 in a pool
    layer; LPIPS is undefined there), when an input is non-finite, and when the
    net itself returns a non-finite value."""
    if min(gt01.shape[:2]) < 32:
        return float("nan")
    if not (np.all(np.isfinite(gt01)) and np.all(np.isfinite(pred01))):
        return float("nan")
    import torch

    model = _get_lpips_model()
    dev = next(model.parameters()).device

    def _to_t(x):
        t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # [1,3,H,W]
        return (t * 2.0 - 1.0).to(dev)  # → [-1, 1]

    with torch.no_grad():
        d = float(model(_to_t(gt01), _to_t(pred01)).reshape(-1)[0])
    return d if math.isfinite(d) else float("nan")


def _lpips_linear(
    pred: np.ndarray, gt: np.ndarray, p: MetricParams, lin_norm: tuple, view: str
) -> float:
    """LPIPS (AlexNet) on the arrays PSNR/SSIM/rho score (`_lin_pair`), rescaled by
    the data range so the net sees [0, 1]. An RA-map metric: NaN on the 16-bin
    RD map (`_lpips_distance`)."""
    g, q, dr = _lin_pair(pred, gt, lin_norm, view, p)
    g01 = np.clip(np.asarray(g, np.float64) / dr, 0.0, 1.0)
    q01 = np.clip(np.asarray(q, np.float64) / dr, 0.0, 1.0)
    return _lpips_distance(g01, q01)


def _lpips_linear_rad(
    pred_rad: np.ndarray, gt_rad: np.ndarray, p: MetricParams, lin_norm: tuple
) -> float:
    """RAD-cube LPIPS: mean over the Doppler axis of per-slice [R, A] LPIPS.

    Uses the cube's linear normalization range (the same arrays `rad_psnr_lin` scores). One
    normalization for the whole cube, never per slice, so a floor-only slice
    compares floor to floor and a Doppler-misplaced object costs on two slices.
    LPIPS has no 3-D form; averaging 2-D slices is the standard volume usage.
    Non-finite slices are dropped; all-NaN -> NaN.
    """
    g, q, dr = _lin_pair(pred_rad, gt_rad, lin_norm, "rad", p)
    vals = []
    for d in range(g.shape[0]):
        g01 = np.clip(np.asarray(g[d], np.float64) / dr, 0.0, 1.0)
        q01 = np.clip(np.asarray(q[d], np.float64) / dr, 0.0, 1.0)
        v = _lpips_distance(g01, q01)
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# 6. Reconstruction metric families
# ---------------------------------------------------------------------------


def classic_metrics(
    pred_ra: np.ndarray,
    gt_ra: np.ndarray,
    pred_rad: Optional[np.ndarray] = None,
    gt_rad: Optional[np.ndarray] = None,
    p: MetricParams = MetricParams(),
    with_lpips: bool = False,
    lin_norm: Optional[tuple] = None,
) -> dict:
    """PSNR / SSIM (/ LPIPS) on RA, RD ([D, R], mean over azimuth) and full RAD.

    Linear keys: {ra,rd,rad}_{psnr,ssim}_lin. N keys (when `p.n_norm` is set,
    log-domain sensors): ra_{psnr,ssim,corr}_N.

    with_lpips: also emit LPIPS on the full RA and RAD measurements
    (`ra_lpips_lin`, `rad_lpips_lin`; `ra_lpips_N` when n_norm is set).
    score_renders enables it unless `--no-lpips`.

    lin_norm: (lo, hi) robust train-GT range applied as clip((x - lo)/(hi - lo), 0, 1).
    Pass the same (lo, hi) for every frame so val and train share a scale;
    per-marginal normalization ranges come from `p.marginal_lin_norm`.
    """
    out: dict = {}
    out["ra_psnr_lin"], out["ra_ssim_lin"] = _psnr_ssim(
        *_lin_pair(pred_ra, gt_ra, lin_norm, "ra", p), p
    )
    if p.n_norm is not None:
        # N (normalized-log) metrics, matching the training representation of a
        # log-domain sensor; PSNR_N, SSIM_N, rho_N and LPIPS_N share one image pair.
        g, q = _n_pair(pred_ra, gt_ra, p.n_norm)
        out["ra_psnr_N"], out["ra_ssim_N"] = _psnr_ssim(g, q, 1.0, p)
        out["ra_corr_N"] = _corr_N(pred_ra, gt_ra, p.n_norm)
        if with_lpips:
            out["ra_lpips_N"] = _lpips_distance(g, q)
    if with_lpips:
        out["ra_lpips_lin"] = _lpips_linear(pred_ra, gt_ra, p, lin_norm, view="ra")
    if pred_rad is not None and gt_rad is not None:
        # Range-Doppler image: `rd_project` = mean over azimuth [D, R], shared with
        # rd_corr / rd_*_obj.
        gt_rd = rd_project(gt_rad)
        pd_rd = rd_project(np.clip(pred_rad, 0, None))
        out["rd_psnr_lin"], out["rd_ssim_lin"] = _psnr_ssim(
            *_lin_pair(pd_rd, gt_rd, lin_norm, "rd", p), p
        )
        # Full 3-D RAD (Doppler included): volumetric PSNR + SSIM.
        out["rad_psnr_lin"], out["rad_ssim_lin"] = _psnr_ssim(
            *_lin_pair(pred_rad, gt_rad, lin_norm, "rad", p), p
        )
        if with_lpips:
            out["rad_lpips_lin"] = _lpips_linear_rad(pred_rad, gt_rad, p, lin_norm)
    return out


def correlation_metrics(
    pred_ra: np.ndarray,
    gt_ra: np.ndarray,
    pred_rad: Optional[np.ndarray] = None,
    gt_rad: Optional[np.ndarray] = None,
    lin_norm: Optional[tuple] = None,
    p: MetricParams = MetricParams(),
) -> dict:
    """Pearson correlation ra_corr / rd_corr / rad_corr.

    rho is computed in the capped normalized space shared with the PSNR/SSIM keys
    (`_corr_lin`). RD is `rd_project` (mean over azimuth), the same RD image the
    rd_* PSNR/SSIM keys use."""
    out = {"ra_corr": _corr_lin(pred_ra, gt_ra, lin_norm, p, "ra")}
    if pred_rad is not None and gt_rad is not None:
        out["rad_corr"] = _corr_lin(pred_rad, gt_rad, lin_norm, p, "rad")
        pd_rd, gt_rd = rd_project(np.clip(pred_rad, 0, None)), rd_project(gt_rad)
        out["rd_corr"] = _corr_lin(pd_rd, gt_rd, lin_norm, p, "rd")
    return out


def region_reconstruction(
    pred_ra,
    gt_ra,
    pred_rad,
    gt_rad,
    masks_ra,
    lin_norm,
    p: MetricParams = MetricParams(),
) -> dict:
    """Object-region linear PSNR / SSIM / Pearson on RA, full RAD and RD.

    masks_ra: {"obj": boolean [R, A]} (region_masks.object_mask_ra). Emits {ra,rd,rad}_{psnr,ssim}_lin_obj and
    {ra,rd,rad}_corr_obj (Pearson in the same capped normalized space as the
    PSNR/SSIM keys), plus the N-domain RA twins when `p.n_norm` is set. The RA
    mask broadcasts over D for RAD; RD uses the range marginal of the RA mask
    (an [R, A] mask does not localize in Doppler, so the RD object region holds
    every Doppler bin at the occupied ranges)."""
    m = (masks_ra or {}).get("obj")
    if m is None:
        return {}
    out = {}
    out["ra_psnr_lin_obj"], out["ra_ssim_lin_obj"] = _psnr_ssim(
        *_lin_pair(pred_ra, gt_ra, lin_norm, "ra", p), p, mask=m
    )
    out["ra_corr_obj"] = _corr_lin(pred_ra, gt_ra, lin_norm, p, "ra", mask=m)
    if p.n_norm is not None:
        out["ra_psnr_N_obj"], out["ra_ssim_N_obj"] = _psnr_ssim(
            *_n_pair(pred_ra, gt_ra, p.n_norm), 1.0, p, mask=m
        )
        out["ra_corr_N_obj"] = _corr_N(pred_ra, gt_ra, p.n_norm, mask=m)
    if pred_rad is None or gt_rad is None:
        return out
    D = gt_rad.shape[0]
    pd = np.clip(pred_rad, 0, None)
    gt_rd, pd_rd = rd_project(gt_rad), rd_project(pd)  # mean over azimuth
    m_full = np.broadcast_to(m[None], (D, *m.shape))
    m_rd = np.broadcast_to(m.any(1)[None], (D, m.shape[0]))
    out["rad_psnr_lin_obj"], out["rad_ssim_lin_obj"] = _psnr_ssim(
        *_lin_pair(pred_rad, gt_rad, lin_norm, "rad", p), p, mask=m_full
    )
    out["rad_corr_obj"] = _corr_lin(pd, gt_rad, lin_norm, p, "rad", mask=m_full)
    out["rd_corr_obj"] = _corr_lin(pd_rd, gt_rd, lin_norm, p, "rd", mask=m_rd)
    out["rd_psnr_lin_obj"], out["rd_ssim_lin_obj"] = _psnr_ssim(
        *_lin_pair(pd_rd, gt_rd, lin_norm, "rd", p), p, mask=m_rd
    )
    return out


# ---------------------------------------------------------------------------
# Marginal projections (RA / RD) of a [D, R, A] cube: mean over the collapsed axis
# ---------------------------------------------------------------------------


def ra_project(rad):
    """Collapse the Doppler axis of a [D, R, A] cube to an RA map (mean over Doppler).

    RADIal builds its RA map by summing over Doppler (`SignalProcessing/rpl.py`:
    `RA_map = np.sum(np.abs(Azimuth_spec), axis=2)`). The mean is the same linear
    projection up to the constant 1/D, which keeps the RA map on the same linear
    normalization range as the cube instead of clipping at the ceiling.

    Do not use max over Doppler: it is not a linear projection and it is
    Doppler-position dependent (a reflector's max varies strongly with its sub-bin
    Doppler position while the sum barely moves), so it leaks velocity
    information into a view that is meant to be Doppler-collapsed.
    """
    return np.asarray(rad).mean(axis=0)


def rd_project(rad):
    """Collapse the azimuth axis of a [D, R, A] cube to an RD map (mean over azimuth).

    The single RD definition used by every RD metric (`rd_psnr_lin`, `rd_ssim_lin`,
    `rd_corr`, `rd_*_obj`). Max over azimuth is avoided for the same reason as in
    `ra_project`: it tracks a single argmax bin, so a sub-bin azimuth shift changes
    a cell whose total power did not move. The mean (rather than the sum) keeps
    the RD image on the shared linear normalization range (`lin_norm`,
    `lin_clip_ceiling`); a 751-bin sum would clip.
    """
    return np.asarray(rad).mean(axis=2)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def score_frame(
    pred_ra: np.ndarray,
    gt_ra: np.ndarray,
    pred_rad: Optional[np.ndarray] = None,
    gt_rad: Optional[np.ndarray] = None,
    frame_labels: Optional[Sequence[dict]] = None,
    range_bins_m: Optional[np.ndarray] = None,
    az_bins_deg: Optional[np.ndarray] = None,
    p: MetricParams = MetricParams(),
    doppler_bins_mps: Optional[np.ndarray] = None,
    v_ego_xy: Optional[np.ndarray] = None,
    with_lpips: bool = False,
    lin_norm: Optional[tuple] = None,
    region_masks: Optional[dict] = None,
) -> dict:
    """Score one frame. A metric whose inputs are missing is omitted, not NaN:
    the RD/RAD keys need pred_rad + gt_rad, the `*_obj` keys region_masks, the
    `*_N` keys p.n_norm, the LPIPS keys with_lpips, and the label Doppler keys
    labels, axes and a Doppler axis with D > 1 (then v_ego_xy is required).

    lin_norm: global linear-normalization range (lo, hi) over train GT, used by
    the linear PSNR/SSIM/Pearson/LPIPS keys. region_masks: {"obj": boolean [R, A]}
    (from region_masks.object_mask_ra) -> object-region PSNR/SSIM/Pearson. The
    point-cloud family is scored at sequence level (`pointcloud_metrics`), not
    here.
    """
    out: dict = {}
    out.update(
        classic_metrics(
            pred_ra,
            gt_ra,
            pred_rad,
            gt_rad,
            p,
            with_lpips=with_lpips,
            lin_norm=lin_norm,
        )
    )
    out.update(peak_detection_metrics(pred_ra, gt_ra, p))
    out.update(scale_alpha(pred_ra, gt_ra, p))
    if region_masks is not None:
        out.update(
            region_reconstruction(
                pred_ra, gt_ra, pred_rad, gt_rad, region_masks, lin_norm, p
            )
        )
    # The label Doppler metrics need a Doppler axis (D > 1; not Boreas Navtech).
    has_doppler_axis = doppler_bins_mps is not None and len(doppler_bins_mps) > 1
    if (
        pred_rad is not None
        and gt_rad is not None
        and frame_labels is not None
        and range_bins_m is not None
        and az_bins_deg is not None
        and has_doppler_axis
    ):
        if v_ego_xy is None:
            raise ValueError(
                "label Doppler metrics need the ego velocity (v_ego_xy) to remove "
                "the ego-static band"
            )
        out.update(
            label_doppler_metrics(
                pred_rad,
                gt_rad,
                frame_labels,
                range_bins_m,
                az_bins_deg,
                p,
                doppler_bins_mps=doppler_bins_mps,
                v_ego_xy=v_ego_xy,
            )
        )
    out.update(
        correlation_metrics(pred_ra, gt_ra, pred_rad, gt_rad, lin_norm=lin_norm, p=p)
    )
    return out


def aggregate(per_frame: Sequence[dict]) -> dict:
    """nanmean per key across frames; count keys (n_*) are summed. Keys holding
    strings (the trainer's "split" tag) are skipped."""
    if not per_frame:
        return {}
    keys: list = []
    for d in per_frame:
        for k in d:
            if k not in keys:
                keys.append(k)
    out = {}
    for k in keys:
        raw = [d[k] for d in per_frame if k in d]
        if any(isinstance(v, str) for v in raw):
            continue
        vals = np.array([float(v) for v in raw], dtype=np.float64)
        if k.startswith("n_"):
            out[k] = int(np.nansum(vals))
        else:
            out[k] = (
                float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float("nan")
            )
    return out
