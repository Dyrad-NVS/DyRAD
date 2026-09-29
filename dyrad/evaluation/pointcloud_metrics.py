"""Detection metrics: a per-frame radar point cloud scored with the RadarGen protocol.

  radar_point_cloud            RADIal's RD-CFAR detector (SignalProcessing/rpl.py,
                               method 'PC') on one [D, R, A] cube -> [M, 4] cloud.
  radargen_pointcloud_metrics  RadarGen's entire-area family: CD-Loc, CD-Full,
                               IoU@1m, DA precision/recall/F1, MMD.
  radargen_box_metrics         RadarGen's foreground family inside the annotated
                               boxes: hit/miss rate, density similarity, FP boxes.
  object_detection_metrics     vehicle recall, the object criterion of the
                               hyperparameter selection (Appendix B.2).

The same detector runs on the pred and the GT cube, so any difference between the
two clouds is render error, not a processing asymmetry. The reference cloud is our
detector on the real cube, not a vendor point cloud: the definitions are RadarGen's
(Borreda et al., arXiv 2512.17897), but the numbers are not comparable across papers.

Amplitude and power. The cube is linear amplitude (see `radar_metrics`). The CFAR
thresholds the azimuth-summed amplitude with RADIal's window, guard and 2 dB threshold;
RADIal's rpl.py squares to power first, so this detector is stricter (roughly a 4 dB
power threshold). It is the detector behind the paper's detection numbers.
The cloud's fourth column keeps RADIal's layout and is called "power", but it holds
the peak's azimuth-summed amplitude; the DA power gate, CD-Full's power axis and the
MMD power term are therefore 10 log10 (or log10) of amplitude ratios.

Clouds are [M, 4] = (x, y, doppler_mps, power) in the sensor frame, xy = (R cos A,
R sin A). The family needs a Doppler axis: score_renders skips it when D == 1.
Doppler differences are wrap-aware with the DDMA period `p.doppler_wrap_mps`.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from dyrad.axes import wrap_bins_signed
from dyrad.evaluation.radar_metrics import MetricParams

_EPS = 1e-10
# Floor on amplitudes before a log, so an empty cell gives a finite dB value.
_POW_FLOOR = 1e-30
# Above this many point pairs, nearest neighbours come from a KD-tree (exact) instead
# of a dense `cdist` matrix; CD-Full's blocked search uses it as the block size.
_DENSE_NN_MAX_PAIRS = 4_000_000


# ---------------------------------------------------------------------------
# 1. Per-frame radar point cloud (RD-CFAR detections, sensor frame)
# ---------------------------------------------------------------------------


def _radial_rd_cfar_mask(A_rd: np.ndarray, p: MetricParams) -> np.ndarray:
    """CA-CFAR on a Range-Doppler amplitude map [R, D] -> boolean hit mask.

    RADIal SignalProcessing/rpl.py `CA_CFAR` applied to amplitude: a rectangular
    estimation window with a guard ring zeroed out gives the noise as the window mean,
    and a cell is a hit where amplitude/noise > 10**(threshold_db/10). rpl.py squares
    the map to power first (`rd_matrix = np.abs(rd_matrix) ** 2`); thresholding
    amplitude is the paper's protocol (see the module docstring). win = (win_R, win_D,
    guard_R, guard_D) half-widths; convolution is zero-padded at borders (mode='same'),
    matching RADIal."""
    wr, wd, gr, gd = (int(v) for v in p.pc_cfar_win)
    mask = np.ones((2 * wr + 1, 2 * wd + 1), dtype=float)
    mask[wr - gr : wr + gr + 1, wd - gd : wd + gd + 1] = 0.0
    n_valid = convolve2d(np.ones_like(A_rd), mask, mode="same")
    w_sum = convolve2d(A_rd, mask, mode="same")
    noise = w_sum / np.maximum(n_valid, 1.0)
    snr = A_rd / np.maximum(noise, _EPS)
    return snr > (10.0 ** (p.pc_cfar_threshold_db / 10.0))


def radar_point_cloud(
    rad: np.ndarray,
    range_bins_m: np.ndarray,
    az_bins_deg: np.ndarray,
    doppler_bins_mps: np.ndarray,
    p: MetricParams = MetricParams(),
) -> np.ndarray:
    """RADIal-faithful per-frame radar point cloud from a [D, R, A] linear-amplitude cube.

    1. azimuth-reduced RD amplitude A_rd[r,d] = sum_a cube[d,r,a]  (analogue of RADIal's
       Rx-summed amplitude spectrum `np.sum(np.abs(RD), axis=2)`; our cube is already
       AoA-beamformed).
    2. CA-CFAR on A_rd (RADIal window and threshold) -> RD peaks (r, d).
    3. one azimuth per peak: a* = argmax_a cube[d, r, a]  (mirrors RADIal's per-peak AoA).
    Returns [M, 4]: (x, y, doppler_mps, A_rd at the peak)."""
    if rad.ndim != 3:
        raise ValueError("radar_point_cloud expects a [D, R, A] cube")
    A_rd = np.clip(rad, 0, None).sum(axis=2).T  # [R, D] amplitude, summed over azimuth
    hit = _radial_rd_cfar_mask(A_rd, p)  # [R, D]
    rb, db = np.where(hit)
    if len(rb) == 0:
        return np.zeros((0, 4))
    a_star = rad[db, rb, :].argmax(axis=1)  # azimuth bin per RD peak
    R = range_bins_m[rb]
    A = np.deg2rad(az_bins_deg[a_star])
    x, y = R * np.cos(A), R * np.sin(A)
    power = A_rd[rb, db]  # the peak's azimuth-summed amplitude
    return np.stack([x, y, doppler_bins_mps[db], power], axis=1)


# ---------------------------------------------------------------------------
# 2. The RadarGen point-cloud protocol, entire area
# ---------------------------------------------------------------------------


def _directed_nn(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Distance from every point of P to its nearest neighbour in Q (empty → empty).

    Uses a KD-tree for large sets (exact NN) to avoid the O(|P||Q|) dense `cdist`
    intermediate."""
    if len(P) == 0 or len(Q) == 0:
        return np.zeros(0)
    if len(P) * len(Q) > _DENSE_NN_MAX_PAIRS:
        return cKDTree(Q).query(P, k=1)[0]
    return cdist(P, Q).min(axis=1)


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, p: MetricParams) -> float:
    """Multi-scale RBF MMD^2 between two [n, k] point sets, numpy port of RadarGen's
    `MMDLoss` + `RBF`.

    K kernels with bandwidths `bw * mul^(k - K//2)`, `bw` = mean pairwise squared
    distance over the joint set (their `get_bandwidth`). Subsampled to
    `pc_mmd_max_pts` with a fixed seed, so the value is deterministic but
    dataset-dependent through the bandwidth; MMD values are not comparable across
    benchmarks."""
    if len(X) == 0 or len(Y) == 0:
        return float("nan")
    rng = np.random.default_rng(p.pc_mmd_seed)

    def _sub(A):
        if len(A) <= p.pc_mmd_max_pts:
            return A
        return A[rng.choice(len(A), p.pc_mmd_max_pts, replace=False)]

    X = _sub(np.asarray(X, np.float64))
    Y = _sub(np.asarray(Y, np.float64))
    Z = np.vstack([X, Y])
    d2 = cdist(Z, Z, "sqeuclidean")
    n = len(Z)
    bw = d2.sum() / max(n * n - n, 1)
    if not np.isfinite(bw) or bw <= 0:
        return float("nan")
    mults = p.pc_mmd_mul ** (np.arange(p.pc_mmd_kernels) - p.pc_mmd_kernels // 2)
    K = sum(np.exp(-d2 / (bw * m)) for m in mults)
    nx = len(X)
    return float(K[:nx, :nx].mean() - 2 * K[:nx, nx:].mean() + K[nx:, nx:].mean())


def _da_match(pred: np.ndarray, gt: np.ndarray, p: MetricParams) -> tuple:
    """RadarGen's distance+attribute matching: optimal one-to-one assignment gated
    on location AND attributes. Returns (TP, n_gt, n_pred).

    Port of their `distance_attr_recall_precision_f1_hungarian`: gates are
    absolute differences, the Hungarian cost is the distance with invalid pairs
    set to `delta_loc + 1e5`, and the validity mask (not the solver, which always
    returns a full assignment) decides what counts.

    Gates: xy distance <= pc_match_tau_m; |10 log10(P_pred / P_gt)| <=
    pc_da_pow_gate_db (their delta_rcs = 8, applied to the ratio because the
    power column is uncalibrated); and a wrap-aware Doppler difference <=
    pc_da_dop_gate_mps. The Doppler gate differs from RadarGen's 2.5 m/s, which
    exceeds the DDMA unambiguous span (period 1.7968 m/s) and would admit every
    velocity; 10% of the period is used instead (~1.6 bins).

    There is no gain-invariant variant: every method's cloud is detected on the
    same clip-normalized cube range, so the absolute level is real information
    about the render. A cloud at the wrong level can fail the power gate for
    every pair (DA = 0) while its locations are right."""
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, len(gt), len(pred)
    d = cdist(gt[:, :2], pred[:, :2])
    ok = d <= p.pc_match_tau_m
    dd = np.abs(wrap_bins_signed(gt[:, None, 2] - pred[None, :, 2], p.doppler_wrap_mps))
    ok &= dd <= p.pc_da_dop_gate_mps
    # Absolute difference, as their |rcs_gt - rcs_syn| <= 8 dBsm.
    pd_db = 10.0 * np.log10(np.maximum(pred[:, 3], _POW_FLOOR))
    gt_db = 10.0 * np.log10(np.maximum(gt[:, 3], _POW_FLOOR))
    ok &= np.abs(gt_db[:, None] - pd_db[None, :]) <= p.pc_da_pow_gate_db
    cost = np.where(ok, d, p.pc_match_tau_m + 1e5)
    gi, pi = linear_sum_assignment(cost)
    return float(ok[gi, pi].sum()), len(gt), len(pred)


def _da_prf(tp: float, n_gt: int, n_pred: int) -> tuple:
    """Their per-sample (recall, precision, f1) from one frame's TP, including the
    empty-cloud early returns of `distance_attr_recall_precision_f1_hungarian`.

    RadarGen reports DA scores per sample and averages them over frames, so they
    must not be micro-pooled as sum(TP)/sum(N) across frames."""
    eps = 1e-10  # RadarGen's own eps: a perfect score is 0.9999999999
    if n_gt == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0
    if n_gt == 0:
        return 1.0, 0.0, 0.0
    if n_pred == 0:
        return 0.0, 1.0, 0.0
    rec = tp / (n_gt + eps)
    prec = tp / (n_pred + eps)
    return rec, prec, 2 * prec * rec / (prec + rec + eps)


def _cd_full_coords(
    pc: np.ndarray, r_max: float, pow_ref: float, pow_range_db: tuple
) -> np.ndarray:
    """CD-Full's Euclidean coordinates (x, y, power), each mapped to [0, 1] (their
    `normalize_pcl_for_chamfer`); the Doppler term is added by `_cd_full_distance`.

    One normalization shared by both clouds, built from dataset constants (their
    `NormalizationConfig`): `r_max` from the range axis, and for power the dB
    offset from `pow_ref` clipped to `pow_range_db`. `pow_ref` is the GT cloud's
    median, used for both clouds, so the power axis is a method-independent
    normalization range that still measures global gain."""
    xy = (pc[:, :2] + r_max) / (2.0 * max(r_max, 1e-9))
    lo_db, hi_db = float(pow_range_db[0]), float(pow_range_db[1])
    db = 10.0 * np.log10(np.maximum(pc[:, 3], _POW_FLOOR) / max(pow_ref, _POW_FLOOR))
    pw = (np.clip(db, lo_db, hi_db) - lo_db) / max(hi_db - lo_db, 1e-9)
    return np.column_stack([xy, pw])


def _cd_full_distance(
    pc: np.ndarray, gc: np.ndarray, r_max: float, pow_ref: float, p: MetricParams
) -> float:
    """CD-Full between two non-empty clouds, with a circular Doppler axis.

    Three axes (x, y, power) are Euclidean in the [0, 1] coordinates of
    `_cd_full_coords`. The Doppler term is
      |wrap(v_p - v_g, T)| / (T / 2)    (T = p.doppler_wrap_mps, the DDMA period)
    instead of their linear |u_p - u_g|. Dividing by the maximum circular
    separation T/2 keeps that axis in [0, 1], so all attributes still count
    equally. The linear map is wrong on a wrapped axis: physically adjacent bins
    across the seam land at opposite ends of [0, 1], while a maximally different
    velocity gets only half weight. This matters most for aliased vehicles, and
    makes `pc_cd_full` a variant of RadarGen's CD-Full.

    Uses exact nearest neighbours (no KD-tree), since a circular coordinate is not
    embeddable in Euclidean space without distorting distances. Computed in blocks
    so the pairwise matrix stays bounded."""
    P = _cd_full_coords(pc, r_max, pow_ref, p.pc_pow_range_db)
    G = _cd_full_coords(gc, r_max, pow_ref, p.pc_pow_range_db)
    T = float(p.doppler_wrap_mps)
    pv = np.asarray(pc[:, 2], np.float64)
    gv = np.asarray(gc[:, 2], np.float64)
    row_min = np.full(len(P), np.inf)
    col_min = np.full(len(G), np.inf)
    blk = max(1, int(_DENSE_NN_MAX_PAIRS // len(G)))
    for a in range(0, len(P), blk):
        b = min(a + blk, len(P))
        d2 = cdist(P[a:b], G, "sqeuclidean")
        dv = np.abs(wrap_bins_signed(pv[a:b, None] - gv[None, :], T)) / (T / 2.0)
        d2 = d2 + dv**2
        row_min[a:b] = d2.min(axis=1)
        np.minimum(col_min, d2.min(axis=0), out=col_min)
    return float(0.5 * (np.sqrt(row_min).mean() + np.sqrt(col_min).mean()))


def radargen_pointcloud_metrics(
    pred_clouds: Sequence[np.ndarray],
    gt_clouds: Sequence[np.ndarray],
    range_bins_m: np.ndarray,
    p: MetricParams = MetricParams(),
) -> dict:
    """RadarGen's ENTIRE-AREA family.

    Follows the published RadarGen evaluation code: every value is a frame mean
    (their `finalize_global_metrics` convention).

    Clouds are the `radar_point_cloud` output on the pred and GT cubes (identical
    detector both sides). `range_bins_m` sets CD-Full's location scale; Doppler
    distances use `p.doppler_wrap_mps`.

    Keys (RadarGen definitions):
      pc_cd_loc_m       : CD-Loc, (d(P,G) + d(G,P)) / 2 over xy.
      pc_cd_full        : CD-Full, the same over (x, y, Doppler, power) each in
                          [0,1] under one shared normalization.
      pc_iou_tau        : IoU@1m -- their reported geometry number.
      pc_da_precision / _recall / _f1 : one-to-one Hungarian on location,
                          Doppler and power.
      pc_mmd_xy / _dop / _pow : MMD Loc. / Dopp. / RCS-analogue, multi-scale RBF.
      n_pc_empty_pred / n_pc_empty_gt : frames the distance/MMD means dropped.
    """
    nan = float("nan")
    r_max = float(np.max(range_bins_m))
    acc: dict = {
        k: []
        for k in (
            "cd_loc",
            "iou",
            "cd_full",
            "da_r",
            "da_p",
            "da_f",
            "n_empty_pred",
            "n_empty_gt",
            "mmd_xy",
            "mmd_dop",
            "mmd_pow",
        )
    }
    for pc, gc in zip(pred_clouds, gt_clouds):
        pxy, gxy = pc[:, :2], gc[:, :2]
        d_p2g, d_g2p = _directed_nn(pxy, gxy), _directed_nn(gxy, pxy)
        if len(d_p2g) and len(d_g2p):
            # Their `chamfer_distance(direction='bi')` returns the SUM of the two
            # directed means and every call site multiplies by 0.5.
            acc["cd_loc"].append(0.5 * float(d_p2g.mean() + d_g2p.mean()))
            pr = float((d_p2g < p.pc_match_tau_m).mean())
            rc = float((d_g2p < p.pc_match_tau_m).mean())
            # RadarGen's `_metrics_pointcloud`: both near zero → report P rather
            # than 0/0.
            if pr < 1e-3 and rc < 1e-3:
                acc["iou"].append(pr)
            else:
                acc["iou"].append(pr * rc / (pr + rc - pr * rc))
        n_p, n_g = len(pc), len(gc)
        # DA runs on every frame, empty clouds included: RadarGen defines the empty
        # cases (ported in `_da_prf`), and skipping them would forgive a method
        # that emits nothing.
        r_, p_, f_ = _da_prf(*_da_match(pc, gc, p))
        acc["da_r"].append(r_)
        acc["da_p"].append(p_)
        acc["da_f"].append(f_)
        # CD-Loc / CD-Full / IoU / MMD are undefined on an empty cloud. Those frames
        # are excluded from the means, and the exclusions are counted and reported.
        acc["n_empty_pred"].append(1.0 if n_p == 0 else 0.0)
        acc["n_empty_gt"].append(1.0 if n_g == 0 else 0.0)
        if n_p and n_g:
            # One shared normalization for both clouds, referenced to the GT cloud
            # (method-independent) -- see _cd_full_coords.
            ref = float(np.median(gc[:, 3]))
            acc["cd_full"].append(_cd_full_distance(pc, gc, r_max, ref, p))
            acc["mmd_xy"].append(_mmd_rbf(pc[:, :2], gc[:, :2], p))
            acc["mmd_dop"].append(_mmd_rbf(pc[:, 2:3], gc[:, 2:3], p))
            # Their MMD RCS is on absolute calibrated dBsm, so a global gain is an
            # error: log10 of the raw column, not re-referenced (consistent with the
            # DA power gate).
            acc["mmd_pow"].append(
                _mmd_rbf(
                    np.log10(np.maximum(pc[:, 3:4], _POW_FLOOR)),
                    np.log10(np.maximum(gc[:, 3:4], _POW_FLOOR)),
                    p,
                )
            )

    def m(k):
        v = [x for x in acc[k] if np.isfinite(x)]
        return float(np.mean(v)) if v else nan

    return {
        "pc_cd_loc_m": m("cd_loc"),
        "pc_cd_full": m("cd_full"),
        "pc_iou_tau": m("iou"),
        "pc_da_precision": m("da_p"),
        "pc_da_recall": m("da_r"),
        "pc_da_f1": m("da_f"),
        "pc_mmd_xy": m("mmd_xy"),
        "pc_mmd_dop": m("mmd_dop"),
        "pc_mmd_pow": m("mmd_pow"),
        # How many frames the distance/MMD means had to drop (see the loop).
        "n_pc_empty_pred": int(sum(acc["n_empty_pred"])),
        "n_pc_empty_gt": int(sum(acc["n_empty_gt"])),
    }


# ---------------------------------------------------------------------------
# 3. RadarGen's foreground family -- inside the annotated boxes
# ---------------------------------------------------------------------------
# The box is the same per-object footprint the init-cloud partition uses
# (`dyrad/ra_partition.car_extent_m`); the caller resolves half-extents in metres.
# It is not the object region of the reconstruction metrics (`*_obj`), which is
# the fixed box of `region_masks.object_mask_ra`.
#   RADIal: width from the camera bbox (w_px * R / fx), times
#           `car_len_over_width` along range.
#   Boreas: per-object w / l / rot_y from the labels -> exact half-extents.
# The along-range extent on RADIal is an assumption (a bbox cannot give length,
# and RADIal carries no yaw); it is too deep and too narrow for crossing traffic.


def _box_axes(R_m: float, A_deg: float) -> tuple:
    """(centre xy, along-range unit vector, across-range unit vector).

    The box is axis-aligned in the line-of-sight frame, which is what
    `ra_partition.footprint_bins` rasterizes (`half_range_m` along range,
    `half_cross_m` across it) — so the point-in-box test here and the RA-grid
    footprint the partition carves are the same rectangle, one in metres and one
    in bins."""
    a = math.radians(float(A_deg))
    u_r = np.array([math.cos(a), math.sin(a)])
    u_a = np.array([-math.sin(a), math.cos(a)])
    return float(R_m) * u_r, u_r, u_a


def points_in_box(pc: np.ndarray, box: dict) -> np.ndarray:
    """Boolean mask of the points of `pc` inside one box."""
    if len(pc) == 0:
        return np.zeros(0, bool)
    c, u_r, u_a = _box_axes(box["R_m"], box["A_deg"])
    d = np.asarray(pc[:, :2], np.float64) - c
    lr, la = d @ u_r, d @ u_a
    return (np.abs(lr) <= float(box["half_range_m"])) & (
        np.abs(la) <= float(box["half_cross_m"])
    )


def radargen_box_metrics(
    pred_clouds: Sequence[np.ndarray],
    gt_clouds: Sequence[np.ndarray],
    box_lists: Sequence[Optional[Sequence[dict]]],
) -> dict:
    """RadarGen's foreground family: the box half of `compute_box_metrics_and_stats`
    + `finalize_global_metrics` that the paper reports (Appendix C.1). The
    conditional foreground CD / CD-Full / MMD are omitted: the baselines miss
    most objects, which leaves them undefined or evaluated on a small detected
    subset.

    box_lists: per frame, a list of {"R_m", "A_deg", "half_cross_m",
    "half_range_m", ...} -- half-extents in metres from `ra_partition.car_extent_m`.

    Emitted keys (RadarGen definitions):
      pc_box_hit_rate     : n_hits / (n_hits + n_fn) — the fraction of boxes that
                            have GT points in which the render also puts points.
      pc_box_miss_rate    : its complement.
      pc_box_density_sim  : Density Similarity, min(N,M)/max(N,M) inside the box,
                            averaged over every box, including empty ones (1.0
                            when both are empty, 0.0 when one is).
      pc_box_n_fp         : boxes with no GT points where the render put points
                            (raw count).
    plus the raw counts pc_box_n_hits / pc_box_n_fn / n_pc_boxes.

    Aggregation is over boxes, not frames (as in their `aggregate_metrics`), and
    hit/miss pool raw counts, so a frame with six vehicles weighs six times a
    frame with one."""
    nan = float("nan")
    dens = []
    n_hits = n_fn = n_fp = 0
    n_boxes = 0
    for pc, gc, boxes in zip(pred_clouds, gt_clouds, box_lists):
        if not boxes:
            continue
        for box in boxes:
            n_boxes += 1
            n_s = int(points_in_box(pc, box).sum())
            n_g = int(points_in_box(gc, box).sum())
            # Unconditional, as in their loop.
            dens.append(
                1.0
                if n_s == 0 and n_g == 0
                else (0.0 if n_s == 0 or n_g == 0 else min(n_s, n_g) / max(n_s, n_g))
            )
            if n_g:
                if n_s:
                    n_hits += 1
                else:
                    n_fn += 1
            elif n_s:
                n_fp += 1

    tot = n_hits + n_fn
    return {
        "pc_box_density_sim": float(np.mean(dens)) if dens else nan,
        "pc_box_hit_rate": (n_hits / tot) if tot else nan,
        "pc_box_miss_rate": (n_fn / tot) if tot else nan,
        "pc_box_n_hits": int(n_hits),
        "pc_box_n_fn": int(n_fn),
        "pc_box_n_fp": int(n_fp),
        "n_pc_boxes": int(n_boxes),
    }


# ---------------------------------------------------------------------------
# 4. Vehicle recall (selection criterion)
# ---------------------------------------------------------------------------


def object_detection_metrics(
    pred_clouds: Sequence[np.ndarray],
    gt_clouds: Sequence[np.ndarray],
    label_lists: Sequence[Optional[Sequence[dict]]],
    p: MetricParams = MetricParams(),
) -> dict:
    """Object-level detection of the labelled vehicles from the radar point cloud
    (the object recall of the selection protocol, Appendix B.2).

    GT is the sparse set of labelled vehicles ({'R_m', 'A_deg'} per frame); a
    label is detected if the cloud has a point within `pc_obj_gate_m`. Precision
    is not reported, since the cloud is dominated by real but unlabelled
    clutter/structure. Computed for both pred and the GT-tensor cloud:
    `obj_recall_gt` is the ceiling (label detectability in GT under this CFAR),
    `obj_recall_pred` what the render achieves. Pools all labels across frames."""

    def _nn(cloud, lxy):
        if len(cloud) == 0:
            return np.full(len(lxy), np.inf)
        return cKDTree(cloud[:, :2]).query(lxy)[0]

    d_pred, d_gt = [], []
    for pc, gc, labs in zip(pred_clouds, gt_clouds, label_lists):
        if not labs:
            continue
        R = np.array([float(lbl["R_m"]) for lbl in labs])
        A = np.deg2rad(np.array([float(lbl["A_deg"]) for lbl in labs]))
        lxy = np.stack([R * np.cos(A), R * np.sin(A)], axis=1)
        d_pred.append(_nn(pc, lxy))
        d_gt.append(_nn(gc, lxy))
    if not d_pred:
        return {"n_obj_labels": 0}
    dp, dg = np.concatenate(d_pred), np.concatenate(d_gt)
    gate = p.pc_obj_gate_m
    return {
        "obj_recall_pred": float(np.mean(dp < gate)),
        "obj_recall_gt": float(np.mean(dg < gate)),
        "n_obj_labels": int(len(dp)),
    }
