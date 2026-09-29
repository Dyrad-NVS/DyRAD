"""Normalization of radar tensors (paper Sec. 4.1 "each view is clipped and normalized on a fixed
linear range").

`compute_norm_params` derives a sequence's parameters (linear range, log map, counts mode for
Navtech log-compressed data) from its training frames. `normalize` is the reference map of raw
values to [0, 1] with them, checked by the tests; the trainer and the scorers apply the same
formulas in trainer/measurement.py and evaluation/radar_metrics.py. The per-sequence parameters
live in `<seq_dir>/norm.json` (`resolve`), and a derived sequence (an off-path view) inherits
its parent's (`inherit`), so every consumer of a sequence normalizes it the same way.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LOG_CLIP = 1e-10  # floor of log10(x / hi) in the log-map statistics
_EPS = 1e-30


# ---------------------------------------------------------------------------
# Per-marginal linear normalization ranges
# ---------------------------------------------------------------------------
# RADIal only (score_renders applies these ranges when dataset == "radial").
# The cube normalization range `lin_lo/lin_hi = 1833/5.9e6` is derived on the RAD cube. RA/RD/AD
# are averages over 16 Doppler, 751 azimuth and 447 (cropped) range bins, so they have different
# distributions; on the cube's normalization range they would occupy only the bottom few percent
# of [0,1], which inflates PSNR, makes SSIM's C1/C2 constants too large relative to the signal,
# and (for baselines that train on 8-bit RA images) wastes most quantization levels. Each view
# therefore has its own normalization range, derived by the same rule as the cube one over the train frames of the evaluation
# sequences (scoring crop [15:462]):
#       lo = median over sequences of that view's p0.01
#       hi = min    over sequences of that view's p99.99
# Applied to the cube, this rule reproduces the cube normalization range to within a few percent.
#
# AD is the exception. An AD frame is 16 x 751 = 12016 bins, so p99.99 is a ~1-bin noise
# estimate and lands below the median p99.9; AD therefore anchors on `median p99.9`, which
# gives it the same clipping regime as the other views (well under 1% of bins clipped).
#
# The normalization ranges are train-derived, so individual val frames can exceed them; `hi` is an anchor,
# not a maximum.
MARGINAL_LIN_RANGE = {
    "rad": (1833.0, 5.9e6),  # the RAD cube
    "ra": (47164.1, 2.5567e6),  # mean over Doppler
    "rd": (52788.9, 2.6082e6),  # mean over azimuth
    "ad": (147410.2, 6.7744e5),  # mean over range   (hi = median p99.9, see above)
}


def marginal_lin_range(view: str) -> tuple:
    """(lin_lo, lin_hi) for one view: 'rad' | 'ra' | 'rd' | 'ad'.

    Global across sequences (like the cube normalization range) and GT-derived, so it is
    method-independent. See MARGINAL_LIN_RANGE above for the derivation. Score an
    RA/RD/AD image on its own range, not on the 'rad' range.
    """
    v = str(view).lower()
    if v not in MARGINAL_LIN_RANGE:
        raise ValueError(f"unknown view {view!r}; expected one of {sorted(MARGINAL_LIN_RANGE)}")
    return MARGINAL_LIN_RANGE[v]


def compute_norm_params(
    train_frames,
    hi: float,
    *,
    pct_lo: float,
    pct_hi: float,
    k_median: float,
    mode: str,
    lin_lo: float,
    counts_per_decade: float,
    counts_full_scale: float,
    max_elems: int = 1 << 24,
) -> dict:
    """Derive per-sequence normalization params from an iterable of raw GT tensors
    (unscaled, sensor-native units, any shape). Returns a JSON-serializable dict.

    mode="linear": the linear normalization range (lin_lo, hi). norm_lo/norm_range/floor_u
        are the log map with its black point at the data's own pct_lo log-percentile (the
        measured noise floor stays data, not 0) and its top at pct_hi; they serve the
        secondary log views (display panels, *_N metrics).
    mode="counts": the sensor's native log-count convention (see below); norm_lo/norm_range/
        floor_u are the counts map.

    Both modes also emit the counts parameterization (counts_*), the noise floor
    nf = k_median * median(RA) and its inputs. The trainer's render pedestal is floor_u;
    nf is read only by the sensor-transfer gain (evaluation/sensor_transfer.py)."""
    if mode not in ("linear", "counts"):
        raise ValueError(f"mode must be 'linear' or 'counts', got {mode!r}")
    hi = float(hi)
    logs = []  # log10 of scaled GT (for the log-percentile map)
    ra_meds = []  # positive RA values, for the noise floor nf
    for t in train_frames:
        t = np.asarray(t, np.float64)
        logs.append(np.log10(np.clip(t / hi, _LOG_CLIP, None)).ravel())
        # RA = mean over Doppler, the same projection as `radar_metrics.ra_project()`
        # (axis 0 for [D,R,A], axis 1 for [B,D,R,A]). With the mean, k_median=1.5 puts
        # the black point at ~1.5x the GT cube's per-bin background level.
        ra = t.mean(axis=1) if t.ndim == 4 else (t.mean(axis=0) if t.ndim == 3 else t)
        ra = ra.ravel()
        ra_meds.append(ra[ra > 0.0])

    pos_ra = np.concatenate(ra_meds) if ra_meds else np.array([1e-10])
    if pos_ra.size > max_elems:
        pos_ra = np.random.default_rng(0).choice(pos_ra, max_elems, replace=False)
    med_ra = float(np.median(pos_ra)) if pos_ra.size else 1e-10
    nf = float(k_median * med_ra)

    allg = np.concatenate(logs)
    if allg.size > max_elems:
        allg = np.random.default_rng(0).choice(allg, max_elems, replace=False)

    # Log map: black point at the data's own low log-percentile, so the measured
    # noise floor survives as data. pct_hi anchors the top of the range.
    log_lo = float(np.percentile(allg, pct_lo))
    log_hi = float(np.percentile(allg, pct_hi))
    log_range = float(max(log_hi - log_lo, 1e-6))
    log_floor_u = float(10.0**log_lo)  # ceiling-units value of the black-point

    # "counts" (log-compressed 8-bit sensors, e.g. Boreas/Navtech): reproduce the
    # sensor's own convention, N = u / full_scale, where u is the raw count. The
    # converter stores P = 10^(u/k), so u = k*log10(P) recovers it exactly. In the shared
    # affine form N = (log10(x/hi) - norm_lo)/norm_range this is:
    #     norm_lo    = -log10(hi)          (so u=0 -> x=1 -> N=0)
    #     norm_range = full_scale / k
    # This matches RadarSplat's image/255 input, so the baseline and ours consume an
    # identical GT.
    cnt_lo = cnt_range = None
    if mode == "counts":
        if counts_per_decade <= 0:
            raise ValueError(
                "mode='counts' requires counts_per_decade > 0 "
                "(the converter's --db-counts-per-decade)"
            )
        cnt_lo = -math.log10(max(hi, _EPS))
        cnt_range = float(counts_full_scale) / float(counts_per_decade)
        cnt_floor_u = 1.0 / max(hi, _EPS)  # u=0 <=> raw power 1.0
        act_lo, act_range, act_floor_u = cnt_lo, cnt_range, cnt_floor_u
    else:  # linear: normalize() uses lin_lo/lin_hi; the log map is the secondary view
        act_lo, act_range, act_floor_u = log_lo, log_range, log_floor_u
    return {
        "mode": mode,
        # linear parameterization: the shared global normalization range, raw units.
        "lin_lo": float(lin_lo),
        "lin_hi": float(hi),
        # active params, used by normalize() and every consumer
        "hi": hi,
        "norm_lo": float(act_lo),
        "norm_range": float(act_range),
        "floor_u": float(act_floor_u),
        # log-percentile map inputs
        "pct_lo": float(pct_lo),
        "pct_hi": float(pct_hi),
        # counts parameterization (sensor-native 8-bit log counts; None if unused)
        "counts_per_decade": float(counts_per_decade) if counts_per_decade > 0 else None,
        "counts_full_scale": float(counts_full_scale),
        "counts_norm_lo": cnt_lo,
        "counts_norm_range": cnt_range,
        "nf": nf,  # noise floor in unscaled units (= k*median_RA)
        "median_ra": med_ra,
        "k_median": float(k_median),
    }


def normalize(raw_power, params: dict):
    """Map raw GT (unscaled) → canonical N in [0,1]. Accepts np array/scalar.

    mode="linear":  N = clip((x - lo) / (hi - lo), 0, 1)   [x in raw units]
        The shared global normalization range (lin_lo, lin_hi), used for both training and scoring.
        The noise floor keeps its natural position (~9.5% of the range), and the range
        is not spent on noise the way a log map spends it (the noise floor is only
        ~20 dB below the ceiling).
    mode="counts": N = u / full_scale with u = k*log10(power), the sensor's 8-bit
        log counts (identical to RadarSplat's image/255).
    Any other mode raises ValueError.
    """
    if params.get("mode") == "counts":
        # 8-bit log counts: N = u / full_scale, u = k*log10(power).
        k = float(params["counts_per_decade"])
        full = float(params["counts_full_scale"])
        x = np.asarray(raw_power, np.float64)
        u = k * np.log10(np.clip(x, 1.0, None))  # power 1.0 == count 0
        return np.clip(u / full, 0.0, 1.0).astype(np.float32)
    if params.get("mode") == "linear":
        lo, hi = float(params["lin_lo"]), float(params["lin_hi"])
        x = np.asarray(raw_power, np.float64)
        return np.clip((x - lo) / max(hi - lo, 1e-30), 0.0, 1.0).astype(np.float32)
    raise ValueError(f"normalize: unknown mode {params.get('mode')!r}")


SIDECAR = "norm.json"


class NormError(RuntimeError):
    """A sequence's normalization could not be resolved."""


@dataclass(frozen=True)
class NormParams:
    """Resolved normalization for one sequence."""

    params: dict
    source: Path

    # ── the numbers ────────────────────────────────────────────────────────
    @property
    def mode(self) -> str:
        return str(self.params["mode"])

    @property
    def hi(self) -> float:
        """Ceiling in raw power units. The normalization divisor."""
        return float(self.params["hi"])

    @property
    def lin_lo(self) -> float:
        return float(self.params["lin_lo"])

    @property
    def lin_hi(self) -> float:
        return float(self.params["lin_hi"])

    @property
    def norm_scale(self) -> float:
        """Multiplier taking raw power -> CEILING_UNITS (i.e. 1/hi).

        This is RadarDataset.gt_power_scale and the factor a view writer must undo
        before writing to disk.
        """
        return 1.0 / max(self.hi, 1e-30)

    def __repr__(self) -> str:
        return (
            f"NormParams(mode={self.mode}, hi={self.hi:.6g}, "
            f"lin=[{self.lin_lo:.6g}, {self.lin_hi:.6g}], src={self.source})"
        )


def sidecar_path(seq_dir) -> Path:
    return Path(seq_dir) / SIDECAR


def resolve(seq_dir) -> NormParams:
    """Resolve a sequence's normalization from its sidecar; raises NormError without one."""
    p = sidecar_path(seq_dir)
    if not p.is_file():
        raise NormError(
            f"no {SIDECAR} for sequence {seq_dir}.\n"
            f"  Every sequence must declare its normalization. Derived sequences "
            f"(written renders) inherit their parent's via norm.inherit().\n"
            f"  Write it with: python -m dyrad.preprocessing.normalize_dataset "
            f"--configs <run config of the sequence>"
        )
    try:
        params = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise NormError(f"unreadable {p}: {e}") from e
    for k in ("mode", "hi"):
        if k not in params:
            raise NormError(f"{p} is missing required key {k!r}")
    return NormParams(params=params, source=p)


def write_sidecar(seq_dir, params: dict, note: str = "") -> Path:
    """Write a sequence's norm.json."""
    d = Path(seq_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = dict(params)
    if note:
        out["_note"] = note
    p = sidecar_path(d)
    p.write_text(json.dumps(out, indent=2) + "\n")
    return p


def inherit(child_seq_dir, parent_seq_dir, note: str = "") -> Path:
    """Give a derived sequence its parent's normalization, verbatim.

    A derived sequence is a written render of the parent scene and is later scored
    against the parent's real GT, so it must sit on the parent's normalization range. Copying
    (rather than recomputing from the render's own statistics) keeps the
    comparison on one scale.
    """
    par = resolve(parent_seq_dir)
    return write_sidecar(
        child_seq_dir,
        par.params,
        note=note
        or f"inherited from {Path(parent_seq_dir).name} "
        f"(derived sequence: same normalization range as its parent)",
    )
