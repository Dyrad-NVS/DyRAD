"""Invariants of the RadarGen point-cloud protocol (the paper's detection keys).

Properties checked:

  * `pc_da_*` uses the protocol's absolute power gate, so a per-cloud power gain
    is charged (as the protocol's |rcs_gt - rcs_syn| <= 8 dBsm gate charges a
    wrong RCS), while the geometry metrics (`pc_cd_loc_m`, `pc_iou_tau`) are
    unaffected by it.
  * Every DA value carries the protocol's eps = 1e-10, so a perfect score is
    0.9999999999, not 1.0. Compare with a tolerance, never with ==.
  * Geometry metrics are blind to wrong attributes (Doppler, power); DA is not.
  * A flooding cloud keeps full DA recall but loses precision, and its placement
    error shows in CD-Loc.
"""

import numpy as np

from dyrad.evaluation.pointcloud_metrics import radargen_pointcloud_metrics
from dyrad.evaluation.radar_metrics import MetricParams

RB = np.linspace(0, 100, 512)
P = MetricParams()


def _cloud(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(-50, 50, n),
            rng.uniform(0, 80, n),
            rng.uniform(-2.5, 2.5, n),
            10 ** rng.uniform(-1, 1, n),
        ]
    )


def _m(pred, gt):
    return radargen_pointcloud_metrics([pred], [gt], RB, P)


def _is1(x, tol=1e-6):
    """True if x is 1 within tol.

    Protocol recall/precision/F1 divide by (N + 1e-10), so a perfect score is
    0.9999999999 and an exact == 1.0 comparison would fail."""
    return abs(float(x) - 1.0) <= tol


def test_identity_is_perfect():
    gt = _cloud()
    r = _m(gt, gt)
    assert _is1(r["pc_da_f1"]) and _is1(r["pc_iou_tau"])
    assert r["pc_cd_loc_m"] == 0.0 and r["pc_cd_full"] == 0.0 and abs(r["pc_mmd_xy"]) < 1e-9


def test_power_gain_is_charged_by_da_only():
    gt = _cloud()
    dim = gt.copy()
    dim[:, 3] *= 10 ** (-1.81)
    r = _m(dim, gt)
    assert r["pc_da_f1"] < 0.1 and r["pc_cd_full"] > 1e-3
    assert r["pc_cd_loc_m"] == 0.0 and _is1(r["pc_iou_tau"])  # geometry untouched


def test_da_penalizes_corrupted_attributes():
    gt = _cloud()
    rng = np.random.default_rng(7)
    shuf = gt.copy()
    shuf[:, 3] *= 10 ** rng.normal(0, 0.6, len(gt))
    assert _m(shuf, gt)["pc_da_f1"] < 0.9
    flip = gt.copy()
    flip[:, 2] = -flip[:, 2]
    r = _m(flip, gt)
    assert r["pc_da_f1"] < 0.6
    assert r["pc_cd_loc_m"] == 0.0 and _is1(r["pc_iou_tau"])  # geometry is blind to Doppler


def test_flooding_keeps_recall_loses_precision():
    gt = _cloud()
    flood = np.vstack([gt, _cloud(1200, seed=3)])
    r = _m(flood, gt)
    assert _is1(r["pc_da_recall"]) and r["pc_da_precision"] < 0.3
    assert r["pc_cd_loc_m"] > 0.5 and r["pc_iou_tau"] < 0.5
