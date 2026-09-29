"""Object-region mask M_obj for the region-decomposed reconstruction metrics (`*_obj`).

Pure numpy. The mask is built from the trainer's per-frame label dicts
(`{"R_m", "A_deg"}`) by dilating each label centroid to a fixed box in (range,
azimuth), `MetricParams.obj_half_r_m` x `obj_half_a_deg` (±2.5 m x ±2.5 deg). This
approximates object extent in the absence of per-frame segmentation, and it is
partially conditioned on the label-based init, so not leak-free.

The mask is native to the RA (range x azimuth) grid; for the full RAD tensor
[D, R, A] it broadcasts over Doppler (an [R, A] mask does not localize in Doppler).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def object_mask_ra(
    frame_labels: Optional[Sequence[dict]],
    range_m: np.ndarray,
    az_deg: np.ndarray,
    half_r_m: float,
    half_a_deg: float,
) -> np.ndarray:
    """Boolean [R, A]: cells within (half_r_m, half_a_deg) of any label centroid.

    frame_labels: list of {"R_m", "A_deg"} dicts, or None.
    range_m [R], az_deg [A]: bin-centre axes (metres / degrees).
    """
    R, A = len(range_m), len(az_deg)
    mask = np.zeros((R, A), dtype=bool)
    if not frame_labels:
        return mask
    for lbl in frame_labels:
        r_ok = np.abs(range_m - float(lbl["R_m"])) <= half_r_m
        a_ok = np.abs(az_deg - float(lbl["A_deg"])) <= half_a_deg
        if r_ok.any() and a_ok.any():
            mask[np.ix_(r_ok, a_ok)] = True
    return mask
