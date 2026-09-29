"""Shifted-label generator for the real-data off-path benchmark.

Labels for a shifted view come from the GT annotation CSV, never from a model.
For each detection:

  1. rebuild the sensor-frame point from `radar_R_m` / `radar_A_deg`,
  2. project it through the original (unshifted) T0 pose -> a fixed world point,
  3. re-project that world point into the shifted pose -> the (range, azimuth)
     the object subtends from the new viewpoint (the parallax),
  4. keep the GT raw Doppler bin unchanged (it is not re-projected: the shift changes
     the radial component with the azimuth; the field only places the visualization's
     crosshair),
  5. gate on range and field of view.

Taking positions from a model's own track instead would be circular (it would grade
M0 against M0's trajectory).

Two deliberate differences from the source CSV, in every view (T0 included):

* Labels closer than `MIN_RANGE_M` (5 m) are dropped. The on-path scorer has no such
  gate, so at shift = 0 the T0 labels are the source labels minus those rows, and the
  off-path object metrics are computed on that smaller label set.
* The camera bbox `x1_pix..y2_pix` is scaled about its centre by R_T0 / R_view, the
  size a pinhole camera at the shifted pose would see. Every consumer sizes a RADIal
  object as `(x2_pix - x1_pix) * R / fx`, so each view keeps the object's annotated
  physical width. Boreas has no camera bbox (written as 0); its footprint comes from
  the re-projected `box_*` columns.

Field conventions:

* Use `radar_R_m` / `radar_A_deg` only. RADIal's `laser_X_m` / `laser_Y_m` are
  axis-swapped and sign-flipped relative to `radar_A_deg`. The written
  `laser_*_m` / `radar_X_m` / `radar_Y_m` are the shifted sensor-frame point
  (x forward, y left), not the source's convention; readers use R / A.
* Never fabricate `vehicle_id`. The trainer decides whether to run its own
  motion-aware tracker by column presence, so a synthesized id disables it, and
  an id built from rounded range/azimuth would split one car into many
  single-point objects. The column is emitted only when the source has real
  track ids (Boreas does, RADIal does not).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from dyrad.labels import remap_label_index
from dyrad.poses import invert_c2w
from dyrad.ra_partition import box_half_extents

# Written in this order; `vehicle_id` is appended only when the source had one.
FIELDNAMES = [
    "index",
    "numSample",
    "x1_pix",
    "y1_pix",
    "x2_pix",
    "y2_pix",
    "laser_X_m",
    "laser_Y_m",
    "laser_Z_m",
    "radar_X_m",
    "radar_Y_m",
    "radar_R_m",
    "radar_A_deg",
    "radar_D_mps",
    "dataset",
    "dataset_index",
    "difficult",
    "Annotation",
    # Boreas only: the object's real box, re-projected for the shifted view.
    "box_w_m",
    "box_l_m",
    "box_half_cross_m",
    "box_half_range_m",
    "box_yaw_rad",
]

MIN_RANGE_M = 5.0  # near-range label gate; see the module docstring


def reproject(
    csv_path,
    *,
    seq_str: str,
    out_seq_name: str,
    poses_t0: np.ndarray,
    poses_shifted: np.ndarray,
    frame_start: int,
    frame_end: int,
    far_range_m: float,
    az_fov_deg: float,
    remap=None,
) -> list[dict]:
    """Reproject the GT labels of one window into the shifted view.

    Returns CSV-ready row dicts, re-indexed to local frames 0..N-1. Pure
    function of the CSV and the two pose sets -- no model, no GPU.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise SystemExit(f"GT label CSV not found: {csv_path}")
    if not seq_str:
        raise SystemExit(f"no seq_str to select the window's rows from {csv_path}")

    import pandas as pd  # heavy; import late

    df = pd.read_csv(csv_path)
    if "dataset" in df.columns:
        df = df[df["dataset"] == seq_str]
    has_src_vid = "vehicle_id" in df.columns

    rows: list[dict] = []
    for _, row in df.iterrows():
        gfi = remap_label_index(int(row["index"]), remap)
        if gfi < frame_start or gfi >= frame_end:
            continue
        li = gfi - frame_start

        # sensor -> world through the original pose (trusted R/A fields only)
        R_m = float(row["radar_R_m"])
        A_r = math.radians(float(row["radar_A_deg"]))
        p_s0 = np.array([R_m * math.cos(A_r), R_m * math.sin(A_r), 0.0, 1.0])
        p_world = (poses_t0[gfi] @ p_s0)[:3]

        # world -> shifted sensor frame (the parallax)
        Rwc, twc = invert_c2w(poses_shifted[gfi])
        p_r = Rwc @ p_world + twc
        r_m = float(np.linalg.norm(p_r))
        az_deg = float(math.degrees(math.atan2(p_r[1], p_r[0])))

        if r_m < MIN_RANGE_M or r_m > float(far_range_m):
            continue
        # `p_r[0] <= 0` is a forward-looking cull, correct for RADIal's ~150 deg
        # wedge and wrong for a 360 deg scanning radar (Navtech/Boreas), where it
        # would drop every object behind the vehicle.
        fov = float(az_fov_deg)
        if fov < 359.0 and p_r[0] <= 0:
            continue
        # The half-FOV cull is meaningful only for a sensor with a real blind
        # sector. At fov >= 359 it would leave a hairline dead wedge around 180 deg
        # into which a lane shift can move objects directly behind the ego. A
        # 360 deg scanner has no blind sector, so nothing is culled.
        if fov < 359.0 and abs(az_deg) > 0.5 * fov:
            continue

        rec = {
            "index": li,
            "numSample": li,
            **_scaled_bbox(row, R_m / r_m),
            "laser_X_m": float(p_r[0]),
            "laser_Y_m": float(p_r[1]),
            "laser_Z_m": float(p_r[2]),
            "radar_X_m": float(p_r[0]),
            "radar_Y_m": float(p_r[1]),
            "radar_R_m": r_m,
            "radar_A_deg": az_deg,
            # Doppler-less sensors (Navtech/Boreas) have no radar_D column;
            # -1 is the trainer's "no Doppler reading" sentinel.
            "radar_D_mps": (int(row["radar_D_mps"]) if "radar_D_mps" in row.index else -1),
            "dataset": out_seq_name,
            "dataset_index": li,
            "difficult": 0,
            "Annotation": str(row.get("Annotation", "strong")),
        }
        # ── the object's box, re-projected into the shifted view ──────────────
        # Boreas labels carry real w/l/yaw, which the RA partition uses to size the
        # footprint. The shifted pose keeps the T0 rotation (view.shift_poses), so the
        # sensor-frame `box_yaw_rad` is unchanged; the box is seen at a different
        # azimuth, so its along/across-range split is recomputed.
        if "box_yaw_rad" in row.index and pd.notna(row["box_yaw_rad"]):
            yaw = float(row["box_yaw_rad"])
            bw, bl = float(row["box_w_m"]), float(row["box_l_m"])
            hc, hr = box_half_extents(bw, bl, yaw - math.radians(az_deg))
            rec.update(
                {
                    "box_w_m": bw,
                    "box_l_m": bl,
                    "box_yaw_rad": yaw,
                    "box_half_cross_m": hc,
                    "box_half_range_m": hr,
                }
            )
        if has_src_vid:  # real track id only — see module doc
            rec["vehicle_id"] = str(row["vehicle_id"])
        rows.append(rec)
    return rows


def _scaled_bbox(row, scale: float) -> dict:
    """The camera bbox scaled about its centre by `scale`; zeros when the row has none."""
    keys = ("x1_pix", "y1_pix", "x2_pix", "y2_pix")
    try:
        x1, y1, x2, y2 = (float(row[k]) for k in keys)
    except (KeyError, TypeError, ValueError):
        return dict.fromkeys(keys, 0)
    if not all(map(math.isfinite, (x1, y1, x2, y2))) or x2 <= x1:
        return dict.fromkeys(keys, 0)
    cx, cy, hw, hh = 0.5 * (x1 + x2), 0.5 * (y1 + y2), 0.5 * (x2 - x1), 0.5 * (y2 - y1)
    return {  # integer pixels, as in the source annotation
        "x1_pix": round(cx - hw * scale),
        "y1_pix": round(cy - hh * scale),
        "x2_pix": round(cx + hw * scale),
        "y2_pix": round(cy + hh * scale),
    }


def write(out_dir, rows: list[dict], filename: str) -> Path:
    """Write the label CSV. `vehicle_id` appears only if some row has one.

    `filename` is dataset-specific (RADIal provides labels_CVPR.csv, Boreas
    labels_boreas.csv) and must match what the M1 configs' `object_label_path`
    points at — see spec.labels_filename().
    """
    fields = list(FIELDNAMES)
    if any("vehicle_id" in r for r in rows):
        fields.append("vehicle_id")
    path = Path(out_dir) / filename
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def reproject_and_write(out_dir, csv_path, *, filename: str, **kw) -> list[dict]:
    rows = reproject(csv_path, **kw)
    write(out_dir, rows, filename=filename)
    return rows
