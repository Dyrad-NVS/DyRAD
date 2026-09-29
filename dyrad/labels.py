"""Label frame indexing and the label-CSV detection reader, shared by the trainer,
preprocessing and evaluation (pure numpy).

The label CSV's `index` column is the RADIal authors' frame-table position
(generate_database.py, SyncReader tolerance=20000); our preprocessing uses
tolerance=200000, which keeps frames the authors dropped and shifts local indices.
`label_index_remap.npy`, saved per sequence dir next to `rad_tensors/`, maps the official
`index` to our local frame (`remap[official] = local`, -1 = unmapped). Synthetic and
off-path sequences write the identity remap.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

REMAP_FILENAME = "label_index_remap.npy"

#: A camera bbox narrower than this (pixels) is no usable width measurement.
MIN_BBOX_W_PX = 2


def load_index_remap(seq_dir) -> Optional[np.ndarray]:
    """The sequence's official->local frame remap, or None when it has none."""
    p = Path(seq_dir) / REMAP_FILENAME
    return np.load(p) if p.exists() else None


def remap_label_index(idx: int, remap: Optional[np.ndarray]) -> int:
    """Apply the official->local frame remap; returns -1 if unmappable."""
    if remap is None:
        return idx
    if idx < 0 or idx >= len(remap):
        return -1
    return int(remap[idx])


def _float_or_none(row: dict, key: str) -> Optional[float]:
    v = row.get(key)
    return None if v in ("", None) else float(v)


def read_detections(csv_path, seq_name: str, remap: Optional[np.ndarray]) -> dict:
    """{local frame: [detection]} from a label CSV, each detection the inputs of
    `ra_partition.car_extent_m`:

        R_m, A_deg                     the label position
        w_px                           camera-bbox width x2_pix - x1_pix (RADIal), None
                                       without a usable bbox (<= MIN_BBOX_W_PX or absent)
        half_cross_m, half_range_m     the metric box (Boreas), None when absent

    Rows whose `dataset` differs from `seq_name` are skipped (an empty `seq_name` keeps
    every row); `index` goes through `remap_label_index`, and unmapped rows are skipped,
    as are the empty-frame rows of labels_CVPR.csv (radar_R_m = -1, no detection).
    """
    out: dict = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if seq_name and row.get("dataset") != seq_name:
                continue
            fidx = remap_label_index(int(row["index"]), remap)
            if fidx < 0:
                continue
            R_m = float(row["radar_R_m"])
            if R_m <= 0:
                continue
            x1, x2 = _float_or_none(row, "x1_pix"), _float_or_none(row, "x2_pix")
            w_px = None if x1 is None or x2 is None else x2 - x1
            out.setdefault(fidx, []).append(
                {
                    "R_m": R_m,
                    "A_deg": float(row["radar_A_deg"]),
                    "w_px": w_px if w_px is not None and w_px > MIN_BBOX_W_PX else None,
                    "half_cross_m": _float_or_none(row, "box_half_cross_m"),
                    "half_range_m": _float_or_none(row, "box_half_range_m"),
                }
            )
    return out
