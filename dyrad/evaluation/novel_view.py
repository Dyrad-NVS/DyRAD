"""Evaluate a trained scene on a sequence directory it was not trained on.

Used by the synthetic off-path evaluation (displaced ground-truth views) and by the
real-data off-path protocol (M1 rendered back at the original poses). The runner's
evaluation data are swapped for the novel sequence and `Runner._evaluate` writes
`metrics.json` and, optionally, `renders_npy/` into `out_dir`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dyrad.data import RadarDataset, RadarParser
from dyrad.poses import load_ego_velocity


def build_loaders(cfg, seq_dir: Path):
    """RadarParser and datasets over a novel sequence dir, on the config's window; every
    frame goes to the val split."""
    parser = RadarParser(
        rad_tensors_dir=str(seq_dir / "rad_tensors"),
        poses_dir=str(seq_dir / "poses_can"),
        test_every=1,
        test_offset=0,  # (i - 0) % 1 == 0: all frames are val
        num_doppler_bins=cfg.num_doppler_bins,
        num_range_bins=cfg.num_range_bins,
        num_azimuth_bins=cfg.num_azimuth_bins,
        range_crop_first=cfg.range_crop_first,
        range_crop_last=cfg.range_crop_last,
        doppler_roll_bins=cfg.doppler_roll_bins,
    )
    common = dict(
        frame_start=cfg.frame_start,
        frame_end=cfg.frame_end,
        bad_frame_ids=cfg.bad_frame_ids,
    )
    val_ds = RadarDataset(parser, split="val", **common)  # all frames
    empty_ds = RadarDataset(parser, split="train", **common)  # empty
    return parser, val_ds, empty_ds


def view_ego_velocity(seq_dir: Path, n_frames: int, device):
    """The view's own sensor-frame ego velocity (the base one differs off-path: under
    yaw, and under a lateral shift on a curve)."""
    p = seq_dir / "ego_vel_can.npy"
    if not p.exists():
        raise SystemExit(f"{p} missing: every view carries its own sensor-frame ego velocity")
    return torch.from_numpy(load_ego_velocity(p, n_frames)).to(device)


def evaluate_on_sequence(
    runner,
    cfg,
    seq_dir: Path,
    out_dir: Path,
    labels_filename: str = "labels_CVPR.csv",
) -> dict:
    """Swap the novel sequence into `runner`, run its evaluation (renders saved to
    <out_dir>/renders_npy/), return the headline row."""
    parser, val_ds, empty_ds = build_loaders(cfg, seq_dir)
    print(f"  frames: {len(val_ds)} (val) + {len(empty_ds)} (train)")

    # The datasets built here must use the model's normalization scale (1/hi), which the
    # trainer applies only to the datasets it built itself.
    scale = 1.0 / float(runner.gt_hi_power)
    for ds in (val_ds, empty_ds):
        ds.gt_power_scale = scale
        ds._pair_cache.clear()

    runner.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    runner.train_loader = DataLoader(empty_ds, batch_size=1, shuffle=False, num_workers=0)
    runner.v_ego_smooth = view_ego_velocity(seq_dir, len(parser.poses), runner.device)

    # Labels of the novel sequence (for the object-region and label-Doppler metrics); the
    # label loader filters by the config's window. They are filtered on the view's
    # `dataset` value: the dir name, or the CSV's only value when the dir name is absent.
    cfg.object_label_path = str(seq_dir / labels_filename)
    cfg.rad_tensors_dir = str(seq_dir / "rad_tensors")
    label_seq = seq_dir.name
    csv_path = seq_dir / labels_filename
    if csv_path.exists():
        with open(csv_path) as f:
            vals = sorted({row["dataset"] for row in csv.DictReader(f)})
        if vals and label_seq not in vals:
            if len(vals) != 1:
                raise ValueError(
                    f"{csv_path}: no label row has dataset == {label_seq!r} and the "
                    f"CSV holds several sequences {vals}"
                )
            label_seq = vals[0]
    runner._labels_by_frame = runner._load_labels_by_frame(label_seq, poses=parser.poses)

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.result_dir = str(out_dir)
    cfg.save_renders_npy = True
    runner._metrics = []
    runner._evaluate(0)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    return metrics[-1] if metrics else {}
