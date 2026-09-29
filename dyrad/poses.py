"""Radar poses: the preprocessing output loader and the radar->world (c2w) inverses.

`load_poses` reads a poses directory (radar_poses.npy + poses_metadata.json) and
`load_ego_velocity` the per-pose sensor-frame ego velocity.
`invert_c2w` is the numpy rigid-body inverse used by the label reprojection and the
synthetic generator; `c2w_to_w2c` the torch matrix inverse the renderer uses (torch is
imported there, so the numpy consumers stay torch-free).
"""

import json
import os
from typing import Tuple

import numpy as np


def load_poses(poses_dir: str) -> Tuple[np.ndarray, dict]:
    """(poses [N,4,4] radar->world, metadata dict) of a poses directory holding
    radar_poses.npy and poses_metadata.json (the metadata carries the per-pose RAD frame
    ids, `tesseract_indices`)."""
    poses_path = os.path.join(poses_dir, "radar_poses.npy")
    metadata_path = os.path.join(poses_dir, "poses_metadata.json")

    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Poses file not found: {poses_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    poses = np.load(poses_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return poses, metadata


def invert_c2w(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """World->radar rigid-body inverse of one [4,4] c2w pose: R_w2c = R^T, t_w2c = -R^T @ t."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    Rinv = R.T
    return Rinv, -(Rinv @ t)


def c2w_to_w2c(c2w):
    """Invert a batch of radar->world poses: [..., 4, 4] c2w -> [..., 4, 4] world->radar."""
    import torch

    return torch.inverse(c2w).contiguous()


def load_ego_velocity(path, n_frames: int) -> np.ndarray:
    """Sensor-frame ego velocity [n_frames, 2] float32 from an `ego_vel*.npy` file
    (CAN speed or pose-derived), one row per pose."""
    v = np.load(path).astype(np.float32)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError(f"{path}: ego velocity must be [N, 2], got shape {v.shape}")
    if len(v) != n_frames:
        raise ValueError(f"{path}: {len(v)} ego-velocity rows for {n_frames} poses")
    return v
