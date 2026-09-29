"""RAD tensor dataset for DyRAD training and evaluation.

- RAD tensors are stored full-range as .npy, one per frame, named rad_00033.npy.
- poses_dir holds radar_poses.npy + poses_metadata.json (`dyrad.poses.load_poses`); the
  metadata's 'tesseract_indices' lists the RAD frame id of each pose, 1:1 with the poses.
- A missing file or a shape mismatch raises.
"""

import glob
import os
import re
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from dyrad.config import is_held_out
from dyrad.poses import load_poses


def _frame_id_from_rad(path: str) -> int:
    """Frame id of a `rad_00033.npy` file (the name every producer writes)."""
    base = os.path.basename(path)
    m = re.fullmatch(r"rad_(\d+)\.npy", base)
    if m is None:
        raise ValueError(f"RAD filename is not rad_<frame id>.npy: {base}")
    return int(m.group(1))


class RadarParser:
    """Indexes RAD tensors + poses (aligned by frame id)."""

    def __init__(
        self,
        rad_tensors_dir: str,
        poses_dir: str,
        *,
        num_doppler_bins: int,
        num_range_bins: int,
        num_azimuth_bins: int,
        range_crop_first: int,
        range_crop_last: int,
        test_every: int = 0,  # holdout period over global frame indices (config.is_held_out)
        test_offset: int = 0,
        # Circular roll of the GT Doppler axis at load time. RADIal DDMA tensors
        # have bin 0 = DC (raw FFT bin mod 16); the renderer's symmetric axis puts
        # v=0 at bin D/2 — roll by D/2 (=8 for D=16) to align the conventions.
        doppler_roll_bins: int = 0,
        # Tolerate poses whose RAD tensor is absent instead of raising at scan time.
        # Needed when rad_tensors_dir holds only the training window of a recording
        # (the coarse sensor-configuration sequences). Missing entries are stored as
        # None and raise only if such a frame is read, which frame_start/frame_end
        # prevent. Off by default: for a full-sequence dir a missing file is an error.
        allow_missing_rad: bool = False,
    ):
        self.rad_tensors_dir = rad_tensors_dir
        self.allow_missing_rad = bool(allow_missing_rad)
        self.poses_dir = poses_dir
        self.test_every = int(test_every)
        self.test_offset = int(test_offset)

        self.num_doppler_bins = int(num_doppler_bins)
        self.num_range_bins = int(num_range_bins)
        self.num_azimuth_bins = int(num_azimuth_bins)
        self.doppler_roll_bins = int(doppler_roll_bins)
        if self.doppler_roll_bins:
            print(f"  [Doppler] rolling GT tensors by {self.doppler_roll_bins} bins along D")

        self.range_crop_first = int(range_crop_first)
        self.range_crop_last = int(range_crop_last)

        if self.range_crop_first < 0 or self.range_crop_last < 0:
            raise ValueError("range_crop_first/last must be non-negative")
        if self.range_crop_first + self.range_crop_last >= self.num_range_bins:
            raise ValueError("range cropping removes all bins")

        # ---- poses + frame ids ----
        print(f"\nLoading poses from: {poses_dir}")
        poses, meta = load_poses(poses_dir)
        self.poses = poses.astype(np.float32)

        # 'tesseract_indices' is the pose-metadata key for the per-pose RAD frame id,
        # written by every producer of a sequence.
        if not isinstance(meta, dict) or "tesseract_indices" not in meta:
            raise ValueError("poses_metadata must contain 'tesseract_indices' (1:1 with poses).")

        self.rad_indices: List[int] = list(meta["tesseract_indices"])
        if len(self.rad_indices) != len(self.poses):
            raise ValueError(
                f"Mismatch: len(tesseract_indices)={len(self.rad_indices)} "
                f"!= len(poses)={len(self.poses)}"
            )
        print(f"  Loaded {len(self.poses)} poses")

        # ---- RAD files by frame id ----
        print(f"\nScanning RAD tensors in: {rad_tensors_dir}")
        rad_files = sorted(glob.glob(os.path.join(rad_tensors_dir, "rad_*.npy")))
        if not rad_files:
            raise FileNotFoundError(f"No RAD .npy files found in: {rad_tensors_dir}")

        rad_by_t: Dict[int, str] = {}
        for f in rad_files:
            t = _frame_id_from_rad(f)
            rad_by_t[t] = f
        print(f"  Found {len(rad_by_t)} RAD files (unique frame ids)")

        # ---- align everything to poses order ----
        self.rad_tensor_files: List[str] = []

        missing_rad: List[int] = []

        for t in self.rad_indices:
            if t not in rad_by_t:
                missing_rad.append(t)

        if missing_rad and not self.allow_missing_rad:
            preview = ", ".join([f"{x:05d}" for x in missing_rad[:10]])
            raise FileNotFoundError(f"Missing RAD for {len(missing_rad)} frames. First: {preview}")
        if missing_rad:
            have = sorted(set(rad_by_t))
            print(
                f"[RadarParser] allow_missing_rad: {len(missing_rad)} of "
                f"{len(self.rad_indices)} poses have no RAD tensor; present ids "
                f"{have[0]:05d}..{have[-1]:05d} ({len(have)}). Reading any absent frame "
                f"will raise — keep frame_start/frame_end inside that range."
            )

        for t in self.rad_indices:
            self.rad_tensor_files.append(rad_by_t.get(t))

        self.num_frames = len(self.rad_tensor_files)
        print(f"\nAligned {self.num_frames} frames (poses/RAD)")

        # The cropped range window [_r0, _r1) of a full-range tensor.
        self._r0 = self.range_crop_first
        self._r1 = self.num_range_bins - self.range_crop_last


class RadarDataset(Dataset):
    """Returns the RAD tensor, pose and global index of a frame."""

    def __init__(
        self,
        parser: RadarParser,
        split: str = "train",
        frame_start: int = 0,
        frame_end: int = -1,
        bad_frame_ids=(),
        raw_unnormalized: bool = False,
    ):
        self.parser = parser
        self.split = split
        # Ceiling-units normalization: multiply every GT tensor (sensor-native units,
        # see domains.Domain.RAW_POWER) by 1/hi so the data ceiling maps to 1. A pure
        # scale, so it commutes with the incoherent power sum + PSF. Applied in _load_pair.
        #
        # The scale is resolved from the sequence's norm.json, not injected by the
        # caller, so every dataset built for a sequence (training, evaluation, novel
        # views) is in the same normalization. A sequence with no norm.json raises
        # (norm.NormError), and that error is meant to propagate.
        #
        # `raw_unnormalized` is the one exemption, for
        # preprocessing/normalize_dataset.py, the script that writes norm.json:
        # it must measure raw values and cannot resolve a normalization range it has
        # not computed yet (and re-running it on pre-scaled frames would write hi~1).
        # Training and scoring paths should never set it.
        from dyrad.norm import resolve as _resolve_norm

        self.raw_unnormalized = bool(raw_unnormalized)
        if self.raw_unnormalized:
            self._norm = None
            self.gt_power_scale = 1.0
            print(
                f"  [norm] {split}: BOOTSTRAP — normalization bypassed, frames stay in "
                f"raw sensor units. Valid only for writing this sequence's norm.json."
            )
        else:
            self._norm = _resolve_norm(os.path.dirname(parser.rad_tensors_dir.rstrip("/")))
            self.gt_power_scale = self._norm.norm_scale
            print(
                f"  [norm] {split}: scale from {self._norm.source.parent.name}"
                f"/norm.json  hi={self._norm.hi:.6g} "
                f"scale={self.gt_power_scale:.6g}"
            )

        # In-RAM cache of fully-processed frames (load+crop+roll+scale), keyed by
        # frame_idx. _load_pair is a pure function of frame_idx, so caching is
        # bit-for-bit identical to re-reading; it removes the per-step disk I/O on
        # the main thread (num_workers=0). A window is at most 60 frames.
        self._pair_cache: Dict[int, Dict[str, np.ndarray]] = {}

        idx = np.arange(parser.num_frames, dtype=np.int64)
        # Optional frame range restriction (applied before train/val split so
        # test_every modulo is computed against the original global frame indices).
        fs = int(frame_start)
        fe = int(frame_end) if int(frame_end) >= 0 else parser.num_frames
        if fs > 0 or fe < parser.num_frames:
            idx = idx[fs:fe]
            print(f"  [frame subset] using global frames {fs}–{fe - 1} ({len(idx)} frames)")
        if bad_frame_ids:
            bad = np.array(bad_frame_ids, dtype=np.int64)
            n_before = len(idx)
            idx = idx[~np.isin(idx, bad)]
            print(
                f"  [bad frames excluded] {n_before - len(idx)} frames dropped: {sorted(bad_frame_ids)}"
            )
        held_out = np.array(
            [is_held_out(i, parser.test_every, parser.test_offset) for i in idx], dtype=bool
        )
        if split == "train":
            self.indices = idx[~held_out]
        elif split == "val":
            self.indices = idx[held_out]
        else:
            raise ValueError("split must be 'train' or 'val'")

        print(f"\nRadarDataset split={split}: {len(self.indices)} frames")

    def __len__(self) -> int:
        return int(len(self.indices))

    def _load_pair(self, frame_idx: int) -> Dict[str, np.ndarray]:
        cached = self._pair_cache.get(frame_idx)
        if cached is not None:
            return cached

        rad_path = self.parser.rad_tensor_files[frame_idx]
        if rad_path is None:
            raise FileNotFoundError(
                f"No RAD tensor for frame {frame_idx} — rad_tensors_dir holds only a "
                f"frame window (allow_missing_rad). Move frame_start/frame_end inside it, "
                f"or build the missing cubes."
            )

        rad = np.load(rad_path).astype(np.float32)

        # Expect 3D D,R,A
        if rad.ndim != 3:
            raise ValueError(f"RAD must be [D,R,A]. Got {rad.shape} at {rad_path}")

        D, R, A = rad.shape
        if D != self.parser.num_doppler_bins:
            raise ValueError(f"D mismatch RAD {D} != {self.parser.num_doppler_bins} at {rad_path}")
        if A != self.parser.num_azimuth_bins:
            raise ValueError(f"A mismatch RAD {A} != {self.parser.num_azimuth_bins} at {rad_path}")

        # Tensors are stored full-range; crop to the configured range window.
        if R != self.parser.num_range_bins:
            raise ValueError(f"R mismatch RAD {R} != {self.parser.num_range_bins} at {rad_path}")
        rad = rad[:, self.parser._r0 : self.parser._r1, :]

        if self.parser.doppler_roll_bins:
            rad = np.roll(rad, self.parser.doppler_roll_bins, axis=0)

        pose = self.parser.poses[frame_idx].copy()  # [4,4]
        result: Dict[str, np.ndarray] = {"rad": rad, "pose": pose}

        # Ceiling-units normalization (data max → 1). Pure linear scale →
        # commutes with the power sum + PSF; the render matches via the sh_dc=0 pin
        # (each reflector emits opacity ≤ 1) and floor_u = floor/hi. No-op when
        # scale == 1.
        if self.gt_power_scale != 1.0:
            result["rad"] = result["rad"] * np.float32(self.gt_power_scale)

        # Cache the processed frame. Consumers wrap each array in
        # torch.from_numpy(...).float() (shares this buffer) and only ever
        # apply out-of-place ops; the trainer never mutates rad_tensor / poses in
        # place. If that ever changes, copy on return or mark these read-only
        # (v.flags.writeable = False).
        self._pair_cache[frame_idx] = result
        return result

    def __getitem__(self, item: int) -> Dict[str, Any]:
        frame_idx = int(self.indices[item])
        cur = self._load_pair(frame_idx)

        return {
            "rad_tensor": torch.from_numpy(cur["rad"]).float(),  # [D,R,A]
            "radarpose": torch.from_numpy(cur["pose"]).float(),  # [4,4]
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }
