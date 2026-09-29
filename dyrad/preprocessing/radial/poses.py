"""Build the RADIal radar poses (poses_can/) by CAN dead-reckoning along the smoothed GPS track.

The GPS fixes (~1 Hz, spread over the recording by line index and interpolated to the radar
frames by `preprocess`) give an irregular per-frame displacement, and a pose error of a few
metres misprojects every static reflector. The GPS track is therefore used only as a
low-frequency shape prior (a heavily smoothed tangent direction and an anchor point), and
the CAN wheel speed (+-0.1 km/h) is integrated along it:

    p[i+1] = p[i] + tangent[i] * v_can[i] * DT * (slot[i+1] - slot[i])
    R[i]   = [x=tangent[i], y=up x x, z=up]   (sensor x-forward convention)

z is 0 for every pose.

Time step. The radar fires every DT = 0.2 s and frame i is acquired at slot[i] * DT, where
slot counts frame periods (slot[i+1] - slot[i] = 1, or 2 across a dropped frame). The
recorded frame stamps (poses/timestamps_us.npy) do not enter the displacement. Inside the paper's training
windows, a stamp interval that departs from 0.2 s is a recorder-clock stall, not a missing
frame:

  * A long interval is followed by a burst of short ones (seq_31_22 frames 146-153: 0.873,
    0.158, 0.201, 0.156, 0.164, 0.160, 0.154 s). The radar fires at a fixed rate and cannot
    deliver frames 0.154 s apart, so these stamps are not acquisition times.
  * The overlap test agrees. Overlap is the IoU of the 0.5 m world voxels occupied by the
    thresholded returns of two consecutive frames, the second frame placed a step of
    v * 0.2 s or v * (stamp interval) ahead. The fixed step overlaps better across each
    stall: 0.29 vs 0.13 (31_22 146->147), 0.29 vs 0.08 (12_20_50 1->2), 0.32 vs 0.21
    (14_25_06 414->415).

The one exception in the paper windows is the isolated 0.4 s interval of RECORD@2020-11-21_
12.00.45 from frame 360 to 361, a dropped frame (overlap 0.24 with one step vs 0.33 with two,
and no burst of short intervals after it): `DROPPED_FRAMES` lists it, so frame 361 is one
extra period (4.3 m at 21.6 m/s) further on. Outside the paper windows the stamps were not
tested, so poses across a long stamp gap there are approximate.

The slots are written to poses_can/frame_slots.npy, and the trainer's RADIal time axis is
slot * DT (Runner._build_frame_times), so the object tracks and the ego poses share one clock.

Input:  <seq-dir>/poses/radar_poses.npy (the GPS track of `preprocess`; only the
        translations are read) and <seq-dir>/ego_vel_can.npy
        (`python -m dyrad.preprocessing.radial ego-velocity`).
Output: <seq-dir>/poses_can/radar_poses.npy, frame_slots.npy (+ copies of timestamps_us.npy
        and poses_metadata.json so RadarParser can load the dir unchanged).

Usage:
    python -m dyrad.preprocessing.radial poses --seq-dir data/radial_processed/seq_31_22
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

DT = 0.2  # radar frame period (s), the dead-reckoning step (see the module docstring)
#: Dropped frames inside the paper windows, per recording (poses_metadata.json `sequence`):
#: each listed frame is acquired two frame periods after its predecessor.
DROPPED_FRAMES = {"RECORD@2020-11-21_12.00.45": (361,)}
SMOOTH_SIGMA = 10.0  # Gaussian sigma (frames) of the GPS shape prior
EGO_VEL = "ego_vel_can.npy"  # CAN speed, [N,2] sensor-frame (vx forward, vy = 0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--seq-dir",
        type=Path,
        required=True,
        help="processed sequence dir (poses/, ego_vel_can.npy)",
    )
    args = ap.parse_args()

    poses_dir = args.seq_dir / "poses"
    out_dir = args.seq_dir / "poses_can"
    out_dir.mkdir(exist_ok=True)

    poses = np.load(poses_dir / "radar_poses.npy").astype(np.float64)  # [N,4,4] GPS track
    ego = np.load(args.seq_dir / EGO_VEL).astype(np.float64)  # [N,2]
    # Column 0 is the forward speed; direction comes from the GPS tangent below, so
    # only its magnitude is used (CAN speed is >= 0 anyway).
    ego = np.abs(ego)
    N = len(poses)
    if len(ego) != N:
        raise ValueError(f"{EGO_VEL} has {len(ego)} rows for {N} poses; rerun ego-velocity")

    # ── GPS shape prior: heavily smoothed XY trajectory → per-frame tangent ──
    gps_xy = poses[:, :2, 3]
    xy_s = gaussian_filter1d(gps_xy, sigma=SMOOTH_SIGMA, axis=0)
    tang = np.gradient(xy_s, axis=0)  # [N,2]
    # Stuck-GPS plateaus can zero the local gradient — fill from neighbours.
    norms = np.linalg.norm(tang, axis=1)
    good = norms > 1e-6
    if not good.all():
        idx = np.arange(N)
        for c in range(2):
            tang[:, c] = np.interp(idx, idx[good], tang[good, c])
        norms = np.linalg.norm(tang, axis=1)
    tang /= norms[:, None]

    # ── frame clock: one period per frame, two across a dropped frame ──────
    recording = json.loads((poses_dir / "poses_metadata.json").read_text())["sequence"]
    steps = np.ones(N, dtype=np.int64)
    steps[0] = 0
    for fi in DROPPED_FRAMES.get(recording, ()):
        if fi < N:
            steps[fi] = 2
    slots = np.cumsum(steps)  # [N] int64, slot[0] = 0

    # ── dead-reckon translations from CAN speed ─────────────────────────────
    # z stays 0, never the GPS altitude: the scene points are at z=0, and a lifted
    # sensor would be wrong for any consumer that uses the translation directly.
    pos = np.zeros((N, 3))
    pos[0, :2] = xy_s[0]
    for i in range(N - 1):
        v = float(ego[i, 0])
        pos[i + 1, :2] = pos[i, :2] + tang[i] * v * DT * int(steps[i + 1])

    # Re-anchor: minimise mean offset to the smoothed GPS (keeps the world frame
    # aligned with the labels built from the original poses).
    off = (xy_s - pos[:, :2]).mean(axis=0)
    pos[:, :2] += off

    # ── rotations: x = tangent, z = up, y = z × x (sensor x-forward) ────────
    new_poses = np.zeros_like(poses)
    up = np.array([0.0, 0.0, 1.0])
    for i in range(N):
        x = np.array([tang[i, 0], tang[i, 1], 0.0])
        y = np.cross(up, x)
        y /= np.linalg.norm(y)
        new_poses[i, :3, 0] = x
        new_poses[i, :3, 1] = y
        new_poses[i, :3, 2] = up
        new_poses[i, :3, 3] = pos[i]
        new_poses[i, 3, 3] = 1.0

    np.save(out_dir / "radar_poses.npy", new_poses.astype(np.float32))
    np.save(out_dir / "frame_slots.npy", slots)
    for fn in ("timestamps_us.npy", "poses_metadata.json"):
        src = poses_dir / fn
        if src.exists():
            shutil.copy(src, out_dir / fn)
    n_drop = int((steps > 1).sum())
    print(f"[poses] wrote {out_dir / 'radar_poses.npy'}  ({N} frames, {n_drop} dropped frame(s))")


if __name__ == "__main__":
    main()
