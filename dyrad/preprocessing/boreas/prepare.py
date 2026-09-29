"""Stage one labelled clip of Boreas `boreas-objects-v1` for `boreas convert`.

`boreas-objects-v1` is the only Boreas recording with object annotations. It is downloaded
in the raw AWS layout (`boreas fetch`), from which this script writes a staging sequence
directory holding what `convert` reads:

    raw download                    staging directory
    ------------------------------  ---------------------------------
    radar/<t_us>.png (whole 1.5 h)  radar/<k>.png (just this clip, symlinks)
    applanix/lidar_poses.csv        radar_trajectory.tum
    (no sensor.yaml)                sensor.yaml
                                    clip_metadata.json (the clip's radar timestamps)

POSE CONVENTION
---------------
World poses are built from the *lidar* applanix table, not the radar one:

    T_world_lidar(t) = Trans(easting, northing, altitude) @ Rz(-heading)
    T_world_radar(t) = T_world_lidar(t) @ inv(T_radar_lidar)

Under this convention parked cars come out stationary in world. Only the time-varying
part is pinned: a constant global rotation of the world frame is unobservable and
harmless, so no attempt is made to align the world axes to true ENU.

Radar poses are interpolated from the lidar table (~9.6 Hz) to the radar timestamps
(4 Hz) rather than read from applanix/radar_poses.csv, so that poses and the
T_radar_lidar extrinsic share one convention.

AZIMUTH: the Navtech spoke angle equals +atan2(y, x) in the calib radar frame, so the
tensor azimuth axis of `convert` IS the calib radar frame and poses, labels and tensors
need no mirror to reconcile.

Run:
    python -m dyrad.preprocessing.boreas prepare \\
        --clip-start 1598988079915263 --clip-end 1598988092163459 \\
        --out data/boreas/_staging/objects_win55_104
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

from dyrad.constants import NAVTECH_RANGE_RES_M

# Navtech CIR304-H: raw PNGs are 400 x (11 + 3360) uint8.
SENSOR_YAML = f"""sensor_type: "scanning_radar"
use_polar: True
H: 400
W: 3371
W_metadata: 11
range_resolution: {NAVTECH_RANGE_RES_M}
azimuth_resolution: 0.9
azimuth_beamwidth: 1.8
azimuth_coverage: 360
"""


def load_applanix(path: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (timestamps_us [N], [N,4] easting/northing/altitude/heading)."""
    ts, rows = [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            ts.append(int(r["ROSTime"]))
            rows.append(
                [
                    float(r["easting"]),
                    float(r["northing"]),
                    float(r["altitude"]),
                    float(r["heading"]),
                ]
            )
    o = np.argsort(ts)
    return np.asarray(ts)[o], np.asarray(rows)[o]


def _Tz(e: float, n: float, alt: float, ang: float) -> np.ndarray:
    c, s = np.cos(ang), np.sin(ang)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = [e, n, alt]
    return T


def world_from_lidar(lt: np.ndarray, lm: np.ndarray, t: float) -> np.ndarray:
    """T_world_lidar at an arbitrary time, linearly interpolated.

    Heading is interpolated on the unwrapped angle so a +-pi wrap between two
    samples cannot produce a spurious half-turn.
    """
    i = int(np.clip(np.searchsorted(lt, t), 1, len(lt) - 1))
    t0, t1 = lt[i - 1], lt[i]
    w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    a, b = lm[i - 1], lm[i]
    e, n, alt = (1 - w) * a[:3] + w * b[:3]
    h0, h1 = a[3], b[3]
    h = h0 + w * ((h1 - h0 + np.pi) % (2 * np.pi) - np.pi)  # shortest-arc lerp
    return _Tz(e, n, alt, -h)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--boreas-objects",
        default="data/boreas/boreas-objects-v1",
        help="raw download root (radar/, applanix/, calib/)",
    )
    ap.add_argument("--clip-start", type=int, required=True, help="us, inclusive")
    ap.add_argument("--clip-end", type=int, required=True, help="us, inclusive")
    ap.add_argument("--out", required=True, help="staging sequence dir to write")
    args = ap.parse_args()

    src = args.boreas_objects
    T_rl = np.loadtxt(os.path.join(src, "calib", "T_radar_lidar.txt"))
    lt, lm = load_applanix(os.path.join(src, "applanix", "lidar_poses.csv"))

    stamps = sorted(
        int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(src, "radar", "*.png"))
    )
    stamps = [t for t in stamps if args.clip_start <= t <= args.clip_end]
    if not stamps:
        raise SystemExit("no radar frames in the requested clip -- is it downloaded?")
    if not (lt[0] <= stamps[0] and stamps[-1] <= lt[-1]):
        raise SystemExit("clip is not covered by applanix/lidar_poses.csv")

    rad_out = os.path.join(args.out, "radar")
    os.makedirs(rad_out, exist_ok=True)
    tum = []
    for k, t in enumerate(stamps):
        # `convert` globs radar/*.png and sorts by NAME, pairing row k of the .tum with
        # the k-th name. Renaming to a zero-padded index keeps the name order equal to
        # the time order. The PNGs are symlinked, never copied.
        dst = os.path.join(rad_out, f"{k:05d}.png")
        srcp = os.path.abspath(os.path.join(src, "radar", f"{t}.png"))
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(srcp, dst)

        T_wr = world_from_lidar(lt, lm, t) @ np.linalg.inv(T_rl)
        q = Rotation.from_matrix(T_wr[:3, :3]).as_quat()  # x, y, z, w
        tum.append([t * 1e-6, *T_wr[:3, 3], *q])

    np.savetxt(os.path.join(args.out, "radar_trajectory.tum"), np.asarray(tum), fmt="%.9f")
    with open(os.path.join(args.out, "sensor.yaml"), "w") as f:
        f.write(SENSOR_YAML)
    with open(os.path.join(args.out, "clip_metadata.json"), "w") as f:
        json.dump(
            {
                "source": "boreas-objects-v1",
                "clip_start_us": args.clip_start,
                "clip_end_us": args.clip_end,
                "n_frames": len(stamps),
                "radar_timestamps_us": stamps,
                "pose_source": "applanix/lidar_poses.csv @ Rz(-heading), "
                "interpolated to radar stamps, then inv(T_radar_lidar)",
            },
            f,
            indent=2,
        )

    dur = (stamps[-1] - stamps[0]) / 1e6
    print(f"[prepare] {len(stamps)} radar frames, {dur:.1f} s -> {args.out}")
    print(
        f"[prepare] next: python -m dyrad.preprocessing.boreas convert "
        f"--boreas-seq {args.out} --out data/boreas_processed/<name>"
    )


if __name__ == "__main__":
    main()
