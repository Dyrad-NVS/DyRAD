"""Convert a staged Boreas Navtech clip to the trainer's sequence format.

Input (the output of `python -m dyrad.preprocessing.boreas prepare`):
  <boreas-seq>/radar/*.png            400 az rows x (11 metadata + 3360 range) uint8
  <boreas-seq>/radar_trajectory.tum   N lines: t x y z qx qy qz qw  (radar-to-world)
  <boreas-seq>/sensor.yaml            range_resolution 0.0596, azimuth_resolution 0.9

Output:
  <out>/rad_tensors/rad_XXXXX.npy     [D=1, R=3360, A=400] float32 linear power
  <out>/poses_can/radar_poses.npy     [N,4,4] c2w  (+ poses_metadata.json)
  <out>/poses_can/timestamps_us.npy   [N] int64 frame times (us), the trainer's time axis
  <out>/ego_vel.npy                   [N,2] sensor-frame (vx, vy)

Conventions:
  * Our azimuth axis: linspace(-179.55, 179.55, 400) deg, step exactly 0.9
    (set radar_az_fov_deg: 359.1); bin j centre az_math_j = -179.55 + 0.9 j.
  * Navtech spoke i points at az_nav = 0.9 i deg; for boreas-objects-v1 staged by
    `prepare` this equals the math azimuth atan2(y, x) of the calib radar frame.
  * Intensity: the uint8 image is log-compressed counts u. It is decompressed to linear
    power P = 10^(u/k), k = --db-counts-per-decade (20 fits the observed count lattice);
    the renderer sums incoherent power, which is only additive in the decompressed domain.
  * Near range: the first int(near_range_blackout_m / range_res) range bins are zeroed
    (Navtech near-field blind zone); the first 2.5 m are blanked by default.

Usage:
    python -m dyrad.preprocessing.boreas convert \\
        --boreas-seq data/boreas/_staging/objects_win55_104 \\
        --out data/boreas_processed/boreas_obj_win55_104
"""

import argparse
import glob
import json
import os

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from dyrad.constants import NAVTECH_COUNTS_PER_DECADE, NAVTECH_NEAR_RANGE_BLACKOUT_M
from dyrad.domains import Domain, save_rad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--boreas-seq", required=True, help="staging clip dir from `prepare`")
    ap.add_argument("--out", required=True, help="processed sequence dir to write")
    ap.add_argument(
        "--db-counts-per-decade",
        type=float,
        default=NAVTECH_COUNTS_PER_DECADE,
        help="k: decompress the uint8 log-compressed counts u to linear power P = 10^(u/k) "
        f"(default {NAVTECH_COUNTS_PER_DECADE:g}, the Navtech value). The renderer sums incoherent POWER, which is only additive in "
        "the decompressed domain.",
    )
    ap.add_argument(
        "--near-range-blackout-m",
        type=float,
        default=NAVTECH_NEAR_RANGE_BLACKOUT_M,
        help="zero the first int(m/range_res) range bins (Navtech near-field blind "
        "zone; u=0 -> power 1.0 = count 0). 0 disables.",
    )
    args = ap.parse_args()

    with open(os.path.join(args.boreas_seq, "sensor.yaml")) as f:
        sensor = yaml.safe_load(f)
    n_az = int(sensor["H"])  # 400
    w_meta = int(sensor["W_metadata"])  # 11
    n_rng = int(sensor["W"]) - w_meta  # 3360
    az_res = float(sensor["azimuth_resolution"])  # 0.9 deg
    rng_res = float(sensor["range_resolution"])  # 0.0596 m

    pngs = sorted(glob.glob(os.path.join(args.boreas_seq, "radar", "*.png")))
    tum = np.loadtxt(os.path.join(args.boreas_seq, "radar_trajectory.tum"))
    assert len(pngs) == len(tum), (len(pngs), len(tum))
    n = len(pngs)

    # Our azimuth grid (deg, math convention) -> source Navtech spoke index.
    fov = n_az * az_res - az_res  # 359.1
    az_math = np.linspace(-fov / 2.0, fov / 2.0, n_az)
    az_nav = np.mod(az_math, 360.0)
    # az_nav/az_res lands on an exact .5 tie for every bin (the linspace grid is a
    # half-bin off the Navtech spoke grid, which is at exact multiples of az_res).
    # np.round breaks those ties inconsistently and would drop/duplicate ~30% of the
    # spokes. floor() is bijective (400/400 unique); the residual is a constant
    # 0.45 deg registration offset, far inside the 1.8 deg beamwidth.
    src_row = np.floor(az_nav / az_res).astype(int) % n_az  # [A]
    _uniq = len(np.unique(src_row))
    assert _uniq == n_az, f"azimuth remap not bijective: {_uniq}/{n_az} spokes used"

    # Near-field blind-zone blackout (see arg help): u -> 0 -> value-map floor -> N=0.
    nr_bins = int(args.near_range_blackout_m / rng_res) if args.near_range_blackout_m > 0 else 0

    rad_dir = os.path.join(args.out, "rad_tensors")
    os.makedirs(rad_dir, exist_ok=True)
    for k, p in enumerate(pngs):
        im = np.asarray(Image.open(p))  # [400, 3371] uint8
        data = im[:, w_meta:].astype(np.float32)  # [az, range]
        if nr_bins:
            data[:, :nr_bins] = 0.0  # near-range blackout
        power = 10.0 ** (data / args.db_counts_per_decade)  # log-compressed -> linear
        rad = power[src_row, :].T[None]  # [1, R, A]
        save_rad(
            os.path.join(rad_dir, f"rad_{k:05d}.npy"),
            rad.astype(np.float32),
            Domain.RAW_POWER,
            note=f"boreas convert: db_counts_per_decade={args.db_counts_per_decade}",
        )

    # Poses: TUM r2w -> [N,4,4] c2w.
    poses = np.tile(np.eye(4, dtype=np.float64)[None], (n, 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(tum[:, 4:8]).as_matrix()
    poses[:, :3, 3] = tum[:, 1:4]
    # "poses_can" is the pose dir name every sequence uses (the RADIal convention; the
    # cloud builder and the trainer read it); these are the applanix poses of `prepare`.
    pose_dir = os.path.join(args.out, "poses_can")
    os.makedirs(pose_dir, exist_ok=True)
    np.save(os.path.join(pose_dir, "radar_poses.npy"), poses.astype(np.float32))
    ts = tum[:, 0]
    np.save(os.path.join(pose_dir, "timestamps_us.npy"), np.round(ts * 1e6).astype(np.int64))
    with open(os.path.join(pose_dir, "poses_metadata.json"), "w") as f:
        json.dump({"tesseract_indices": list(range(n))}, f, indent=2)

    # Ego velocity, sensor frame [N,2]: smoothed world FD rotated into radar frame.
    dt = np.gradient(ts)  # the TUM times are seconds
    # Gaussian sigma of 2 frames (0.5 s at 4 Hz): smooths the positions before differencing
    pos_s = gaussian_filter1d(poses[:, :3, 3], sigma=2.0, axis=0)
    v_world = np.gradient(pos_s, axis=0) / dt[:, None]
    v_sensor = np.einsum("nij,nj->ni", poses[:, :3, :3].transpose(0, 2, 1), v_world)
    np.save(os.path.join(args.out, "ego_vel.npy"), v_sensor[:, :2].astype(np.float32))

    print(f"[convert] {n} frames -> {args.out}")
    print(f"  rad [1,{n_rng},{n_az}]  rng_res {rng_res} m  fov {fov:.1f} deg")
    print(
        f"  ego speed m/s: mean {np.linalg.norm(v_sensor[:, :2], axis=1).mean():.2f} "
        f"max {np.linalg.norm(v_sensor[:, :2], axis=1).max():.2f}"
    )


if __name__ == "__main__":
    main()
