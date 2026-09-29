"""Write a recentred copy of a Boreas window's sequence directory.

Boreas poses are raw UTM (|y| ~ 4.84e6 m), where one float32 ULP is 0.5 m in northing.
This damages three things:

  a  the pose file is a float32 cast of a float64 source -> up to ~0.25 m error
  b  learned parameters cannot move: a 0.25 m half-ULP is far larger than any Adam
     step at the configured learning rates, so every update rounds away
  c  `w2c . x` subtracts two ~4.84e6 quantities in float32 -> the render carries up to
     ~10 range bins of error before it reaches the rasterizer

Recentring reduces the render range error to ~3e-4 bins. It is required for the object
frame and the learnable control points alike: x = R.mu + P is summed in float32, so while P
sits at 4.84e6 the sum lands on the 0.5 m lattice regardless of how finely mu is resolved.

The recentred copy is a new sequence directory on disk (`<dir>_rc`), not a trainer-side
offset, because many tools load poses independently and a trainer-only offset would
desynchronise them. Poses are rebuilt from the float64 TUM text source of `boreas prepare`
and only then recentred, so they carry no float32 rounding of the UTM values. Originals are
left untouched.
Two checks run before writing: the float64 rebuild must equal the on-disk poses after a
float32 cast (so the staging file is the source of those poses), and the recentred float32
poses plus the stored offset must reproduce the float64 source.

  python -m dyrad.preprocessing.boreas recentre --window sparse2
  python -m dyrad.preprocessing.boreas recentre --all --dry-run

Init clouds and norm.json are not copied: `prepare-window` builds both on the recentred
directory afterwards (build_init_cloud against the recentred poses).
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from dyrad.constants import NAVTECH_RANGE_RES_M
from dyrad.paths import ROOT
from dyrad.poses import load_poses

#: the paper's windows -> the float64 UTM TUM of `boreas prepare` that poses_can was built from
STAGING = {
    w: f"data/boreas/_staging/objects_{w}/radar_trajectory.tum"
    for w in ("sparse2", "sparse4", "win55_104")
}
#: sequence-directory prefix: a window `<win>` lives under data/boreas_processed/boreas_obj_<win>
PREFIX = "boreas_obj_"

# Translation-invariant files, copied verbatim.
#   labels_boreas.csv : emitted in the radar frame via the T_rl lidar->radar extrinsic from
#                       raw Boreas per-frame boxes (`boreas labels`); poses do not enter it
#                       except for a world-frame median-speed moving/parked threshold
#                       computed from float64 applanix data.
#   ego_vel.npy       : derived from the in-memory float64 poses in `boreas convert` (the
#                       float32 cast is only applied when saving), and a velocity.
#   clip_metadata.json: timestamps + provenance strings, no coordinates.
COPY_VERBATIM = [
    "labels_boreas.csv",
    "ego_vel.npy",
    "clip_metadata.json",
]


def build_poses_float64(tum_path):
    """Rebuild [N,4,4] c2w in float64, exactly as `boreas convert` does."""
    tum = np.loadtxt(tum_path)
    if tum.ndim == 1:
        tum = tum[None]
    n = len(tum)
    poses = np.tile(np.eye(4, dtype=np.float64)[None], (n, 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(tum[:, 4:8]).as_matrix()
    poses[:, :3, 3] = tum[:, 1:4]
    return poses, tum


def recentre(win, force=False, dry=False):
    src = os.path.join(ROOT, "data", "boreas_processed", f"{PREFIX}{win}")
    out = src + "_rc"
    stg = os.path.join(ROOT, STAGING[win])
    for p in (src, stg):
        if not os.path.exists(p):
            print(f"[{win}] MISSING {p}", file=sys.stderr)
            return False

    poses64, tum = build_poses_float64(stg)

    # ── Check 1: the staging text must be the source of the on-disk poses, so the only
    # lossy step in the chain is the float32 cast. Refuse rather than guess.
    on_disk, meta = load_poses(os.path.join(src, "poses_can"))
    if len(on_disk) != len(poses64):
        print(
            f"[{win}] REFUSED: {len(on_disk)} poses on disk vs {len(poses64)} in staging "
            f"({stg}).",
            file=sys.stderr,
        )
        return False
    if not np.array_equal(poses64.astype(np.float32), on_disk):
        d = float(np.abs(poses64.astype(np.float32) - on_disk).max())
        print(
            f"[{win}] REFUSED: the staging rebuild does not equal poses_can after a float32 "
            f"cast (max|diff| {d:.3e}). Wrong source or the window moved.",
            file=sys.stderr,
        )
        return False

    t64 = poses64[:, :3, 3]
    pose_err = np.linalg.norm(t64 - on_disk[:, :3, 3].astype(np.float64), axis=1)

    # ── The offset: the exact float64 mean translation. No rounding is needed; the
    # recentred values are exactly representable.
    off = t64.mean(axis=0, dtype=np.float64)

    poses_rc = poses64.copy()
    poses_rc[:, :3, 3] = t64 - off
    poses_rc32 = poses_rc.astype(np.float32)  # readers expect float32 poses

    # ── Check 2: the recentred float32 file + the stored offset must reproduce the
    # float64 source. At |t| ~ 50 m one float32 ULP is 4e-6 m.
    back = poses_rc32[:, :3, 3].astype(np.float64) + off
    rt = float(np.abs(back - t64).max())
    if rt > 1e-4:
        print(f"[{win}] REFUSED: recentred round-trip error {rt:.3e} m > 1e-4", file=sys.stderr)
        return False

    resid = float(np.abs(poses_rc32[:, :3, 3]).max())
    print(f"[{win}]")
    print(f"   staging source        {os.path.relpath(stg, ROOT)}")
    print(
        f"   pose repair           max {pose_err.max():.4f} m  rms "
        f"{np.sqrt((pose_err**2).mean()):.4f} m   "
        f"({pose_err.max() / NAVTECH_RANGE_RES_M:.2f} range bins max)"
    )
    print(f"   offset (float64)      [{off[0]:.6f}, {off[1]:.6f}, {off[2]:.6f}]")
    print(f"   residual |t| after    {resid:.3f} m   round-trip {rt:.2e} m")
    print(f"   -> {os.path.relpath(out, ROOT)}")

    if dry:
        return True
    if os.path.exists(out):
        if not force:
            print(f"[{win}] exists, use --force: {out}", file=sys.stderr)
            return False
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "poses_can"))

    np.save(os.path.join(out, "poses_can", "radar_poses.npy"), poses_rc32)
    # The frame times (the trainer's time axis) are unaffected by the shift.
    shutil.copy2(
        os.path.join(src, "poses_can", "timestamps_us.npy"),
        os.path.join(out, "poses_can", "timestamps_us.npy"),
    )

    # `meta` is the source's poses_metadata.json (loaded with the poses above).
    meta["recentre"] = {
        "offset_m": [float(v) for v in off],
        "offset_hex": [float(v).hex() for v in off],  # exact, for reversal
        "frame": "UTM minus offset_m; add offset_m to return to UTM",
    }
    with open(os.path.join(out, "poses_can", "poses_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    for f in COPY_VERBATIM:
        s = os.path.join(src, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out, f))

    # rad_tensors are measurements, unaffected by any coordinate change.
    os.symlink(
        os.path.join("..", f"{PREFIX}{win}", "rad_tensors"), os.path.join(out, "rad_tensors")
    )

    recentre_meta = {
        "window": win,
        "src": f"data/boreas_processed/{PREFIX}{win}",
        "float64_source": STAGING[win],
        "offset_m": [float(v) for v in off],
        "offset_hex": [float(v).hex() for v in off],
        "n_poses": int(len(poses64)),
        "residual_abs_t_max_m": resid,
        "pose_repair_max_m": float(pose_err.max()),
        "pose_repair_rms_m": float(np.sqrt((pose_err**2).mean())),
        "pose_repair_max_range_bins": float(pose_err.max() / NAVTECH_RANGE_RES_M),
        "roundtrip_max_m": rt,
        "copied_verbatim": [f for f in COPY_VERBATIM if os.path.exists(os.path.join(src, f))],
        "symlinked": ["rad_tensors"],
        "written_by": "dyrad.preprocessing.boreas.recentre",
    }
    with open(os.path.join(out, "recentre.json"), "w") as f:
        json.dump(recentre_meta, f, indent=2)
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--window",
        choices=sorted(STAGING),
        action="append",
        default=[],
        help="window to recentre (repeatable)",
    )
    ap.add_argument("--all", action="store_true", help="recentre every window")
    ap.add_argument("--force", action="store_true", help="overwrite an existing <dir>_rc")
    ap.add_argument("--dry-run", action="store_true", help="run the checks, write nothing")
    a = ap.parse_args()
    wins = sorted(STAGING) if a.all else a.window
    if not wins:
        ap.error("pass --window <w> (repeatable) or --all")

    ok = [w for w in wins if recentre(w, a.force, a.dry_run)]
    print(f"\n{len(ok)}/{len(wins)} recentred" + (" (dry run)" if a.dry_run else ""))
    if ok and not a.dry_run:
        print(
            "\nNext: build norm.json and the init clouds on the recentred directory "
            "(`prepare-window` does both)."
        )
    return 0 if len(ok) == len(wins) else 1


if __name__ == "__main__":
    sys.exit(main())
