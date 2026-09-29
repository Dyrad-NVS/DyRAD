"""Generate a paired set of synthetic RADIal sequences for off-path novel-view
synthesis: one base (training) trajectory plus the off-path trajectories of the
same scene (lateral lane shifts and sensor yaws).

The scene (world-frame reflectors, scene_spec.py) is fully decoupled from the ego
trajectory (trajectory.py), so re-rendering the identical scene (same --seed, --scene,
--num-frames) from a shifted/rotated trajectory yields ground-truth RAD tensors for
a viewpoint that was never on the training path. Frame i of every sequence shares
the same world state, so the off-path evaluation renders the trained model at the
novel poses and scores against the novel GT (dyrad.evaluation.evaluate_offpath_synthetic).

Each output is a full RADIal seq dir (rad_tensors/, poses/ + poses_can/,
ego_vel_can.npy, doppler/range/az axes, labels_CVPR.csv, scene_gt.json) -- no LiDAR.

Usage:
    python -m dyrad.synthetic.generate_offpath_views \\
        --parent-dir data/synthetic/curve_ramp \\
        --scene curve_ramp --num-frames 40 --seed 42
    # -> curve_ramp/{base, lat+1.75, lat-1.75, lat+3.5, yaw+5, yaw+10}/

The default views are the paper's: lateral +-1.75 m (half lane) and +3.5 m (full lane),
yaw +5 / +10 deg; --lateral / --yaw take space-separated lists.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from dyrad.synthetic.generate_scene import NUM_FRAMES, SEED

# The paper's off-path views: lateral lane shifts (+Y m) and sensor yaws (deg).
LATERAL_M = (1.75, -1.75, 3.5)
YAW_DEG = (5.0, 10.0)


def _tag(lateral: float, yaw: float) -> str:
    """Stable directory suffix encoding the novel-view offset."""
    if lateral == 0.0 and yaw == 0.0:
        return "base"
    parts = []
    if lateral:
        parts.append(f"lat{lateral:+g}")
    if yaw:
        parts.append(f"yaw{yaw:+g}")
    return "_".join(parts)


def view_offsets(lateral=LATERAL_M, yaw=YAW_DEG) -> list[tuple[float, float]]:
    """(lateral, yaw) of every view: the base (0, 0), then the lateral, then the yaw views."""
    return [(0.0, 0.0)] + [(lat, 0.0) for lat in lateral] + [(0.0, y) for y in yaw]


def view_tags(lateral=LATERAL_M, yaw=YAW_DEG) -> list[str]:
    """Directory names of every view, in `view_offsets` order."""
    return [_tag(lat, y) for lat, y in view_offsets(lateral, yaw)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--parent-dir",
        required=True,
        help="parent dir; each view is written to <parent-dir>/<tag>/",
    )
    # Shared scene args (forwarded verbatim, so every view renders the identical scene)
    ap.add_argument(
        "--scene",
        required=True,
        help="scene spec name under dyrad/synthetic/scenes/ (or a .yaml path)",
    )
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--seed", type=int, default=SEED)
    # Novel-view sweep
    ap.add_argument(
        "--lateral",
        type=float,
        nargs="*",
        default=list(LATERAL_M),
        help="lateral lane-shift offsets in +Y metres (each -> one novel seq)",
    )
    ap.add_argument(
        "--yaw",
        type=float,
        nargs="*",
        default=list(YAW_DEG),
        help="sensor yaw offsets in degrees (each -> one novel seq)",
    )
    args = ap.parse_args()

    parent = Path(args.parent_dir)
    parent.mkdir(parents=True, exist_ok=True)

    # The base (offset-0) training sequence, then the (lateral, yaw) views.
    views = view_offsets(args.lateral, args.yaw)

    shared = [
        "--scene",
        args.scene,
        "--num-frames",
        str(args.num_frames),
        "--seed",
        str(args.seed),
    ]

    print(f"[offpath_views] {len(views)} sequences -> {parent}")
    made = []
    for lat, yaw in views:
        tag = _tag(lat, yaw)
        out = parent / tag
        cmd = [
            sys.executable,
            "-m",
            "dyrad.synthetic.generate_scene",
            "--out",
            str(out),
            *shared,
            "--ego-lateral-offset",
            str(lat),
            "--ego-yaw-deg",
            str(yaw),
        ]
        kind = "BASE (train)" if tag == "base" else "novel-view"
        print(f"\n========== {tag}  [{kind}]  lat={lat:+g} yaw={yaw:+g} ==========")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[offpath_views] FAILED on {tag} (exit {r.returncode})")
            return r.returncode
        made.append(out)

    print(f"\n[offpath_views] Done. {len(made)} sequences:")
    for m in made:
        print(f"  {m}")
    print(
        "\nNext: python -m dyrad.synthetic.prepare builds the normalization sidecars and init "
        "clouds; after training, score the views with dyrad.evaluation.evaluate_offpath_synthetic."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
