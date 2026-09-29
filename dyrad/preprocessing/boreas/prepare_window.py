"""Run the whole Boreas chain for one of the paper's three windows.

The windows are 50-frame clips (4 Hz, ~12.25 s) of `boreas-objects-v1`, staged directly on
their own timestamps; their stubs are `configs/sequences/boreas_<window>.yaml`. Per window:

  1. `fetch`      radar frames of the window and the tracking labels around it (labels are
                  keyed by lidar stamps), plus calib/T_radar_lidar.txt and
                  applanix/lidar_poses.csv, into --boreas-objects
  2. `prepare`    -> data/boreas/_staging/objects_<window>  (poses, sensor.yaml)
  3. `convert`    -> data/boreas_processed/boreas_obj_<window>  (--db-counts-per-decade k,
                  k = NAVTECH_COUNTS_PER_DECADE = 20)
  4. `labels`     -> <processed>/labels_boreas.csv  (the paper's object selection: Car
                  tracks moving >= 1.0 m/s with >= 50 lidar points, within 80 m, labels
                  within 0.15 s of a radar frame; dataset = the stub's seq_name)
  5. `recentre`   -> data/boreas_processed/boreas_obj_<window>_rc  (the stub's seq_dir)
  6. normalize_dataset --configs configs/boreas/<window>_dyrad.yaml --mode counts
                  --counts-per-decade k
  7. build_init_cloud --noise-factor 2.0 (voxel: the config's init_cloud_voxel_m) for
                  configs/boreas/<window>_dyrad.yaml (--partition; also used by the
                  ablation) and configs/boreas/<window>_dyrad_static.yaml (unpartitioned)

Steps whose output exists are skipped, so the command can be re-run; --force rebuilds
everything from the downloaded frames.

    python -m dyrad.preprocessing.boreas prepare-window --window win55_104
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from dyrad.constants import NAVTECH_COUNTS_PER_DECADE
from dyrad.paths import ROOT
from dyrad.preprocessing.boreas import fetch
from dyrad.preprocessing.boreas.recentre import PREFIX, STAGING

PY = sys.executable
#: first and last radar timestamp (us, inclusive) of each window
WINDOWS = {
    "sparse2": (1598991981484792, 1598991993733497),
    "sparse4": (1598987711906340, 1598987724156947),
    "win55_104": (1598988079915263, 1598988092163459),
}
# `labels` object selection of the paper's windows
LABEL_TYPES = "Car"
MIN_SPEED_MPS = 1.0  # --min-speed: slower tracks are parked cars (static scene)
LABEL_MIN_POINTS = 50  # --min-points: median lidar points of a track
LABEL_MAX_RANGE_M = 80.0  # --max-range
LABEL_MAX_DT_US = 150_000  # --max-dt-us: label-to-radar-frame association window
NOISE_FACTOR = 2.0  # init cloud threshold multiplier
LABEL_MARGIN_US = 500_000  # labels (lidar stamps, ~10 Hz) fetched this far beyond the window


def sh(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def boreas(cmd, *args):
    sh([PY, "-m", "dyrad.preprocessing.boreas", cmd, *args])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--window", required=True, choices=sorted(WINDOWS), help="paper window")
    ap.add_argument(
        "--boreas-objects",
        default="data/boreas/boreas-objects-v1",
        help="raw download root of boreas-objects-v1 (filled by `fetch`)",
    )
    ap.add_argument(
        "--force", action="store_true", help="rebuild every product from the downloaded frames"
    )
    a = ap.parse_args()

    win = a.window
    t0, t1 = WINDOWS[win]
    raw = Path(a.boreas_objects)
    raw = raw if raw.is_absolute() else ROOT / raw  # the subprocesses run from ROOT
    stub = yaml.safe_load((ROOT / "configs" / "sequences" / f"boreas_{win}.yaml").read_text())
    seq_name = stub["seq_name"]
    staging = (ROOT / STAGING[win]).parent
    proc = ROOT / "data" / "boreas_processed" / f"{PREFIX}{win}"
    rc = ROOT / stub["seq_dir"]
    assert rc == proc.with_name(proc.name + "_rc"), (stub["seq_dir"], proc)
    cfg = ROOT / "configs" / "boreas" / f"{win}_dyrad.yaml"
    cfg_static = ROOT / "configs" / "boreas" / f"{win}_dyrad_static.yaml"
    print(f"[prepare-window] {win}: {t0}..{t1}  seq_name={seq_name}  -> {rc.relative_to(ROOT)}")

    # 1. raw frames, labels and the per-recording files
    boreas("fetch", "--sensor", "radar", "--start", t0, "--end", t1, "--out", raw / "radar")
    boreas(
        "fetch",
        "--sensor",
        "labels_tracking",
        "--start",
        t0 - LABEL_MARGIN_US,
        "--end",
        t1 + LABEL_MARGIN_US,
        "--out",
        raw / "labels_tracking",
    )
    fetch.fetch_files(str(raw))

    # 2. staging clip (cheap and idempotent: always rewritten)
    boreas(
        "prepare",
        "--boreas-objects",
        raw,
        "--clip-start",
        t0,
        "--clip-end",
        t1,
        "--out",
        staging,
    )

    # 3. RAD tensors, poses, ego velocity
    if a.force or not (proc / "rad_tensors").is_dir():
        boreas(
            "convert",
            "--boreas-seq",
            staging,
            "--out",
            proc,
            "--db-counts-per-decade",
            NAVTECH_COUNTS_PER_DECADE,
        )
    else:
        print(f"[prepare-window] {proc.name}/rad_tensors exists; convert skipped")

    # 4. labels of the moving objects
    labels = proc / "labels_boreas.csv"
    if a.force or not labels.is_file():
        boreas(
            "labels",
            "--boreas-objects",
            raw,
            "--clip-meta",
            staging / "clip_metadata.json",
            "--out",
            labels,
            "--dataset",
            seq_name,
            "--types",
            LABEL_TYPES,
            "--min-speed",
            MIN_SPEED_MPS,
            "--min-points",
            LABEL_MIN_POINTS,
            "--max-range",
            LABEL_MAX_RANGE_M,
            "--max-dt-us",
            LABEL_MAX_DT_US,
        )
    else:
        print(f"[prepare-window] {labels.name} exists; labels skipped")

    # 5. float32-safe poses
    if a.force or not (rc / "poses_can" / "radar_poses.npy").is_file():
        boreas("recentre", "--window", win, *(["--force"] if a.force else []))
    else:
        print(f"[prepare-window] {rc.name} exists; recentre skipped")

    # 6. normalization sidecar (per-window robust range in the sensor's count convention)
    if a.force or not (rc / "norm.json").is_file():
        sh(
            [
                PY,
                "-m",
                "dyrad.preprocessing.normalize_dataset",
                "--configs",
                cfg.relative_to(ROOT),
                "--mode",
                "counts",
                "--counts-per-decade",
                NAVTECH_COUNTS_PER_DECADE,
            ]
        )
    else:
        print(f"[prepare-window] {rc.name}/norm.json exists; normalize_dataset skipped")

    # 7. init clouds: partitioned (dyrad, abl_no_interp) and unpartitioned (dyrad_static)
    for c, partition in ((cfg, True), (cfg_static, False)):
        cloud = ROOT / yaml.safe_load(c.read_text())["init_cloud_path"]
        if not a.force and cloud.is_file():
            print(f"[prepare-window] {cloud.name} exists; build_init_cloud skipped")
            continue
        sh(
            [
                PY,
                "-m",
                "dyrad.preprocessing.build_init_cloud",
                "--config",
                c.relative_to(ROOT),
                "--noise-factor",
                NOISE_FACTOR,
            ]
            + (["--partition"] if partition else [])
        )
    print(
        f"[prepare-window] done. Train with:\n  python -m dyrad.train --config {cfg.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
