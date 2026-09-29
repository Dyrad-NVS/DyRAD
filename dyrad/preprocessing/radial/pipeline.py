"""Process one RADIal recording end to end.

Steps, in order:
  1. unzip the recording if a .zip is given (or found next to it)
  2. `preprocess`    -> rad_tensors/, poses/ (GPS track), sensor.json, label_index_remap.npy
  3. `ego-velocity`  -> ego_vel_can.npy (CAN wheel speed; needs *_can.bin)
  4. print a per-frame summary (ego speed + label count) to help choose a training window

Usage:

  python -m dyrad.preprocessing.radial pipeline \\
      --recording data/radial_raw/RECORD@2020-11-22_12.31.22 \\
      --out-dir   data/radial_processed/seq_31_22

  # Coarse sensor configuration (sensor-configuration transfer experiment):
  python -m dyrad.preprocessing.radial pipeline \\
      --recording data/radial_raw/RECORD@2020-11-22_12.31.22 \\
      --out-dir   data/radial_processed/seq_31_22_coarse --sensor coarse

Then pick a training window from the summary and prepare it with
`python -m dyrad.preprocessing.radial prepare`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from dyrad.labels import load_index_remap, remap_label_index
from dyrad.paths import ROOT
from dyrad.preprocessing.radial import RADIAL_LABELS

PYTHON = sys.executable


def run(cmd: list[str], step: str) -> None:
    print(f"\n{'=' * 60}\n[STEP] {step}\n{'=' * 60}")
    print("CMD:", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def unzip_if_needed(recording_dir: Path, zip_path: Path | None) -> None:
    if recording_dir.exists() and any(recording_dir.iterdir()):
        print(f"[unzip] {recording_dir} already exists and is non-empty; skipping unzip")
        return
    if zip_path is None:
        candidate = recording_dir.parent / f"{recording_dir.name}.zip"
        if not candidate.exists():
            print(f"[unzip] no zip found for {recording_dir.name}; assumed extracted or pass --zip")
            return
        zip_path = candidate
    print(f"[unzip] extracting {zip_path} -> {recording_dir.parent}")
    recording_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(recording_dir.parent)


def print_frame_summary(out_dir: Path, seq_name: str) -> None:
    """Per-frame ego speed (CAN) and label count, to help pick a training window.

    Label rows are placed on local frames through label_index_remap.npy (the frame rule of
    the trainer and the cloud builder); empty-frame rows (radar_R_m = -1) are not counted.
    """
    print(f"\n{'=' * 60}\n[SUMMARY] per-frame overview\n{'=' * 60}")
    vel_path = out_dir / "ego_vel_can.npy"
    if not vel_path.exists():
        print("  no ego_vel_can.npy; skipping speed summary")
        return
    speed = np.linalg.norm(np.load(vel_path), axis=1)
    N = len(speed)
    ts_path = out_dir / "poses/timestamps_us.npy"
    ts = np.load(ts_path).astype(np.float64) if ts_path.exists() else None
    t_s = (ts - ts[0]) / 1e6 if ts is not None else np.arange(N) * 0.2

    label_counts = np.zeros(N, dtype=int)
    if RADIAL_LABELS.exists():
        import pandas as pd

        remap = load_index_remap(out_dir)
        df = pd.read_csv(RADIAL_LABELS)
        df_seq = df[(df["dataset"] == seq_name) & (df["radar_R_m"] > 0)]
        for idx, cnt in df_seq.groupby("index").size().items():
            fi = remap_label_index(int(idx), remap)
            if 0 <= fi < N:
                label_counts[fi] += int(cnt)

    print(f"  {'frame':>5}  {'t(s)':>6}  {'speed m/s':>10}  {'labels':>7}")
    for fi in range(N):
        print(f"  {fi:>5}  {t_s[fi]:>6.1f}  {speed[fi]:>10.2f}  {label_counts[fi]:>7}")
    print(
        f"\n  {N} frames, {label_counts.sum()} labels on {(label_counts > 0).sum()} frames; "
        f"speed mean={speed.mean():.1f} min={speed.min():.1f} max={speed.max():.1f} m/s"
    )
    print(
        "  Pick a contiguous window with stable speed and label coverage, then run "
        "python -m dyrad.preprocessing.radial prepare."
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--recording", required=True, type=Path, help="raw RECORD@... folder")
    p.add_argument("--out-dir", required=True, type=Path, help="output directory for the sequence")
    p.add_argument("--zip", type=Path, default=None, help="recording .zip (optional)")
    p.add_argument(
        "--sensor",
        default="native",
        choices=["native", "coarse"],
        help="sensor configuration of `preprocess`: native, or coarse (128 of 512 ADC "
        "samples, 128 of 256 chirps)",
    )
    p.add_argument("--az-min", type=float, default=-75.0, help="min azimuth to keep (deg)")
    p.add_argument("--az-max", type=float, default=75.0, help="max azimuth to keep (deg)")
    p.add_argument(
        "--max-frames", type=int, default=0, help="limit to the first N frames (0 = all)"
    )
    p.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="reuse an existing out-dir; only run the ego-velocity step and summary",
    )
    args = p.parse_args()

    recording_dir = args.recording.resolve()
    out_dir = args.out_dir.resolve()
    seq_name = recording_dir.name  # e.g. RECORD@2020-11-22_12.31.22

    unzip_if_needed(recording_dir, args.zip)
    if not recording_dir.exists():
        sys.exit(f"ERROR: recording dir {recording_dir} not found; pass --zip or extract it")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_preprocess:
        cmd = [
            PYTHON,
            "-m",
            "dyrad.preprocessing.radial",
            "preprocess",
            "--seq-dir",
            str(recording_dir),
            "--out-dir",
            str(out_dir),
            "--az-min",
            str(args.az_min),
            "--az-max",
            str(args.az_max),
            "--sensor",
            args.sensor,
        ]
        if args.max_frames > 0:
            cmd += ["--max-frames", str(args.max_frames)]
        run(cmd, "preprocess")

    can_file = next(recording_dir.glob("*_can.bin"), None)
    if can_file is None:
        print(f"\n[SKIP] CAN ego velocity: no *_can.bin in {recording_dir}")
    elif not (out_dir / "ego_vel_can.npy").exists():
        run(
            [
                PYTHON,
                "-m",
                "dyrad.preprocessing.radial",
                "ego-velocity",
                "--recording",
                str(recording_dir),
                "--timestamps",
                str(out_dir / "poses/timestamps_us.npy"),
                "--out-dir",
                str(out_dir),
            ],
            "ego-velocity (CAN wheel speed)",
        )

    print_frame_summary(out_dir, seq_name)
    print(f"\n[DONE] {out_dir}")


if __name__ == "__main__":
    main()
