"""One-command preparation of a processed RADIal sequence for training.

Given a sequence produced by `python -m dyrad.preprocessing.radial pipeline` (rad_tensors,
poses, CAN ego velocity) and its annotations in labels_CVPR.csv, this prepares one
training window using radar + CAN + annotations only (no LiDAR):

  1. build poses_can (CAN dead-reckoning) with `radial poses` if absent,
  2. write the 256-bin raw-FFT Doppler axis (rd_doppler_bins_mps.npy) if absent,
  3. write the sequence stub `configs/sequences/radial_<ID>.yaml` (data paths, frame
     window, track_max_displacement_m, render frames) and generate its run configs
     `configs/radial/<ID>_<variant>.yaml` with dyrad.generate_configs,
  4. compute the normalization sidecar norm.json from the window's TRAIN frames
     (normalize_dataset --configs configs/radial/<ID>_dyrad.yaml),
  5. build the two holdout-aware radar pseudo-LiDAR init clouds under the builder's
     default names, which are the `init_cloud_path` of the generated run configs: the
     RA-partitioned cloud (build_init_cloud --partition) used by `dyrad` and
     the ablations, and the unpartitioned cloud used by `dyrad_static`. The threshold
     multiplier (noise factor) is bisected on the partitioned cloud so it holds about
     `--target-density` points per train frame, and the unpartitioned cloud is built with
     the same value. This bisection targets a density; it does not recover the noise
     factor of a given paper cloud (those were built at per-sequence values that are
     recorded only in their own .build.json sidecars).

The coarse sequence of the sensor-transfer experiment (`seq_<ID>_coarse`, from
`radial pipeline --sensor coarse`) is prepared the same way; its stub
`radial_<ID>_coarse.yaml` generates the `dyrad` config only, so only the partitioned
cloud is built.

Usage:
  python -m dyrad.preprocessing.radial prepare --seq seq_31_22 --frame-start 119
  python -m dyrad.preprocessing.radial prepare --seq seq_31_22 --frame-start 119 --n-frames 60
"""

import argparse
import csv
import json
import re
import subprocess
import sys

import numpy as np
import yaml

from dyrad import generate_configs
from dyrad.config import is_held_out, load_config
from dyrad.constants import DOPPLER_BIN_MPS
from dyrad.paths import ROOT
from dyrad.preprocessing.radial import RADIAL_LABELS

PROC = ROOT / "data" / "radial_processed"
PY = sys.executable
#: default --target-density: init-cloud points per train frame the noise factor is bisected
#: to. It is a density target only; the paper clouds were built at per-sequence noise
#: factors (recorded in their .build.json) and hold 1261-2634 points per train frame.
TARGET_DENSITY = 1793.0
# Bound (m) of the motion model's per-frame track offset, which under the object frame is
# dominated by the object's world position (its distance from the pose origin), not its
# travel. It must exceed the largest |world position| of an object in the window; the
# paper's sequences use 6200-30300 (the trainer warns if it ever binds).
TRACK_MAX_DISPLACEMENT_M = 10000.0


def sh(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(ROOT))


def seq_name_from_labels(short):
    """seq_HH_MM_SS -> full 'RECORD@DATE_HH.MM.SS' from the labels CSV dataset col.

    The trainer filters labels by seq_name == dataset, so this must be the exact
    dataset string (date included). Match by the HH.MM.SS time suffix; a coarse
    sequence (`<ID>_coarse`) is the same recording, so its suffix is stripped first.
    """
    time_part = re.sub(re.escape(generate_configs.COARSE_SUFFIX) + "$", "", short)
    dotted = ".".join(time_part.split("_"))  # 12_25_47 -> 12.25.47
    keys = set()
    with open(RADIAL_LABELS) as f:
        for r in csv.DictReader(f):
            keys.add(r["dataset"])
    # exact HH.MM.SS match (full dir names like seq_12_25_47)
    hits = [k for k in keys if k.split("_")[-1] == dotted]
    # 2-part dir names (seq_31_22 -> 12.31.22): match the MM.SS tail
    if not hits:
        hits = [k for k in keys if k.split("_")[-1].endswith("." + dotted)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise SystemExit(f"[radial] AMBIGUOUS seq_name for {short}: {hits}")
    raise SystemExit(
        f"[radial] no labels dataset matches time suffix {dotted} "
        f"(seq has no annotations -> cannot build the rigid tracks)"
    )


def write_configs(seq, short, seq_name, fs, fe, n_total, track_max_displacement_m, force=False):
    """Write the sequence stub and generate its run configs; return the `dyrad` run config.

    The stub follows the `configs/sequences/radial_*.yaml` files of the paper's sequences;
    the run configs `configs/radial/<ID>_<variant>.yaml` are produced by
    dyrad.generate_configs from the stub, the recipes and the variant files. An existing
    stub (e.g. one of the paper's sequences) is kept unless `force` is set.
    """
    stub = ROOT / "configs" / "sequences" / f"radial_{short}.yaml"
    cfg = ROOT / "configs" / "radial" / f"{short}_dyrad.yaml"
    if stub.exists() and not force:
        kept = yaml.safe_load(stub.read_text())
        if (int(kept["frame_start"]), int(kept["frame_end"])) != (fs, fe):
            raise SystemExit(
                f"[radial] {stub.relative_to(ROOT)} exists with window "
                f"[{kept['frame_start']}, {kept['frame_end']}), not [{fs}, {fe}); "
                f"pass --force to overwrite it"
            )
        print(f"[radial] {stub.relative_to(ROOT)} exists; keeping it (pass --force to overwrite)")
    else:
        # render_frame_idxs span the whole recording (as in the paper's stubs), not the window
        render_idxs = sorted(set(int(round(x)) for x in np.linspace(0, n_total - 1, 7)))
        stub.write_text(
            f"# RADIal sequence {short}: recording {seq_name}, {fe - fs}-frame training window.\n"
            + yaml.safe_dump(
                {
                    "dataset": "radial",
                    "seq_dir": f"data/radial_processed/{seq}",
                    "rad_tensors_dir": f"data/radial_processed/{seq}/rad_tensors",
                    "radar_poses_dir": f"data/radial_processed/{seq}/poses_can",
                    "ego_vel_npy": f"data/radial_processed/{seq}/ego_vel_can.npy",
                    "seq_name": seq_name,
                    "frame_start": fs,
                    "frame_end": fe,
                    "track_max_displacement_m": float(track_max_displacement_m),
                    "render_frame_idxs": render_idxs,
                },
                sort_keys=False,
                default_flow_style=None,
            )
        )
        print(f"[radial] stub    -> {stub.relative_to(ROOT)}")
    # the run configs of this sequence only (generate_configs.main would rewrite them all)
    n = 0
    wanted = {f"{short}_{v}.yaml" for v in generate_configs.VARIANTS["radial"]}
    for p, c in generate_configs.all_configs().items():
        if p.parent == generate_configs.CONFIGS / "radial" and p.name in wanted:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(generate_configs.render(c))
            n += 1
    print(f"[radial] configs -> {cfg.parent.relative_to(ROOT)}/{short}_*.yaml ({n} variants)")
    assert cfg.exists(), f"generate_configs did not produce {cfg}"
    return cfg


def run_builder(cfg, nf, partition):
    cmd = [
        PY,
        "-m",
        "dyrad.preprocessing.build_init_cloud",
        "--config",
        cfg,
        "--noise-factor",
        round(nf, 4),
    ]
    sh(cmd + (["--partition"] if partition else []))


def build_cloud(cfg, cloud, n_train, target, nf0=2.0, tol=0.15, max_iter=7):
    """Bisect the noise factor (log scale) until the partitioned cloud has ~target points per
    train frame; return the value used.

    Density decreases monotonically with the noise factor, so step geometrically until the
    target is bracketed, then bisect. The builder writes its default output name, which
    is the `init_cloud_path` of the generated run configs (`cloud`).
    """
    lo = hi = None
    nf, best = nf0, None
    for _ in range(max_iter):
        run_builder(cfg, nf, True)
        assert cloud.exists(), f"cloud builder did not write {cloud}"
        dens = len(np.load(cloud)) / n_train
        err = abs(dens - target) / target
        print(f"[radial] noise_factor={nf:.4f}  density={dens:.0f} pts/frame  err={err:.3f}")
        if best is None or err < best[1]:
            best = (nf, err)
        if err <= tol:
            return nf
        if dens > target:
            lo = nf
        else:
            hi = nf
        if lo is not None and hi is not None:
            nf = float(np.sqrt(lo * hi))
        elif lo is not None:
            nf = lo * 1.5
        else:
            nf = hi / 1.5
    nf = best[0]
    run_builder(cfg, nf, True)
    print(
        f"[radial] no noise factor within {tol:.0%} of the target; kept the closest "
        f"({nf:.4f}, err {best[1]:.3f})"
    )
    return nf


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seq", required=True, help="processed seq dir name, e.g. seq_31_22")
    ap.add_argument("--frame-start", type=int, default=0, help="first frame of the window")
    ap.add_argument("--n-frames", type=int, default=60, help="window length in frames")
    ap.add_argument(
        "--target-density",
        type=float,
        default=TARGET_DENSITY,
        help=f"init-cloud points per train frame the noise factor is bisected to "
        f"(default {TARGET_DENSITY:g}, a density target; the paper clouds were built at "
        f"per-sequence noise factors)",
    )
    ap.add_argument(
        "--track-max-displacement-m",
        type=float,
        default=TRACK_MAX_DISPLACEMENT_M,
        help="bound on an object's world position, m (its distance from the pose origin, "
        "not its travel): must exceed the largest |world position| of an object in the "
        f"window (default {TRACK_MAX_DISPLACEMENT_M:g})",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing sequence stub")
    ap.add_argument(
        "--skip-cloud",
        action="store_true",
        help="write poses/configs/norm.json but skip the cloud builds",
    )
    args = ap.parse_args()

    seq = args.seq
    short = seq[len("seq_") :]
    seq_dir = PROC / seq
    assert seq_dir.exists(), f"{seq_dir} not found"
    assert (seq_dir / "ego_vel_can.npy").exists(), (
        f"{seq} has no ego_vel_can.npy (no CAN) -> cannot build poses_can"
    )
    n_total = len(list((seq_dir / "rad_tensors").glob("rad_*.npy")))
    fs, fe = args.frame_start, min(args.frame_start + args.n_frames, n_total)
    assert 0 <= fs < fe, f"bad window [{fs}, {fe}) for {n_total} frames"
    seq_name = seq_name_from_labels(short)
    print(f"[radial] {seq}  window [{fs}, {fe})  seq_name={seq_name}")

    # 1. poses_can
    if not (seq_dir / "poses_can" / "radar_poses.npy").exists():
        sh([PY, "-m", "dyrad.preprocessing.radial", "poses", "--seq-dir", seq_dir])
    # 2. raw-FFT Doppler axis
    rd = seq_dir / "rd_doppler_bins_mps.npy"
    if not rd.exists():
        # fftshifted 256-bin raw-FFT velocity axis, DC at index 128 (labels store the raw bin)
        np.save(rd, (np.arange(256) - 128) * DOPPLER_BIN_MPS)
        print(f"[radial] wrote {rd.name}")
    # 3. sequence stub + run configs
    cfg = write_configs(
        seq, short, seq_name, fs, fe, n_total, args.track_max_displacement_m, args.force
    )
    cfg_rel = cfg.relative_to(ROOT)
    cloud = ROOT / yaml.safe_load(cfg.read_text())["init_cloud_path"]
    cfg_static = cfg.with_name(f"{short}_dyrad_static.yaml")
    cloud_static = (
        ROOT / yaml.safe_load(cfg_static.read_text())["init_cloud_path"]
        if cfg_static.exists()
        else None
    )
    # 4. normalization sidecar
    if not (seq_dir / "norm.json").exists():
        sh([PY, "-m", "dyrad.preprocessing.normalize_dataset", "--configs", cfg_rel])
    # 5. init clouds: partitioned (dyrad, ablations) and unpartitioned (dyrad_static)
    run_cfg = load_config(str(cfg))
    n_train = sum(
        1 for f in range(fs, fe) if not is_held_out(f, run_cfg.test_every, run_cfg.test_offset)
    )
    if args.skip_cloud:
        print("[radial] --skip-cloud: not building the radar clouds")
    else:
        nf = None
        if cloud.exists():
            print(f"[radial] cloud exists, skipping ({cloud.name})")
            bj = cloud.with_name(cloud.name + ".build.json")
            if bj.exists():
                nf = float(json.loads(bj.read_text())["noise_factor"])
        else:
            nf = build_cloud(cfg_rel, cloud, n_train, args.target_density)
        if cloud_static is None:
            print("[radial] no dyrad_static config for this sequence; unpartitioned cloud skipped")
        elif cloud_static.exists():
            print(f"[radial] cloud exists, skipping ({cloud_static.name})")
        elif nf is None:
            print(f"[radial] {cloud.name} has no .build.json; delete it to rebuild both clouds")
        else:
            run_builder(cfg_static.relative_to(ROOT), nf, False)
            print(f"[radial] unpartitioned cloud -> {cloud_static.name} (noise_factor={nf:.4f})")
    print(f"[radial] done. Train with:\n  python -m dyrad.train --config {cfg_rel}")


if __name__ == "__main__":
    main()
