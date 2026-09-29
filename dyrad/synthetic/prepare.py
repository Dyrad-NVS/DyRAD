#!/usr/bin/env python
"""Generate and prepare the five synthetic off-path benchmark scenes.

For each scene this
  1. generates the base trajectory and the five off-path views
     (lateral -1.75 / +1.75 / +3.5 m, yaw +5 / +10 deg) with generate_offpath_views.py
     (skipped when all six view directories already hold NUM_FRAMES tensors),
  2. writes the base's normalization sidecar (norm.json) and gives every view the
     base's normalization range, so all views of a scene are scored on one range
     (recomputed whenever the views are generated),
  3. builds the two radar pseudo-LiDAR init clouds: the RA-partitioned cloud used by
     the full method, and the unpartitioned cloud used by the static-only variant,
  4. measures the scene's RA / RD / AD normalization ranges for the scorer
     (configs/benchmarks/synthetic_norm_ranges.json): the p0.01 / p99.9 percentiles of
     each projection, pooled over all base-trajectory frames on the cropped range axis.

    python -m dyrad.synthetic.prepare                     # all five scenes
    python -m dyrad.synthetic.prepare --scenes intersection
"""

import argparse
import json
import subprocess
import sys

import numpy as np

from dyrad.config import load_config, load_yaml_config
from dyrad.evaluation.radar_metrics import ra_project, rd_project
from dyrad.paths import ROOT
from dyrad.synthetic.generate_offpath_views import view_tags
from dyrad.synthetic.generate_scene import NUM_FRAMES, SEED
from dyrad.synthetic.scene_spec import scene_names

SCENES = scene_names()
# Threshold multiplier of the init-cloud builder, per scene (bins kept: > nf * median):
# the paper's fixed per-scene thresholds.
NOISE_FACTOR = {
    "urban_canyon": 1.50,
    "open_road_sparse": 1.22,
    "curve_ramp": 1.40,
    "intersection": 1.38,
    "dual_carriageway": 1.38,
}
assert set(NOISE_FACTOR) == set(SCENES), "every scene yaml needs a NOISE_FACTOR entry"


NORM_RANGES = ROOT / "configs/benchmarks/synthetic_norm_ranges.json"
# The percentiles of the scene's cube range, applied per projection.
PCT_LO, PCT_HI = 0.01, 99.9


def measure_norm_ranges(cfg_path) -> tuple[dict, tuple[int, int]]:
    """({n_frames, ra, rd, ad: [lo, hi]}, (crop_first, crop_last)) of one scene: the
    percentiles pooled over its base frames on the cropped range axis."""
    cfg = load_config(str(cfg_path))
    frames = sorted((ROOT / cfg.rad_tensors_dir).glob("rad_*.npy"))
    projections = {
        "ra": ra_project,
        "rd": rd_project,
        "ad": lambda rad: rad.mean(axis=1),  # [D, A], mean over range
    }
    pooled = {v: [] for v in projections}
    for f in frames:
        rad = np.load(f).astype(np.float64)
        rad = rad[:, cfg.range_crop_first : rad.shape[1] - cfg.range_crop_last, :]
        for v, proj in projections.items():
            pooled[v].append(proj(rad).ravel())
    out = {"n_frames": len(frames)}
    for v, chunks in pooled.items():
        arr = np.concatenate(chunks)
        out[v] = [float(np.percentile(arr, PCT_LO)), float(np.percentile(arr, PCT_HI))]
    return out, (cfg.range_crop_first, cfg.range_crop_last)


def write_norm_ranges(scene: str, ranges: dict, crop: tuple) -> None:
    doc = (
        json.loads(NORM_RANGES.read_text()) if NORM_RANGES.is_file() else {"scenes": {}}
    )
    doc["description"] = (
        "Per-scene RA/RD/AD marginal normalization ranges of the synthetic benchmark: "
        f"p{PCT_LO}/p{PCT_HI} of each projection pooled over the base-trajectory frames "
        "(written by python -m dyrad.synthetic.prepare)."
    )
    doc["crop"] = {"first": crop[0], "last": crop[1]}
    doc["scenes"][scene] = ranges
    doc["scenes"] = dict(sorted(doc["scenes"].items()))
    NORM_RANGES.write_text(json.dumps(doc, indent=1) + "\n")


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenes", nargs="*", default=SCENES, choices=SCENES)
    a = ap.parse_args()
    py = sys.executable
    from dyrad.norm import inherit, sidecar_path

    for scene in a.scenes:
        print(f"[synthetic] {scene}")
        parent = ROOT / "data/synthetic" / scene
        cfg = ROOT / f"configs/synthetic/{scene}_dyrad.yaml"
        cfg_static = ROOT / f"configs/synthetic/{scene}_dyrad_static.yaml"
        n_tensors = [len(list((parent / t / "rad_tensors").glob("rad_*.npy"))) for t in view_tags()]
        generate = n_tensors != [NUM_FRAMES] * len(n_tensors)
        if not generate:
            print(f"  reusing the generated views in {parent} (delete them to regenerate)")
        else:
            run(
                [
                    py,
                    "-m",
                    "dyrad.synthetic.generate_offpath_views",
                    "--parent-dir",
                    parent,
                    "--scene",
                    scene,
                    "--num-frames",
                    NUM_FRAMES,
                    "--seed",
                    SEED,
                ]
            )
        # The sidecars depend on the tensors, so fresh views get fresh sidecars.
        if sidecar_path(parent / "base").is_file() and not generate:
            print(f"  reusing {sidecar_path(parent / 'base')} (delete it to recompute)")
        else:
            run([py, "-m", "dyrad.preprocessing.normalize_dataset", "--configs", cfg])
        for v in sorted(parent.iterdir()):
            if (
                v.name != "base"
                and (v / "rad_tensors").is_dir()
                and (generate or not sidecar_path(v).is_file())
            ):
                inherit(
                    v,
                    parent / "base",
                    note="off-path view; inherits the base scene's normalization range",
                )
        # The clouds land under the builder's default names, which are the run configs'
        # `init_cloud_path` (generate_configs derives both from the same rule, with the
        # config's init_cloud_voxel_m).
        nf = NOISE_FACTOR[scene]
        run(
            [
                py,
                "-m",
                "dyrad.preprocessing.build_init_cloud",
                "--config",
                cfg,
                "--noise-factor",
                nf,
                "--voxel",
                load_yaml_config(cfg)["init_cloud_voxel_m"],
                "--partition",
            ]
        )
        run(
            [
                py,
                "-m",
                "dyrad.preprocessing.build_init_cloud",
                "--config",
                cfg_static,
                "--noise-factor",
                nf,
                "--voxel",
                load_yaml_config(cfg_static)["init_cloud_voxel_m"],
            ]
        )
        ranges, crop = measure_norm_ranges(cfg)
        write_norm_ranges(scene, ranges, crop)
        print(
            "  norm ranges: "
            + "  ".join(
                f"{v}=({ranges[v][0]:.4g}, {ranges[v][1]:.4g})" for v in ("ra", "rd", "ad")
            )
        )
    print("[synthetic] done")


if __name__ == "__main__":
    main()
