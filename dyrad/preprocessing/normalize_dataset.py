"""Compute the per-sequence normalization sidecar (norm.json).

Writes one norm.json per sequence next to its rad_tensors_dir. Every consumer (trainer
loss and eval, scoring, visualization) reads this sidecar via dyrad.norm, so no component
re-derives its own normalization.

Params are computed over the train split only, in the exact tensor domain the trainer
sees: the RadarDataset output (post range-crop + Doppler roll, sensor-native, unscaled).
`hi` is the global constant lin_hi_global (shared across sequences), or a per-scene
robust percentile when that is unset (Boreas, the coarse RADIal sequences). Under `--mode linear` (RADIal, synthetic)
the normalization is affine on the linear range (lin_lo_global, hi); under `--mode counts`
(Boreas) it reproduces the sensor's 8-bit log-count convention, N = u / full_scale with
u = k * log10(power). See dyrad/norm.py for the exact maps.

Usage:
    python -m dyrad.preprocessing.normalize_dataset --configs "configs/radial/*_dyrad.yaml"
    python -m dyrad.preprocessing.normalize_dataset --configs configs/boreas/win55_104_dyrad.yaml \\
        --mode counts --counts-per-decade 20
"""

from __future__ import annotations

import argparse
import glob
import json
import traceback
from pathlib import Path

from dyrad import norm as cn
from dyrad.config import load_config
from dyrad.data import RadarDataset, RadarParser

#: `--mode counts`: the sensor's full-scale count (8-bit)
COUNTS_FULL_SCALE = 255.0


def _train_frames(cfg):
    """Yield unscaled train-split rad tensors in the trainer's dataset domain."""
    parser = RadarParser(
        rad_tensors_dir=cfg.rad_tensors_dir,
        poses_dir=cfg.radar_poses_dir,
        test_every=cfg.test_every,
        test_offset=cfg.test_offset,
        num_doppler_bins=cfg.num_doppler_bins,
        num_range_bins=cfg.num_range_bins,
        num_azimuth_bins=cfg.num_azimuth_bins,
        range_crop_first=cfg.range_crop_first,
        range_crop_last=cfg.range_crop_last,
        doppler_roll_bins=cfg.doppler_roll_bins,
    )
    # raw_unnormalized: this script defines the normalization range, so it must not be measured
    # through one. Without it a fresh sequence raises NormError (no sidecar to
    # resolve) and an existing one gets its frames pre-scaled by 1/hi, so a re-run
    # would write hi~1 over a correct range.
    ds = RadarDataset(
        parser,
        split="train",
        frame_start=cfg.frame_start,
        frame_end=cfg.frame_end,
        bad_frame_ids=cfg.bad_frame_ids,
        raw_unnormalized=True,
    )
    for i in range(len(ds)):
        yield ds[i]["rad_tensor"].numpy()  # [D,R,A], sensor-native, unscaled


def process_config(
    cfg_path: str,
    mode: str = "linear",
    counts_per_decade: float = 0.0,
) -> dict:
    if mode == "counts" and counts_per_decade <= 0:
        raise ValueError("--mode counts needs --counts-per-decade > 0 (Navtech: 20)")
    cfg = load_config(cfg_path)
    hi = float(cfg.lin_hi_global)
    hi_source = "lin_hi_global (shared across sequences)"
    if hi <= 0.0:
        # No global normalization range for this sensor (e.g. Boreas, where the RADIal globals
        # map every bin to 0). Fall back to the per-scene robust hi, computed exactly
        # as the trainer's lin_range() does, so the sidecar and the trainer agree.
        from dyrad.evaluation.radar_metrics import robust_lin_range

        hi_pct = float(cfg.lin_hi_pct)
        _, hi = robust_lin_range(_train_frames(cfg), hi_pct=hi_pct)
        hi = float(hi)
        hi_source = f"per-scene robust p{hi_pct} (no global normalization range)"
        print(f"  [hi] lin_hi_global unset -> per-scene robust p{hi_pct} = {hi:.6g}")
    if hi <= 0.0:
        raise SystemExit(f"{cfg_path}: could not determine hi; got {hi}")
    params = cn.compute_norm_params(
        _train_frames(cfg),
        hi,
        pct_lo=float(cfg.norm_percentile_lo),
        pct_hi=float(cfg.norm_percentile_hi),
        k_median=float(cfg.noise_floor_median_k),
        mode=mode,
        lin_lo=float(cfg.lin_lo_global),
        counts_per_decade=counts_per_decade,
        counts_full_scale=COUNTS_FULL_SCALE,
    )
    params["hi_source"] = hi_source
    params["seq_rad_dir"] = str(cfg.rad_tensors_dir)
    params["gt_is_amplitude"] = str(cfg.measurement_domain).endswith("amplitude")
    params["source_config"] = str(cfg_path)
    params["split"] = {
        "test_every": int(cfg.test_every),
        "test_offset": int(cfg.test_offset),
        "frame_start": cfg.frame_start,
        "frame_end": cfg.frame_end,
    }

    out = Path(cfg.rad_tensors_dir).parent / "norm.json"
    print(
        f"[{Path(cfg_path).name}] mode={params['mode']} hi={hi:.4g} ({hi_source}) "
        f"norm_lo={params['norm_lo']:.4f} norm_range={params['norm_range']:.4f} "
        f"floor_u={params['floor_u']:.4e} (nf={params['nf']:.4e}) -> {out}"
    )
    out.write_text(json.dumps(params, indent=2))
    return params


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--configs", required=True, help="glob or single path to run config yaml(s)")
    ap.add_argument(
        "--counts-per-decade",
        type=float,
        default=0.0,
        help="mode=counts only: the converter's --db-counts-per-decade "
        "(Boreas/Navtech = 20). u = k*log10(power) recovers the raw count.",
    )
    ap.add_argument(
        "--mode",
        default="linear",
        choices=["linear", "counts"],
        help="linear (default): affine on the linear range (lin_lo_global, hi); "
        "counts: the sensor's 8-bit log-count convention (Boreas)",
    )
    args = ap.parse_args()

    cfgs = (
        sorted(glob.glob(args.configs)) if any(c in args.configs for c in "*?[") else [args.configs]
    )
    if not cfgs:
        raise SystemExit(f"no configs matched {args.configs}")
    print(f"[normalize_dataset] {len(cfgs)} config(s)")
    failed = []
    for c in cfgs:
        try:
            process_config(
                c,
                mode=args.mode,
                counts_per_decade=args.counts_per_decade,
            )
        except Exception:
            print(f"[FAIL] {c}:\n{traceback.format_exc()}")
            failed.append(c)
    # Exit non-zero on any failure, so a calling script does not proceed to
    # training without a sidecar.
    if failed:
        print(
            f"[normalize_dataset] {len(failed)}/{len(cfgs)} FAILED — "
            f"no sidecar written for: {', '.join(failed)}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
