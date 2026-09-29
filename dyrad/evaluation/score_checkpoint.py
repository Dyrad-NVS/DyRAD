"""Score a checkpoint on demand: render, score, delete. No persistent cache.

The trainer does not persist renders_npy by default (save_renders_npy=False,
~3.5 GB per run). This wrapper regenerates them transiently:

  1. restore the checkpoint into a Runner,
  2. run the trainer's own eval loop with save_renders_npy forced on, into a
     temporary directory inside the result dir (so the training run's
     metrics.json is left untouched), giving renders_npy/ in exactly the
     conventions score_renders expects,
  3. score them with score_renders.score_run_dir (GPU-free, the full panel
     including the point-cloud family),
  4. move metrics_extended[_<split>].json (and renders_npy/ with --keep) into the
     result dir and delete the temporary directory.

Usage:
    python -m dyrad.evaluation.score_checkpoint --config CFG [--ckpt ckpt_final.pt]
        [--split val] [--keep] [--no-lpips]

Writes <result_dir>/metrics_extended[_<split>].json. Run from the repository root
(the config's data paths are relative to it).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from dyrad.evaluation.score_renders import score_run_dir
from dyrad.config import load_config
from dyrad.trainer.runner import Runner


def _sig10(v) -> float:
    """`v` rounded to 10 significant digits."""
    return float(f"{float(v):.10g}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="the run's training config")
    ap.add_argument("--ckpt", default=None, help="default: <result_dir>/ckpt_final.pt")
    ap.add_argument("--split", default="val", choices=["all", "train", "val"])
    ap.add_argument(
        "--keep",
        action="store_true",
        help="keep renders_npy/ after scoring (default: delete)",
    )
    ap.add_argument(
        "--no-lpips", action="store_true", help="skip LPIPS (diagnostic scoring only)"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.save_renders_npy = True  # force the transient dump
    result_dir = Path(cfg.result_dir)
    ckpt = args.ckpt or str(result_dir / "ckpt_final.pt")
    if not Path(ckpt).exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    runner = Runner(cfg)
    runner.load_checkpoint(ckpt)
    tmp = Path(tempfile.mkdtemp(prefix=".score_checkpoint_", dir=result_dir))
    try:
        cfg.result_dir = str(tmp)  # _evaluate writes renders_npy/ and metrics.json here
        print(f"[score_checkpoint] rendering renders_npy from {ckpt} ...")
        runner._evaluate(0)
        # The trainer's normalization, rounded to 10 significant digits: the values
        # score_renders would resolve from the sidecar, up to that rounding.
        lo, hi = runner.lin_range()
        n_norm = (runner.norm_floor_u, runner.norm_lo_log, runner.norm_hi_log)
        score_run_dir(
            tmp,
            args.config,
            split=args.split,
            with_lpips=not args.no_lpips,
            lin_norm=(_sig10(lo), _sig10(hi)),
            n_norm=tuple(_sig10(v) for v in n_norm),
        )
        name = (
            "metrics_extended.json"
            if args.split == "all"
            else f"metrics_extended_{args.split}.json"
        )
        shutil.move(str(tmp / name), str(result_dir / name))
        print(f"[score_checkpoint] wrote {result_dir / name}")
        if args.keep:
            shutil.rmtree(result_dir / "renders_npy", ignore_errors=True)
            shutil.move(str(tmp / "renders_npy"), str(result_dir / "renders_npy"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
