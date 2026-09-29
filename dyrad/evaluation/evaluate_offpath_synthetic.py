"""Synthetic off-path evaluation (paper Sec. 4.2, Tables 9-10).

A scene fitted to a synthetic scene's base trajectory is rendered, without refitting,
along the displaced trajectories of that scene (lateral -1.75 / +1.75 / +3.5 m, yaw
+5 / +10 deg) and scored against their ground-truth measurements. The scene is identical
across views and the reflectors and tracks live in world coordinates, so this measures
generalization to unseen viewpoints rather than interpolation between training poses.

    python -m dyrad.evaluation.evaluate_offpath_synthetic \\
        --config configs/synthetic/intersection_dyrad.yaml --views data/synthetic/intersection \\
        [--ckpt results/synthetic/intersection/dyrad/ckpt_final.pt] [--out DIR] [--keep-renders]

`--views` is the scene's parent dir: every subdir with rad_tensors/ is a view, except the
config's own training sequence (`base/`). `--view DIR` (repeatable) names views explicitly
instead. `--ckpt` is the one override of the config (default <result_dir>/ckpt_final.pt).

Each view is rendered into <out>/<view>/renders_npy/ (next to the trainer's reduced
panel, metrics.json) and scored into <out>/<view>/metrics_extended.json (the paper's
Table 9/10 metrics), the same as running
`score_renders <out>/<view> --config <config> --labels <views>/<view>/labels_CVPR.csv --label-seq <view>`
(the view's own labels, filtered on the view name; the config's labels are the base's).
The renders are deleted afterwards unless `--keep-renders`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dyrad.evaluation.novel_view import evaluate_on_sequence
from dyrad.evaluation.score_renders import score_run_dir
from dyrad.config import load_config
from dyrad.trainer.runner import Runner


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="BASE training config (the trained scene)")
    ap.add_argument(
        "--ckpt", default=None, help="checkpoint to render (default: <result_dir>/ckpt_final.pt)"
    )
    ap.add_argument(
        "--views",
        default=None,
        help="parent dir of the scene's views; evaluates every subdir with rad_tensors/ "
        "except the config's own training sequence",
    )
    ap.add_argument(
        "--view",
        action="append",
        default=[],
        help="explicit view dir (repeatable); overrides --views",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output root for per-view metrics (default <result_dir>/offpath_eval)",
    )
    ap.add_argument(
        "--keep-renders",
        action="store_true",
        help="keep each view's renders_npy/ after scoring (default: delete)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    result_dir = Path(cfg.result_dir)
    ckpt = args.ckpt or str(result_dir / "ckpt_final.pt")
    if not Path(ckpt).exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    if args.view:
        seq_dirs = [Path(s) for s in args.view]
    elif args.views:
        base_dir = Path(cfg.seq_dir).resolve()  # the trained (on-path) sequence
        seq_dirs = sorted(
            d
            for d in Path(args.views).iterdir()
            if (d / "rad_tensors").is_dir() and d.resolve() != base_dir
        )
    else:
        raise SystemExit("provide --views or one or more --view")
    if not seq_dirs:
        raise SystemExit("no view dirs found")
    out_root = Path(args.out) if args.out else (result_dir / "offpath_eval")

    print(f"[offpath_synthetic] building Runner from the base config + checkpoint {ckpt}")
    runner = Runner(cfg)
    runner.load_checkpoint(ckpt)
    runner.lin_range()  # cache the base scene's normalization range before swapping loaders

    # Render every view, then score them.
    for sd in seq_dirs:
        print(f"\n========== view: {sd.name}  ({sd}) ==========")
        evaluate_on_sequence(runner, cfg, sd, out_root / sd.name)
    for sd in seq_dirs:
        print(f"\n========== scoring view: {sd.name} ==========")
        view_out = out_root / sd.name
        score_run_dir(
            view_out,
            args.config,
            labels_csv=str(sd / "labels_CVPR.csv"),
            label_seq=sd.name,
        )
        if not args.keep_renders:
            shutil.rmtree(view_out / "renders_npy", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
