"""Render a variant's M1 at the original poses (the `T0` view) for scoring."""

from __future__ import annotations

from pathlib import Path

from . import spec


def evaluate(seq: str, variant: str, ckpt=None) -> int:
    from dyrad.evaluation.novel_view import evaluate_on_sequence
    from dyrad.config import load_config
    from dyrad.trainer.runner import Runner

    cfg = load_config(str(spec.config_path(seq, variant, "M1")))
    ckpt = Path(ckpt) if ckpt else spec.result_dir(seq, variant, "M1") / "ckpt_final.pt"
    if not ckpt.exists():
        raise SystemExit(f"M1 checkpoint not found: {ckpt}")
    t0 = spec.definition_dir(seq, "T0")
    if not (t0 / "rad_tensors").is_dir():
        raise SystemExit(
            f"no T0 view at {t0}; run `build-view --seq {seq} --view T0` first"
        )
    print(f"[evaluate] {seq}/{variant}: rendering M1 ({ckpt}) at the original poses")
    runner = Runner(cfg)
    runner.load_checkpoint(str(ckpt))
    # Cache the linear range from M1's training view (its norm.json) before
    # evaluate_on_sequence swaps the loaders to T0, so scoring uses the range M1 was fit with.
    runner.lin_range()
    head = evaluate_on_sequence(
        runner,
        cfg,
        t0,
        spec.eval_dir(seq, variant),
        labels_filename=spec.labels_filename(),
    )
    print(f"[evaluate] ra_psnr_lin={head.get('ra_psnr_lin')}  -> {spec.eval_dir(seq, variant)}")
    return 0
