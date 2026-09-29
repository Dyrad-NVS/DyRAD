"""Score a variant's M1 renders at T0 against the real measurements with the shared scorer."""

from __future__ import annotations

import json
import shutil

from . import spec

METRICS_NAME = "metrics_extended.json"


def score(seq: str, variant: str, keep_renders: bool = False) -> dict:
    from dyrad.evaluation.score_renders import score_run_dir

    eval_dir = spec.eval_dir(seq, variant)
    if not (eval_dir / "renders_npy").is_dir():
        raise SystemExit(
            f"no renders_npy under {eval_dir}; run `evaluate --seq {seq} --variant {variant}` first"
        )
    stub = spec.score_stub(seq)
    labels = spec.definition_dir(seq, "T0") / spec.labels_filename()
    for p in (stub, labels):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    print(f"[score] {seq}/{variant}: {eval_dir}")
    # The stub's data paths are relative to the repository root (__main__ chdirs there).
    score_run_dir(eval_dir, str(stub), labels_csv=str(labels), split="all")
    out = eval_dir / METRICS_NAME
    if not out.exists():
        raise SystemExit(f"score_renders wrote no {METRICS_NAME} in {eval_dir}")
    # The trainer's reduced panel; the full one supersedes it.
    (eval_dir / "metrics.json").unlink(missing_ok=True)
    if not keep_renders:
        shutil.rmtree(eval_dir / "renders_npy", ignore_errors=True)
    return json.loads(out.read_text())["aggregate"]
