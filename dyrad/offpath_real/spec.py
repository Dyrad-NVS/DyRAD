"""The real-data off-path benchmark definition (paper Sec. 4.2, Table 2).

Backed by `configs/benchmarks/offpath_real_<dataset>.json` and the sequence stubs in
`configs/sequences/`. Needs only json and yaml, so the CLI can read it from any environment.

Protocol for one sequence and one method variant:

    M0  fit to the recorded window                    results/offpath_real/<ds>_<seq>/<variant>_M0
    ->  render M0 along the laterally shifted path    <views_root>/seq_<seq>/T1_<variant>
    M1  fit to those renders                          results/offpath_real/<ds>_<seq>/<variant>_M1
    ->  render M1 at the original poses and score it  <...>/<variant>_M1/eval_T0
        against the real measurements                 <views_root>/seq_<seq>/T0

Two view definitions are shared by every variant: `T0` (the real tensors of the
window, re-indexed 0..N-1) and `T1` (the shifted poses, axes, ego velocity,
re-projected labels and normalization range, no tensors). Each variant renders its
own copy of `T1`.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dyrad.paths import ROOT

DEFAULT_BENCHMARK = "radial"
DEFINITIONS = ("T0", "T1")
STAGES = ("M0", "M1")

_ACTIVE = DEFAULT_BENCHMARK
_CACHE: dict[str, dict] = {}


def available() -> list[str]:
    return sorted(
        p.stem.removeprefix("offpath_real_")
        for p in (ROOT / "configs" / "benchmarks").glob("offpath_real_*.json")
    )


def select(benchmark: str) -> None:
    global _ACTIVE
    if benchmark not in available():
        raise SystemExit(f"unknown benchmark {benchmark!r}; have: {', '.join(available())}")
    _ACTIVE = benchmark


def active() -> str:
    return _ACTIVE


def load() -> dict:
    if _ACTIVE not in _CACHE:
        _CACHE[_ACTIVE] = json.loads(
            (ROOT / "configs" / "benchmarks" / f"offpath_real_{_ACTIVE}.json").read_text()
        )
    return _CACHE[_ACTIVE]


# ── protocol ─────────────────────────────────────────────────────────────────


def dataset() -> str:
    return load()["dataset"]["id"]


def seq_ids() -> list[str]:
    return [s["id"] for s in load()["sequences"]]


def variants() -> list[str]:
    return list(load()["variants"])


def n_frames() -> int:
    return int(load()["n_frames"])


def shift_m() -> float:
    return float(load()["lateral_shift_m"])


def labels_filename() -> str:
    return load()["dataset"]["labels_filename"]


def init_cloud() -> dict:
    """`{voxel_m, noise_factor}`: the build_init_cloud flags of the M0 and M1 clouds
    (`offpath_real init-cloud`); `voxel_m` also names the clouds in the generated configs."""
    return load()["init_cloud"]


def window(seq: str) -> tuple[int, int]:
    """(frame_start, frame_end) of `seq` in the source sequence's frame indexing.

    The M0 config trains on the sequence stub's window (through `base:`), so the stub is
    the source; the benchmark JSON's `frame_start` and `n_frames` must agree with it.
    """
    entry = next((s for s in load()["sequences"] if s["id"] == seq), None)
    if entry is None:
        raise KeyError(
            f"{seq!r} is not in the {active()} off-path benchmark (have: {', '.join(seq_ids())})"
        )
    stub = yaml.safe_load(sequence_stub(seq).read_text())
    fs, fe = int(stub["frame_start"]), int(stub["frame_end"])
    if (fs, fe) != (int(entry["frame_start"]), int(entry["frame_start"]) + n_frames()):
        raise SystemExit(
            f"{sequence_stub(seq).name} window {fs}..{fe} disagrees with the {active()} "
            f"benchmark (frame_start {entry['frame_start']}, n_frames {n_frames()})"
        )
    return fs, fe


# ── paths ────────────────────────────────────────────────────────────────────


def sequence_stub(seq: str) -> Path:
    return ROOT / "configs" / "sequences" / f"{dataset()}_{seq}.yaml"


def source_seq_dir(seq: str) -> Path:
    """The real, unmodified sequence a view derives from (from its sequence stub)."""
    return ROOT / yaml.safe_load(sequence_stub(seq).read_text())["seq_dir"]


def views_root() -> Path:
    return ROOT / load()["views_root"]


def definition_dir(seq: str, view: str) -> Path:
    """The variant-free view definition: `T0` (real measurements) or `T1` (shifted poses)."""
    if view not in DEFINITIONS:
        raise KeyError(f"unknown view definition {view!r} (have: {', '.join(DEFINITIONS)})")
    return views_root() / f"seq_{seq}" / view


def rendered_view_dir(seq: str, variant: str) -> Path:
    """Where `variant`'s M0 render of the shifted view lives (M1's training data)."""
    return views_root() / f"seq_{seq}" / f"T1_{variant}"


def view_seq_name(seq: str, lateral_m: float) -> str:
    """The `dataset` string stamped into a view's label CSV and set as the config's
    `seq_name`. The trainer keeps only label rows whose `dataset` equals `seq_name`,
    so both sides must come from this one function."""
    return f"offpath_{source_seq_dir(seq).name}_lat{float(lateral_m):+g}"


def variant_stage(variant: str, stage: str) -> str:
    if stage not in STAGES:
        raise KeyError(f"stage must be one of {STAGES}, got {stage!r}")
    return f"{variant}_{stage}"


def result_dir(seq: str, variant: str, stage: str) -> Path:
    return ROOT / "results" / "offpath_real" / f"{dataset()}_{seq}" / variant_stage(variant, stage)


def eval_dir(seq: str, variant: str) -> Path:
    """Where M1's render at the original poses, and its score, are written."""
    return result_dir(seq, variant, "M1") / "eval_T0"


def config_path(seq: str, variant: str, stage: str) -> Path:
    return (
        ROOT
        / "configs"
        / "offpath_real"
        / f"{dataset()}_{seq}_{variant_stage(variant, stage)}.yaml"
    )


def score_stub(seq: str) -> Path:
    """The shared scoring config of a sequence: axes, labels and normalization of T0."""
    return ROOT / "configs" / "offpath_real" / f"{dataset()}_{seq}_score_T0.yaml"
