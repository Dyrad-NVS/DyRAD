"""Real-data off-path evaluation (paper Sec. 4.2, Table 2): a render-and-refit protocol.

T0 is the original trajectory: the real measurements of the window, re-indexed 0..N-1.
For one sequence and one method variant:

    python -m dyrad.offpath_real build-view --seq 31_22 --view T0          # real measurements, re-indexed
    python -m dyrad.offpath_real build-view --seq 31_22 --view T1          # shifted poses, labels, axes
    python -m dyrad.offpath_real init-cloud --seq 31_22 --variant dyrad --stage M0
    python -m dyrad.train --config configs/offpath_real/radial_31_22_dyrad_M0.yaml
    python -m dyrad.offpath_real render     --seq 31_22 --variant dyrad    # M0 along the shifted path
    python -m dyrad.offpath_real init-cloud --seq 31_22 --variant dyrad --stage M1
    python -m dyrad.train --config configs/offpath_real/radial_31_22_dyrad_M1.yaml
    python -m dyrad.offpath_real evaluate   --seq 31_22 --variant dyrad    # M1 back at the original poses
    python -m dyrad.offpath_real score      --seq 31_22 --variant dyrad    # against the real measurements

`init-cloud` runs dyrad.preprocessing.build_init_cloud on the stage's config with the
benchmark's `init_cloud` block (configs/benchmarks/offpath_real_<dataset>.json:
`--noise-factor`, `--voxel`), and with `--partition` unless the variant is `static_only`
(DyRAD-static trains on the unpartitioned cloud). The M1 cloud is built from the view
written by `render`.

The dataset is chosen with --benchmark radial|boreas. It is a top-level option and goes
before the subcommand:

    python -m dyrad.offpath_real --benchmark boreas build-view --seq win55_104 --view T0

`spec` prints the benchmark definition. Run from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import spec


def main(argv=None) -> int:
    os.chdir(spec.ROOT)
    # Select the benchmark before building the parser: `--variant` choices depend on it.
    pre = argparse.ArgumentParser(prog="dyrad.offpath_real", add_help=False)
    pre.add_argument("--benchmark", default=None, choices=spec.available())
    benchmark = pre.parse_known_args(argv)[0].benchmark
    if benchmark:
        spec.select(benchmark)
    ap = argparse.ArgumentParser(
        prog="dyrad.offpath_real",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--benchmark",
        default=None,
        choices=spec.available(),
        help=f"dataset benchmark (default: {spec.DEFAULT_BENCHMARK})",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("spec", help="print the benchmark definition")
    p.set_defaults(fn=_spec)
    p = sub.add_parser("build-view", help="write a view definition")
    p.add_argument("--seq", required=True)
    p.add_argument("--view", required=True, choices=spec.DEFINITIONS)
    p.set_defaults(fn=_build)
    p = sub.add_parser("init-cloud", help="build a stage's init cloud with the benchmark's flags")
    p.add_argument("--seq", required=True)
    p.add_argument("--variant", required=True, choices=spec.variants())
    p.add_argument("--stage", required=True, choices=spec.STAGES)
    p.set_defaults(fn=_init_cloud)
    for name, fn, help_ in (
        ("render", _render, "render a variant's M0 into the shifted view"),
        ("evaluate", _evaluate, "render a variant's M1 at the original poses"),
        ("score", _score, "score a variant's M1 renders at T0"),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--seq", required=True)
        p.add_argument("--variant", required=True, choices=spec.variants())
        if name != "score":
            p.add_argument(
                "--ckpt", default=None, help="checkpoint (default: the variant's result dir)"
            )
        else:
            p.add_argument(
                "--keep-renders",
                action="store_true",
                help="keep eval_T0/renders_npy after scoring (deleted by default)",
            )
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


def _spec(a) -> int:
    print(json.dumps(spec.load(), indent=2))
    return 0


def _build(a) -> int:
    from . import build

    return build.build_view(a.seq, a.view)


def _init_cloud(a) -> int:
    from dyrad.config import load_config

    config = spec.config_path(a.seq, a.variant, a.stage)
    if a.stage == "M1" and not (spec.rendered_view_dir(a.seq, a.variant) / "rad_tensors").is_dir():
        raise SystemExit(
            f"no rendered view for {a.seq}/{a.variant}; run `render --seq {a.seq} "
            f"--variant {a.variant}` first"
        )
    ic = spec.init_cloud()
    cmd = [
        sys.executable,
        "-m",
        "dyrad.preprocessing.build_init_cloud",
        "--config",
        str(config.relative_to(spec.ROOT)),
        "--noise-factor",
        str(float(ic["noise_factor"])),
        "--voxel",
        str(float(ic["voxel_m"])),
    ]
    if not load_config(str(config)).static_only:
        cmd.append("--partition")
    print("[init-cloud] " + " ".join(cmd[1:]))
    return subprocess.run(cmd, check=False).returncode


def _render(a) -> int:
    from . import build

    return build.render(a.seq, a.variant, ckpt=a.ckpt)


def _evaluate(a) -> int:
    from . import evaluate

    return evaluate.evaluate(a.seq, a.variant, ckpt=a.ckpt)


def _score(a) -> int:
    from . import score

    m = score.score(a.seq, a.variant, keep_renders=a.keep_renders)
    print(
        f"  ra_psnr_lin={m.get('ra_psnr_lin')}  ra_corr={m.get('ra_corr')}  rad_corr={m.get('rad_corr')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
