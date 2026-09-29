"""Write every per-run training config from the sequence stubs and the variant recipes.

    configs/<dataset>/<seq>_<variant>.yaml
        base: [../recipes/radial.yaml, (../recipes/<dataset>.yaml), ../sequences/<dataset>_<seq>.yaml,
               ../variants/<variant>.yaml]
        init_cloud_path: <seq_dir>/radar_cloud_f<S>_<E>_<split>_<vox>m[_free_psf<k>].npy
        result_dir:      results/<dataset>/<seq>/<variant>

The init-cloud name follows dyrad.preprocessing.build_init_cloud: window [S, E], holdout split
(`no_every<K>` or `full`), voxel size, and the `_free_psf<k>` tail of an RA-partitioned cloud (DyRAD-static uses the
unpartitioned cloud). Everything else a run needs lives in the layered yaml files, so this
script only encodes the layout.

Off-path configs (configs/offpath_real/, from configs/benchmarks/offpath_real_<dataset>.json) follow the same
scheme: `<dataset>_<seq>_<variant>_M0.yaml` trains on the recorded window,
`..._M1.yaml` trains on the variant's rendered shifted view, and `<dataset>_<seq>_score_T0.yaml`
is the shared scoring config of the real measurements.

    python -m dyrad.generate_configs            # rewrite every generated config
    python -m dyrad.generate_configs --check    # exit 1 if the committed files differ

Every generated file starts with a comment saying so; edit the stubs, recipes and
variants, not the generated files. The M0 / M1 / score_T0 configs set `test_every: 0`: the
off-path fits train on every frame of their window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from dyrad.paths import ROOT
from dyrad.preprocessing.build_init_cloud import cloud_filename

CONFIGS = ROOT / "configs"

#: Variants trained per dataset: RADIal runs the full method, DyRAD-static and the four
#: ablations (Tables 1, 2, 12-15); Boreas and synthetic run DyRAD and DyRAD-static.
VARIANTS = {
    "radial": [
        "dyrad",
        "dyrad_static",
        "abl_no_doppler",
        "abl_no_interp",
        "abl_learned_extent",
        "abl_learned_psf",
    ],
    "boreas": ["dyrad", "dyrad_static"],
    "synthetic": ["dyrad", "dyrad_static"],
}
#: Coarse-sensor sequences (sensor-configuration transfer) train the full method only.
COARSE_SUFFIX = "_coarse"


def load_layered(files: list[Path]) -> dict:
    """Merge yaml files in order (later files win), following nested `base:` lists."""
    from dyrad.config import load_yaml_config

    merged: dict = {}
    for f in files:
        merged.update(load_yaml_config(str(f)))
    return merged


def recipe_chain(dataset: str) -> list[str]:
    chain = ["recipes/radial.yaml"]
    if dataset != "radial":
        chain.append(f"recipes/{dataset}.yaml")
    return chain


def cloud_name(cfg: dict, frame_start: int, frame_end: int, test_every=None, voxel_m=None) -> str:
    """The run's init-cloud filename from its layered config: the full method and the
    ablations use the RA-partitioned cloud (`_free[_psf<k>]`), DyRAD-static
    (`static_only`) the unpartitioned one. `test_every` / `voxel_m` override the
    config's (the off-path fits)."""
    return cloud_filename(
        frame_start,
        frame_end,
        int(cfg["test_every"] if test_every is None else test_every),
        cfg["init_cloud_voxel_m"] if voxel_m is None else voxel_m,
        partition=not cfg["static_only"],
        psf_margin=float(cfg["car_psf_margin"]),
    )


def run_config(stub_path: Path, variant: str) -> tuple[Path, dict]:
    stub = yaml.safe_load(stub_path.read_text())
    dataset = stub["dataset"]
    seq = stub_path.stem[len(dataset) + 1 :]
    # recipes first, then the sequence stub (sequence-specific values win), then the variant
    base = (
        [f"../{r}" for r in recipe_chain(dataset)]
        + [f"../sequences/{stub_path.name}"]
        + [f"../variants/{variant}.yaml"]
    )
    cfg = load_layered([CONFIGS / b[3:] for b in base])
    out = {
        "base": base,
        "init_cloud_path": f"{stub['seq_dir']}/{cloud_name(cfg, stub['frame_start'], stub['frame_end'])}",
        "result_dir": f"results/{dataset}/{seq}/{variant}",
    }
    return CONFIGS / dataset / f"{seq}_{variant}.yaml", out


def offpath_configs(benchmark: str) -> dict[Path, dict]:
    """M0 / M1 / score_T0 configs of the real-data off-path benchmark of one dataset."""
    from dyrad.offpath_real import spec

    spec.select(benchmark)
    ds, n, shift = spec.dataset(), spec.n_frames(), spec.shift_m()
    recipes = [f"../{r}" for r in recipe_chain(ds)]
    out = {}
    for seq in spec.seq_ids():
        stub_path = spec.sequence_stub(seq)
        stub = yaml.safe_load(stub_path.read_text())
        fs, fe = spec.window(seq)
        render_idxs = sorted(
            {int(round(x)) for x in [0, (n - 1) / 4, (n - 1) / 2, 3 * (n - 1) / 4, n - 1]}
        )
        t0 = spec.definition_dir(seq, "T0").relative_to(ROOT)
        for variant in spec.variants():
            cfg = load_layered(
                [CONFIGS / r[3:] for r in recipes]
                + [stub_path, CONFIGS / "variants" / f"{variant}.yaml"]
            )
            # No holdout (test_every 0) and the benchmark's own voxel size; the
            # rendered view's cloud is the same rule over its local frames 0..n-1.
            vox = spec.init_cloud()["voxel_m"]
            m0_cloud = cloud_name(cfg, fs, fe, test_every=0, voxel_m=vox)
            m1_cloud = cloud_name(cfg, 0, n, test_every=0, voxel_m=vox)
            out[spec.config_path(seq, variant, "M0")] = {
                "base": recipes + [f"../sequences/{stub_path.name}", f"../variants/{variant}.yaml"],
                "test_every": 0,
                "init_cloud_path": f"{stub['seq_dir']}/{m0_cloud}",
                "result_dir": str(spec.result_dir(seq, variant, "M0").relative_to(ROOT)),
            }
            view = spec.rendered_view_dir(seq, variant).relative_to(ROOT)
            out[spec.config_path(seq, variant, "M1")] = {
                "base": recipes + [f"../variants/{variant}.yaml"],
                "dataset": ds,
                "seq_dir": str(view),
                "rad_tensors_dir": f"{view}/rad_tensors",
                "radar_poses_dir": f"{view}/poses_can",
                "ego_vel_npy": f"{view}/ego_vel_can.npy",
                "object_label_path": f"{view}/{spec.labels_filename()}",
                "seq_name": spec.view_seq_name(seq, shift),
                "frame_start": 0,
                "frame_end": n,
                "test_every": 0,
                "render_frame_idxs": render_idxs,
                "track_max_displacement_m": stub["track_max_displacement_m"],
                "init_cloud_path": f"{view}/{m1_cloud}",
                "result_dir": str(spec.result_dir(seq, variant, "M1").relative_to(ROOT)),
            }
        out[spec.score_stub(seq)] = {
            "base": recipes + [f"../sequences/{stub_path.name}"],
            "seq_dir": str(t0),
            "rad_tensors_dir": f"{t0}/rad_tensors",
            "radar_poses_dir": f"{t0}/poses_can",
            "ego_vel_npy": f"{t0}/ego_vel_can.npy",
            "object_label_path": f"{t0}/{spec.labels_filename()}",
            "seq_name": spec.view_seq_name(seq, 0.0),
            "frame_start": 0,
            "frame_end": n,
            "test_every": 0,
            "result_dir": str(
                (ROOT / "results" / "offpath_real" / f"{ds}_{seq}" / "score_T0").relative_to(ROOT)
            ),
        }
    return out


def all_configs() -> dict[Path, dict]:
    out = {}
    for stub_path in sorted((CONFIGS / "sequences").glob("*.yaml")):
        dataset = yaml.safe_load(stub_path.read_text())["dataset"]
        variants = ["dyrad"] if stub_path.stem.endswith(COARSE_SUFFIX) else VARIANTS[dataset]
        for v in variants:
            path, cfg = run_config(stub_path, v)
            out[path] = cfg
    for benchmark in sorted(
        p.stem.removeprefix("offpath_real_")
        for p in (CONFIGS / "benchmarks").glob("offpath_real_*.json")
    ):
        out.update(offpath_configs(benchmark))
    return out


HEADER = (
    "# generated by python -m dyrad.generate_configs from the stub, recipe and variant; "
    "edit those instead\n"
)


def render(cfg: dict) -> str:
    return HEADER + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check", action="store_true", help="verify the committed configs instead of writing"
    )
    a = ap.parse_args(argv)
    generated = all_configs()
    stale = []
    for dataset in list(VARIANTS) + ["offpath_real"]:
        for p in (CONFIGS / dataset).glob("*.yaml"):
            if p not in generated:
                stale.append(p)
    if a.check:
        bad = [p for p, c in generated.items() if not p.exists() or p.read_text() != render(c)]
        for p in bad + stale:
            print(f"[generate_configs] out of date: {p.relative_to(ROOT)}")
        print(
            f"[generate_configs] {len(generated)} configs checked, {len(bad)} differ, {len(stale)} stale"
        )
        return 1 if (bad or stale) else 0
    for p in stale:
        p.unlink()
    for p, c in generated.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render(c))
    print(f"[generate_configs] wrote {len(generated)} configs, removed {len(stale)} stale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
