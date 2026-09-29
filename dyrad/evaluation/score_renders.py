"""The paper's scorer: score a result directory from its saved renders_npy/ arrays.

GPU-free and checkpoint-free (no gsplat import). Every frame pair in
<run_dir>/renders_npy/ is scored with the radar_metrics library (reconstruction,
Doppler, point-cloud detection and selection metrics) against the config's
sequence: its norm.json, axes, labels, ego velocity and holdout split.

`score_run_dir(run_dir, config_path, ...)` is the entry point; score_checkpoint,
sensor_transfer and dyrad.offpath_real call it. The CLI is the synthetic off-path
scorer of Tables 9/10: each view is scored against the scene's base config, with
the view's own labels:

    python -m dyrad.evaluation.score_renders <out>/<view> --config <base config> \\
        --labels <views>/<view>/labels_CVPR.csv --label-seq <view>

Writes <run_dir>/metrics_extended.json (split `all`) or
metrics_extended_<split>.json: per-frame values, the aggregate and the metric
parameters. Run from the repository root (config data paths are relative to it).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dyrad.axes import sensor_axes
from dyrad.config import Config, is_held_out, load_config
from dyrad.poses import load_ego_velocity
from dyrad.domains import Domain, convert, read_domain
from dyrad.evaluation.pointcloud_metrics import (
    object_detection_metrics,
    radar_point_cloud,
    radargen_box_metrics,
    radargen_pointcloud_metrics,
)
from dyrad.evaluation.radar_metrics import MetricParams, aggregate, score_frame
from dyrad.evaluation.region_masks import object_mask_ra
from dyrad.labels import load_index_remap, read_detections
from dyrad.norm import marginal_lin_range
from dyrad.norm import resolve as resolve_norm
from dyrad.paths import ROOT
from dyrad.poses import load_poses
from dyrad.ra_partition import car_extent_m, load_fx

#: The paper's metric keys, in table order, with the column each one fills.
PANELS = [
    ("n_frames_scored", "frames scored"),
    ("ra_corr", "RA rho"),
    ("rd_corr", "RD rho"),
    ("rad_corr", "RAD rho"),
    ("ra_corr_obj", "RA rho_obj"),
    ("rd_corr_obj", "RD rho_obj"),
    ("rad_corr_obj", "RAD rho_obj"),
    ("ra_psnr_lin", "RA PSNR"),
    ("ra_ssim_lin", "RA SSIM"),
    ("ra_lpips_lin", "RA LPIPS"),
    ("rd_psnr_lin", "RD PSNR"),
    ("rd_ssim_lin", "RD SSIM"),
    ("rad_psnr_lin", "RAD PSNR"),
    ("rad_ssim_lin", "RAD SSIM"),
    ("rad_lpips_lin", "RAD LPIPS"),
    ("ra_psnr_lin_obj", "RA PSNR_obj"),
    ("ra_ssim_lin_obj", "RA SSIM_obj"),
    ("rd_psnr_lin_obj", "RD PSNR_obj"),
    ("rd_ssim_lin_obj", "RD SSIM_obj"),
    ("rad_psnr_lin_obj", "RAD PSNR_obj"),
    ("rad_ssim_lin_obj", "RAD SSIM_obj"),
    ("ra_corr_N", "RA rho (N, Boreas)"),
    ("ra_psnr_N", "RA PSNR (N, Boreas)"),
    ("ra_ssim_N", "RA SSIM (N, Boreas)"),
    ("ra_lpips_N", "RA LPIPS (N, Boreas)"),
    ("ra_corr_N_obj", "RA rho_obj (N, Boreas)"),
    ("ra_psnr_N_obj", "RA PSNR_obj (N, Boreas)"),
    ("ra_ssim_N_obj", "RA SSIM_obj (N, Boreas)"),
    ("lbl_dop_peak_mae_bins_dyn", "Doppler MAE [bins]"),
    ("pc_cd_loc_m", "CD [m]"),
    ("pc_cd_full", "CD-Full"),
    ("pc_iou_tau", "IoU@1m"),
    ("pc_da_precision", "Precision"),
    ("pc_da_recall", "Recall"),
    ("pc_da_f1", "F1"),
    ("pc_mmd_xy", "MMD loc"),
    ("pc_mmd_dop", "MMD Doppler"),
    ("pc_mmd_pow", "MMD power"),
    ("pc_box_hit_rate", "Hit rate"),
    ("pc_box_miss_rate", "Miss rate"),
    ("pc_box_density_sim", "Density sim."),
    ("pc_box_n_fp", "FP boxes"),
    ("peak_f1", "peak F1 (selection)"),
    ("obj_recall_pred", "vehicle recall (selection)"),
]

#: Plausible band for the scale-align alpha = <pred,gt>/<pred,pred> over signal
#: bins. A healthy model sits at O(1). A prediction/GT domain mismatch drives it
#: orders of magnitude out (e.g. a prediction ~hi times too dim collapses to 0
#: under the shared normalization range, leaving PSNR/SSIM a function of the GT alone). Wide on
#: purpose: this is a blunder detector.
ALPHA_SANE_LO, ALPHA_SANE_HI = 1e-3, 1e3

# ---------------------------------------------------------------------------
# Config inputs: labels, foreground boxes
# ---------------------------------------------------------------------------


def boxes_for_frame(labels, box_cfg: dict):
    """Per-frame detections (`labels.read_detections`) -> RadarGen foreground boxes
    (half-extents in metres).

    Uses `ra_partition.car_extent_m`, the same footprint function as the
    static/dynamic init partition. A label whose extent cannot be resolved (no
    bbox, no calibration, no label extents) falls back to the median car width, so
    the box set never depends on which method is being scored."""
    if not labels:
        return []
    fx, cfg = box_cfg["fx"], box_cfg["cfg"]
    out = []
    for i, lbl in enumerate(labels):
        hc, hr = car_extent_m(
            lbl.get("w_px"),
            float(lbl["R_m"]),
            fx,
            cfg,
            lbl.get("half_cross_m"),
            lbl.get("half_range_m"),
        )
        out.append(
            {
                "R_m": float(lbl["R_m"]),
                "A_deg": float(lbl["A_deg"]),
                "half_cross_m": hc,
                "half_range_m": hr,
                "key": i,
            }
        )
    return out


def labels_from_cfg(
    cfg: Config, labels_override: str | None, seq_name_override: str | None = None
) -> dict:
    """The config's label CSV as {local frame: [detection]} (`labels.read_detections`)
    over the config's frame window; {} when the config names none.

    Only rows whose `dataset` column == seq_name are kept. An off-path view's labels
    carry the view's name (`lat+1.75`, `yaw+5`, ...) while the config it is scored
    against says `seq_name: base`; pass `seq_name_override` (the view name) in that
    case. A named CSV that does not exist, or one with rows but none for seq_name,
    raises (a wrong label_seq would otherwise empty every foreground metric)."""
    csv_path = labels_override or cfg.object_label_path
    if not csv_path:
        return {}
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"label CSV {csv_path} does not exist")
    seq_name = seq_name_override if seq_name_override is not None else cfg.seq_name
    dets = read_detections(csv_path, seq_name, load_index_remap(cfg.seq_dir))
    if seq_name and not dets:
        with open(csv_path, newline="") as fh:
            seqs = sorted({row.get("dataset") for row in csv.DictReader(fh)})
        if seqs and seq_name not in seqs:
            raise ValueError(
                f"{csv_path}: no label row has dataset == {seq_name!r} (found {seqs}); "
                "pass the view's name as label_seq"
            )
    fs, fe = int(cfg.frame_start), int(cfg.frame_end)
    return {k: v for k, v in dets.items() if k >= fs and (fe < 0 or k < fe)}


# ---------------------------------------------------------------------------
# Normalization and render domain
# ---------------------------------------------------------------------------


def _marginal_ranges(cfg: Config, np_, scale: float):
    """{'ra','rd','ad': (lo, hi)} marginal normalization ranges in the renders' units,
    or None (see MetricParams.marginal_lin_norm). Which set applies is a property of
    the dataset, read from the config's `dataset` key:
      radial     RADIal's marginal ranges (dyrad/norm.py). They were derived on the
                 native cube range, so a sequence whose norm.json carries another
                 cube range (the coarse re-processings, `seq_*_coarse`) gets none.
      synthetic  the scene's own ranges from configs/benchmarks/synthetic_norm_ranges.json.
      boreas     none: with num_doppler_bins == 1 the RA map is the cube.
    """
    if cfg.dataset == "radial":
        is_native_radial_range = (
            abs(float(np_.lin_hi) - float(marginal_lin_range("rad")[1])) < 1.0
        )
        if not is_native_radial_range:
            print(
                f"[score_renders] {np_.source.parent.name}: cube range hi={np_.lin_hi:.6g} is not "
                f"the native RADIal range; no marginal normalization ranges"
            )
            return None
        ranges = {v: marginal_lin_range(v) for v in ("ra", "rd", "ad")}
    elif cfg.dataset == "synthetic":
        scene = Path(cfg.seq_dir).parent.name  # data/synthetic/<scene>/base
        ranges = json.loads(
            (ROOT / "configs/benchmarks/synthetic_norm_ranges.json").read_text()
        )["scenes"][scene]
    else:
        return None
    marg = {v: (ranges[v][0] * scale, ranges[v][1] * scale) for v in ("ra", "rd", "ad")}
    print(
        "[score_renders] marginal normalization ranges: "
        + "  ".join(
            f"{v}=({marg[v][0]:.4g}, {marg[v][1]:.4g})" for v in ("ra", "rd", "ad")
        )
    )
    return marg


def _prepare(
    config_path,
    run_dir: Path,
    p: MetricParams,
    labels_csv=None,
    label_seq=None,
    lin_norm=None,
    n_norm=None,
) -> dict:
    """Everything `score_run` needs besides the frames, resolved once from the config
    and the run's renders_npy/_domain.json. Mutates `p` (marginal ranges, clip
    ceiling, Doppler wrap period, N map).

    Normalization. The saved renders are in the domain their producer declares in
    renders_npy/_domain.json (dyrad/domains.py); a renders dir without one raises.
    LOG_CEILING renders (Boreas) are converted to CEILING_UNITS at load, so every
    metric sees a linear array in either CEILING_UNITS (the trainer's renders,
    raw / hi) or RAW_POWER (sensor_transfer). The ranges all come from the
    sequence's norm.json (a missing one raises) and are expressed in that domain:
      * lin_norm (lo, hi), the cube's linear range: the sidecar's (lin_lo, lin_hi),
        times 1/hi for ceiling-unit renders.
      * p.marginal_lin_norm, the RA/RD/AD ranges: per dataset (`_marginal_ranges`),
        scaled by the same factor.
      * p.n_norm (floor_u, lo_log, hi_log), the log map N: the sidecar's floor_u,
        norm_lo and norm_lo + norm_range. These are ceiling-space constants; for
        RAW_POWER renders the map is shifted by hi (floor_u * hi, lo_log +
        log10 hi, hi_log + log10 hi), which equals rescaling the data by 1/hi.
    An explicit `lin_norm` or `n_norm` replaces the sidecar's value; it is given in
    the renders' units (the RAW_POWER shift of n_norm still applies).
    """
    cfg = load_config(config_path)
    domain = read_domain(run_dir / "renders_npy")
    if domain is None:
        raise FileNotFoundError(f"{run_dir}/renders_npy has no readable _domain.json")
    np_ = resolve_norm(Path(cfg.seq_dir))
    ceiling = domain in (Domain.CEILING_UNITS, Domain.LOG_CEILING)
    scale = np_.norm_scale if ceiling else 1.0

    if lin_norm is None:
        s = np_.norm_scale
        lin_norm = (
            (np_.lin_lo * s, np_.lin_hi * s) if ceiling else (np_.lin_lo, np_.lin_hi)
        )
    lin_norm = tuple(float(x) for x in lin_norm)
    p.marginal_lin_norm = _marginal_ranges(cfg, np_, scale)
    if n_norm is None:
        prm = np_.params
        n_norm = (
            float(prm["floor_u"]),
            float(prm["norm_lo"]),
            float(prm["norm_lo"]) + float(prm["norm_range"]),
        )
    n_norm = tuple(float(x) for x in n_norm)
    if domain is Domain.RAW_POWER:
        sh = math.log10(np_.hi)
        n_norm = (n_norm[0] * np_.hi, n_norm[1] + sh, n_norm[2] + sh)
    p.n_norm = n_norm
    # Clip ceiling: the trainer's norm_clip_ceiling, so offline eval and loss agree.
    p.lin_clip_ceiling = float(cfg.norm_clip_ceiling)
    # The DDMA period from the run's own config, so the wrap-aware Doppler gates use
    # the renderer's units.
    if float(cfg.doppler_wrap_period_mps) > 0.0:
        p.doppler_wrap_mps = float(cfg.doppler_wrap_period_mps)
    print(
        f"[score_renders] {run_dir}: renders {domain.value}; norm from "
        f"{np_.source.parent.name}/norm.json: lin_norm=({lin_norm[0]:.6g}, {lin_norm[1]:.6g}) "
        f"N map=(floor_u {n_norm[0]:.4e}, lo_log {n_norm[1]:.3f}, hi_log {n_norm[2]:.3f}) "
        f"clip ceiling {p.lin_clip_ceiling}"
    )

    # float64 range (m), azimuth (deg) and the renderer's folded Doppler axis (m/s);
    # the saved gt_rad is already doppler-rolled to that axis.
    range_m, az_deg, doppler_mps = sensor_axes(cfg)
    # Sensor-frame ego velocity, one row per pose (indexed by global frame index).
    ego_vel = None
    if cfg.ego_vel_npy:
        n_poses = len(load_poses(cfg.radar_poses_dir)[0])
        ego_vel = load_ego_velocity(cfg.ego_vel_npy, n_poses).astype(np.float64)
    labels_by_frame = labels_from_cfg(cfg, labels_csv, label_seq)
    if labels_by_frame and len(doppler_mps) > 1 and ego_vel is None:
        raise ValueError(
            "labels and a Doppler axis but no ego_vel_npy: the label Doppler error needs it"
        )
    if labels_by_frame:
        n = sum(len(v) for v in labels_by_frame.values())
        print(f"[score_renders] {n} labels over {len(labels_by_frame)} frames")
    return dict(
        domain=domain,
        lin_norm=lin_norm,
        labels_by_frame=labels_by_frame,
        range_m=range_m,
        az_deg=az_deg,
        doppler_mps=doppler_mps,
        ego_vel=ego_vel,
        box_cfg={"fx": load_fx(cfg, ROOT), "cfg": cfg},
        test_every=int(cfg.test_every),
        test_offset=int(cfg.test_offset),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def cfar_input(cube: np.ndarray, lin_norm) -> np.ndarray:
    """The cube the point-cloud CFAR runs on: floored at 0, normalized by the linear
    range (lo, hi) and clipped to [0, 1], so every method is detected on one scale."""
    lo, hi = lin_norm
    return np.clip((np.clip(cube, 0, None) - lo) / max(hi - lo, 1e-30), 0.0, 1.0)


def find_renders(run_dir: Path) -> dict:
    """{frame_idx: {"pred": path, "gt": path, "pred_rad": ..., "gt_rad": ...}} from
    <run_dir>/renders_npy/; raises if the directory holds no pred/gt pair."""
    nd = run_dir / "renders_npy"
    frames: dict = {}
    pat = re.compile(r"^(pred|gt)(_rad)?_(\d+)\.npy$")
    for f in nd.iterdir():
        m = pat.match(f.name)
        if not m:
            continue
        kind = m.group(1) + ("_rad" if m.group(2) else "")
        frames.setdefault(int(m.group(3)), {})[kind] = f
    frames = {i: d for i, d in sorted(frames.items()) if "pred" in d and "gt" in d}
    if not frames:
        raise FileNotFoundError(f"no pred/gt render pairs in {nd}")
    return frames


def score_run(
    run_dir: Path,
    p: MetricParams,
    ctx: dict,
    split: str,
    with_lpips: bool,
    write_json: bool = True,
    allow_degenerate: bool = False,
) -> dict:
    """Score the `split` frames of run_dir's renders; returns the aggregate."""
    frames = find_renders(run_dir)
    if split != "all":
        if ctx["test_every"] <= 0:
            raise ValueError(
                f"--split {split} needs a holdout, but the config has test_every="
                f"{ctx['test_every']}; score with --split all"
            )
        want_val = split == "val"
        frames = {
            i: d
            for i, d in frames.items()
            if is_held_out(i, ctx["test_every"], ctx["test_offset"]) == want_val
        }
        if not frames:
            raise ValueError(f"no {split} frames in {run_dir}")
        print(
            f"[score_renders] split={split} (test_every={ctx['test_every']}, test_offset={ctx['test_offset']})"
        )

    lin_norm, labels_by_frame = ctx["lin_norm"], ctx["labels_by_frame"]
    range_m, az_deg, doppler_mps = ctx["range_m"], ctx["az_deg"], ctx["doppler_mps"]
    ego_vel = ctx["ego_vel"]
    from_log = ctx["domain"] is Domain.LOG_CEILING

    def _load_lin(path):
        a = np.load(path).astype(np.float64)
        return convert(a, Domain.LOG_CEILING, Domain.CEILING_UNITS) if from_log else a

    per_frame: dict = {}
    # per-frame point clouds, labels, boxes
    pc_pred, pc_gt, pc_labels, pc_boxes = [], [], [], []
    pc_ra_only = False  # set when the sensor has no Doppler axis
    for idx, paths in frames.items():
        pred_ra = _load_lin(paths["pred"])
        gt_ra = _load_lin(paths["gt"])
        pred_rad = _load_lin(paths["pred_rad"]) if "pred_rad" in paths else None
        gt_rad = _load_lin(paths["gt_rad"]) if "gt_rad" in paths else None
        labels = labels_by_frame.get(idx)
        masks = {
            "obj": object_mask_ra(
                labels, range_m, az_deg, p.obj_half_r_m, p.obj_half_a_deg
            )
        }
        v_ego = ego_vel[idx] if ego_vel is not None else None
        per_frame[idx] = score_frame(
            pred_ra,
            gt_ra,
            pred_rad=pred_rad,
            gt_rad=gt_rad,
            frame_labels=labels,
            range_bins_m=range_m,
            az_bins_deg=az_deg,
            p=p,
            lin_norm=lin_norm,
            doppler_bins_mps=doppler_mps,
            v_ego_xy=v_ego,
            region_masks=masks,
            with_lpips=with_lpips,
        )
        # RADIal-style per-frame radar point cloud: the same RD-CFAR on GT and pred
        # cubes, no pose, no aggregation across frames.
        #
        # Not computed on a sensor with no Doppler axis (D == 1). The family is a
        # RAD-domain construction and does not transfer:
        #   1. `radar_point_cloud` runs CFAR on the Range-Doppler plane, which does
        #      not exist; any RA substitute is a different detector.
        #   2. The Doppler attribute disappears: DA drops to loc+power, CD-Full
        #      becomes 3-D, MMD-Doppler is undefined.
        if pred_rad is None or gt_rad is None:
            continue
        if gt_rad.shape[0] <= 1:
            pc_ra_only = True
            continue
        pc_pred.append(
            radar_point_cloud(cfar_input(pred_rad, lin_norm), range_m, az_deg, doppler_mps, p)
        )
        pc_gt.append(
            radar_point_cloud(cfar_input(gt_rad, lin_norm), range_m, az_deg, doppler_mps, p)
        )
        pc_labels.append(labels)
        pc_boxes.append(boxes_for_frame(labels, ctx["box_cfg"]))
    agg = aggregate(list(per_frame.values()))
    agg["n_frames_scored"] = len(per_frame)
    _domain_sanity(agg, run_dir, allow_degenerate)
    if pc_ra_only:
        agg["pc_skipped"] = "no_doppler_axis"
        print(
            "[score_renders] point-cloud family skipped: this sensor has no Doppler "
            "axis (D=1), so the RD-CFAR detector and the Doppler-gated metrics "
            "are not defined."
        )
    if pc_pred:
        # The RadarGen protocol on the same clouds (entire area), vehicle recall,
        # and RadarGen's foreground half inside the annotated boxes.
        agg.update(radargen_pointcloud_metrics(pc_pred, pc_gt, range_m, p))
        agg.update(object_detection_metrics(pc_pred, pc_gt, pc_labels, p))
        if any(pc_boxes):
            agg.update(radargen_box_metrics(pc_pred, pc_gt, pc_boxes))
    if write_json:
        out = {
            "params": asdict(p),
            "frames": {str(k): v for k, v in per_frame.items()},
            "aggregate": agg,
        }
        # The point-cloud family is a run-level accumulation over the scored frames
        # with no per-frame entries, so it cannot be re-subset to a split afterwards;
        # compare split files only against split files.
        name = (
            "metrics_extended.json"
            if split == "all"
            else f"metrics_extended_{split}.json"
        )
        with open(run_dir / name, "w") as f:
            json.dump(out, f, indent=2, default=float)
    return agg


def domain_violations(agg: dict) -> list:
    """Domain-mismatch evidence in an aggregate, as a list of messages."""
    msgs = []
    a = agg.get("alpha")
    if a is not None and np.isfinite(a) and a != 0.0:
        if not (ALPHA_SANE_LO <= abs(a) <= ALPHA_SANE_HI):
            msgs.append(
                f"alpha={a:.4g} outside [{ALPHA_SANE_LO:g}, {ALPHA_SANE_HI:g}] "
                f"-> pred is ~{abs(a):.3g}x off GT; prediction and GT are almost "
                f"certainly in DIFFERENT DOMAINS (see dyrad/domains.py)"
            )
    # A prediction that never fires a peak while the GT has plenty is another
    # signature of a domain/scale mismatch.
    gp, pp = agg.get("n_gt_peaks"), agg.get("n_pred_peaks")
    if gp and pp is not None and gp > 100 and pp == 0:
        msgs.append(
            f"pred_peaks=0 while gt_peaks={gp:g} -> prediction has no "
            f"structure above threshold; check the domain/scale"
        )
    # Degenerate photometrics: pred and GT collapsed to the same value under the
    # normalization (typically both clipped to 0 because the normalization range is in the wrong
    # domain), so they compare as identical (PSNR=100 / SSIM=1.0). alpha does not
    # catch this: it is computed before normalization.
    for key, lim, what in (
        ("ra_psnr_lin", 99.0, "PSNR"),
        ("rd_psnr_lin", 99.0, "PSNR"),
        ("rad_psnr_lin", 99.0, "PSNR"),
        ("ra_ssim_lin", 0.999, "SSIM"),
        ("rd_ssim_lin", 0.999, "SSIM"),
        ("rad_ssim_lin", 0.999, "SSIM"),
    ):
        v = agg.get(key)
        if v is not None and np.isfinite(v) and v >= lim:
            msgs.append(
                f"{key}={v:.4g} -> degenerate {what}: pred and GT are identical "
                f"after normalization, which means the linear normalization range is in the "
                f"WRONG DOMAIN for these renders (both clipped to the same "
                f"value), not that the fit is perfect"
            )
    return msgs


def _domain_sanity(agg: dict, run_dir: Path, allow_degenerate: bool = False) -> None:
    """Raise on any domain-mismatch evidence in the aggregate (nothing is written).

    With `allow_degenerate` the evidence is printed and recorded in the aggregate as
    `sanity_warnings` instead: for a prediction that is genuinely degenerate (a collapsed fit
    with no structure above threshold) rather than in the wrong domain.
    """
    msgs = domain_violations(agg)
    if msgs and allow_degenerate:
        print(f"[score_renders] WARNING, scored anyway (--allow-degenerate) {run_dir}:")
        for m in msgs:
            print(f"  - {m}")
        agg["sanity_warnings"] = msgs
    elif msgs:
        raise ValueError(
            f"domain sanity failed for {run_dir}:\n" + "\n".join(f"  - {m}" for m in msgs)
        )


def print_summary(agg: dict) -> None:
    """Print the paper keys of one aggregate (PANELS order)."""
    for key, col in PANELS:
        v = agg.get(key)
        if v is None:
            continue
        s = f"{v:d}" if isinstance(v, int) else f"{v:.4f}"
        print(f"  {col:<28s} {key:<28s} {s}")


def score_run_dir(
    run_dir,
    config_path,
    labels_csv=None,
    split: str = "all",
    with_lpips: bool = True,
    label_seq=None,
    *,
    lin_norm=None,
    n_norm=None,
    params: MetricParams | None = None,
    write_json: bool = True,
    allow_degenerate: bool = False,
) -> dict:
    """Score one result dir (its renders_npy/) against `config_path`. Returns the
    aggregate and writes <run_dir>/metrics_extended[_<split>].json.

    labels_csv   label CSV in place of the config's `object_label_path`.
    label_seq    the `dataset` value the labels are filtered on (an off-path view's name).
    lin_norm, n_norm  explicit normalization in place of the sidecar's (see `_prepare`).
    params       a MetricParams to score with (default: the paper's).
    allow_degenerate  score a prediction that fails the domain sanity check (see
                 `_domain_sanity`) instead of raising.
    Relative paths in the config resolve against the working directory, so call
    from the repository root.
    """
    run_dir = Path(run_dir)
    p = params if params is not None else MetricParams()
    ctx = _prepare(
        config_path,
        run_dir,
        p,
        labels_csv=labels_csv,
        label_seq=label_seq,
        lin_norm=lin_norm,
        n_norm=n_norm,
    )
    agg = score_run(
        run_dir,
        p,
        ctx,
        split=split,
        with_lpips=with_lpips,
        write_json=write_json,
        allow_degenerate=allow_degenerate,
    )
    print(f"[score_renders] {run_dir} ({split}):")
    print_summary(agg)
    return agg


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dirs", nargs="+", help="result dirs with renders_npy/")
    ap.add_argument(
        "--config", required=True, help="training YAML (axes, labels, normalization)"
    )
    ap.add_argument("--labels", default=None, help="labels CSV override")
    ap.add_argument(
        "--label-seq",
        default=None,
        help="override the `dataset` value labels are filtered on (see "
        "labels_from_cfg). Needed when scoring an OFF-PATH view, whose "
        "labels carry the view name while the config says `base`.",
    )
    ap.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="score only train or val frames (test_every/test_offset of --config)",
    )
    ap.add_argument(
        "--no-lpips",
        action="store_true",
        help="skip LPIPS (faster; the paper metrics include it)",
    )
    ap.add_argument(
        "--allow-degenerate",
        action="store_true",
        help="score a prediction that fails the domain sanity check (e.g. a collapsed fit "
        "with no peaks) instead of refusing it; the failed checks are printed and stored "
        "in the aggregate as `sanity_warnings`",
    )
    args = ap.parse_args()
    for rd in args.run_dirs:
        score_run_dir(
            rd,
            args.config,
            labels_csv=args.labels,
            split=args.split,
            with_lpips=not args.no_lpips,
            label_seq=args.label_seq,
            allow_degenerate=args.allow_degenerate,
        )


if __name__ == "__main__":
    main()
