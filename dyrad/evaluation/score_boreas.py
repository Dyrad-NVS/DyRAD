"""Score Boreas range-azimuth reconstruction in the sensor's log-count domain (Tables 1 and 5).

Navtech counts are log-compressed (u = 20*log10(power), 20 counts per decade), so the sensor's
measurement domain is N = u/255 on the sequence's counts-mode norm.json, and every column is
scored in N. The point-cloud family is not reported on Boreas: Navtech has no Doppler axis.

Two tables, on the run config's held-out frames and polar grid
(`configs/boreas/<seq>_<variant>.yaml`):

  full frame (default)   rho_N, SSIM_N, PSNR_N, LPIPS_N over the whole RA map
                         (Tables 1 and 5, Boreas Full).
  --dynamic              rho_N, PSNR_N, SSIM_N restricted to the labelled moving-vehicle
                         windows (+-WIN_R = 20 range bins = +-1.19 m, +-WIN_A = 3 azimuth
                         bins = +-2.7 deg around each label centroid, all objects of a frame;
                         Tables 1 and 5, Boreas Object).

The off-path Boreas object columns (Tables 2 and 7) come from `score_renders` (`ra_*_N_obj`),
not from this module, and use a +-2.5 m / +-2.5 deg box around each label instead of the
window above.

DyRAD's renders come from `<result_dir>/renders_npy` of the run config, the layout
`score_checkpoint --split val --keep` leaves (LOG_CEILING or CEILING_UNITS, as declared by
`_domain.json`; any other declaration is an error). The RadarSplat and RadarFields rows are
scored only when a directory is supplied, and a supplied directory must hold every held-out
frame; those renders come from the baselines' own repositories, which are not part of this
release:
  --radarfields DIR   its `val_frames_polar_raw/`: rendered_ra_polar_frame_<f>.npy,
                      (400, 764) in N, az-descending, cropped at min_range_bin=76
                      (so it is transposed, reversed and placed at range offset 75).
  --radarsplat DIR    its `<run>/<step>/val/raw_npy/`: {pred,gt}_XXXX.npy in N (RS fits
                      image/255), (400, 838) az-major; scored pred-vs-its-own-gt since
                      its azimuth frame is ours rolled by RS_AZ_ROLL bins.
Each is converted to ceiling-linear before scoring so one code path sees every method.

Label -> grid: a labelled vehicle at (radar_R_m, radar_A_deg) lands at the nearest range bin of
the config's range axis, range_bin = round((R_m - range_m[0]) / DR), and at
az_bin = round(((A_deg + 180) mod 360) / DAZ), with DR = radar_far_range / num_range_bins and
DAZ = 360 / num_azimuth_bins. The azimuth rule puts bin j at
0.9 j - 180 deg, half a bin from the config's azimuth axis (-179.55 + 0.9 j), which is small
against the +-3-bin window. The +180 deg origin was chosen against the GT: on a 14-label probe,
11-12 label centroids land on a bright GT return with it, against 6-9 with origin 0. For --dynamic,
RadarSplat is rolled RS_AZ_ROLL = 200 azimuth bins (180 deg, no flip) into our azimuth frame so
the one label mask applies to all methods.

Run (Tables 1 and 5 need both):
    python -m dyrad.evaluation.score_boreas --seq win55_104 --variant dyrad \\
        [--radarsplat DIR] [--radarfields DIR]
    python -m dyrad.evaluation.score_boreas --seq win55_104 --variant dyrad --dynamic \\
        [--radarsplat DIR] [--radarfields DIR]

Writes `<result_dir>/score_boreas[_dynamic].{md,json}` (`results/boreas/<seq>/<variant>/`); the
JSON holds the numbers of the markdown table.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from dyrad import norm as normmod
from dyrad.axes import sensor_axes
from dyrad.config import held_out_frames, load_config
from dyrad.domains import Domain, convert, read_domain
from dyrad.paths import ROOT

# Full-frame table: (score_frame key, column). LPIPS is taken on the same N pair as the
# other columns (one metric domain per sensor).
FULL_KEYS = [
    ("ra_corr_N", "rho_N"),
    ("ra_ssim_N", "SSIM_N"),
    ("ra_psnr_N", "PSNR_N"),
    ("ra_lpips_N", "LPIPS_N"),
]

# Object-region table.
AZ_LABEL_ORIGIN = 180.0  # label azimuth origin (see the module docstring)
RS_AZ_ROLL = 200  # RS az = ours rolled +200 bins
WIN_R, WIN_A = 20, 3  # object window: +-1.19 m range, +-2.7 deg az
# RadarSplat window check: the azimuth-marginal range profiles of RS's GT and ours correlate at
# ~0.9 on the same window and far lower on a different one.
RS_WINDOW_MIN_CORR = 0.60

# Module-level state, bound by _rebind(seq, variant): paths, the config's held-out frames
# (VAL), the (range, azimuth) render grid (GRID), its bin sizes (DR m, DAZ deg) and the centre
# of its first range bin (RANGE0 m).
SEQ_ID = VARIANT = None
SEQ = LAB = RENDERS = CFG = OUT_DIR = None
VAL = GRID = DR = DAZ = RANGE0 = None
RS_DIR = RF_DIR = None  # optional baseline render dirs (--radarsplat / --radarfields)


def _rebind(seq: str, variant: str) -> None:
    """Point every module-level path at the run config `configs/boreas/<seq>_<variant>.yaml`:
    its sequence dir, label file, result dir, held-out frames and grid."""
    global SEQ_ID, VARIANT, SEQ, LAB, RENDERS, CFG, OUT_DIR, VAL, GRID, DR, DAZ, RANGE0
    SEQ_ID, VARIANT = seq, variant
    CFG = ROOT / "configs" / "boreas" / f"{seq}_{variant}.yaml"
    if not CFG.is_file():
        have = sorted(p.stem for p in (ROOT / "configs" / "boreas").glob("*.yaml"))
        raise SystemExit(f"[boreas] no run config {CFG.relative_to(ROOT)}; have {have}")
    cfg = load_config(str(CFG))
    SEQ = ROOT / cfg.seq_dir
    LAB = ROOT / cfg.object_label_path
    OUT_DIR = ROOT / cfg.result_dir
    RENDERS = OUT_DIR / "renders_npy"
    VAL = held_out_frames(cfg)
    range_m, az_deg, _ = sensor_axes(cfg, doppler=False)
    GRID = (len(range_m), len(az_deg))
    DR = float(cfg.radar_far_range) / int(cfg.num_range_bins)
    DAZ = 360.0 / int(cfg.num_azimuth_bins)  # the scan covers the full circle
    RANGE0 = float(range_m[0])  # centre of the grid's first range bin (m)


def _norm():
    """(floor_u, counts_norm_lo, counts_norm_range) of the sequence's norm.json."""
    p = normmod.resolve(SEQ).params
    return float(p["floor_u"]), float(p["counts_norm_lo"]), float(p["counts_norm_range"])


def _N_to_ceiling(n: np.ndarray, fu: float, lo: float, rng: float) -> np.ndarray:
    """Invert N = (log10(x/hi) - lo)/rng  ->  ceiling-linear x/hi."""
    return np.clip(10.0 ** (np.clip(n, 0.0, 1.0) * rng + lo), fu, None)


def _ceiling_to_N(x, fu, lo, rng):
    """Inverse of _N_to_ceiling: ceiling-linear -> N in [0,1]."""
    return np.clip((np.log10(np.clip(x, 1e-12, None)) - lo) / rng, 0.0, 1.0)


def _to_grid(a: np.ndarray, offset: int = 0) -> np.ndarray:
    """-> GRID (range, az), placing a short render at its true range offset.

    DyRAD's is already on GRID (839, 400). RS is (400, 838) az-major -- it crops the last range
    bin, so offset 0 is right (the missing bin is 49.94-50.00 m, outside any
    labelled object). RF is (400, 764) because upstream crops the near range at
    min_range_bin=76, so its first row is range bin 75, i.e. offset=75.

    The offset must be passed per source: padding a short render at the far edge
    would shift it toward the sensor and score a registration error as quality
    (and, in the object-region table, misregister the label windows).
    """
    R, A = GRID
    if a.shape[0] == A and a.shape[1] != A:  # az-major (RF, RS) -> range-major
        a = a.T
    if a.shape[1] != A:
        raise ValueError(f"cannot map {a.shape} onto (range, {A})")
    r = a.shape[0]
    if r == R and offset == 0:
        return a
    if offset + r > R:
        raise ValueError(f"render of {r} rows at offset {offset} overflows {R}")
    out = np.zeros((R, A), dtype=a.dtype)
    out[offset : offset + r] = a
    return out


# ── loaders (ceiling-linear on DyRAD's GRID and azimuth frame) ───────────────
def load_dyrad():
    """DyRAD's renders in ceiling-linear power. The trainer persists Ŷ/Y_t in the sensor's
    measurement domain (Boreas: LOG_CEILING, declared in renders_npy/_domain.json);
    this scorer is defined on linear power, so LOG_CEILING is converted on load
    (dyrad.domains.convert) and CEILING_UNITS is read as is. Any other or no declaration is
    an error. Every held-out frame must be present."""
    dom = read_domain(RENDERS)
    if dom is Domain.LOG_CEILING:
        print("[boreas] renders declared log_ceiling -> converted to ceiling units on load")

        def to_lin(a):
            return convert(a, Domain.LOG_CEILING, Domain.CEILING_UNITS)
    elif dom is Domain.CEILING_UNITS:

        def to_lin(a):
            return a
    else:
        raise SystemExit(
            f"[boreas] {RENDERS} declares domain {dom.value if dom else None!r}; this scorer "
            f"reads {Domain.LOG_CEILING.value!r} or {Domain.CEILING_UNITS.value!r} renders"
        )
    missing = [
        f
        for f in VAL
        if not (RENDERS / f"pred_{f:05d}.npy").is_file()
        or not (RENDERS / f"gt_{f:05d}.npy").is_file()
    ]
    if missing:
        raise SystemExit(
            f"[boreas] {RENDERS} is missing held-out frames {missing}; "
            "run score_checkpoint --split val --keep first"
        )
    out = {}
    for f in VAL:
        p, g = RENDERS / f"pred_{f:05d}.npy", RENDERS / f"gt_{f:05d}.npy"
        out[f] = (_to_grid(to_lin(np.load(p))), _to_grid(to_lin(np.load(g))))
    return out


def load_rf(gt_by_frame, fu, lo, rng):
    """RF renders only rows [min_range_bin-1 : 839] because upstream crops the near
    range (its P_r = alpha*log10(sigma/d^2) blows up at bin 1). Place them at that
    offset, inferred from the render's own height so a re-run at a different
    min_range_bin cannot silently misregister. Every held-out frame must be present.

    Azimuth: RF's Boreas scenes are written az-descending so its clockwise spoke
    grid lands on our counter-clockwise one, so the render comes back descending and
    must be reversed here (omitting it mirrors the scene front-to-back).
    """
    out = {}
    R = GRID[0]
    if RF_DIR is None:
        return out
    missing = [f for f in VAL if not (RF_DIR / f"rendered_ra_polar_frame_{f}.npy").is_file()]
    if missing:
        raise SystemExit(f"[boreas] RadarFields dir {RF_DIR} is missing held-out frames {missing}")
    for f in VAL:
        a = np.load(RF_DIR / f"rendered_ra_polar_frame_{f}.npy")
        rows = a.shape[1] if a.shape[0] == GRID[1] else a.shape[0]
        off = R - rows  # 839 - 764 = 75  (== min_range_bin - 1)
        g = _to_grid(a, offset=off)[:, ::-1]  # descending -> ascending azimuth
        out[f] = (_N_to_ceiling(g, fu, lo, rng), gt_by_frame[f])
    return out


def load_rs(fu, lo, rng, az_roll: int = 0):
    """RS dumps <run>/<step>/val/raw_npy/{pred,gt}_XXXX.npy; RS_DIR is that raw_npy dir.

    RS renders only its val frames, in order, so the dir must hold exactly one pred/gt pair per
    held-out frame. Score against RadarSplat's own gt (same dir), not
    ours: RadarSplat's azimuth frame is ours rolled by RS_AZ_ROLL bins, so cross-scoring would
    measure the azimuth mismatch rather than quality. The GT content is the same
    (az-marginal range-profile corr ~0.9). `az_roll` (RS_AZ_ROLL for the object-region
    table) rolls both into ours' az frame so the label mask applies.
    """
    if RS_DIR is None:
        return {}
    d = RS_DIR
    print(f"  [rs] using {d}")
    preds = sorted(glob.glob(str(d / "pred_*.npy")))
    no_gt = [pf for pf in preds if not os.path.isfile(pf.replace("pred_", "gt_"))]
    if len(preds) != len(VAL) or no_gt:
        raise SystemExit(
            f"[boreas] RadarSplat dir {d} holds {len(preds)} pred files for {len(VAL)} held-out "
            f"frames{f', {len(no_gt)} without a gt file' if no_gt else ''}"
        )
    out = {}
    for i, pf in enumerate(preds):
        gf = pf.replace("pred_", "gt_")
        pr = _N_to_ceiling(_to_grid(np.load(pf)), fu, lo, rng)
        gt = _N_to_ceiling(_to_grid(np.load(gf)), fu, lo, rng)
        if az_roll:
            pr, gt = np.roll(pr, az_roll, axis=1), np.roll(gt, az_roll, axis=1)
        out[VAL[i]] = (pr, gt)
    return out


def _rs_window_check(rs, gt_by_frame):
    """Check that the RS run belongs to this window.

    RS builds a per-window pyboreas layout re-indexed from 0, so every run dir is
    named "<rec>_frame_0_50_<ts>" whatever the window; a stale directory looks fine
    while being the wrong 50 frames.

    RadarSplat's azimuth frame differs from ours, so compare the azimuth-marginal range
    profile, which does not depend on the azimuth frame: same window -> ~0.9, a different
    window -> far lower. A run below RS_WINDOW_MIN_CORR, or one that cannot be checked, is an
    error.
    """
    if not rs:
        return rs
    cs = []
    for f, (_, g_rs) in sorted(rs.items()):
        g_ours = gt_by_frame.get(f)
        if g_ours is None:
            continue
        a = np.asarray(g_rs, float).mean(axis=1)
        b = np.asarray(g_ours, float).mean(axis=1)
        n = min(len(a), len(b))
        if a[:n].std() > 1e-12 and b[:n].std() > 1e-12:
            cs.append(np.corrcoef(a[:n], b[:n])[0, 1])
    if not cs:
        raise SystemExit(
            f"[boreas] RadarSplat window check impossible: no comparable GT frame in {RS_DIR}"
        )
    c = float(np.mean(cs))
    if c < RS_WINDOW_MIN_CORR:
        raise SystemExit(
            f"[boreas] RadarSplat REJECTED: its GT does not match this window "
            f"(az-marginal range-profile corr {c:.3f}, expect ~0.9).\n"
            f"  That run is almost certainly a DIFFERENT 50-frame window -- RS run\n"
            f"  dirs are all named frame_0_50. Re-run RS on this window and pass\n"
            f"  the right --radarsplat dir."
        )
    print(f"  [rs] window check OK (GT range-profile corr {c:.3f})")
    return rs


def _load_all(az_roll: int = 0):
    """(fu, lo, rng), ours, rs, rf."""
    fu, lo, rng = _norm()
    ours = load_dyrad()
    gt_by_frame = {f: g for f, (_, g) in ours.items()}
    rf = load_rf(gt_by_frame, fu, lo, rng)
    rs = load_rs(fu, lo, rng, az_roll=az_roll)
    rs = _rs_window_check(rs, gt_by_frame)
    return (fu, lo, rng), ours, rs, rf


# ── full-frame table ─────────────────────────────────────────────────────────
def main_full() -> int:
    from dyrad.evaluation.radar_metrics import (
        MetricParams,
        robust_lin_range,
        score_frame,
    )

    (fu, lo, rng), ours, rs, rf = _load_all()

    # Only the four _N keys are reported. score_frame requires a linear normalization range for
    # its linear-domain keys, which this table discards.
    lin_norm = robust_lin_range(iter([g for _, g in ours.values()]))
    # n_norm is (floor_u, lo_log, hi_log): the third entry is an upper bound
    # (lo + range), not the span, matching score_renders.
    mp = MetricParams(n_norm=(fu, lo, lo + rng))

    rows = []
    for name, frames in (("Ours", ours), ("RadarSplat", rs), ("RadarFields", rf)):
        if not frames:
            rows.append((name, 0, None))
            continue
        per = [
            score_frame(pr, gt, p=mp, lin_norm=lin_norm, with_lpips=True)
            for _, (pr, gt) in sorted(frames.items())
        ]
        agg = {k: float(np.nanmean([d[k] for d in per if k in d])) for k, _ in FULL_KEYS}
        rows.append((name, len(frames), agg))

    keys = FULL_KEYS
    hdr = "| method | n | " + " | ".join(h for _, h in keys) + " |"
    sep = "|" + "---|" * (len(keys) + 2)
    lines = [
        f"# Boreas {SEQ_ID} — full-frame RA reconstruction ({VARIANT} vs RadarSplat vs RadarFields)",
        "",
        f"Identical split ({len(VAL)} held-out frames {VAL}), identical GT normalization range",
        f"(counts-mode norm.json, N = u/255), identical polar grid ({GRID[0]} x {GRID[1]}).",
        "Every column is in N: Navtech counts are log-compressed, so N is the sensor's own",
        "domain and the one RadarSplat fits. RadarFields rows 0-74 are structural zeros",
        "(it does not render below its min_range_bin).",
        "",
        f"Ours: `{RENDERS}`",
        f"RadarSplat renders: `{RS_DIR or '(not supplied)'}`",
        f"RadarFields renders: `{RF_DIR or '(not supplied)'}`",
        "",
        hdr,
        sep,
    ]
    for name, n, agg in rows:
        if agg is None:
            lines.append(f"| {name} | 0 | " + " | ".join("—" for _ in keys) + " |")
        else:
            lines.append(
                f"| {name} | {n} | "
                + " | ".join(f"{agg.get(k, float('nan')):.4f}" for k, _ in keys)
                + " |"
            )
    md = "\n".join(lines) + "\n"
    out_md, out_json = OUT_DIR / "score_boreas.md", OUT_DIR / "score_boreas.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    print(md)
    print(f"-> {out_md}")

    # Machine-readable twin of the markdown table.
    out_json.write_text(
        json.dumps(
            {
                "seq": SEQ_ID,
                "variant": VARIANT,
                "ours_dir": str(RENDERS),
                "val_frames": VAL,
                "grid": list(GRID),
                "normalization": "counts_N",
                "range_rows": [0, GRID[0] - 1],
                # Which baseline renders produced the RadarFields / RadarSplat rows (RS run dirs
                # do not encode the window, so this is needed to check for cross-scored windows).
                "rf_dir": str(RF_DIR) if RF_DIR else None,
                "rs_dir": str(RS_DIR) if RS_DIR else None,
                "methods": {name: ({"n": n} | (agg or {})) for name, n, agg in rows},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"-> {out_json}")
    return 0


# ── object-region table ──────────────────────────────────────────────────────
def load_label_centres():
    """labels -> per-frame (range_bin, az_bin) object centres inside the range grid."""
    with open(LAB) as fh:
        rows = list(csv.DictReader(fh))
    by_f = {}
    for r in rows:
        f = int(r["index"])
        Rm, Ad = float(r["radar_R_m"]), float(r["radar_A_deg"])
        rb = int(round((Rm - RANGE0) / DR))
        ab = int(round(((Ad + AZ_LABEL_ORIGIN) % 360.0) / DAZ)) % GRID[1]
        if 0 <= rb < GRID[0]:
            by_f.setdefault(f, []).append((rb, ab))
    return by_f


def frame_mask(centres):
    R, A = GRID
    m = np.zeros((R, A), bool)
    for rb, ab in centres:
        r0, r1 = max(0, rb - WIN_R), min(R, rb + WIN_R + 1)
        cols = [(ab + j) % A for j in range(-WIN_A, WIN_A + 1)]
        m[r0:r1][:, cols] = True
    return m


def _masked_metrics(pr, gt, mask, fu, lo, rng):
    from skimage.metrics import structural_similarity as _ssim

    pN, gN = _ceiling_to_N(pr, fu, lo, rng), _ceiling_to_N(gt, fu, lo, rng)
    a, b = pN[mask], gN[mask]
    if a.size < 8 or a.std() < 1e-9 or b.std() < 1e-9:
        return dict(rho_N=np.nan, psnr_N=np.nan, ssim_N=np.nan)
    rho = float(np.corrcoef(a, b)[0, 1])
    mse = float(np.mean((a - b) ** 2))
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else 99.0
    # SSIM needs a 2-D neighbourhood, so it cannot be taken on the masked vector.
    # As in radar_metrics._psnr_ssim (region path): compute the full-image SSIM map,
    # then average it over the mask. Same N arrays as rho_N/psnr_N above.
    md = min(gN.shape)
    win = min(7, md if md % 2 == 1 else md - 1)
    if win < 3:
        ssim = float("nan")
    else:
        _, smap = _ssim(gN, pN, data_range=1.0, win_size=win, full=True)
        ssim = float(smap[mask].mean())
    return dict(rho_N=rho, psnr_N=psnr, ssim_N=ssim)


def main_dynamic() -> int:
    (fu, lo, rng), ours, rs, rf = _load_all(az_roll=RS_AZ_ROLL)
    labels = load_label_centres()

    rows = []
    for name, frames in (("Ours", ours), ("RadarSplat", rs), ("RadarFields", rf)):
        if not frames:
            rows.append((name, 0, None))
            continue
        per = []
        for f, (pr, gt) in sorted(frames.items()):
            centres = labels.get(f, [])
            if not centres:
                continue
            per.append(_masked_metrics(pr, gt, frame_mask(centres), fu, lo, rng))
        agg = {
            k: float(np.nanmean([d[k] for d in per if k in d]))
            for k in ("rho_N", "psnr_N", "ssim_N")
        }
        rows.append((name, len(per), agg))

    n_obj = sum(len(labels.get(f, [])) for f in ours)
    lines = [
        f"# Boreas {SEQ_ID} — object-region RA reconstruction ({VARIANT} vs RadarSplat vs RadarFields)",
        "",
        f"Metrics restricted to labelled moving-vehicle cells only "
        f"(+-{WIN_R} range bins = +-{WIN_R * DR:.2f} m, +-{WIN_A} az bins = +-{WIN_A * DAZ:.1f} deg "
        f"around each label centroid). {n_obj} vehicle labels across the {len(VAL)} held-out frames.",
        f"Same counts-N normalization range and {GRID} grid as the full-frame table. RS rolled",
        f"+{RS_AZ_ROLL} az bins into ours' frame and scored vs its OWN gt.",
        "",
        "| method | n_frames | rho_N (dyn) | PSNR_N (dyn) | SSIM_N (dyn) |",
        "|---|---|---|---|---|",
    ]
    for name, n, agg in rows:
        if agg is None:
            lines.append(f"| {name} | 0 | — | — | — |")
        else:
            lines.append(
                f"| {name} | {n} | {agg['rho_N']:.4f} | {agg['psnr_N']:.4f} | {agg['ssim_N']:.4f} |"
            )
    md = "\n".join(lines) + "\n"
    out_md, out_json = OUT_DIR / "score_boreas_dynamic.md", OUT_DIR / "score_boreas_dynamic.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    print(md)
    print(f"-> {out_md}")

    # Machine-readable twin of the markdown table.
    out_json.write_text(
        json.dumps(
            {
                "seq": SEQ_ID,
                "variant": VARIANT,
                "ours_dir": str(RENDERS),
                "rf_dir": str(RF_DIR) if RF_DIR else None,
                "rs_dir": str(RS_DIR) if RS_DIR else None,
                "val_frames": VAL,
                "n_vehicle_labels": n_obj,
                "normalization": "counts_N",
                "methods": {
                    name: {"n_frames": n, **{f"{k}_obj": v for k, v in (agg or {}).items()}}
                    for name, n, agg in rows
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"-> {out_json}")
    return 0


def _cli() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--seq",
        required=True,
        help="Boreas sequence id: win55_104, sparse2 or sparse4 "
        "(with --variant, names the run config configs/boreas/<seq>_<variant>.yaml)",
    )
    ap.add_argument(
        "--variant",
        default="dyrad",
        help="run variant supplying the Ours row: dyrad or dyrad_static, i.e. the "
        "<result_dir>/renders_npy of configs/boreas/<seq>_<variant>.yaml (default dyrad)",
    )
    ap.add_argument(
        "--dynamic",
        action="store_true",
        help="score the labelled moving-vehicle regions instead of the full frame",
    )
    ap.add_argument(
        "--radarsplat",
        default=None,
        metavar="DIR",
        help="optional: a RadarSplat raw_npy dir ({pred,gt}_XXXX.npy) for the RadarSplat "
        "row (renders from the RadarSplat repository, not part of this release). RS run "
        "dirs are all named frame_0_50 regardless of window, so pass the one that "
        "belongs to this sequence (it is checked against GT).",
    )
    ap.add_argument(
        "--radarfields",
        default=None,
        metavar="DIR",
        help="optional: a RadarFields val_frames_polar_raw dir for the RadarFields row "
        "(renders from the RadarFields repository, not part of this release)",
    )
    a = ap.parse_args()
    global RS_DIR, RF_DIR
    RS_DIR = Path(a.radarsplat) if a.radarsplat else None
    RF_DIR = Path(a.radarfields) if a.radarfields else None
    _rebind(a.seq, a.variant)
    print(f"[boreas] seq = {SEQ_ID}  variant = {VARIANT}  ({RENDERS})")
    for nm, d in (("RadarSplat", RS_DIR), ("RadarFields", RF_DIR)):
        if d is not None and not d.is_dir():
            raise SystemExit(f"[boreas] {nm} dir not found: {d}")
    return main_dynamic() if a.dynamic else main_full()


if __name__ == "__main__":
    raise SystemExit(_cli())
