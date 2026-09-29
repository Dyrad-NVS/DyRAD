#!/usr/bin/env python
"""Sensor-configuration transfer (paper Sec. 4.3, Appendix E, Table 16): coarse -> native RADIal.

A model fitted on the coarse configuration (128 of 512 ADC samples, 128 of 256 chirps) is
scored against the native configuration's real measurements on the held-out frames. The
representation stores world-frame reflectors in metres and velocities in m/s, and the render
grid and PSF are config-declared (`psf_learned: false`, physical kernels in bins), so the
model can be rendered on another grid without retraining. Two routes, selected by `--route`:

  render    (default) direct rendering: the checkpoint rendered on the target grid with the
            target configuration's PSF, by swapping the live Runner's grid after the
            checkpoint is loaded.
  upsample  linear upsampling: the same model's render on its OWN coarse grid
            (`score_checkpoint --split val --keep`) interpolated onto the target grid:
            linear along range on the metre axis and circular linear along Doppler over the
            shared wrap period. Nothing is fitted; the target cubes are read only as GT.

Level conventions (Table 16). The preprocessing applies unnormalized FFTs, so a point target's
amplitude scales with the number of ADC samples and of chirps: the analytic amplitude gain of
native over coarse is (512/128) * (256/128) = 8.
  * render:   amplitude gain 8, applied as `sensor_power_scale = 8^2 = 64`; the render
              pedestal is the TARGET sequence's `norm.json:floor_u * hi`.
  * upsample: amplitude gain sqrt(8) = 2.83; the pedestal is swapped: the source render's
              own `floor_u * hi` is subtracted before interpolation and the target's
              `floor_u * hi` is added back after the gain, so both routes sit on the same
              pedestal.
The two gains differ on purpose. The render route evaluates the forward model on the target
grid, where the unnormalized FFTs make a point target 8x stronger in amplitude, so it takes the
analytic gain. Linear
interpolation has rows that sum to 1 and already keeps the per-bin level, so the analytic x8
would count the gain twice. sqrt(8) is the level difference the two configurations' real cubes
show, and it is the upsampling control's best case: with it the fitted scale alpha lands near
the direct render's, while x8 puts it several times lower. Both are recorded in
`<out>/sensor_transfer.json` (`gain`, `pedestal`).

Both routes write `<out>/renders_npy/` in the trainer's layout (pred_/gt_/pred_rad_/gt_rad_,
raw amplitude, declared as Domain.RAW_POWER in _domain.json) for the target configuration's
held-out frames, and score them with
`score_renders.score_run_dir(<out>, <target config>, split="val")`: the target sequence's own
norm.json, axes, labels and split. Run from the repository root (the configs' data paths are
relative to it).

    # direct rendering
    python -m dyrad.evaluation.sensor_transfer \\
        --config configs/radial/31_22_coarse_dyrad.yaml \\
        --target-config configs/radial/31_22_dyrad.yaml \\
        [--ckpt ...] [--out DIR] [--no-score]

    # linear upsampling of the coarse model's own val-split renders
    python -m dyrad.evaluation.score_checkpoint \\
        --config configs/radial/31_22_coarse_dyrad.yaml --split val --keep
    python -m dyrad.evaluation.sensor_transfer --route upsample \\
        --config configs/radial/31_22_coarse_dyrad.yaml \\
        --target-config configs/radial/31_22_dyrad.yaml \\
        [--src-renders DIR] [--out DIR] [--no-score]

`--config` is the coarse training config (its checkpoint, or its renders_npy for upsample);
`--target-config` is a native training config (grid, norm.json, labels, split).
Default outputs: results/sensor_transfer/<seq>/coarse_to_native           (render)
                 results/sensor_transfer/<seq>/coarse_to_native_upsample  (upsample)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from dyrad import norm as normmod
from dyrad.axes import sensor_axes
from dyrad.config import held_out_frames, load_config
from dyrad.domains import D_PROJECTION, Domain, convert, read_domain, save_rad
from dyrad.evaluation import radar_metrics as rm
from dyrad.paths import ROOT
from dyrad.poses import load_poses
from dyrad.preprocessing.radial.preprocess import SENSOR_PRESETS

@dataclass
class Run:
    cfg: object
    runner: object
    poses: np.ndarray  # [N,4,4] poses_can (the trainer's poses)
    norm: object  # norm.NormParams of the training sequence
    seq_dir: Path
    test_every: int
    test_offset: int
    ckpt: Path


def sensor_of(seq_dir: Path) -> dict:
    """The sequence dir's declared sensor configuration (`sensor.json`); a dir without one is
    the native configuration."""
    p = Path(seq_dir) / "sensor.json"
    if p.exists():
        return json.loads(p.read_text())
    return SENSOR_PRESETS["native"].to_dict()


def check_same_split(src_cfg, dst_cfg, tag: str) -> None:
    """The source and target windows and holdout must agree: the model only knows its window."""
    for keys, what in (
        (("frame_start", "frame_end"), "window"),
        (("test_every", "test_offset"), "split"),
    ):
        a = tuple(int(getattr(src_cfg, k)) for k in keys)
        b = tuple(int(getattr(dst_cfg, k)) for k in keys)
        if a != b:
            raise SystemExit(f"[{tag}] {what} mismatch: source {a} vs target {b} ({', '.join(keys)})")


def save_frame(npy_dir: Path, f: int, pred: np.ndarray, gt: np.ndarray, note: str) -> None:
    """Write frame `f`'s RAD cubes and their RA projections, declared RAW_POWER."""
    for nm, arr in (
        (f"pred_{f:05d}.npy", rm.ra_project(pred)),
        (f"gt_{f:05d}.npy", rm.ra_project(gt)),
        (f"pred_rad_{f:05d}.npy", pred),
        (f"gt_rad_{f:05d}.npy", gt),
    ):
        save_rad(
            npy_dir / nm,
            arr.astype(np.float32),
            Domain.RAW_POWER,
            note=note,
            d_projection=D_PROJECTION,
        )


#: Config fields the run's provenance.json must agree with: the split, the init cloud and the
#: frame clock decide which Runner the checkpoint belongs to.
PROVENANCE_KEYS = ("test_every", "test_offset", "init_cloud_path", "use_frame_timestamps")


def load_run(config: str, ckpt: Optional[str], scratch: Path) -> Run:
    """Runner on the TRAINING config, checkpoint loaded, ready for `render_all`.

    The config must agree with the run's provenance.json on PROVENANCE_KEYS."""
    import torch

    from dyrad.trainer.runner import Runner

    cfg = load_config(config)
    res_dir = ROOT / cfg.result_dir
    ckpt_p = Path(ckpt) if ckpt else res_dir / "ckpt_final.pt"
    if not ckpt_p.exists():
        raise SystemExit(f"[src] no checkpoint at {ckpt_p}")
    prov = res_dir / "provenance.json"
    if not prov.is_file():
        raise SystemExit(f"[src] no {prov}; the run's split and init cloud cannot be checked")
    pc = json.loads(prov.read_text())["config"]
    bad = [
        f"{k}: config {getattr(cfg, k)!r} vs provenance {pc.get(k, '<absent>')!r}"
        for k in PROVENANCE_KEYS
        if k not in pc or getattr(cfg, k) != pc[k]
    ]
    if bad:
        raise SystemExit(
            f"[src] {config} does not describe the run in {res_dir}:\n  " + "\n  ".join(bad)
        )
    cfg.save_renders_npy = False
    cfg.result_dir = str(scratch / "_runner_src")
    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runner = Runner(cfg)
    runner.load_checkpoint(str(ckpt_p))
    runner.tracks.eval()
    ck = torch.load(str(ckpt_p), map_location="cpu", weights_only=True)
    n_dyn_ck = int((ck["obj_idx"] >= 1).sum())
    n_dyn_now = int((runner.obj_idx >= 1).sum())
    if n_dyn_ck != n_dyn_now or runner.params["means"].shape[0] != ck["params"]["means"].shape[0]:
        raise SystemExit(
            f"[src] init/ckpt mismatch: dynamic {n_dyn_now} vs {n_dyn_ck}, "
            f"N {runner.params['means'].shape[0]} vs {ck['params']['means'].shape[0]}"
        )
    seq_dir = (ROOT / cfg.rad_tensors_dir).parent
    poses = load_poses(str(ROOT / cfg.radar_poses_dir))[0].astype(np.float64)
    print(
        f"[src] Runner ready in {time.time() - t0:.0f}s: N={ck['params']['means'].shape[0]} "
        f"(dynamic {n_dyn_ck}), split every {cfg.test_every}/offset {cfg.test_offset}, "
        f"seq {seq_dir.name}, hi={normmod.resolve(seq_dir).hi:.4g}"
    )
    return Run(
        cfg,
        runner,
        poses,
        normmod.resolve(seq_dir),
        seq_dir,
        int(cfg.test_every),
        int(cfg.test_offset),
        ckpt_p,
    )


@dataclass
class Grid:
    """A target render grid and its PSF (see grid_from_cfg)."""

    num_range_bins: int
    range_crop_first: int
    range_crop_last: int
    num_azimuth_bins: int
    num_doppler_bins: int
    doppler_min: float  # Doppler bin centres (m/s)
    doppler_max: float
    physical: bool  # physical kernels (see physical_ok)
    wR: float  # sinc-Hann widths in bins, used when not physical
    kR: int
    wA: float
    kA: int
    wD: float
    kD: int
    tag: str

    def describe(self):
        psf = (
            "Hamming-FFT range/Doppler + measured beamformer"
            if self.physical
            else f"sinc-Hann wR={self.wR}/kR={self.kR} wA={self.wA}/kA={self.kA} "
            f"wD={self.wD}/kD={self.kD} (bins)"
        )
        return (
            f"{self.num_doppler_bins}x{self.num_range_bins}x{self.num_azimuth_bins}  PSF: {psf}"
        )


#: Doppler bin counts the physical kernels are used for: 16 = native, 8 = the coarse configuration
#: (halved CPI). The Doppler PSF is the Hamming-FFT response in bin units, which does not
#: depend on the transform length, so a proportional CPI change needs no kernel change.
PHYSICAL_PSF_D = (16, 8)


def physical_ok(num_doppler_bins: int, num_azimuth_bins: int) -> bool:
    """Whether the physical kernels (RasterizeRadarRAE.cu: Hamming-FFT range and Doppler
    responses, measured beamformer) apply to a grid. Range and Doppler PSFs are in bins, so any
    FFT length works (native 512 or the coarse re-processing) and a proportional CPI change is fine
    (see PHYSICAL_PSF_D). Azimuth must stay 751: the measured beamformer response table is
    [751, 751] and a sub-aperture response is not implemented. Other grids use the analytic
    sinc-Hann with the widths held in bins."""
    return num_doppler_bins in PHYSICAL_PSF_D and num_azimuth_bins == 751


def grid_from_cfg(cfg: object, tag: str) -> Grid:
    """The render grid a training config declares (used for the target configuration)."""
    D, A = int(cfg.num_doppler_bins), int(cfg.num_azimuth_bins)
    return Grid(
        int(cfg.num_range_bins),
        int(cfg.range_crop_first),
        int(cfg.range_crop_last),
        A,
        D,
        float(cfg.radar_doppler_min_mps),
        float(cfg.radar_doppler_max_mps),
        physical=physical_ok(D, A) and bool(cfg.use_physical_dr_psf),
        wR=float(cfg.psf_w_R),
        kR=int(cfg.psf_k_R),
        wA=float(cfg.psf_w_A),
        kA=int(cfg.psf_k_A),
        wD=float(cfg.psf_w_D),
        kD=int(cfg.psf_k_D),
        tag=tag,
    )


def retarget_runner(run: Run, g: Grid) -> None:
    """Swap the live Runner's render grid (+ PSF) after the checkpoint is loaded.

    Everything set here is what `render_all` reads at call time. The Runner is built on the
    training config (the dataset and norm.json need GT of the training shape), the checkpoint is
    loaded, and only then is the grid swapped: a later `load_checkpoint` would restore the
    training-bin PSF widths. The PSF module is rebuilt because its max_k clamp is fixed at
    construction. Render times come from `Runner.t_of_frame` (render_frame).
    """
    import torch

    from dyrad.psf import SensorPSF
    from dyrad.trainer.initialization import make_render_bins

    r, cfg = run.runner, run.runner.cfg
    # 1. the fixed-PSF CUDA render path is the one retargeted here
    if not (cfg.psf_in_cuda and cfg.render_mode == "psf" and not cfg.psf_learned):
        raise SystemExit(
            "[retarget] needs the fixed-PSF CUDA render path "
            "(psf_in_cuda, render_mode: psf, psf_learned: false)"
        )
    # 2. cfg fields read by make_render_bins and by the render path
    cfg.num_range_bins, cfg.range_crop_first, cfg.range_crop_last = (
        g.num_range_bins,
        g.range_crop_first,
        g.range_crop_last,
    )
    cfg.num_azimuth_bins, cfg.num_doppler_bins = g.num_azimuth_bins, g.num_doppler_bins
    cfg.radar_doppler_min_mps, cfg.radar_doppler_max_mps = g.doppler_min, g.doppler_max
    cfg.use_physical_dr_psf, cfg.psf_az_sidelobes = bool(g.physical), bool(g.physical)
    cfg.psf_w_R, cfg.psf_k_R, cfg.psf_w_A, cfg.psf_k_A, cfg.psf_w_D, cfg.psf_k_D = (
        g.wR,
        g.kR,
        g.wA,
        g.kA,
        g.wD,
        g.kD,
    )
    # widen the kernel-support clamps to the target widths (read by the sinc-Hann kernels; the
    # physical range/Doppler kernels have a fixed support)
    for nm, k in (("psf_max_k_R", g.kR), ("psf_max_k_A", g.kA), ("psf_max_k_D", g.kD)):
        m = max(int(getattr(cfg, nm)), 2 * int(k) + 1)
        setattr(cfg, nm, m if m % 2 == 1 else m + 1)
    # 3. bins
    rb, ab, eb, db = make_render_bins(cfg)
    dev = r.device
    r.range_bins = torch.from_numpy(rb).float().to(dev)
    r.az_bins = torch.from_numpy(ab).float().to(dev)
    r.el_bins = torch.from_numpy(eb).float().to(dev)
    r.doppler_bins = torch.from_numpy(db).float().to(dev)
    # 4. the PSF module is rebuilt (psf_params clamps k_eff to max_k//2 at construction)
    r.psf = SensorPSF(
        max_kD=cfg.psf_max_k_D,
        max_kR=cfg.psf_max_k_R,
        max_kA=cfg.psf_max_k_A,
        init_k_D=cfg.psf_k_D,
        init_k_R=cfg.psf_k_R,
        init_k_A=cfg.psf_k_A,
        init_w_D=cfg.psf_w_D,
        init_w_R=cfg.psf_w_R,
        init_w_A=cfg.psf_w_A,
        init_strength=cfg.psf_strength_init,
        learnable=False,
    ).to(dev)
    r.psf_optimizer = None
    # 5. the azimuth-response cache is grid-shaped
    r._az_response_cache = None
    print(
        f"[retarget] {g.tag or 'grid'}: {g.describe()} | "
        f"R_crop={len(rb)} A={len(ab)} D={len(db)} dv={db[1] - db[0]:.4f} "
        f"psf={tuple(round(float(x), 3) for x in r.psf.psf_params())}"
    )


def analytic_amplitude_gain(sensor_src: dict, sensor_dst: dict) -> float:
    """Spec-only amplitude gain of configuration `dst` relative to `src`.

    preprocessing/radial/preprocess.py applies unnormalized FFTs (`np.fft.fft(frame * window, n=n)`) with
    the same-shape Hamming-type window at every length, so a point target's range-bin
    amplitude scales with the number of ADC samples (sum of the window ~ 0.54 n) and, through
    the Doppler FFT, with the number of chirps. The cube is amplitude
    (`np.abs(calib_mat @ mimo_win.T)`, `measurement_domain: amplitude`), so this ratio
    applies directly: (512/128) * (256/128) = 8 for coarse -> native. The noise floor does
    not follow this gain; each route sets its pedestal from the sequences' norm.json.
    """
    return (float(sensor_dst["n_samples"]) / float(sensor_src["n_samples"])) * (
        float(sensor_dst["n_chirps"]) / float(sensor_src["n_chirps"])
    )


def render_frame(run: Run, fi: int) -> np.ndarray:
    """RAW-amplitude RAD cube [D,R_crop,A] of frame `fi` on the runner's CURRENT grid.

    `pred_lin` is in the training sequence's ceiling units; x hi gives raw amplitude in the
    training configuration's scale (before any cross-configuration gain, which the caller
    applies through `cfg.sensor_power_scale`)."""
    import torch

    r, cfg, dev = run.runner, run.runner.cfg, run.runner.device
    c2w = torch.from_numpy(run.poses[fi].astype(np.float32)).unsqueeze(0).to(dev)
    with torch.no_grad():
        _, meta = r.render_all(c2w, sh_degree=int(cfg.sh_degree), t_frame=r.t_of_frame(fi))
    pred_lin = meta["pred_lin"][0].detach().float().cpu().numpy()  # ceiling units
    return pred_lin.astype(np.float64) * run.norm.hi  # RAW amplitude


def gt_cube_of(seq_dir: Path, fi: int, num_range_bins: int, crop: tuple, roll: int) -> np.ndarray:
    """A REAL cube of `seq_dir`, range-cropped and Doppler-rolled to the renderer's axis
    (the dataset loader's convention), raw amplitude."""
    raw = np.load(Path(seq_dir) / "rad_tensors" / f"rad_{fi:05d}.npy").astype(np.float64)
    R = raw.shape[1]
    if R != num_range_bins:
        raise SystemExit(f"{seq_dir.name} frame {fi}: R={R} != declared {num_range_bins}")
    r0, r1 = crop[0], R - crop[1]
    return np.roll(raw[:, r0:r1, :], int(roll), axis=0)


# ---------------------------------------------------------------------------
# Route 2: linear upsampling of the coarse model's own render (paper App. E, Table 16).
# ---------------------------------------------------------------------------


def check_doppler_period(cfg) -> None:
    """The config's Doppler axis (bin centres min..max, `dyrad.axes.doppler_bins_mps`, the
    post-roll axis the renders and the loader's GT share) must span its declared wrap period.
    Called for D >= 2 and a declared period > 0.

    Both RADIal configurations share the 1.7968 m/s period: the coarse configuration halves the
    CPI, which halves the bin count at the same period. It changes Doppler resolution without
    changing Doppler ambiguity, so native bin 2k lands exactly on coarse bin k.
    """
    D = int(cfg.num_doppler_bins)
    dv = (float(cfg.radar_doppler_max_mps) - float(cfg.radar_doppler_min_mps)) / (D - 1)
    per_cfg = float(cfg.doppler_wrap_period_mps)
    if abs(D * dv - per_cfg) > 1e-6:
        raise SystemExit(
            f"[upsample] Doppler axis inconsistent: D*dv={D * dv:.6f} != declared period "
            f"{per_cfg:.6f}; refusing to guess the axis"
        )


def circular_linear_matrix(v_src: np.ndarray, v_dst: np.ndarray, period: float) -> np.ndarray:
    """[D_dst, D_src] circular bilinear interpolation on the periodic Doppler axis
    (`searchsorted` cannot be used, it breaks across the wrap)."""
    n = len(v_src)
    dv = period / n
    pos = (v_dst - v_src[0]) / dv  # fractional source index
    i0 = np.floor(pos).astype(int)
    a = pos - i0
    M = np.zeros((len(v_dst), n))
    k = np.arange(len(v_dst))
    np.add.at(M, (k, i0 % n), 1.0 - a)
    np.add.at(M, (k, (i0 + 1) % n), a)
    return M


def linear_matrix(r_src: np.ndarray, r_dst: np.ndarray) -> np.ndarray:
    """[R_dst, R_src] bilinear interpolation on the metre axis (rows sum to 1; clamped at the ends)."""
    M = np.zeros((len(r_dst), len(r_src)))
    idx = np.clip(np.searchsorted(r_src, r_dst) - 1, 0, len(r_src) - 2)
    for k, (i, rt) in enumerate(zip(idx, r_dst)):
        a = (rt - r_src[i]) / (r_src[i + 1] - r_src[i])
        a = min(max(a, 0.0), 1.0)
        M[k, i], M[k, i + 1] = 1.0 - a, a
    return M


def default_out(src_seq: Path, target_seq: Path, suffix: str = "") -> Path:
    """results/sensor_transfer/<seq>/<src sensor>_to_<dst sensor><suffix>: a tree of its own, so
    the output is never nested inside another run's result dir."""
    s_src, s_dst = sensor_of(src_seq)["name"], sensor_of(target_seq)["name"]
    tag = src_seq.name.replace("seq_", "", 1)
    if s_src != "native" and tag.endswith("_" + s_src):
        tag = tag[: -len(s_src) - 1]
    return ROOT / "results/sensor_transfer" / tag / f"{s_src}_to_{s_dst}{suffix}"


def run_upsample(a) -> int:
    """The linear-upsampling route (torch-free: configs, numpy, norm.json, the saved renders)."""
    if a.ckpt is not None:
        raise SystemExit("[upsample] --ckpt is a render-route option")
    src_cfg, dst_cfg = load_config(a.config), load_config(a.target_config)
    src_seq = (ROOT / src_cfg.rad_tensors_dir).parent
    target_seq = (ROOT / dst_cfg.rad_tensors_dir).parent
    check_same_split(src_cfg, dst_cfg, "upsample")
    # azimuth is never resampled: the configurations share the azimuth grid, and an azimuth
    # change alters no grid a resampler could act on
    if int(src_cfg.num_azimuth_bins) != int(dst_cfg.num_azimuth_bins):
        raise SystemExit(
            f"[upsample] num_azimuth_bins differs ({src_cfg.num_azimuth_bins} vs "
            f"{dst_cfg.num_azimuth_bins}); azimuth resampling is not defined here"
        )
    src_npy = Path(a.src_renders) if a.src_renders else ROOT / src_cfg.result_dir / "renders_npy"
    if not src_npy.is_dir() or not any(src_npy.glob("pred_rad_[0-9]*.npy")):
        raise SystemExit(
            f"[upsample] no pred_rad cubes at {src_npy}; produce them with\n  python -m "
            f"dyrad.evaluation.score_checkpoint --config {a.config} --split val --keep\n"
            f"or pass --src-renders"
        )
    src_domain = read_domain(src_npy)
    if src_domain is None:
        raise SystemExit(
            f"[upsample] {src_npy} has no _domain.json; refusing to guess the cubes' domain"
        )
    out = Path(a.out) if a.out else default_out(src_seq, target_seq, "_upsample")
    npy_dir = out / "renders_npy"
    if npy_dir.exists() and any(npy_dir.glob("pred_0*.npy")) and not a.overwrite:
        print(f"[upsample] {npy_dir} already has renders; use --overwrite to redo")
    else:
        r_src, _, v_src = sensor_axes(src_cfg)
        r_dst, _, v_dst = sensor_axes(dst_cfg)
        src_sensor, dst_sensor = sensor_of(src_seq), sensor_of(target_seq)
        n_src, n_dst = normmod.resolve(src_seq), normmod.resolve(target_seq)
        # Table 16 conventions: sqrt of the spec gain, and the pedestal swapped src -> dst.
        gain_spec = analytic_amplitude_gain(src_sensor, dst_sensor)
        gain = float(np.sqrt(gain_spec))
        floor_src_raw = float(n_src.params["floor_u"]) * n_src.hi
        floor_dst_raw = float(n_dst.params["floor_u"]) * n_dst.hi
        print(
            f"[upsample] {src_seq.name} ({src_sensor['name']}) render -> {target_seq.name} "
            f"({dst_sensor['name']}) grid: amplitude gain {gain:.4f} = sqrt(spec {gain_spec:.4f}); "
            f"pedestal swap src {floor_src_raw:.4g} -> dst {floor_dst_raw:.4g}; "
            f"source cubes are {src_domain.value}"
        )
        te, to = int(src_cfg.test_every), int(src_cfg.test_offset)
        fs, fe = int(src_cfg.frame_start), int(src_cfg.frame_end)
        val = held_out_frames(src_cfg)
        roll = int(dst_cfg.doppler_roll_bins)
        n_dr = int(dst_cfg.num_range_bins)
        crop = (int(dst_cfg.range_crop_first), int(dst_cfg.range_crop_last))
        missing = [f for f in val if not (src_npy / f"pred_rad_{f:05d}.npy").exists()]
        if missing:
            raise SystemExit(f"[upsample] {src_npy} is missing val cubes {missing}")

        def load_pred(f):
            cube = np.load(src_npy / f"pred_rad_{f:05d}.npy").astype(np.float64)
            return convert(cube, src_domain, Domain.RAW_POWER, hi=n_src.hi)

        # The Doppler axes are the post-roll ones: the source renders are on the trainer's rolled
        # axis, and gt_cube_of rolls the target cube the same way.
        per_s = float(src_cfg.doppler_wrap_period_mps)
        per_d = float(dst_cfg.doppler_wrap_period_mps)
        dop_differs = len(v_src) != len(v_dst)
        if dop_differs and min(per_s, per_d) <= 0:
            raise SystemExit(
                "[upsample] resampling Doppler needs doppler_wrap_period_mps > 0 on both "
                f"configs (got {per_s} and {per_d})"
            )
        if dop_differs and min(len(v_src), len(v_dst)) < 2:
            raise SystemExit(
                f"[upsample] resampling Doppler needs >= 2 bins on both configs "
                f"(got {len(v_src)} and {len(v_dst)})"
            )
        for c in (src_cfg, dst_cfg):
            if int(c.num_doppler_bins) >= 2 and float(c.doppler_wrap_period_mps) > 0:
                check_doppler_period(c)
        if dop_differs and abs(per_s - per_d) > 1e-6:
            raise SystemExit(
                f"[upsample] the two grids have DIFFERENT wrap periods ({per_s} vs {per_d}); "
                f"that is an AMBIGUITY change, not a resolution change, and a resampler "
                f"cannot unfold it -- refusing"
            )
        rng_differs = len(r_src) != len(r_dst)
        if not (rng_differs or dop_differs):
            raise SystemExit(
                "[upsample] src and dst share both the range and the Doppler grid; "
                "nothing to interpolate"
            )
        M = MD = None
        if rng_differs:
            M = linear_matrix(r_src, r_dst)
            print(
                f"[upsample] linear matrix [{M.shape[0]}, {M.shape[1]}] on the metre axis "
                f"({r_src[1] - r_src[0]:.4f} -> {r_dst[1] - r_dst[0]:.4f} m/bin), row sums "
                f"{M.sum(1).min():.4f}..{M.sum(1).max():.4f}"
            )
        else:
            print(
                f"[upsample] range grid identical on both sensors ({len(r_src)} bins) -- range skipped"
            )
        if dop_differs:
            MD = circular_linear_matrix(v_src, v_dst, per_d)
            print(
                f"[upsample] circular linear matrix [{MD.shape[0]}, {MD.shape[1]}] on the Doppler axis "
                f"({per_s / len(v_src):.4f} -> {per_d / len(v_dst):.4f} m/s/bin, wrap period "
                f"{per_d:.4f} m/s), row sums {MD.sum(1).min():.4f}..{MD.sum(1).max():.4f}"
            )
        elif len(v_src) > 1:
            print(
                f"[upsample] Doppler grid identical on both sensors ({len(v_src)} bins) -- Doppler skipped"
            )

        npy_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        note = (
            f"sensor_transfer upsample: {src_seq.name} render linearly interpolated onto the "
            f"{target_seq.name} grid; RAW amplitude x sqrt-spec gain {gain:.4f}, pedestal "
            f"{floor_src_raw:.4g}->{floor_dst_raw:.4g}; gt = {target_seq.name} real cube"
        )
        for i, f in enumerate(val):
            pred = load_pred(f) - floor_src_raw  # the SIGNAL, with src's render pedestal removed
            res = np.einsum("dra,tr->dta", pred, M) if M is not None else pred
            if MD is not None:
                res = np.einsum("dra,ed->era", res, MD)
            res = res * gain
            res = res + floor_dst_raw  # ... and the TARGET's put back on
            gt = gt_cube_of(target_seq, f, n_dr, crop, roll)
            if res.shape != gt.shape:
                raise SystemExit(f"[upsample] frame {f}: pred {res.shape} vs gt {gt.shape}")
            save_frame(npy_dir, f, res, gt, note)
            if i == 0:
                print(
                    f"  frame {f}: pred med {np.median(res):.4g} max {res.max():.4g} | gt med {np.median(gt):.4g}"
                )
        prov = dict(
            written_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            route="upsample",
            source=dict(
                config=a.config,
                seq=src_seq.name,
                sensor=src_sensor,
                hi=n_src.hi,
                renders=str(src_npy),
                renders_domain=src_domain.value,
                result_dir=str(src_cfg.result_dir),
            ),
            target=dict(
                config=a.target_config, seq=target_seq.name, sensor=dst_sensor, hi=n_dst.hi
            ),
            interpolation=dict(
                range=("linear on the metre axis" if M is not None else "identity (same grid)"),
                doppler=(
                    "circular linear over the wrap period"
                    if MD is not None
                    else "identity (same grid)"
                ),
                n_range=dict(src=len(r_src), dst=len(r_dst)),
                n_doppler=dict(src=len(v_src), dst=len(v_dst), wrap_period_mps=per_d),
            ),
            gain=dict(mode="sqrt", amplitude=gain, spec=gain_spec),
            pedestal=dict(
                mode="swap",
                floor_src_raw=floor_src_raw,
                floor_dst_raw=floor_dst_raw,
                source="each configuration's own norm.json floor_u * hi -- the same pedestal "
                "the render route puts on the direct render",
            ),
            split=dict(test_every=te, test_offset=to, frame_start=fs, frame_end=fe),
            frames=val,
            seconds=time.time() - t0,
            domain="raw amplitude (Domain.RAW_POWER); score with --config <target config> --split val",
            compare_against=str(default_out(src_seq, target_seq)),
        )
        (out / "sensor_transfer.json").write_text(json.dumps(prov, indent=1, default=float))
        print(f"[upsample] {len(val)} frames -> {npy_dir} in {time.time() - t0:.0f}s")

    if a.no_score:
        return 0
    _score(out, a.target_config)
    return 0


def _score(out: Path, target_config: str) -> None:
    """score_renders on `out`'s held-out frames with the target config, from the repository root."""
    from dyrad.evaluation.score_renders import score_run_dir

    out = Path(out).resolve()
    print(f"[sensor_transfer] scoring {out} --config {target_config} --split val")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        score_run_dir(out, target_config, split="val")
    finally:
        os.chdir(cwd)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--route",
        choices=["render", "upsample"],
        default="render",
        help="render: the checkpoint rendered on the target grid (default); "
        "upsample: the coarse run's own renders linearly interpolated onto it",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="coarse training config of the checkpoint; with --route upsample, the "
        "coarse run whose <result_dir>/renders_npy is interpolated",
    )
    ap.add_argument("--ckpt", default=None, help="[render] default <result_dir>/ckpt_final.pt")
    ap.add_argument(
        "--target-config",
        required=True,
        help="a native training config of the target sequence: grid, norm.json, labels, split",
    )
    ap.add_argument(
        "--src-renders",
        default=None,
        help="[upsample] renders_npy holding the coarse run's pred_rad_<f>.npy cubes "
        "(default <result_dir>/renders_npy, from `score_checkpoint --split val --keep`)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="default results/sensor_transfer/<seq>/<src sensor>_to_<dst sensor>[_upsample]",
    )
    ap.add_argument(
        "--no-score", action="store_true", help="write the renders without scoring them"
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="redo the renders when <out>/renders_npy already has some",
    )
    a = ap.parse_args()
    if a.route == "upsample":
        return run_upsample(a)
    if a.src_renders is not None:
        raise SystemExit("[sensor_transfer] --src-renders is an upsample-route option")

    tcfg = load_config(a.target_config)
    target_seq = (ROOT / tcfg.rad_tensors_dir).parent
    src_cfg = load_config(a.config)
    src_seq = (ROOT / src_cfg.rad_tensors_dir).parent
    out = Path(a.out) if a.out else default_out(src_seq, target_seq)
    npy_dir = out / "renders_npy"
    if npy_dir.exists() and any(npy_dir.glob("pred_0*.npy")) and not a.overwrite:
        print(f"[sensor_transfer] {npy_dir} already has renders; use --overwrite to redo")
    else:
        run = load_run(a.config, a.ckpt, out / "_scratch")
        check_same_split(run.cfg, tcfg, "sensor_transfer")
        src_sensor, dst_sensor = sensor_of(run.seq_dir), sensor_of(target_seq)
        tnorm = normmod.resolve(target_seq)
        g = grid_from_cfg(tcfg, tag=target_seq.name)
        retarget_runner(run, g)
        # pedestal: the trainer's render pedestal is the sidecar's active `floor_u` (ceiling
        # units), not `nf`. Use the target configuration's own floor_u, converted into the
        # source's ceiling units (pred_lin is in those).
        floor_dst_raw = float(tnorm.params["floor_u"]) * tnorm.hi
        floor_src_raw = float(run.norm.params["floor_u"]) * run.norm.hi
        nf_dst, nf_src = float(tnorm.params["nf"]), float(run.norm.params["nf"])
        import torch

        run.runner.noise_floor = torch.tensor(
            floor_dst_raw / run.norm.hi, device=run.runner.device, dtype=torch.float32
        )
        gain = analytic_amplitude_gain(src_sensor, dst_sensor)
        crop, roll = (g.range_crop_first, g.range_crop_last), int(tcfg.doppler_roll_bins)
        run.runner.cfg.sensor_power_scale = float(gain) ** 2
        print(
            f"[sensor_transfer] {run.seq_dir.name} ({src_sensor['name']}) -> {target_seq.name} "
            f"({dst_sensor['name']}): analytic amplitude gain {gain:.4f}; pedestal floor "
            f"src {floor_src_raw:.4g} dst {floor_dst_raw:.4g}"
        )

        frames = held_out_frames(run.cfg)  # scored on the held-out split only
        npy_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        note = (
            f"sensor_transfer: {run.seq_dir.name} ckpt rendered as {target_seq.name}; "
            f"gt = {target_seq.name} real cube; RAW amplitude"
        )
        for i, fi in enumerate(frames):
            pred = render_frame(run, fi).astype(np.float32)  # [D,R,A] raw amp
            gt = gt_cube_of(target_seq, fi, g.num_range_bins, crop, roll).astype(np.float32)
            if pred.shape != gt.shape:
                raise SystemExit(
                    f"[sensor_transfer] frame {fi}: pred {pred.shape} vs gt {gt.shape}"
                )
            save_frame(npy_dir, fi, pred, gt, note)
            if i % 10 == 0:
                print(
                    f"  frame {fi} ({i + 1}/{len(frames)}) pred med {np.median(pred):.4g} "
                    f"max {pred.max():.4g} | gt med {np.median(gt):.4g} max {gt.max():.4g}"
                )
        prov = dict(
            written_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            route="render",
            source=dict(
                config=a.config,
                ckpt=str(run.ckpt),
                seq=run.seq_dir.name,
                sensor=src_sensor,
                hi=run.norm.hi,
                nf=nf_src,
                result_dir=str(src_cfg.result_dir),
            ),
            target=dict(
                config=a.target_config,
                seq=target_seq.name,
                sensor=dst_sensor,
                hi=tnorm.hi,
                nf=nf_dst,
                grid=g.__dict__,
            ),
            gain=dict(mode="analytic", amplitude=gain, sensor_power_scale=float(gain) ** 2),
            pedestal=dict(
                source="target norm.json floor_u * hi (the trainer's render pedestal)",
                floor_dst_raw=floor_dst_raw,
                floor_src_raw=floor_src_raw,
                nf_dst=nf_dst,
                nf_src=nf_src,
                in_source_ceiling_units=floor_dst_raw / run.norm.hi,
            ),
            split=dict(
                test_every=run.test_every,
                test_offset=run.test_offset,
                frame_start=int(run.cfg.frame_start),
                frame_end=int(run.cfg.frame_end),
            ),
            frames=frames,
            seconds=time.time() - t0,
            domain="raw amplitude (Domain.RAW_POWER); score with --config <target config> --split val",
        )
        (out / "sensor_transfer.json").write_text(json.dumps(prov, indent=1, default=float))
        print(f"[sensor_transfer] {len(frames)} frames -> {npy_dir} in {time.time() - t0:.0f}s")
        del run
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(out / "_scratch", ignore_errors=True)  # the throw-away result_dir of the Runner

    if a.no_score:
        return 0
    _score(out, a.target_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
