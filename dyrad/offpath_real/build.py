"""Write the view definitions of a sequence and render a variant's M0 into the shifted view."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import spec, view


def _config(seq: str):
    """The geometry (poses, window, bins, labels) is shared by every variant's M0 config."""
    from dyrad.config import load_config

    return load_config(str(spec.config_path(seq, spec.variants()[0], "M0")))


def _poses_and_frames(cfg, seq: str):
    from dyrad.data import RadarParser

    parser = RadarParser(
        rad_tensors_dir=cfg.rad_tensors_dir,
        poses_dir=cfg.radar_poses_dir,
        num_doppler_bins=cfg.num_doppler_bins,
        num_range_bins=cfg.num_range_bins,
        num_azimuth_bins=cfg.num_azimuth_bins,
        range_crop_first=cfg.range_crop_first,
        range_crop_last=cfg.range_crop_last,
        allow_missing_rad=bool(cfg.allow_missing_rad),
    )
    fs, fe = spec.window(seq)
    return parser.poses.astype(np.float64), list(range(fs, fe))


def _view_kwargs(cfg, seq: str, lateral_m: float, poses_t0, poses_view, frames, content: str):
    return dict(
        base_seq=Path(cfg.rad_tensors_dir).parent,
        frames=frames,
        poses_t0=poses_t0,
        poses_view=poses_view,
        lateral_m=lateral_m,
        content=content,
        range_crop_first=int(cfg.range_crop_first),
        range_crop_last=int(cfg.range_crop_last),
        dt=float(cfg.dt),
        ego_vel_npy=cfg.ego_vel_npy,
        label_csv=cfg.object_label_path,
        label_seq_str=str(cfg.seq_name),
        far_range_m=float(cfg.radar_far_range),
        az_fov_deg=float(cfg.radar_az_fov_deg),
        labels_filename=spec.labels_filename(),
        seq_name=spec.view_seq_name(seq, lateral_m),
    )


def build_view(seq: str, which: str) -> int:
    """Write a view definition: `T0` (the real tensors of the window, re-indexed 0..N-1)
    or `T1` (shifted poses, axes, ego velocity, re-projected labels and normalization
    range, no tensors: each variant's `render` adds its own M0 renders)."""
    cfg = _config(seq)
    poses_t0, frames = _poses_and_frames(cfg, seq)
    out = spec.definition_dir(seq, which)
    if which == "T0":
        kw = _view_kwargs(cfg, seq, 0.0, poses_t0, poses_t0, frames, "real_gt")
    else:
        poses_view = view.shift_poses(poses_t0, spec.shift_m())
        kw = _view_kwargs(cfg, seq, spec.shift_m(), poses_t0, poses_view, frames, "definition")
    view.write_view(out, **kw)
    print(f"[build-view] {seq}/{which}: window {frames[0]}..{frames[-1]} -> {out}")
    return 0


def render(seq: str, variant: str, ckpt=None) -> int:
    """Render `variant`'s M0 along the shifted trajectory into its own copy of the `T1` view."""
    import torch

    from dyrad.config import load_config
    from dyrad.trainer.runner import Runner

    src = spec.definition_dir(seq, "T1")
    if not (src / "poses_can/poses_metadata.json").exists():
        raise SystemExit(
            f"no view definition at {src}; run `build-view --seq {seq} --view T1` first"
        )
    cfg = load_config(str(spec.config_path(seq, variant, "M0")))
    ckpt = Path(ckpt) if ckpt else spec.result_dir(seq, variant, "M0") / "ckpt_final.pt"
    if not ckpt.exists():
        raise SystemExit(f"M0 checkpoint not found: {ckpt}")
    poses_t0, frames = _poses_and_frames(cfg, seq)
    poses_view = view.shift_poses(poses_t0, spec.shift_m())

    print(f"[render] {seq}/{variant}: loading M0 from {ckpt}")
    runner = Runner(cfg)
    runner.load_checkpoint(str(ckpt))
    # The model renders in the normalized (ceiling) units the loader scaled GT to;
    # multiply by the range's hi to write raw power like the real tensors.
    gt_hi = float(runner.gt_hi_power)
    if not gt_hi > 0.0:
        raise SystemExit(f"normalization hi of {cfg.seq_dir} is {gt_hi}, expected > 0")
    roll, r0 = int(cfg.doppler_roll_bins), int(cfg.range_crop_first)
    r1 = int(cfg.num_range_bins) - int(cfg.range_crop_last)

    def frame(fi: int) -> np.ndarray:
        with torch.no_grad():
            c2w = torch.from_numpy(poses_view[fi]).float().to(runner.device).unsqueeze(0)
            _, meta = runner.render_all(
                c2w,
                sh_degree=int(cfg.sh_degree),
                t_frame=runner.t_of_frame(fi),
                v_ego_sensor=runner.ego_vel_sensor(fi),
            )
        pl = meta["pred_lin"][0].detach().cpu().numpy() * gt_hi
        # Undo the renderer's conventions for on-disk tensors: DC back to bin 0 and the
        # cropped range re-embedded into a full-length tensor.
        full = np.zeros((pl.shape[0], int(cfg.num_range_bins), pl.shape[2]), dtype=np.float32)
        full[:, r0:r1, :] = np.roll(pl, -roll, axis=0)
        return full

    out = spec.rendered_view_dir(seq, variant)
    view.derive_view(
        out,
        src,
        (frame(fi) for fi in frames),
        rendered_from=str(ckpt),
        labels_filename=spec.labels_filename(),
    )
    return 0
