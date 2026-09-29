"""Visualize a trained DyRAD checkpoint: render frames of its sequence next to the measurement.

Uses the trainer's own frame renderer (the one that writes the `step<N>_frame<i>.png`
panels during training), so the panels here are identical in layout and normalization
to the training-time renders: GT and prediction RA / RD maps in the run's measurement
domain, with the annotated objects overlaid, plus a Cartesian bird's-eye view
(`*_cart.png`). Writes to <result_dir>/render_<ckpt stem>/ (or --out), and with
--video stitches the frames into render.mp4.

Usage (from the repository root):
    python -m dyrad.evaluation.render_checkpoint --config configs/radial/31_22_dyrad.yaml
        [--ckpt <result_dir>/ckpt_final.pt] [--out DIR] [--stride N] [--video --fps 10]

The paper's metrics are not computed here; use score_checkpoint for those.
"""

import argparse
import re
from pathlib import Path
from typing import List

import imageio
import numpy as np
from PIL import Image

from dyrad.config import load_config
from dyrad.trainer.runner import Runner


def _make_video(frame_pngs: List[str], out_path: str, fps: int = 10) -> None:
    imgs = [np.array(Image.open(p).convert("RGB")) for p in frame_pngs]
    # crop to the smallest common size (matplotlib's bbox_inches can jitter by a pixel)
    h = min(im.shape[0] for im in imgs)
    w = min(im.shape[1] for im in imgs)
    imageio.mimsave(out_path, [im[:h, :w] for im in imgs], fps=fps)
    print(f"[render] video: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="the run's training config")
    ap.add_argument(
        "--ckpt", default=None, help="checkpoint (default: <result_dir>/ckpt_final.pt)"
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output directory (default: <result_dir>/render_<ckpt stem>)",
    )
    ap.add_argument(
        "--stride", type=int, default=1, help="render every Nth frame of the window"
    )
    ap.add_argument("--video", action="store_true", help="also write render.mp4")
    ap.add_argument("--fps", type=int, default=10, help="video frame rate")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ckpt = Path(args.ckpt or Path(cfg.result_dir) / "ckpt_final.pt")
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    out_dir = (
        Path(args.out) if args.out else Path(cfg.result_dir) / f"render_{ckpt.stem}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = Runner(cfg)
    runner.load_checkpoint(str(ckpt))
    runner.tracks.eval()

    # The trainer's renderer takes its frames and output directory from the config.
    frame_end = int(cfg.frame_end)
    if frame_end < 0:  # -1: to the end of the sequence
        frame_end = runner.val_loader.dataset.parser.num_frames
    frames = list(range(int(cfg.frame_start), frame_end))[:: max(1, args.stride)]
    cfg.result_dir = str(out_dir)
    cfg.render_val_frames = False
    cfg.render_frame_idxs = frames
    m = re.search(r"(\d+)$", ckpt.stem)
    step = int(m.group(1)) if m else int(cfg.max_steps)
    print(
        f"[render] {len(frames)} frames of window {cfg.frame_start}..{frame_end - 1} -> {out_dir}"
    )
    runner._save_render(step)

    if args.video:
        pngs = sorted(
            str(p)
            for p in out_dir.glob(f"step{step:06d}_frame*.png")
            if not p.stem.endswith("_cart")
        )
        if pngs:
            _make_video(pngs, str(out_dir / "render.mp4"), args.fps)


if __name__ == "__main__":
    main()
