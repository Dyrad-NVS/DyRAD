"""Checkpoints, provenance and run statistics."""

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import torch


def _git(*args: str) -> str:
    """stdout of a git command run in this package's checkout ("" without git)."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent),
        ).stdout.strip()
    except OSError:
        return ""  # no git executable: the commit stays unrecorded


def _write_provenance(cfg) -> None:
    """Write `<result_dir>/provenance.json`: git commit (and whether the working tree had
    uncommitted changes), command line, torch / CUDA / GPU versions and the full resolved
    config.

    Called once by `Runner.train()`, so only a training process writes it.
    """
    prov = {
        "written_at": time.time(),
        "written_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "argv": sys.argv,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "config": dataclasses.asdict(cfg),
    }
    p = Path(cfg.result_dir) / "provenance.json"
    p.write_text(json.dumps(prov, indent=1, default=str))
    print(f"[Provenance] wrote {p}")


def _check_same_keys(what: str, ckpt_keys, run_keys) -> None:
    """Raise when a checkpoint section and the Runner built from the config differ in keys."""
    missing, extra = sorted(set(run_keys) - set(ckpt_keys)), sorted(set(ckpt_keys) - set(run_keys))
    if missing or extra:
        raise ValueError(
            f"checkpoint {what} do not match this config's Runner: missing {missing[:5]}, "
            f"unexpected {extra[:5]}. Load a checkpoint with the config it was trained with."
        )


class CheckpointMixin:
    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint into a Runner built from the config it was trained with.

        The reflector parameters are replaced (densification changes their count). The
        track layout is rebuilt from the labels and the config at init, so its tensors
        must match the checkpoint's key for key and shape for shape; the PSF must be
        present in both or in neither.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        _check_same_keys("params", ckpt["params"], self.params)
        for k, v in ckpt["params"].items():
            self.params[k] = torch.nn.Parameter(v.to(self.device))

        _cur = self.tracks.state_dict()
        _check_same_keys("track tensors", ckpt["tracks"], _cur)
        for k, v in ckpt["tracks"].items():
            if tuple(v.shape) != tuple(_cur[k].shape):
                raise ValueError(
                    f"checkpoint track tensor {k} has shape {tuple(v.shape)}, this config's "
                    f"track {tuple(_cur[k].shape)}: the knot layout differs (labels, holdout "
                    f"split or track_ctrl_interp_every). Load it with its training config."
                )
        self.tracks.load_state_dict(ckpt["tracks"], strict=True)

        if (ckpt["psf"] is None) != (self.psf is None):
            raise ValueError(
                f"checkpoint {'has no' if ckpt['psf'] is None else 'has a'} PSF but this "
                f"config's render_mode is {self.cfg.render_mode!r}"
            )
        if self.psf is not None:
            self.psf.load_state_dict(ckpt["psf"], strict=True)
        if (ckpt["obj_idx"] is None) != (self.obj_idx is None):
            raise ValueError(
                "checkpoint and this config's Runner disagree on whether the scene has "
                "dynamic reflectors (obj_idx)"
            )
        if ckpt["obj_idx"] is not None:
            self.obj_idx = ckpt["obj_idx"].to(self.device)
        print(f"[Checkpoint] loaded {path}")

    def _write_run_stats(self) -> None:
        """Write `<result_dir>/train_meta.json`: runtime, peak GPU memory, final size.

        Written once, at the end of training (provenance.json is written before
        training and cannot carry these).
        """
        cfg = self.cfg
        peak_alloc = peak_reserved = None
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        wall = time.time() - self._t_train_start
        meta = {
            "scene": cfg.seq_name,
            "result_dir": str(cfg.result_dir),
            # Includes the final checkpoint and train-log plot; see
            # `train_loop_seconds` for the optimisation loop alone.
            "wall_time_seconds": wall,
            "train_loop_seconds": self._t_train_loop_end - self._t_train_start,
            # PyTorch counters: `allocated` is what the model needs; `reserved` is what
            # the caching allocator held (what nvidia-smi shows).
            "peak_gpu_mem_mib": peak_alloc,
            "peak_gpu_mem_reserved_mib": peak_reserved,
            "max_steps": int(cfg.max_steps),
            # Steps skipped because the loss was not finite (no update was applied).
            "nonfinite_loss_steps": int(self._n_nonfinite_steps),
            "n_primitives_final": int(self.params["means"].shape[0]),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "written_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        p = Path(cfg.result_dir) / "train_meta.json"
        p.write_text(json.dumps(meta, indent=1, default=str))
        print(
            f"[RunStats] {p}  wall={wall / 60:.1f} min  "
            f"peak={peak_alloc:.0f} MiB (reserved {peak_reserved:.0f})"
            if peak_alloc
            else f"[RunStats] wrote {p}"
        )

    def _save_checkpoint(self, tag) -> None:
        path = Path(self.cfg.result_dir) / f"ckpt_{tag}.pt"
        torch.save(
            {
                "params": {k: v.detach().cpu() for k, v in self.params.items()},
                "tracks": self.tracks.state_dict(),
                "psf": self.psf.state_dict() if self.psf else None,
                "obj_idx": self.obj_idx.cpu() if self.obj_idx is not None else None,
            },
            path,
        )
