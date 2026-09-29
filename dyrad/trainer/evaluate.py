"""Evaluation pass over every frame of a trained scene: the persisted renders and a reduced
metric panel.

`renders_npy/` (Ŷ and Y in the measurement domain) is what score_renders scores into the
paper's metrics_extended*.json. metrics.json holds the reduced panel of the headline split
(the same `score_frame` as score_renders plus the RD-CFAR point-cloud and object metrics),
read by the off-path summaries.
"""

import json
import math
from pathlib import Path
from typing import List

import numpy as np
import torch

from dyrad.domains import D_PROJECTION, save_rad
from dyrad.evaluation.pointcloud_metrics import (
    object_detection_metrics,
    radar_point_cloud,
    radargen_pointcloud_metrics,
)
from dyrad.evaluation.radar_metrics import MetricParams, aggregate, ra_project, score_frame


def _cfar_norm(cube: np.ndarray, lin_norm: tuple) -> np.ndarray:
    """The cube on the shared linear map, clip((x - lo) / (hi - lo), 0, 1), as CFAR input."""
    lo, hi = lin_norm
    return np.clip((cube - lo) / max(hi - lo, 1e-30), 0.0, 1.0)


class EvalMixin:
    def _evaluate(self, step: int) -> None:
        """Render and score ALL frames (train + val) with split tags.

        The aggregate in metrics.json is computed over the val split only; when it is
        empty it falls back to train. The per-frame table (each row tagged
        "train"/"val") is stored under entry["frames"]. With save_renders_npy, renders
        for every evaluated frame are written to renders_npy/ for offline scoring.
        """
        cfg = self.cfg
        ext_params = MetricParams()
        # Keep eval's linear clip ceiling identical to the loss.
        ext_params.lin_clip_ceiling = float(cfg.norm_clip_ceiling)
        # The shared log map N: eval also emits the log-domain ra_psnr_N / ra_ssim_N /
        # ra_corr_N keys alongside the linear ones.
        ext_params.n_norm = (self.norm_floor_u, self.norm_lo_log, self.norm_hi_log)

        _lbf = self._labels_by_frame
        _range_np = self.range_bins.cpu().numpy()
        _az_deg_np = np.rad2deg(self.az_bins.cpu().numpy())
        _dop_bins_np = self.doppler_bins.cpu().numpy()

        headline = "val" if len(self.val_loader.dataset) > 0 else "train"

        save_npy = bool(cfg.save_renders_npy)
        npy_dir = Path(cfg.result_dir) / "renders_npy"
        if save_npy:
            npy_dir.mkdir(parents=True, exist_ok=True)

        # Global linear-normalisation range (lo, hi) for the linear PSNR/SSIM metrics
        # (see lin_range()), shared with _save_render.
        lin_norm = self.lin_range()

        all_frames: List[dict] = []
        # Headline-split radar point clouds + labels for the sequence-level RadarGen /
        # object-detection metrics (RADIal-style RD-CFAR), with the same pc_*/obj_*
        # keys and MetricParams as offline scoring.
        pc_pred: list = []
        pc_gt: list = []
        pc_lbl: list = []

        with torch.no_grad():
            for split, loader in (("train", self.train_loader), ("val", self.val_loader)):
                for data in loader:
                    c2w = data["radarpose"].to(self.device)
                    gt_linear = data["rad_tensor"].to(self.device)
                    frame_idx = int(data["frame_idx"][0].item())
                    t_frame = self._t_frame(data["frame_idx"][0])

                    _, _ev_meta = self.render_all(c2w, sh_degree=cfg.sh_degree, t_frame=t_frame)
                    # Native linear render for metrics defined on linear values (RA PSNR,
                    # CFAR, Pearson).
                    pred_lin = _ev_meta["pred_lin"]
                    # Ŷ and Y_t in the measurement domain (Config.measurement_domain): what
                    # the persisted renders hold.
                    pred_meas = _ev_meta["pred_meas"]
                    gt_meas = self._to_measurement(gt_linear)

                    fr: dict = {"frame": frame_idx, "split": split}

                    # Metric panel (ra/rd/rad PSNR/SSIM, Pearson corrs, label Doppler,
                    # peak F1), the same score_frame as score_renders.
                    pred_ra_np = pred_lin[0].clamp(min=0).max(dim=0).values.cpu().numpy()
                    gt_ra_np = gt_linear[0].max(dim=0).values.cpu().numpy()
                    fr.update(
                        score_frame(
                            pred_ra_np,
                            gt_ra_np,
                            pred_rad=pred_lin[0].clamp(min=0).cpu().numpy(),
                            gt_rad=gt_linear[0].cpu().numpy(),
                            frame_labels=_lbf.get(frame_idx),
                            range_bins_m=_range_np,
                            az_bins_deg=_az_deg_np,
                            p=ext_params,
                            doppler_bins_mps=_dop_bins_np,
                            v_ego_xy=self.ego_vel_sensor(frame_idx)[0].cpu().numpy(),
                            lin_norm=lin_norm,
                        )
                    )
                    all_frames.append(fr)

                    # Per-frame radar point cloud (RADIal-style RD-CFAR) for the
                    # sequence-level point-cloud / object-detection metrics. CFAR runs on
                    # the normalised and clipped cube (the shared train robust (lo, hi),
                    # the same map as loss/eval/viz) rather than raw linear values, so
                    # bright reflectors do not dominate the CA-CFAR background.
                    if split == headline:
                        for _cube, _pcs in (
                            (pred_lin[0].clamp(min=0).cpu().numpy(), pc_pred),
                            (gt_linear[0].cpu().numpy(), pc_gt),
                        ):
                            _pcs.append(
                                radar_point_cloud(
                                    _cfar_norm(_cube, lin_norm),
                                    _range_np,
                                    _az_deg_np,
                                    _dop_bins_np,
                                    ext_params,
                                )
                            )
                        pc_lbl.append(_lbf.get(frame_idx))

                    # Save renders for every evaluated frame (off by default; renders can be
                    # regenerated from the checkpoint).
                    if save_npy:
                        # Persist Ŷ and Y_t in the sensor's measurement domain (the tensors
                        # the loss compares), with the domain declared (CEILING_UNITS for a
                        # linear sensor, LOG_CEILING for a log one). Readers that need linear
                        # power convert explicitly (dyrad.domains).
                        gt_save_np = gt_meas[0].cpu().numpy().astype(np.float32)  # [D, R, A]
                        pred_save_np = pred_meas[0].cpu().numpy().astype(np.float32)
                        _dom = self._render_domain()
                        _note = (
                            f"trainer _evaluate: gt + pred in the measurement domain "
                            f"({self._meas_domain}), same normalization"
                        )
                        # The 2D dumps are the mean over D (radar_metrics.ra_project, RADIal's
                        # own convention up to 1/D) and the *_rad_ ones are the full cube; the
                        # projection is declared in the file metadata (D_PROJECTION). Mean, not max: a max
                        # over D depends on where in Doppler the energy sits.
                        for _n, _a in (
                            (f"pred_{frame_idx:05d}.npy", ra_project(pred_save_np)),
                            (f"gt_{frame_idx:05d}.npy", ra_project(gt_save_np)),
                            (f"pred_rad_{frame_idx:05d}.npy", pred_save_np),
                            (f"gt_rad_{frame_idx:05d}.npy", gt_save_np),
                        ):
                            save_rad(npy_dir / _n, _a, _dom, note=_note, d_projection=D_PROJECTION)

        # ── Headline aggregate: VAL split only (train fallback) ──────────────
        def _agg(key: str) -> float:
            vals = [
                f[key]
                for f in all_frames
                if f["split"] == headline
                and key in f
                and isinstance(f[key], (int, float))
                and not math.isnan(float(f[key]))
            ]
            return float(np.mean(vals)) if vals else float("nan")

        ra_psnr_lin = _agg("ra_psnr_lin")  # linear, shared norm.json normalisation
        entry = {"step": step, "headline_split": headline, "ra_psnr_lin": ra_psnr_lin}
        ext_agg = aggregate([f for f in all_frames if f["split"] == headline])
        entry.update(
            {k: v for k, v in ext_agg.items() if k not in entry and k not in ("frame", "split")}
        )
        # Sequence-level detection metrics: the RadarGen entire-area protocol on the
        # RD-CFAR clouds + vehicle recall at the labels.
        _pc = radargen_pointcloud_metrics(pc_pred, pc_gt, _range_np, ext_params)
        _od = object_detection_metrics(pc_pred, pc_gt, pc_lbl, ext_params)
        entry.update(_pc)
        entry.update(_od)
        entry["frames"] = all_frames
        self._metrics.append(entry)

        metrics_path = Path(cfg.result_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._metrics, f, indent=2)

        n_val = sum(1 for f in all_frames if f["split"] == "val")
        print(
            f"[Eval {step}] ({headline}: {n_val if headline == 'val' else len(all_frames)} frames, "
            f"{len(all_frames)} total)  "
            f"RA_corr={ext_agg.get('ra_corr', float('nan')):.3f}  "
            f"RA_PSNR_LIN={ra_psnr_lin:.2f} dB  "
            f"RA_SSIM_LIN={ext_agg.get('ra_ssim_lin', float('nan')):.4f}  "
            f"| RA_PSNR_N={ext_agg.get('ra_psnr_N', float('nan')):.2f} dB "
            f"RA_SSIM_N={ext_agg.get('ra_ssim_N', float('nan')):.4f} "
            f"RA_corr_N={ext_agg.get('ra_corr_N', float('nan')):.3f} | "
            f"N={len(self.params['means'])}"
        )
        print(
            f"[Eval {step}] peak_F1={ext_agg.get('peak_f1', float('nan')):.3f} "
            f"(P={ext_agg.get('peak_precision', float('nan')):.3f} "
            f"R={ext_agg.get('peak_recall', float('nan')):.3f})  "
            f"lblDopPkDyn={ext_agg.get('lbl_dop_peak_mae_bins_dyn', float('nan')):.2f} bins  "
            f"corr RA={ext_agg.get('ra_corr', float('nan')):.3f}"
            f"/RAD={ext_agg.get('rad_corr', float('nan')):.3f}"
            f"/RD={ext_agg.get('rd_corr', float('nan')):.3f}  "
            f"obj corr RA={ext_agg.get('ra_corr_obj', float('nan')):.3f}"
            f"/RAD={ext_agg.get('rad_corr_obj', float('nan')):.3f}"
        )
        print(
            f"[Eval {step}] PC: CD={_pc.get('pc_cd_loc_m', float('nan')):.2f} m  "
            f"CD-Full={_pc.get('pc_cd_full', float('nan')):.4f}  "
            f"IoU@1m={_pc.get('pc_iou_tau', float('nan')):.3f}  "
            f"DA-F1={_pc.get('pc_da_f1', float('nan')):.3f} "
            f"(P={_pc.get('pc_da_precision', float('nan')):.3f} "
            f"R={_pc.get('pc_da_recall', float('nan')):.3f})  | "
            f"obj_recall pred={_od.get('obj_recall_pred', float('nan')):.3f} "
            f"vs GT={_od.get('obj_recall_gt', float('nan')):.3f}  "
            f"({_od.get('n_obj_labels', 0)} labels)"
        )
