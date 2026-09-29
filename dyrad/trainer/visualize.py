"""Training curves and rendered frames."""

import os
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import torch

from dyrad.axes import wrap_bins_signed


def _write_train_log(log: List[dict], path: Path) -> None:
    import csv

    if not log:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        writer.writeheader()
        writer.writerows(log)


def _plot_train_log(log: List[dict], path: Path) -> None:
    steps = [r["step"] for r in log]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    def _plot(ax, keys, title):
        for k in keys:
            vals = [r.get(k, float("nan")) for r in log]
            ax.plot(steps, vals, label=k)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    _plot(axes[0, 0], ["loss", "rad_l1", "interp"], "Total loss, RAD L1, interpolation term")
    _plot(axes[0, 1], ["dyn_l1"], "Dynamic-region L1")
    _plot(axes[1, 0], ["v_mean", "dx_mean"], "Mean object velocity (m/s) / displacement (m)")
    _plot(axes[1, 1], ["gn_track"], "Track grad norm")

    plt.suptitle(f"Training curves — {path.parent.name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[TrainLog] saved {path}")


class VizMixin:
    def _render_frame_ids(self, train_global: np.ndarray, val_global: np.ndarray) -> List[int]:
        """Global frame indices to render: the val frames (render_val_frames), else the
        configured render_frame_idxs inside this run's frame window [frame_start,
        frame_end), else an even spread of in-window frames."""
        cfg = self.cfg
        _in_window = set(train_global.tolist()) | set(val_global.tolist())
        if cfg.render_val_frames and len(val_global) > 0:
            idxs = sorted(val_global.tolist())  # holdout frames (all in-window)
        else:
            _req = cfg.render_frame_idxs if cfg.render_frame_idxs else [cfg.render_frame_idx]
            idxs = [i for i in _req if i in _in_window]
            _drop = [i for i in _req if i not in _in_window]
            if _drop:
                print(
                    f"[_save_render] WARNING: {len(_drop)} render_frame_idxs are OUTSIDE the "
                    f"frame window [{min(_in_window)}, {max(_in_window)}] and were dropped: "
                    f"{_drop}  (set render_val_frames: true, or fix render_frame_idxs)"
                )
            if not idxs:
                # Fall back to an evenly-spaced spread of in-window frames (prefer val).
                _pool = sorted(val_global.tolist()) or sorted(_in_window)
                _k = min(7, len(_pool))
                idxs = (
                    [_pool[round(j * (len(_pool) - 1) / (_k - 1))] for j in range(_k)]
                    if _k > 1
                    else _pool
                )
                print(f"[_save_render] no in-window render_frame_idxs; using {idxs}")
        return idxs

    def _save_render(self, step: int) -> None:
        """Save GT vs Pred RA/RD panels (polar and Cartesian PNGs) of the
        `_render_frame_ids` frames."""
        cfg = self.cfg
        train_global = np.asarray(self.train_loader.dataset.indices)
        val_global = np.asarray(self.val_loader.dataset.indices)
        idxs = self._render_frame_ids(train_global, val_global)

        r0, r1 = cfg.range_crop_first, cfg.num_range_bins - cfg.range_crop_last
        r_min = cfg.radar_far_range * r0 / cfg.num_range_bins
        r_max = cfg.radar_far_range * r1 / cfg.num_range_bins
        az_bins_deg = np.rad2deg(self.az_bins.cpu().numpy())
        az_lim_deg = (float(az_bins_deg[0]), float(az_bins_deg[-1]))
        dop_lim = (float(self.doppler_bins[0]), float(self.doppler_bins[-1]))
        # RADIal convention: range = rows (Y-axis), azimuth = cols (X-axis)
        ra_ext = [az_lim_deg[0], az_lim_deg[1], r_min, r_max]
        BG = "#0d1117"
        TC = "white"
        cmap = "inferno"

        train_ds = self.train_loader.dataset
        val_ds = self.val_loader.dataset

        for idx in idxs:
            # idx may be a global frame index or a local item index.
            # Check train split first, then val split (test frames live there).
            matches_train = np.where(train_global == idx)[0]
            matches_val = np.where(val_global == idx)[0]
            if len(matches_train) > 0:
                dataset, item = train_ds, int(matches_train[0])
            elif len(matches_val) > 0:
                dataset, item = val_ds, int(matches_val[0])
            else:
                # No global-index match. Do not treat idx as a local item index, which
                # would render a different frame under this label.
                print(f"[_save_render] frame idx {idx} not in train/val split — skipping")
                continue
            data = dataset[item]
            c2w = data["radarpose"].unsqueeze(0).to(self.device)
            gt_linear = data["rad_tensor"].unsqueeze(0).to(self.device)
            t_frame = self._t_frame(data["frame_idx"])

            with torch.no_grad():
                pred_log, _sr_meta = self.render_all(c2w, sh_degree=cfg.sh_degree, t_frame=t_frame)

            gt_log = self._norm_log_map(gt_linear)

            pred_np = pred_log[0].cpu().numpy()
            gt_np = gt_log[0].cpu().numpy()

            # RADIal convention: range=rows, azimuth=cols → no .T (shape [R, A])
            gt_ra = gt_np.max(axis=0)
            pred_ra = pred_np.max(axis=0)

            gt_vmin, gt_vmax = 0.0, 1.0

            # linear panels
            gt_lin_disp = gt_linear[0].cpu().numpy()  # [D, R, A]
            pred_lin_disp = _sr_meta["pred_lin"][0].cpu().numpy()  # [D, R, A] linear pred
            # Linear RA panels: max over D (display only), on the global (lo, hi) used by
            # the *_psnr_lin metrics (no per-frame rescale).
            gt_lin_ra = gt_lin_disp.max(axis=0)
            pred_lin_ra = pred_lin_disp.max(axis=0)
            # Display range = the shared (lo, hi) and clip ceiling of loss/eval:
            # imshow(x, vmin=lo, vmax=lo+ceiling*span) equals
            # clamp((x-lo)/span, 0, ceiling) as used by the loss and metrics.
            _lo, _hi = self.lin_range()
            _ceil = self.lin_clip_ceiling
            lin_vmin = _lo
            lin_vmax = _lo + _ceil * (_hi - _lo) if _ceil > 0 else _hi

            # RD panels: sum over azimuth in the linear domain -> [D, R] -> [R, D].
            # Summing in log would give log(product), not log(sum).
            gt_lin_rd_sum = gt_lin_disp.sum(axis=2)  # [D, R], linear
            pred_lin_rd_sum = pred_lin_disp.sum(axis=2)  # [D, R], linear

            # Log RD panels: log10 of the linear sum, then per-RD percentile normalisation
            # (the RA normalisation does not apply: summing A bins shifts log10 by log10(A)).
            _gt_rd_log = np.log10(np.maximum(gt_lin_rd_sum, 1e-30))  # [D, R]
            _pred_rd_log = np.log10(np.maximum(pred_lin_rd_sum, 1e-30))  # [D, R]
            _rd_valid = _gt_rd_log[np.isfinite(_gt_rd_log)]
            _rd_lo = float(np.percentile(_rd_valid, 1.0)) if len(_rd_valid) else -6.0
            _rd_hi = float(np.percentile(_rd_valid, 99.9)) if len(_rd_valid) else 0.0
            _rd_span = max(_rd_hi - _rd_lo, 1e-6)
            gt_rd = ((_gt_rd_log - _rd_lo) / _rd_span).T  # [R, D], ~0–1
            pred_rd = ((_pred_rd_log - _rd_lo) / _rd_span).T
            rd_log_vmin, rd_log_vmax = 0.0, 1.0

            # Linear RD panels: max over azimuth, [D,R] -> [R,D], on the same global
            # (lo, hi) scale as the RA linear panel.
            gt_lin_rd = gt_lin_disp.max(axis=2).T  # [R, D]
            pred_lin_rd = pred_lin_disp.max(axis=2).T

            # RD extent: X=Doppler, Y=Range (consistent with RA where Y=Range)
            rd_ext_t = [dop_lim[0], dop_lim[1], r_min, r_max]

            # Square-bin aspect for RA: each (az_bin × range_bin) appears square in display.
            # aspect = y_unit_display / x_unit_display = (az_bin_size_deg) / (r_bin_size_m)
            n_R_bins = gt_ra.shape[0]
            n_A_bins = gt_ra.shape[1]
            r_span = r_max - r_min
            az_span = az_lim_deg[1] - az_lim_deg[0]
            ra_aspect = (az_span / n_A_bins) / (r_span / n_R_bins)

            # 2 rows (GT / Pred) × 4 cols (RA log, RD log, RA lin, RD lin);
            n_rows = 2
            fig, axes = plt.subplots(
                n_rows,
                4,
                figsize=(15, 3 * n_rows),
                facecolor=BG,
                gridspec_kw={"width_ratios": [4, 1, 4, 1]},
                constrained_layout=True,
            )
            fig.suptitle(f"Step {step} | frame {idx}", color=TC, fontsize=10)

            panels = [
                # row, col, data,         extent,      vmin,           vmax,           vline, xlabel,          ylabel
                (0, 0, gt_ra, ra_ext, gt_vmin, gt_vmax, None, "Azimuth (°)", "Range (m)"),
                (
                    0,
                    1,
                    gt_rd,
                    rd_ext_t,
                    rd_log_vmin,
                    rd_log_vmax,
                    0.0,
                    "Doppler (m/s)",
                    "Range (m)",
                ),
                (0, 2, gt_lin_ra, ra_ext, lin_vmin, lin_vmax, None, "Azimuth (°)", "Range (m)"),
                (
                    0,
                    3,
                    gt_lin_rd,
                    rd_ext_t,
                    lin_vmin,
                    lin_vmax,
                    0.0,
                    "Doppler (m/s)",
                    "Range (m)",
                ),
            ]
            row_labels = ["GT"]
            _pr = len(row_labels)
            panels += [
                (_pr, 0, pred_ra, ra_ext, gt_vmin, gt_vmax, None, "Azimuth (°)", "Range (m)"),
                (
                    _pr,
                    1,
                    pred_rd,
                    rd_ext_t,
                    rd_log_vmin,
                    rd_log_vmax,
                    0.0,
                    "Doppler (m/s)",
                    "Range (m)",
                ),
                (_pr, 2, pred_lin_ra, ra_ext, lin_vmin, lin_vmax, None, "Azimuth (°)", "Range (m)"),
                (
                    _pr,
                    3,
                    pred_lin_rd,
                    rd_ext_t,
                    lin_vmin,
                    lin_vmax,
                    0.0,
                    "Doppler (m/s)",
                    "Range (m)",
                ),
            ]
            col_titles = ["RA (log)", "RD (log)", "RA (linear)", "RD (linear)"]
            row_labels.append("Pred")
            # panels tuple: (..., vline_x, xlabel, ylabel)  — RD cols use axvline at 0
            rd_cols = {1, 3}
            for row, col, dat, ext, vmin, vmax, vline, xl, yl in panels:
                ax = axes[row, col]
                ax.set_facecolor(BG)
                asp = "auto" if col in rd_cols else ra_aspect
                ax.imshow(
                    dat,
                    aspect=asp,
                    origin="lower",
                    interpolation="nearest",
                    extent=ext,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                title = f"{row_labels[row]} {col_titles[col]}"
                ax.set_title(title, color=TC, fontsize=8)
                ax.tick_params(colors=TC, labelsize=6)
                for sp in ax.spines.values():
                    sp.set_edgecolor(TC)
                ax.set_xlabel(xl, color=TC, fontsize=7)
                ax.set_ylabel(yl, color=TC, fontsize=7)
                if vline is not None:
                    if col in rd_cols:
                        ax.axvline(vline, color="white", lw=0.5, alpha=0.4)
                    else:
                        ax.axhline(vline, color="white", lw=0.5, alpha=0.4)
                # Pin axes to the data extent — overlay markers outside it must
                # not auto-extend the view (black stripe beyond the image edge).
                ax.set_xlim(ext[0], ext[1])
                ax.set_ylim(ext[2], ext[3])

            # ── Label GT overlays: green box on RA panels, compact crosshair on RD ──
            _frame_labels = self._labels_by_frame.get(idx, [])
            if _frame_labels:
                # Marker boxes around each label (display only).
                _box_r_half = 5.0  # ±5 m range box half-width
                _box_a_half = 5.0  # ±5° azimuth box half-width
                # Doppler crosshair uses CSV radar_D_raw -> rd_doppler_bins_mps fftshift,
                # the same source as the label Doppler targets.
                _rd_dop_render = getattr(self, "_rd_dop_bins_render", None)
                if _rd_dop_render is None:
                    _rdb_path = Path(cfg.rad_tensors_dir).parent / "rd_doppler_bins_mps.npy"
                    # Sensors without Doppler have no axis file and no radar_D column:
                    # draw the RA boxes and skip the RD crosshair. `False` (not None) is
                    # the cached "absent" sentinel.
                    _rd_dop_render = (
                        np.load(_rdb_path).astype(np.float64) if _rdb_path.is_file() else False
                    )
                    self._rd_dop_bins_render = _rd_dop_render

                for _lbl in _frame_labels:
                    _r = _lbl["R_m"]
                    _a = _lbl["A_deg"]
                    _d_raw = int(_lbl.get("radar_D_raw", -1))
                    _dop_mps = None
                    if _d_raw >= 0 and _rd_dop_render is not False:
                        # radar_D_raw is the raw RD FFT index; a half-length roll is the
                        # fftshift into rd_doppler_bins_mps order.
                        _n_rd = len(_rd_dop_render)
                        _dop_mps = float(_rd_dop_render[(_d_raw + _n_rd // 2) % _n_rd])
                        # Fold the physical Doppler into the rendered axis range when the
                        # axis wraps, so the crosshair lands inside the image extent.
                        _wrap_p = float(cfg.doppler_wrap_period_mps)
                        if _wrap_p > 0.0:
                            _dop_mps = float(wrap_bins_signed(_dop_mps, _wrap_p))
                    # Draw on all four RA panels (cols 0 and 2, every row)
                    for _row in range(n_rows):
                        for _col in (0, 2):
                            _ax = axes[_row, _col]
                            # RA extent: x=azimuth, y=range (imshow with origin='lower')
                            _ax.add_patch(
                                Rectangle(
                                    (_a - _box_a_half, _r - _box_r_half),
                                    2 * _box_a_half,
                                    2 * _box_r_half,
                                    linewidth=1.2,
                                    edgecolor="lime",
                                    facecolor="none",
                                    linestyle="--",
                                    alpha=0.85,
                                    zorder=10,
                                )
                            )
                        if _dop_mps is None:
                            continue
                        # Compact + marker on RD panels.
                        for _col in (1, 3):
                            _ax = axes[_row, _col]
                            _ax.plot(
                                _dop_mps,
                                _r,
                                "+",
                                color="lime",
                                ms=14,
                                mew=1.8,
                                alpha=0.95,
                                zorder=10,
                            )

            base = os.path.join(cfg.result_dir, f"step{step:06d}_frame{idx:02d}")
            plt.savefig(base + ".png", dpi=100, facecolor=BG, bbox_inches="tight")
            plt.close(fig)

            # This is the visualisation path and writes PNGs only; renders_npy/ is
            # written by _evaluate alone, so scored arrays have a single producer and
            # a single domain convention.

            # ── Cartesian RA figure ───────────────────────────────────────────────────
            # Convert polar (az_deg, range_m) → top-down Cartesian (x=r·sin(az), y=r·cos(az))
            # with equal-aspect metres so physical proportions are correct.
            # Bounding box of the polar wedge over az in [az_lo, az_hi], r in
            # [r_min, r_max], found by sampling the arc so it is correct for any FOV
            # (including 360° sensors).
            _az_s = np.deg2rad(np.linspace(az_lim_deg[0], az_lim_deg[1], 1441))
            _rs = np.array([r_min, r_max])
            _X = np.outer(_rs, np.sin(_az_s))
            _Y = np.outer(_rs, np.cos(_az_s))
            x_lo_c, x_hi_c = float(_X.min()), float(_X.max())
            y_lo_c, y_hi_c = float(_Y.min()), float(_Y.max())
            # ~square pixels; cap the longer axis at 600 samples
            _span_x, _span_y = x_hi_c - x_lo_c, y_hi_c - y_lo_c
            _n_long = 600
            n_cx = _n_long if _span_x >= _span_y else max(120, int(_n_long * _span_x / _span_y))
            n_cy = _n_long if _span_y > _span_x else max(120, int(_n_long * _span_y / _span_x))
            xc = np.linspace(x_lo_c, x_hi_c, n_cx)
            yc = np.linspace(y_lo_c, y_hi_c, n_cy)
            XX_c, YY_c = np.meshgrid(xc, yc)
            RR_c = np.sqrt(XX_c**2 + YY_c**2)
            AZ_g_deg = np.rad2deg(np.arctan2(XX_c, YY_c))
            ri_c = np.clip(((RR_c - r_min) / r_span * n_R_bins).astype(int), 0, n_R_bins - 1)
            ai_c = np.clip(
                ((AZ_g_deg - az_lim_deg[0]) / az_span * n_A_bins).astype(int), 0, n_A_bins - 1
            )
            in_fov_c = (
                (RR_c >= r_min)
                & (RR_c <= r_max)
                & (AZ_g_deg >= az_lim_deg[0])
                & (AZ_g_deg <= az_lim_deg[1])
            )

            def _to_cart(ra_img):
                c = np.full((n_cy, n_cx), np.nan)
                c[in_fov_c] = ra_img[ri_c[in_fov_c], ai_c[in_fov_c]]
                return c

            cart_ext = [xc[0], xc[-1], yc[0], yc[-1]]
            cart_data = [
                ("GT log", _to_cart(gt_ra), gt_vmin, gt_vmax),
                ("GT linear", _to_cart(gt_lin_ra), lin_vmin, lin_vmax),
                ("Pred log", _to_cart(pred_ra), gt_vmin, gt_vmax),
                ("Pred linear", _to_cart(pred_lin_ra), lin_vmin, lin_vmax),
            ]
            fig_c, axes_c = plt.subplots(
                2, 2, figsize=(10, 8), facecolor=BG, constrained_layout=True
            )
            fig_c.suptitle(f"Step {step} | frame {idx} — Cartesian RA", color=TC, fontsize=10)
            for (ctitle, cdat, vmin_, vmax_), ax in zip(cart_data, axes_c.ravel()):
                ax.set_facecolor(BG)
                ax.imshow(
                    cdat,
                    aspect="equal",
                    origin="lower",
                    extent=cart_ext,
                    cmap=cmap,
                    vmin=vmin_,
                    vmax=vmax_,
                    interpolation="bilinear",
                )
                ax.set_title(f"Cartesian RA — {ctitle}", color=TC, fontsize=8)
                ax.tick_params(colors=TC, labelsize=6)
                for sp in ax.spines.values():
                    sp.set_edgecolor(TC)
                ax.set_xlabel("X (m, lateral)", color=TC, fontsize=7)
                ax.set_ylabel("Y (m, forward)", color=TC, fontsize=7)
                ax.axvline(0, color="white", lw=0.5, alpha=0.3)
                # Label GT overlay on Cartesian: green cross at (x=R·sin(A), y=R·cos(A))
                for _lbl in _frame_labels:
                    _cx = _lbl["R_m"] * np.sin(np.deg2rad(_lbl["A_deg"]))
                    _cy = _lbl["R_m"] * np.cos(np.deg2rad(_lbl["A_deg"]))
                    ax.plot(_cx, _cy, "g+", ms=10, mew=1.5, alpha=0.9, zorder=10)
                    ax.add_patch(
                        Rectangle(
                            (_cx - 3.0, _cy - 3.0),
                            6.0,
                            6.0,
                            linewidth=1.2,
                            edgecolor="lime",
                            facecolor="none",
                            linestyle="--",
                            alpha=0.85,
                            zorder=10,
                        )
                    )
            plt.savefig(base + "_cart.png", dpi=100, facecolor=BG, bbox_inches="tight")
            plt.close(fig_c)
