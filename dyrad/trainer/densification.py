"""Residual-guided densification of the static background (paper B.3)."""

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree
from torch import Tensor


class DensifyMixin:
    # -----------------------------------------------------------------------
    # Parameter-rebuild helpers
    # -----------------------------------------------------------------------

    def _param_lrs(self) -> dict:
        """Learning-rate map for every entry in self.params (used to rebuild optimisers)."""
        cfg = self.cfg
        lrs = {
            "means": cfg.means_lr,
            "sh_coeffs": cfg.sh_coeffs_lr,
            "opacities": cfg.opacities_lr,
        }
        # Keep scales/quats optimisers alive across densification rebuilds when they
        # are trained (omitting them would drop them at the first rebuild).
        if cfg.train_scales:
            lrs["scales"] = cfg.scales_lr
        if cfg.train_quats:
            lrs["quats"] = cfg.quats_lr
        return lrs

    # -----------------------------------------------------------------------
    # Densification (residual-peak seeding of static reflectors)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _means_world_np(self) -> np.ndarray:
        """`params["means"]` as world xyz [N, 3].

        The dynamic rows hold on-object positions, so they are mapped through the
        object's position at its canonical time, x = means + P_j(t_canon). Any consumer
        that assumes one frame for all N rows must use this.
        """
        m = self.params["means"].detach().cpu().numpy().astype(np.float64).copy()
        tr = self.tracks
        if self.obj_idx is None or not tr._ridx:
            return m
        oi = self.obj_idx.detach().cpu().numpy()
        for ridx in tr._ridx:
            sel = oi == ridx
            if not sel.any():
                continue
            # The object's rotation is left out: R_j(t_canon) is the identity at init
            # but drifts once the control points move. The only consumer is the
            # candidate dedup (densify_dedup_radius_m).
            tc = tr.tcanon(ridx)
            Pn = tr._interp_pos(ridx, tc).detach().cpu().numpy().astype(np.float64)
            m[sel, :2] = m[sel, :2] + Pn[None, :]
        return m

    def _densify_accum(
        self,
        pred_log: Tensor,
        gt_log: Tensor,
        gt_linear: Tensor,
        c2w: Tensor,
        t_frame: float,
    ) -> None:
        """Collect densification candidates from the underprediction residual.

        The log-space residual (gt_log - pred_log).clamp(min=0) at signal bins marks
        where the model underpredicts a bright return, i.e. where new reflectors
        are needed. It is residual-based rather than gradient-based, so it works
        when parameter gradients are near zero.

        Static-band scoring: new reflectors are always static, so they are only
        seeded for residual they can produce, i.e. power in the static Doppler band.
        Otherwise a moving object's Doppler residual would attract static seeds.
        When cfg.sigma_static_bins > 0, the residual is masked to the static band
        (wrap-aware gate, measured ego velocity).

        Args:
            pred_log:   [B, D, R, A] log-space prediction (normalised).
            gt_log:     [B, D, R, A] log-space GT (normalised).
            gt_linear:  [B, D, R, A] linear GT, used for the signal mask.
            c2w:        [B, 4, 4] current pose, for the static Doppler gate.
            t_frame:    frame time in seconds, for the ego-velocity lookup.
        """
        with torch.no_grad():
            # Residual: underprediction at signal bins only (log10 normalised units).
            sig_mask = gt_linear > self.noise_floor  # [B, D, R, A]
            residual = torch.zeros_like(pred_log)
            residual[sig_mask] = (gt_log - pred_log)[sig_mask].clamp(min=0.0)

            # Collapse Doppler -> RA residual (max over D of a candidate signal),
            # restricted to the static band when possible (new reflectors are static).
            if self.cfg.sigma_static_bins > 0:
                static_mask, _ = self._compute_doppler_gate(
                    c2w, self._c2w_prev_can(c2w, t_frame)
                )  # [D, R, A]
                residual = residual * static_mask.unsqueeze(0)
            residual_ra = residual.max(dim=1).values  # [B, R, A]

        # ── Residual-peak seeding: find NMS peaks of the residual and back-project to
        # world XY (elevation 0 in the sensor frame; z is unsupervised), so new
        # reflectors land in the underpredicted cells. _densify_grow dedups, takes the
        # top K and seeds.
        if self._dens_cand is None:
            self._dens_cand = {"xyz": [], "score": [], "peak": []}

        cfg = self.cfg
        thresh = float(cfg.densify_residual_thresh)
        nms_size = 2 * int(cfg.rad_peak_nms_radius) + 1
        rbins = self.range_bins.cpu().numpy()
        abins = self.az_bins.cpu().numpy()
        res_np = residual_ra.detach().cpu().numpy()  # [B, R, A]
        # GT peak over Doppler at each candidate cell, in ceiling units (data max ~ 1).
        gt_peak_np = gt_linear.max(dim=1).values.detach().cpu().numpy()  # [B, R, A]
        c2w_np = c2w.detach().cpu().numpy().reshape(-1, 4, 4)
        for b in range(res_np.shape[0]):
            ra = res_np[b]
            lm = maximum_filter(ra, size=nms_size, mode="nearest")
            ri, ai = np.where((ra == lm) & (ra > thresh))
            if len(ri) == 0:
                continue
            # Vectorised back-projection of all peaks.
            rm = rbins[ri]  # [P]
            az = abins[ai]  # [P]
            p_cam = np.stack(
                [np.cos(az) * rm, np.sin(az) * rm, np.zeros_like(rm)], axis=1
            )  # [P, 3]
            pw = (p_cam @ c2w_np[b, :3, :3].T + c2w_np[b, :3, 3]).astype(np.float32)  # [P, 3] world
            sc = ra[ri, ai].astype(np.float32)
            self._dens_cand["xyz"].append(pw)
            self._dens_cand["score"].append(sc)
            self._dens_cand["peak"].append(gt_peak_np[b][ri, ai].astype(np.float32))

    @torch.no_grad()
    def _densify_grow(self, step: int) -> int:
        """Seed new reflectors at back-projected residual peaks (coverage gaps).

        Candidates are world-XY positions collected in _densify_accum. They are
        deduplicated among themselves and against existing reflectors within
        densify_dedup_radius_m; the top K by residual (densify_max_per_grow) are
        appended with zero SH coefficients and opacity init_alpha * GT peak. New reflectors are
        appended after the [:n_lbl] dynamic prefix, so they are static by
        construction. Returns the number added.
        """
        cfg = self.cfg
        cand = self._dens_cand
        if not cand or len(cand["xyz"]) == 0:
            print(f"[Densify {step}] no residual-peak candidates", flush=True)
            return 0

        device = self.device
        N = len(self.params["means"])
        xyz = np.concatenate(cand["xyz"], axis=0)  # [C, 3]
        score = np.concatenate(cand["score"], axis=0)  # [C]
        peak = np.concatenate(cand["peak"], axis=0)  # [C]
        n_raw = len(xyz)
        rad = float(cfg.densify_dedup_radius_m)

        # ── Voxel-dedup among candidates: snap XY to a rad grid and keep the
        # highest-residual candidate per cell. ─────────────────────────────────
        key = np.round(xyz[:, :2] / rad).astype(np.int64)
        kk = key[:, 0] * 1_000_003 + key[:, 1]
        order = np.argsort(-score)  # best score first
        kk_sorted = kk[order]
        _, first = np.unique(kk_sorted, return_index=True)
        sel = order[first]  # one (best) per voxel
        xyz, score, peak = xyz[sel], score[sel], peak[sel]

        # ── remove cells already covered by an existing reflector (batched). ────
        # World xy, not raw `means`: the dynamic rows hold on-object positions (origin
        # p0), which would otherwise appear as phantom points near the world origin.
        tree = cKDTree(self._means_world_np()[:, :2])
        dist, _ = tree.query(xyz[:, :2])
        free = dist >= rad
        xyz, score, peak = xyz[free], score[free], peak[free]

        # ── top-K by residual, honouring per-grow cap and global cap. ───────────
        room = (cfg.densify_max_reflectors - N) if cfg.densify_max_reflectors > 0 else 10**9
        k = int(min(cfg.densify_max_per_grow, room, len(xyz)))
        bg_max = float(score.max()) if len(score) else 0.0
        if k <= 0:
            print(
                f"[Densify {step}] {n_raw} cand → 0 after dedup/cap (bg_max={bg_max:.2e})",
                flush=True,
            )
            self._dens_cand = {"xyz": [], "score": [], "peak": []}
            return 0
        top = np.argsort(-score)[:k]
        xyz, peak = xyz[top], peak[top]
        n_new = k
        print(
            f"[Densify {step}] residual-peaks: {n_raw} raw → +{n_new} seeds "
            f"→ {N + n_new} total (bg_max={bg_max:.2e})",
            flush=True,
        )
        # Clear candidate buffer for the next accumulation window.
        self._dens_cand = {"xyz": [], "score": [], "peak": []}

        new_means = torch.from_numpy(np.ascontiguousarray(xyz)).to(device)  # [n_new,3]

        # Per-reflector parameters of the seeds: means at the candidates, SH zero,
        # opacity from the GT peak, scales the background median, identity quats.
        bg0 = self._n_dynamic
        for key in self.params:
            old_val = self.params[key].data
            if key == "means":
                new_rows = new_means
            elif key == "sh_coeffs":
                # As create_reflectors: every SH coefficient 0 (eta_i0 = 0 is pinned), so
                # a seed's power comes from its opacity alone.
                new_rows = torch.zeros(
                    (n_new,) + tuple(old_val.shape[1:]), device=device, dtype=old_val.dtype
                )
            elif key == "scales":
                med = (
                    old_val[bg0:].median(dim=0).values
                    if old_val.shape[0] > bg0
                    else old_val.median(dim=0).values
                )
                new_rows = med.unsqueeze(0).repeat(n_new, 1).clone()
            elif key == "opacities":
                # A seed starts at init_alpha times the GT peak of its cell (ceiling
                # units), so it enters at the brightness it was seeded for. Stored as
                # logit(alpha), clipped into the open interval.
                alpha = np.clip(cfg.init_alpha * peak, 1e-4, 1.0 - 1e-4)
                new_rows = (
                    torch.from_numpy(np.log(alpha / (1.0 - alpha)))
                    .to(device=device, dtype=old_val.dtype)
                    .reshape((n_new,) + tuple(old_val.shape[1:]))
                )
            elif key == "quats":
                q = torch.zeros(
                    (n_new,) + tuple(old_val.shape[1:]), device=device, dtype=old_val.dtype
                )
                q[:, 0] = 1.0
                new_rows = q
            else:
                raise KeyError(f"no densification rule for reflector parameter {key!r}")
            # Keep the parameter's trainability: the frozen scales/quats stay frozen.
            self.params[key] = torch.nn.Parameter(
                torch.cat([old_val, new_rows], dim=0),
                requires_grad=self.params[key].requires_grad,
            )

        # New static reflectors: obj_idx -1 (any id outside the tracks is static).
        if self.obj_idx is not None:
            self.obj_idx = torch.cat(
                [self.obj_idx, torch.full((n_new,), -1, device=device, dtype=self.obj_idx.dtype)]
            )

        lrs = self._param_lrs()
        self.optimizers = {k: torch.optim.Adam([self.params[k]], lr=lrs[k]) for k in lrs}
        return n_new
