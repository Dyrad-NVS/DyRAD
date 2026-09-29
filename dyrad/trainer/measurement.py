"""Measurement domain of the rendered tensor (paper Sec. 3.2): the sensor's reporting units
(amplitude, power or log-compressed), its noise pedestal, and the per-sequence normalization range.
"""

from torch import Tensor

from dyrad.domains import Domain as _Domain


class MeasurementMixin:
    # ── Measurement domain (Config.measurement_domain) ─────────────────────────────
    _MEAS_G = {
        "amplitude": "sqrt(Ψ) + b_s",
        "power": "Ψ + b_s",
        "log_power": "log10(Ψ + b_s)",
        "log_amplitude": "log10(sqrt(Ψ) + b_s)",
    }

    def _meas_is_log(self) -> bool:
        return self._meas_domain.startswith("log_")

    def _to_measurement(self, x_lin: Tensor) -> Tensor:
        """Native linear quantity (amplitude or power, floor included) -> measurement domain.

        Identity for linear sensors, log10 for log-compressed ones. The only place a
        log10 is applied to a render or GT for loss, eval and persisted renders.
        """
        return x_lin.clamp(min=1e-30).log10() if self._meas_is_log() else x_lin

    def _norm_log_map(self, x_lin: Tensor) -> Tensor:
        """Native linear -> normalised log map `(log10(x) - norm_lo) / norm_range`.

        This is the view `render_all` returns as `pred_log` for every sensor, used by
        the visualisation panels and the densification residual. It is not the loss
        domain (the loss reads `pred_meas` / gt_N).
        """
        return (x_lin.clamp(min=1e-30).log10() - self.norm_lo) / self.norm_range

    def _measurement_range(self):
        """(lo, span, clip): affine normalisation of the measurement domain to [0, clip].

        A per-sensor data attribute: the linear (lo, hi) of the sequence's norm.json for
        linear sensors, the log map N (floor at 0, ceiling at 1) for log sensors.
        `norm_clip_ceiling` applies to both (<= 0: no clip).
        """
        if self._meas_is_log():
            lo, span = self.norm_lo, self.norm_range
        else:
            lo, span = self.lin_lo, self.lin_span
        return lo, span, self.lin_clip_ceiling

    def _normalize_measurement(self, x_meas: Tensor) -> Tensor:
        """Measurement domain -> normalized [0, clip]. Applied identically to Ŷ and Y."""
        lo, span, clip = self._measurement_range()
        y = (x_meas - lo) / span
        return y.clamp(min=0.0, max=clip) if clip > 0 else y

    def _render_domain(self):
        """Domain of the tensors this run writes to renders_npy/ (i.e. of Ŷ).

        The loader scales GT power by 1/hi, so a linear sensor's `gt`/`pred_meas` are in
        CEILING_UNITS and a log sensor's in LOG_CEILING (log10 of ceiling units).
        Declared at runtime because it depends on the config.
        """
        if self._meas_is_log():
            return _Domain.LOG_CEILING
        return _Domain.CEILING_UNITS

    def lin_range(self) -> tuple:
        """Global (lo, hi) linear-normalization range, in ceiling units.

        The sequence's norm.json (lin_lo, lin_hi), as resolved by the training dataset,
        times its 1/hi scale. Cached and shared by the *_psnr_lin metrics (_evaluate),
        the loss normalisation and the linear display panels (_save_render), so what
        is displayed equals what is scored.
        """
        if self._lin_range_cache is None:
            _n = self.train_loader.dataset._norm
            _s = _n.norm_scale
            self._lin_range_cache = (_n.lin_lo * _s, _n.lin_hi * _s)
            print(
                f"[Norm] linear range from {_n.source.parent.name}/norm.json: "
                f"(lo, hi)=({self._lin_range_cache[0]:.4g}, {self._lin_range_cache[1]:.4g}) "
                f"[ceiling units]",
                flush=True,
            )
        return self._lin_range_cache
