"""Dispatch of the radar rasterizer: the sensor-PSF path needs the CUDA extension and never
falls back silently to the PyTorch rasterizer; one reflector lands in its own bins."""

import pytest
import torch

backend = pytest.importorskip("gsplat.cuda._backend")
from gsplat.cuda._wrapper import rasterize_radar_rae  # noqa: E402

D, R, A, N = 4, 16, 12, 20
PSF = dict(psf_w_D=1.0, psf_k_eff_D=2.0, psf_w_R=1.0, psf_k_eff_R=3.0, psf_w_A=1.0, psf_k_eff_A=3.0)


def _inputs():
    g = torch.Generator().manual_seed(0)
    zeros = torch.zeros(1, 1, N, dtype=torch.int32)
    ranges = torch.rand(1, 1, N, generator=g) * 14 + 1
    az = torch.rand(1, 1, N, generator=g) - 0.5
    return (
        torch.rand(1, 1, N, generator=g), zeros + 5, zeros + 6, zeros, zeros,
        torch.full((1, 1, N), 0.3), ranges, az,
        torch.linspace(0, 15, R), torch.linspace(-0.6, 0.6, A), D, R, A,
    )


def test_psf_path_refuses_without_extension(monkeypatch):
    monkeypatch.setattr(backend, "_C", None)
    with pytest.raises(RuntimeError, match="CUDA extension"):
        rasterize_radar_rae(*_inputs(), **PSF)


def test_learned_extent_path_runs_without_extension(monkeypatch):
    monkeypatch.setattr(backend, "_C", None)
    sigma = torch.full((1, 1, N), 0.5)
    out = rasterize_radar_rae(*_inputs(), sigma_r_m=sigma, sigma_az_rad=sigma * 0.1)
    assert out.shape == (1, 1, D, R, A) and out.sum() > 0


def test_non_psf_path_needs_the_projected_sigmas():
    with pytest.raises(ValueError, match="sigma_r_m"):
        rasterize_radar_rae(*_inputs())


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or backend._C is None, reason="needs a GPU and the built extension"
)
def test_psf_render_peaks_at_the_reflector():
    dev = "cuda"
    nd, nr, na, dv = 16, 32, 24, 0.1123
    rb = torch.linspace(0, 31, nr, device=dev)
    ab = torch.linspace(-0.6, 0.6, na, device=dev)
    ri, ai = 20, 10
    one = torch.ones(1, 1, 1, device=dev)
    idx = torch.zeros(1, 1, 1, dtype=torch.int32, device=dev)
    # v_r = 0 sits at Doppler bin D/2 of the axis linspace(-8 dv, 7 dv)
    out = rasterize_radar_rae(
        one, idx + ri, idx + ai, idx, idx + nd // 2, one * 0.3, one * rb[ri], one * ab[ai],
        rb, ab, nd, nr, na, v_r_cont=one * 0.0, doppler_bin_spacing=dv,
        use_physical_dr_psf=True, **PSF,
    )[0, 0]
    assert out.shape == (nd, nr, na)
    peak = divmod(int(out.argmax()), nr * na)
    assert (peak[0], *divmod(peak[1], na)) == (nd // 2, ri, ai)
