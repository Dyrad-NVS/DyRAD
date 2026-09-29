"""Paper statements that are cheap to check on CPU: projections, the constant-prediction rho,
the sensor-transfer gain, the coarse grid and the Doppler wrap, the radial velocity, the
synthetic off-path views and the synthetic scene's determinism."""

import numpy as np
import pytest
import torch

from dyrad.axes import sensor_axes
from dyrad.config import load_config
from dyrad.evaluation.radar_metrics import MetricParams, correlation_metrics, ra_project, rd_project


def test_ra_and_rd_maps_are_means():
    rad = np.random.default_rng(0).random((16, 20, 30))  # [D, R, A]
    np.testing.assert_array_equal(ra_project(rad), rad.mean(axis=0))
    np.testing.assert_array_equal(rd_project(rad), rad.mean(axis=2))


def test_constant_prediction_has_zero_correlation():
    gt = np.random.default_rng(1).uniform(2e3, 5e6, (40, 60))
    rho = correlation_metrics(np.full_like(gt, 1e5), gt, lin_norm=(1833.0, 5.9e6))["ra_corr"]
    assert rho == 0.0  # Table 4: a structureless prediction scores rho = 0, not NaN


def test_analytic_transfer_gain_is_8():
    from dyrad.evaluation.sensor_transfer import analytic_amplitude_gain
    from dyrad.preprocessing.radial.preprocess import SENSOR_PRESETS

    native, coarse = SENSOR_PRESETS["native"].to_dict(), SENSOR_PRESETS["coarse"].to_dict()
    assert analytic_amplitude_gain(coarse, native) == (512 / 128) * (256 / 128) == 8.0


def test_grids_and_the_shared_doppler_wrap():
    native = load_config("configs/radial/31_22_dyrad.yaml")
    coarse = load_config("configs/radial/31_22_coarse_dyrad.yaml")
    r, a, d = sensor_axes(native)
    assert (len(d), len(r), len(a)) == (16, 512 - 15 - 50, 751)
    r, a, d = sensor_axes(coarse)
    assert (len(d), len(r), len(a)) == (8, 128 - 4 - 12, 751)  # 8 x 112 x 751
    for cfg in (native, coarse):
        _, _, d = sensor_axes(cfg)
        assert len(d) * (d[1] - d[0]) == pytest.approx(16 * 0.1123)  # one wrap period
    assert MetricParams().doppler_wrap_mps == pytest.approx(16 * 0.1123) == pytest.approx(1.7968)


def test_radial_velocity_of_static_and_moving_reflectors():
    from gsplat.cuda._wrapper import compute_doppler_radial_velocity

    dt = 0.1
    w2c = torch.eye(4).view(1, 1, 4, 4)
    w2c_prev = w2c.clone()
    w2c_prev[..., 0, 3] = 1.0  # the sensor 1 m further back (-x) one frame earlier
    means = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])  # ahead, abeam
    v = compute_doppler_radial_velocity(means, w2c, w2c_prev, dt)[0, 0]
    assert torch.allclose(v, torch.tensor([-10.0, 0.0]))  # (r_t - r_prev) / dt; abeam: none
    assert torch.allclose(compute_doppler_radial_velocity(means, w2c, w2c_prev, dt, doppler_sign=-1.0)[0, 0], -v)

    vel = torch.tensor([[[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]], requires_grad=True)
    v = compute_doppler_radial_velocity(means, w2c, w2c, dt, velocities=vel)[0, 0]
    assert torch.allclose(v, torch.tensor([-5.0, 0.0]))  # a receding target lowers v_r
    v.sum().backward()
    assert torch.allclose(vel.grad[0, 0], torch.tensor([-1.0, 0.0, 0.0]))


def test_synthetic_offpath_views():
    from dyrad.synthetic.generate_offpath_views import view_offsets

    assert view_offsets() == [(0.0, 0.0), (1.75, 0.0), (-1.75, 0.0), (3.5, 0.0), (0.0, 5.0), (0.0, 10.0)]


def test_synthetic_scene_is_deterministic_at_seed_42():
    from dyrad.synthetic import scene_spec
    from dyrad.synthetic.generate_scene import DT, NUM_FRAMES, SEED

    assert (SEED, NUM_FRAMES) == (42, 40)
    spec = scene_spec.load_spec("intersection")
    v_ego = scene_spec.ego_params(spec)["v_ego"]

    def build():
        rng = np.random.default_rng(SEED)
        return scene_spec.build_scene(spec, rng, v_ego=v_ego, dt=DT, num_frames=NUM_FRAMES)

    a, b = build(), build()
    assert len(a) == len(b) > 0
    for ra, rb in zip(a, b):
        assert ra.keys() == rb.keys()
        for k in ra:
            assert np.array_equal(np.asarray(ra[k]), np.asarray(rb[k])), k
