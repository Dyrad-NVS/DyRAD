"""The synthetic generator: its sensor constants agree with the training recipe, and
the ego velocity is the central difference of the poses."""

import math

import numpy as np

from dyrad import constants
from dyrad.config import load_config
from dyrad.paths import ROOT
from dyrad.synthetic import generate_scene, sensor
from dyrad.synthetic.trajectory import compute_ego_vel_sensor, ego_vel_world


def test_sensor_constants_match_recipe():
    # The renderer takes the grid, Doppler axis and PSF widths from the config, not
    # from the generated axis files, so the generator must use the same values.
    cfg = load_config(str(ROOT / "configs/synthetic/intersection_dyrad.yaml"))
    assert sensor.DOPPLER_BIN_MPS == constants.DOPPLER_BIN_MPS
    assert (cfg.num_range_bins, cfg.num_azimuth_bins, cfg.num_doppler_bins) == (
        sensor.NUM_RANGE_BINS,
        sensor.NUM_AZ_BINS,
        sensor.NUM_DOPPLER_BINS,
    )
    assert cfg.radar_far_range == sensor.RANGE_MAX_M == constants.RADIAL_RANGE_MAX_M
    assert cfg.radar_az_fov_deg == sensor.AZ_MAX_DEG - sensor.AZ_MIN_DEG
    assert math.isclose(cfg.radar_doppler_min_mps, sensor.DOPPLER_MIN, abs_tol=1e-6)
    assert math.isclose(cfg.radar_doppler_max_mps, sensor.DOPPLER_MAX, abs_tol=1e-6)
    assert math.isclose(cfg.doppler_wrap_period_mps, sensor.DOPPLER_PERIOD_MPS, abs_tol=1e-6)
    assert cfg.doppler_roll_bins == sensor.NUM_DOPPLER_BINS // 2
    assert cfg.psf_w_A == sensor.AZ_PSF_W0_BINS
    assert cfg.psf_max_k_A == sensor.AZ_PSF_K
    assert cfg.psf_max_k_R == sensor.PSF_KR
    assert cfg.range_crop_last == sensor.RANGE_CROP_LAST
    assert cfg.dt == generate_scene.DT


def test_ego_vel_central_difference():
    dt, n = 0.1, 6
    t = np.arange(n) * dt
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = 10.0 * t + 2.0 * t**2  # accelerating along x
    v = ego_vel_world(poses, dt)
    assert np.allclose(v[1:-1, 0], 10.0 + 4.0 * t[1:-1])  # exact for a quadratic
    assert np.allclose(compute_ego_vel_sensor(poses, dt)[:, 0], v[:, 0])  # identity rotation
