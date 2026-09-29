"""Config plumbing: `base:` inheritance, the holdout rule, init-cloud names, generated configs."""

import sys

import pytest

from dyrad import generate_configs
from dyrad.config import is_held_out, load_config
from dyrad.preprocessing.build_init_cloud import cloud_filename


def test_generated_configs_are_up_to_date(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_configs", "--check"])
    assert generate_configs.main() == 0


def test_is_held_out():
    held = [f for f in range(20) if is_held_out(f, 5)]
    assert held == [0, 5, 10, 15]
    assert [f for f in range(12) if is_held_out(f, 5, test_offset=2)] == [2, 7]
    assert [f for f in range(913, 925) if is_held_out(f, 5)] == [915, 920]  # global index
    assert not any(is_held_out(f, 0) for f in range(50))  # test_every 0: no holdout


def test_cloud_filename():
    # frame_end is exclusive, so the name carries the inclusive last frame
    assert (
        cloud_filename(913, 973, 5, 0.5, True, 1.0)
        == "radar_cloud_f913_972_no_every5_0p5m_free_psf1.npy"
    )
    assert cloud_filename(0, 60, 0, 0.5, False) == "radar_cloud_f0_59_full_0p5m.npy"
    assert cloud_filename(0, 60, 5, 0.1, True) == "radar_cloud_f0_59_no_every5_0p1m_free.npy"


def test_load_config_base_inheritance(tmp_path):
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "r.yaml").write_text("means_lr: 1.0e-4\nsh_degree: 1\nframe_start: 3\n")
    (tmp_path / "seq.yaml").write_text("base: recipes/r.yaml\nframe_start: 10\nframe_end: 70\n")
    (tmp_path / "other.yaml").write_text("sh_degree: 2\nframe_end: 40\n")
    (tmp_path / "run.yaml").write_text("base: [seq.yaml, other.yaml]\nmeans_lr: 2.0e-4\n")
    cfg = load_config(str(tmp_path / "run.yaml"))
    assert cfg.means_lr == 2.0e-4  # the declaring file overrides every base
    assert cfg.frame_start == 10  # a base overrides its own base
    assert cfg.sh_degree == 2 and cfg.frame_end == 40  # later list entries override earlier ones

    (tmp_path / "bad.yaml").write_text("base: seq.yaml\nnot_a_config_key: 1\n")
    with pytest.raises(ValueError, match="not_a_config_key"):
        load_config(str(tmp_path / "bad.yaml"))


def test_init_cloud_range_axis_is_checked(tmp_path):
    import json
    from types import SimpleNamespace

    import pytest

    from dyrad.axes import range_axis_identity
    from dyrad.trainer.initialization import InitMixin

    cfg = SimpleNamespace(
        test_every=5, test_offset=0, num_range_bins=512, radar_far_range=103.0, range_bin_offset=0.0
    )
    runner = SimpleNamespace(cfg=cfg)
    cloud = tmp_path / "cloud.npy"
    sidecar = tmp_path / "cloud.npy.build.json"

    def write(axis):
        sidecar.write_text(json.dumps({"test_every": 5, "test_offset": 0, **axis}))

    write({"range_axis": range_axis_identity(cfg)})
    InitMixin._check_cloud_split(runner, cloud)  # same axis: accepted
    write({})  # built before the axis was recorded
    with pytest.raises(RuntimeError, match="range axis"):
        InitMixin._check_cloud_split(runner, cloud)
    write({"range_axis": {**range_axis_identity(cfg), "range_bin_offset": 0.5}})
    with pytest.raises(RuntimeError, match="range axis"):
        InitMixin._check_cloud_split(runner, cloud)
