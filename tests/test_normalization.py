"""Normalization and measurement-domain guards (dyrad.norm, dyrad.domains)."""

import numpy as np
import pytest

from dyrad.domains import Domain, DomainError, check_domain, load_rad, save_rad
from dyrad.evaluation.score_renders import domain_violations
from dyrad.norm import normalize

LIN_LO, LIN_HI = 1833.0, 5.9e6  # the RADIal recipe's global linear range


def test_linear_mode_is_affine_and_clipped():
    p = {"mode": "linear", "lin_lo": LIN_LO, "lin_hi": LIN_HI}
    assert normalize(LIN_LO, p) == pytest.approx(0.0)
    assert normalize(LIN_HI, p) == pytest.approx(1.0)
    assert normalize(0.0, p) == pytest.approx(0.0)
    assert normalize(1e9, p) == pytest.approx(1.0)
    x = np.array([5e3, 1e4, 1e5, 1e6, 5e6])
    np.testing.assert_allclose(normalize(x, p), (x - LIN_LO) / (LIN_HI - LIN_LO), atol=1e-6)


def test_counts_mode_matches_8bit_log_counts():
    k, full = 36.0, 255.0
    p = {"mode": "counts", "counts_per_decade": k, "counts_full_scale": full, "hi": LIN_HI}
    assert normalize(1.0, p) == pytest.approx(0.0)  # count 0
    assert normalize(0.5, p) == pytest.approx(0.0)  # below count 0 clips
    assert normalize(10.0 ** (full / k), p) == pytest.approx(1.0, abs=1e-6)
    for u in (10.0, 64.0, 128.0, 200.0):
        assert normalize(10.0 ** (u / k), p) == pytest.approx(u / full, abs=1e-6)


def test_ceiling_units_and_raw_power_are_distinguishable():
    raw = np.array([1.6e5, 3.7e5, 2.5e7, 1e3])
    ceiling = raw * (1.0 / LIN_HI)
    p = {"mode": "linear", "lin_lo": LIN_LO, "lin_hi": LIN_HI}
    assert float(normalize(raw, p).max()) > 0.01
    assert float(normalize(ceiling, p).max()) == 0.0  # wrong domain collapses to zero


def test_domain_guard(tmp_path):
    raw = np.random.default_rng(0).lognormal(np.log(1.6e5), 1.0, (4, 16, 16))
    ceiling = raw / LIN_HI
    with pytest.raises(DomainError):
        save_rad(tmp_path / "a" / "rad_0.npy", ceiling, Domain.RAW_POWER)
    save_rad(tmp_path / "b" / "rad_0.npy", raw, Domain.RAW_POWER)
    with pytest.raises(DomainError):
        load_rad(tmp_path / "b" / "rad_0.npy", expect=Domain.CEILING_UNITS)
    load_rad(tmp_path / "b" / "rad_0.npy", expect=Domain.RAW_POWER)
    with pytest.raises(DomainError):
        save_rad(
            tmp_path / "b" / "rad_1.npy", ceiling, Domain.CEILING_UNITS
        )  # two domains in one dir
    with pytest.raises(DomainError):
        check_domain(np.array([0.5, 1.7]), Domain.N01)
    check_domain(np.array([0.0, 0.5, 1.0]), Domain.N01)


def test_scoring_domain_guard():
    for a in (4.0, 11.07, 1.0, 0.5):
        assert not domain_violations({"alpha": a})
    for a in (4.08e9, 4.73e10):  # pred-to-GT gains of a prediction scored in the wrong domain
        assert domain_violations({"alpha": a})
    assert domain_violations({"n_gt_peaks": 18451, "n_pred_peaks": 0})
    assert not domain_violations({"alpha": float("nan")})


def test_normalize_rejects_unknown_mode():
    with pytest.raises(ValueError):
        normalize(1.0, {"mode": "capped"})
