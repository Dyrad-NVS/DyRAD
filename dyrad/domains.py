"""Explicit domains for radar tensors, checked at file boundaries.

A radar tensor is always float32 of the right shape, so an array whose numbers mean
something other than what the reader assumes (e.g. ceiling-units values written where
raw values are expected) produces plausible but wrong results without any error. This
module makes the meaning explicit.

A directory of radar tensors declares its domain in a `_domain.json` sidecar.
`save_rad` writes the declaration and refuses to mix domains in one directory; readers
check it with `read_domain` (the scorers) or `load_rad`. A mismatch raises instead of
returning bad numbers. For undeclared directories (e.g. the synthetic generator's
output), the magnitude check `check_domain` still catches the gross ceiling-vs-raw
confusion.

This is deliberately not an ndarray subclass: those get silently dropped by
almost every numpy/torch operation. The tag lives with the file, which is where
the boundary is.

    from dyrad.domains import Domain, save_rad, load_rad, check_domain

    save_rad(path, arr, Domain.RAW_POWER)
    arr = load_rad(path, expect=Domain.RAW_POWER)      # raises on mismatch
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

import numpy as np

SIDECAR = "_domain.json"


class Domain(str, Enum):
    """What the numbers in a radar tensor mean."""

    #: Sensor-native units, unscaled (amplitude for RADIal and synthetic, power for
    #: Boreas). What real rad_tensors/ hold, and what every dataset loader expects to
    #: read off disk.
    RAW_POWER = "raw_power"

    #: RAW_POWER * (1/hi): the ceiling-units scale the model trains in, so
    #: also the domain of meta["pred_lin"] and any raw render output.
    CEILING_UNITS = "ceiling_units"

    #: Canonically normalized to [0,1] via norm.normalize(). What the
    #: loss, the metrics and the baselines compare on.
    N01 = "n01"

    #: log10 of CEILING_UNITS, before the affine N-map.
    LOG_CEILING = "log_ceiling"


# Plausible median magnitude per domain. Deliberately wide: this is a
# blunder detector (is this ~1e5 or ~1e-2?), not a calibration check. Bounds
# are (lo, hi) on the median of positive values.
_MEDIAN_BAND = {
    Domain.RAW_POWER: (1.0, 1e12),
    Domain.CEILING_UNITS: (1e-12, 10.0),
    Domain.N01: (0.0, 1.0),
    Domain.LOG_CEILING: (-30.0, 5.0),
}

#: Render-file prefix -> how its RA map was projected from the cube, recorded in the
#: `_domain.json` of a render directory (the trainer's renders_npy/ and the
#: sensor-transfer renders): `pred_`/`gt_` hold the mean over D, `*_rad_` the full cube.
D_PROJECTION = {
    "pred_rad_": "none",
    "gt_rad_": "none",
    "pred_": "mean_over_D",
    "gt_": "mean_over_D",
}


# The bands span orders of magnitude, so they must not be decided by a float32
# ULP at the boundary. Example: Boreas's raw-power floor is floor_u * hi == 1.0
# exactly (in float64), precisely RAW_POWER's lower bound, and more than half of a
# Boreas render sits on that floor, so the median is the bound; computed in float32
# it lands one ULP low (0.99999994). The bounds are therefore padded outward by a
# relative tolerance (by magnitude, so it works for negative bounds too).
_BAND_RTOL = 1e-6


class DomainError(ValueError):
    """A tensor was not in the domain its reader required."""


def _median_pos(a: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    pos = a[a > 0.0]
    return float(np.median(pos if pos.size else a))


def check_domain(arr, domain: Domain, where: str = "") -> None:
    """Raise DomainError if `arr`'s magnitude is impossible for `domain`.

    Catches the gross confusions (ceiling units in a raw file and the
    reverse). Values outside [0,1] in N01 are a hard error; the rest are
    magnitude bands.
    """
    domain = Domain(domain)
    band = _MEDIAN_BAND[domain]
    a = np.asarray(arr, np.float64)
    ctx = f" [{where}]" if where else ""

    if domain is Domain.N01:
        finite = a[np.isfinite(a)]
        if finite.size and (finite.min() < -1e-6 or finite.max() > 1.0 + 1e-6):
            raise DomainError(
                f"N01 tensor out of [0,1]: min={finite.min():.4g} max={finite.max():.4g}{ctx}"
            )
        return

    med = _median_pos(a)
    if not np.isfinite(med):
        return
    lo, hi = band
    lo_ok = lo - abs(lo) * _BAND_RTOL
    hi_ok = hi + abs(hi) * _BAND_RTOL
    if not (lo_ok <= med <= hi_ok):
        hint = ""
        if domain is Domain.RAW_POWER and med < lo_ok:
            hint = "  -> looks like CEILING_UNITS; multiply by hi before writing)"
        elif domain is Domain.CEILING_UNITS and med > 10.0:
            hint = "  -> looks like RAW_POWER; the 1/hi scale was not applied"
        raise DomainError(
            f"median {med:.4g} outside plausible range [{lo:g}, {hi:g}] for "
            f"domain={domain.value}{ctx}{hint}"
        )


def declare_domain(dir_path, domain: Domain, note: str = "", d_projection=None) -> None:
    """Write the `_domain.json` declaration for a directory of tensors.

    `d_projection` records how the Doppler axis was collapsed, the other half of
    "what is in this file": sum-over-D and max-over-D dumps can both be
    `ceiling_units` yet differ by a large factor. Values: "sum_over_D" |
    "max_over_D" | "mean_over_D" | "none" (full [D,R,A]).

    One directory can legitimately hold several projections (a render directory holds
    mean-over-D `pred_*.npy` beside full-cube `pred_rad_*.npy`, see `D_PROJECTION`), so
    this accepts either a single string or a {filename-prefix: projection} mapping.
    """
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)
    body = {"domain": Domain(domain).value, "note": note}
    if d_projection is not None:
        body["d_projection"] = d_projection
    (d / SIDECAR).write_text(json.dumps(body, indent=2) + "\n")


def read_domain(dir_path):
    """Declared Domain for a directory, or None if it has no `_domain.json`.

    A sidecar that exists but cannot be read, or names no known domain, raises
    DomainError rather than passing for an undeclared directory.
    """
    p = Path(dir_path) / SIDECAR
    if not p.is_file():
        return None
    try:
        return Domain(json.loads(p.read_text())["domain"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as e:
        raise DomainError(f"unreadable domain declaration {p}: {e!r}") from e


def save_rad(
    path,
    arr,
    domain: Domain,
    note: str = "",
    check: bool = True,
    d_projection=None,
):
    """np.save `arr`, declaring its domain for the containing directory.

    A domain violation (magnitude outside the band, or a directory already declared
    another domain) raises, so a bad file is never created.
    """
    path = Path(path)
    problems = []
    if check:
        try:
            check_domain(arr, domain, where=str(path))
        except DomainError as e:
            problems.append(str(e))
    declared = read_domain(path.parent)
    if declared is not None and declared != Domain(domain):
        problems.append(
            f"{path.parent} is declared {declared.value} but a {Domain(domain).value} "
            f"tensor is being written into it — one directory, one domain"
        )
    if problems:
        raise DomainError("; ".join(problems))
    if declared is None:
        declare_domain(path.parent, domain, note, d_projection)
    np.save(path, arr)
    return path


def load_rad(path, expect: Domain | None = None, check: bool = True):
    """np.load, verifying the declared domain (and magnitude) against `expect`."""
    path = Path(path)
    arr = np.load(path)
    if expect is None:
        return arr
    expect = Domain(expect)
    declared = read_domain(path.parent)
    if declared is not None and declared != expect:
        raise DomainError(f"{path} is declared {declared.value} but was read as {expect.value}")
    if check:
        check_domain(arr, expect, where=str(path))
    return arr


#: Floor for log10 conversions. Matches the `clamp(min=1e-30)` before the log10 in
#: trainer/measurement.py (the measurement map and the normalised log map), and
#: LOG_CEILING's band lower bound (-30).
_LOG_FLOOR = 1e-30


def convert(arr, src: Domain, dst: Domain, *, hi: float | None = None):
    """Explicitly convert a tensor between domains. Every reader that needs a
    different domain than the one declared on disk goes through here, so the
    conversion is visible instead of being an ad-hoc `10 **` or `log10` at the
    call site.

    Supported (numpy arrays and torch tensors alike):
      CEILING_UNITS <-> LOG_CEILING      log10 / 10**   (no constants needed)
      RAW_POWER     <-> CEILING_UNITS    / hi, * hi     (`hi` required)
      RAW_POWER     <-> LOG_CEILING      via CEILING_UNITS (`hi` required)
    N01 needs the per-sequence norm.json parameters (lin_lo/lin_hi, norm_lo/norm_range,
    counts_per_decade) and is not handled here; use dyrad.norm for it so the constants
    stay in one file.

    A log-native sensor (Boreas) declares its rendered measurement in LOG_CEILING;
    metrics that are defined on linear power (CFAR point clouds, Doppler pooling,
    the capped Pearson rho) call convert(..., LOG_CEILING, CEILING_UNITS) first.
    """
    src, dst = Domain(src), Domain(dst)
    if src is dst:
        return arr
    is_torch = hasattr(arr, "detach") and hasattr(arr, "clamp")

    def _log10(x):
        if is_torch:
            return x.clamp(min=_LOG_FLOOR).log10()
        return np.log10(np.maximum(np.asarray(x, np.float64), _LOG_FLOOR))

    def _pow10(x):
        return (10.0**x) if is_torch else np.power(10.0, np.asarray(x, np.float64))

    def _need_hi():
        if hi is None or not np.isfinite(hi) or hi <= 0:
            raise DomainError(
                f"convert {src.value} -> {dst.value} needs the ceiling `hi` "
                f"(the run's gt_hi_power / norm.json hi), got {hi!r}"
            )
        return float(hi)

    pair = (src, dst)
    if pair == (Domain.CEILING_UNITS, Domain.LOG_CEILING):
        return _log10(arr)
    if pair == (Domain.LOG_CEILING, Domain.CEILING_UNITS):
        return _pow10(arr)
    if pair == (Domain.RAW_POWER, Domain.CEILING_UNITS):
        return arr / _need_hi()
    if pair == (Domain.CEILING_UNITS, Domain.RAW_POWER):
        return arr * _need_hi()
    if pair == (Domain.RAW_POWER, Domain.LOG_CEILING):
        return _log10(arr / _need_hi())
    if pair == (Domain.LOG_CEILING, Domain.RAW_POWER):
        return _pow10(arr) * _need_hi()
    raise DomainError(
        f"no conversion {src.value} -> {dst.value} in domains.convert "
        f"(N01 conversions live in dyrad.norm, they need norm.json)"
    )
