"""Object footprints in the Range-Azimuth grid, and the free space they leave.

The static cloud builder carves each frame's free space (`partition_frame`) from the same
footprint functions (`car_extent_m`, `psf_margin_bins`, `footprint_bins`) that the
trainer's dynamic seeding samples inside, so the two are complementary by construction:
every RA cell is either free space (the static pseudo-LiDAR cloud samples there) or inside
one object's footprint (that object's dynamic seeds sample there). The scorers' foreground
boxes use `car_extent_m` too.

The footprint is defined in metres and converted to bins per object, so it means the
same physical extent at every range (a fixed aperture in bins would be an angle, i.e.
a different width at every range). Masking per frame removes exactly the cells the car
occupies in that frame, so no ghost trail can form and static structure the car drives
past is kept.

Object size
-----------
From the camera bbox, per object: `W = w_px * R / fx` is the car's physical width
(median ≈ 1.94 m over the RADIal labels, stable with range). Detections without a
usable bbox fall back to `car_width_default_m`.

The partition parameters are the Config fields named in `PARTITION_IDENTITY`. A config
dict that lacks one (a raw yaml dict, as the scorers pass) takes the Config default.

The bbox gives width and height, never length along the radial direction, and RADIal
annotations carry no yaw, so the along-range extent is an assumption:
`car_len_over_width` (default 2.3) sets it. A car is ~4.6 m long and ~1.9 m wide, and in
RADIal's highway traffic the length lies roughly along range. For crossing traffic this
footprint is too deep and too narrow; set car_len_over_width to 1.0 for a square,
orientation-free footprint if a sequence is dominated by crossing traffic.

Boreas labels carry real per-object w/l and yaw, so their converter resolves the
footprint exactly and passes metres directly (see car_extent_m).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dyrad.config import Config
from dyrad.labels import MIN_BBOX_W_PX

_DEFAULTS = Config()

#: median bbox-derived car width over the RADIal labels, used when a detection has no bbox
CAR_WIDTH_DEFAULT_M = _DEFAULTS.car_width_default_m
#: lower bound on a label box's half-extent (m), so a thin or degenerate box keeps a footprint
MIN_BOX_HALF_M = 0.25
#: lower bound on a bbox-derived car width (m), against tiny or truncated camera boxes
MIN_CAR_WIDTH_M = 0.5

#: The fields that decide which cells are free. A `*_free.npy` cloud is only valid for a
#: run that shares all of them, because the static and dynamic halves are complementary by
#: construction; carved with different parameters, cells end up in both halves (ghost
#: trails) or in neither. Written beside the cloud, checked at init.
PARTITION_IDENTITY = (
    "car_width_default_m",
    "car_len_over_width",
    "car_margin_m",
    "car_psf_margin",
    "psf_w_R",
    "psf_w_A",
)


def _param(cfg, key: str) -> float:
    """A partition parameter of a Config, or of a config dict (Config default if absent)."""
    if isinstance(cfg, dict):
        return float(cfg.get(key, getattr(_DEFAULTS, key)))
    return float(getattr(cfg, key))


def identity(cfg) -> dict:
    """This config's partition identity, to compare against a cloud's sidecar.

    The PSF widths are only part of the identity when `car_psf_margin` actually uses them;
    otherwise a change to the render-side PSF would falsely invalidate every cloud.
    """
    out = {k: _param(cfg, k) for k in PARTITION_IDENTITY}
    if out["car_psf_margin"] <= 0:
        out.pop("psf_w_R"), out.pop("psf_w_A")
    return out


def load_fx(cfg, root: Path | None = None) -> float | None:
    """Camera focal length in pixels from `camera_calib_path`, or None when it is empty
    (no camera: every detection uses `car_width_default_m`).

    A path that is set but missing or unreadable raises, because falling back to the
    default width would silently change every footprint.
    """
    p = str(cfg.get("camera_calib_path", "") if isinstance(cfg, dict) else cfg.camera_calib_path)
    if not p:
        return None
    q = Path(p)
    if not q.is_absolute() and root is not None:
        q = root / q
    if not q.is_file():
        raise FileNotFoundError(f"camera_calib_path is set but {q} does not exist")
    try:
        cal = np.load(q, allow_pickle=True).item()
        return float(np.array(cal["intrinsic"]["camera_matrix"])[0, 0])
    except Exception as e:
        raise ValueError(f"cannot read the focal length from camera_calib_path {q}: {e!r}") from e


def box_half_extents(w_m: float, l_m: float, yaw_rel_rad: float) -> tuple:
    """(half across range, half along range) for a w x l box whose long axis sits at
    `yaw_rel_rad` to the radial direction: the axis-aligned bound of a rotated rectangle.

    Shared by the Boreas label converter and view re-projection, because a shifted ego
    sees the same object at a different azimuth and hence with a different profile.
    """
    ct, st = abs(np.cos(yaw_rel_rad)), abs(np.sin(yaw_rel_rad))
    return 0.5 * (l_m * st + w_m * ct), 0.5 * (l_m * ct + w_m * st)


def car_extent_m(
    bbox_w_px, R_m: float, fx: float | None, cfg, half_cross_m=None, half_range_m=None
) -> tuple:
    """(half-width across range, half-extent along range) in metres, for one detection.

    Width comes from the camera bbox; the along-range extent is width * car_len_over_width
    because a car's length is not observable from a bbox (see the module docstring).

    `car_margin_m` then dilates the box by that many metres on every side. It is added
    after the aspect scaling, so it is a true isotropic dilation: a 0.3 m margin grows the
    footprint by 0.3 m across range and 0.3 m along range.
    """
    w_def = _param(cfg, "car_width_default_m")
    aspect = _param(cfg, "car_len_over_width")
    margin = _param(cfg, "car_margin_m")
    if half_cross_m is not None and half_range_m is not None:
        # The sensor's own labels carry the box: Boreas gives real per-object w/l and
        # rot_y, so its converter resolves the footprint exactly (axis-aligned bound of
        # the rotated rectangle) and passes metres. No RADIal constant is involved, which
        # matters because on a 360° scanner crossing traffic is common.
        return (
            max(float(half_cross_m), MIN_BOX_HALF_M) + margin,
            max(float(half_range_m), MIN_BOX_HALF_M) + margin,
        )
    w = w_def
    if fx and bbox_w_px and bbox_w_px > MIN_BBOX_W_PX and R_m > 0:
        w = float(bbox_w_px) * float(R_m) / float(fx)
    w = max(w, MIN_CAR_WIDTH_M)
    return 0.5 * w + margin, 0.5 * w * aspect + margin


def psf_margin_bins(cfg) -> tuple:
    """(extra range bins, extra azimuth bins) to dilate the footprint by, from the PSF.

    A car's RA footprint is its geometry convolved with the PSF: the cells that hold its
    energy are wider than the cells it physically occupies. The geometric box alone is
    therefore too small for seeding, by a factor that grows with range, because the
    azimuth PSF is an angle (fixed in bins) while the car's angular width shrinks as 1/R.

    On RADIal (`psf_w_A=21.44`, `psf_w_R=1.47`; FWHM = 0.886*w) the azimuth beam is
    19.0 bins FWHM, i.e. +-9.5 bins, whereas a 1.94 m car subtends only +-6 bins at 50 m
    and +-3 at 100 m. Past ~30 m the geometric footprint is narrower than the beam that
    produced the return, so it would clip the object's own energy.

    `car_psf_margin` is in multiples of the PSF half-width (half-FWHM), so 1.0 means
    "cover the main lobe". 0 (the default) disables it.
    """
    k = _param(cfg, "car_psf_margin")
    if k <= 0:
        return 0.0, 0.0
    w_r = _param(cfg, "psf_w_R")
    w_a = _param(cfg, "psf_w_A")
    return 0.5 * 0.886 * w_r * k, 0.5 * 0.886 * w_a * k


def footprint_bins(
    R_m: float,
    A_deg: float,
    half_cross_m: float,
    half_range_m: float,
    range_centers: np.ndarray,
    az_deg: np.ndarray,
    pad_r_bins: float = 0.0,
    pad_a_bins: float = 0.0,
) -> tuple:
    """The object's footprint as bin index ranges (r_lo, r_hi, a_lo, a_hi), hi exclusive.

    Range is metric, so a fixed number of metres is a fixed number of bins. Azimuth is an
    angle, so the same metres subtend fewer bins the further away the object is.

    `pad_*_bins` then dilates the box in bin space (see `psf_margin_bins`): the PSF spread
    is a property of the sensor, not of the object, so it is a fixed number of bins at
    every range.
    """
    nR, nA = len(range_centers), len(az_deg)
    r0 = int(np.clip(np.argmin(np.abs(range_centers - R_m)), 0, nR - 1))
    a0 = int(np.clip(np.argmin(np.abs(az_deg - A_deg)), 0, nA - 1))
    dr = float(np.mean(np.diff(range_centers))) if nR > 1 else 1.0
    d_az = float(np.mean(np.diff(az_deg))) if nA > 1 else 1.0  # degrees per bin
    n_r = max(int(np.ceil(half_range_m / max(dr, 1e-9) + max(pad_r_bins, 0.0))), 0)
    half_az_deg = np.degrees(np.arctan2(half_cross_m, max(R_m, 1e-6)))
    n_a = max(int(np.ceil(half_az_deg / max(abs(d_az), 1e-9) + max(pad_a_bins, 0.0))), 0)
    return (max(0, r0 - n_r), min(nR, r0 + n_r + 1), max(0, a0 - n_a), min(nA, a0 + n_a + 1))


def partition_frame(
    dets: list, range_centers: np.ndarray, az_deg: np.ndarray, fx: float | None, cfg
) -> np.ndarray:
    """One frame's free-space mask: [len(range_centers), len(az_deg)] bool, False inside
    any detection's footprint.

    dets: iterable of dicts with `R_m`, `A_deg`, and optionally `w_px` (camera bbox width)
          or `half_cross_m` / `half_range_m` (metric box), as `car_extent_m` takes them.
    """
    free = np.ones((len(range_centers), len(az_deg)), bool)
    pad_r, pad_a = psf_margin_bins(cfg)
    for d in dets:
        hc, hr = car_extent_m(
            d.get("w_px"), float(d["R_m"]), fx, cfg, d.get("half_cross_m"), d.get("half_range_m")
        )
        r_lo, r_hi, a_lo, a_hi = footprint_bins(
            float(d["R_m"]), float(d["A_deg"]), hc, hr, range_centers, az_deg, pad_r, pad_a
        )
        free[r_lo:r_hi, a_lo:a_hi] = False
    return free
