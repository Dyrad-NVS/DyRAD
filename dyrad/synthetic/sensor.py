"""The synthetic RADIal sensor: grid constants, bin axes, PSFs, clutter and the frame renderer.

Tensors are [D=16, R=512, A=751] linear AMPLITUDE (RADIal stores |beamform|, never
squared). Doppler is the DDMA-reduced 16-bin axis at 0.1123 m/s/bin (unambiguous
period 1.7968 m/s), not ego-compensated, receding-positive (RADIal's forward-FFT
convention; the renderer matches it with `doppler_sign: -1`), with DC at bin 0 (the
trainer rolls it by doppler_roll_bins=8).

Forward model per reflector and frame (`render_frame`):
  - amplitude = SIGNAL_SCALE * beam_gain(az) * specular_factor * rcs / r^2;
  - spread into the cube by three amplitude-domain PSFs: the Doppler and range
    responses are |DFT| of the Hamming window (the same lobe in bin units), and
    azimuth is a Hann-windowed |sinc| that broadens as 1/cos(az), approximating the
    192-element virtual array;
  - placement: each reflector gets its exact radial velocity, placed at 1/N_FRAC-bin
    resolution on the circular Doppler axis (one Hamming lobe, no DDMA sidelobe
    structure); range and azimuth snap to the nearest bin centre;
  - log-normal, spatially correlated background clutter calibrated to real RADIal
    tensor statistics is added per frame.

At 0.1123 m/s/bin essentially every return in a driving scene (static structure at
ego speed, and every vehicle) lies outside the +-0.8984 m/s unambiguous window, so
the information is in the WRAPPED bin, as on the real sensor.

The grid, Doppler axis and PSF widths must equal the training recipe
(configs/recipes/radial.yaml): the renderer takes them from the config, not from the
saved axis files.
"""

import functools
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from dyrad.axes import wrap_bins_signed
from dyrad.constants import DOPPLER_BIN_MPS, RADIAL_RANGE_MAX_M
from dyrad.poses import invert_c2w
from dyrad.synthetic.scene_spec import reflector_pos, reflector_vel

# ── RADIal hardware constants ────────────────────────────────────────────────

NUM_RANGE_BINS = 512
NUM_DOPPLER_BINS = 16
NUM_AZ_BINS = 751
NUM_CHIRPS = 256

RANGE_MAX_M = RADIAL_RANGE_MAX_M
DR = RANGE_MAX_M / NUM_RANGE_BINS  # 0.2012 m/bin
# DDMA Doppler: 16 bins of DOPPLER_BIN_MPS -> unambiguous period 1.7968 m/s, window
# [DOPPLER_MIN, DOPPLER_MAX] = [-0.8984, +0.7861] m/s, i.e. [-period/2, period/2), so
# wrapping a velocity into it is `axes.wrap_bins_signed(v, DOPPLER_PERIOD_MPS)`.
DOPPLER_PERIOD_MPS = float(NUM_DOPPLER_BINS) * DOPPLER_BIN_MPS
DOPPLER_MIN = -DOPPLER_PERIOD_MPS / 2.0
DOPPLER_MAX = DOPPLER_MIN + (NUM_DOPPLER_BINS - 1) * DOPPLER_BIN_MPS
AZ_MIN_DEG = -75.0
AZ_MAX_DEG = +75.0

# Hamming window coefficients (0.54 - 0.46 cos) for the Doppler and range FFTs
HAMMING_D = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(NUM_CHIRPS) / (NUM_CHIRPS - 1))
HAMMING_R = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(NUM_RANGE_BINS) / (NUM_RANGE_BINS - 1))

# Background statistics of real RADIal tensors (all Doppler bins): log10 mean ~5.14,
# std ~0.39, median linear ~1.6e5. The level is matched; the std is set slightly
# lower (0.30) so the +3 sigma clutter tail stays below dim static structure
# (guardrail rails), keeping the radar-cloud initialization clean.
BG_LOG10_MEAN = 5.14  # diffuse ground/road clutter + thermal floor
BG_LOG10_STD = 0.30  # speckle variation
# Correlation lengths (Gaussian sigma, bins) of the clutter speckle in range and
# azimuth: the scale of one resolution cell, so the speckle is blobby, not per-bin.
BG_CORR_R_BINS = 2.0
BG_CORR_A_BINS = 8.0
# Far-range rolloff: from this range bin on (~90.5 m, near the far edge of the rendered
# reflectors) the clutter decays as exp(-(r - r_start) / BG_FAR_DECAY_M).
BG_FAR_START_BIN = 450
BG_FAR_DECAY_M = 10.0

# Signal scale, calibrated so the ratio of the normalisation ceiling (`hi`) to the
# median RA level matches real RADIal (~32, range 31-34 over the evaluation
# sequences): that ratio is the dynamic range the model is trained on.
SIGNAL_SCALE = 7.5e9

# PSF discretisation: Doppler table resolution (fractional-bin subdivisions),
# range kernel support in bins (odd), and the zero-padding factor of the |DFT| used
# to sample the Doppler and range lobes at fractional offsets.
N_FRAC = 64
PSF_KR = 9
PSF_OVERSAMPLE = 1024
# Azimuth PSF: kernel support in bins (odd) and first-zero width w0 at boresight
# (amplitude FWHM 1.207 * 21.44 ~ 26 bins, close to RADIal's ~24-bin beam).
AZ_PSF_K = 73
AZ_PSF_W0_BINS = 21.44
# Azimuth beam taper: the azimuth (deg) at which the one-way Gaussian gain is 0.5.
BEAM_AZ_DEG = 70.0
# Range falloff of the returned amplitude, rcs / r^RANGE_EXPONENT.
RANGE_EXPONENT = 2.0

# Visibility window of the renderer: 10 m <= r <= the centre of the last range bin the
# trainer keeps (bin 461, 92.74 m, before the recipe's range_crop_last = 50 bins). The
# near field is excluded because 1/r^2 would swamp the scene there.
RANGE_CROP_LAST = 50
RENDER_R_MIN_M = 10.0
RENDER_R_MAX_M = (NUM_RANGE_BINS - 1 - RANGE_CROP_LAST) * DR


# ── Bin arrays ───────────────────────────────────────────────────────────────


def doppler_axis_dc_at_bin0() -> np.ndarray:
    """16-bin DDMA Doppler axis with DC at bin 0, as the real doppler_bins_mps.npy.

    bins_mps[k] = (((k + 8) % 16) - 8) * DOPPLER_BIN_MPS
      -> [0, .1123, ..., 0.7861, -0.8984, -0.7861, ..., -.1123]
    Velocities are receding-positive (real RADIal forward-FFT; renderer doppler_sign=-1).
    The trainer rolls the GT tensor by doppler_roll_bins=8 to its DC-at-bin-8 axis.
    """
    return (
        (
            ((np.arange(NUM_DOPPLER_BINS) + NUM_DOPPLER_BINS // 2) % NUM_DOPPLER_BINS)
            - NUM_DOPPLER_BINS // 2
        )
        * DOPPLER_BIN_MPS
    ).astype(np.float32)


def rd_doppler_axis() -> np.ndarray:
    """Full NUM_CHIRPS-bin FFT Doppler axis (m/s), as the real rd_doppler_bins_mps.npy:
    (arange(256) - 128) * 0.1123 -> [-14.374, ..., +14.262]."""
    return (np.arange(NUM_CHIRPS, dtype=np.float64) - NUM_CHIRPS // 2) * DOPPLER_BIN_MPS


def make_radial_params() -> dict:
    """RADIal bin arrays at full resolution; render_frame uses them and
    save_sequence writes them unchanged (the trainer crops range itself)."""
    # Range bin b sits at b * DR, the FFT convention real RADIal follows and the trainer
    # reads (range_bin_offset 0 in recipes/radial.yaml). Targets snap to the nearest bin
    # centre in range and azimuth; only Doppler is placed fractionally (see render_frame).
    az_centers_deg = np.linspace(AZ_MIN_DEG, AZ_MAX_DEG, NUM_AZ_BINS, dtype=np.float32)
    return {
        "range_centers": np.arange(NUM_RANGE_BINS, dtype=np.float32) * DR,  # [R] m
        "az_deg": az_centers_deg,
        "az_rad": np.deg2rad(az_centers_deg),
        "doppler_centers": doppler_axis_dc_at_bin0(),  # [16] DC-at-bin0 wrap axis
        "rd_doppler": rd_doppler_axis(),  # [256] full-FFT axis
    }


# ── Doppler PSF ──────────────────────────────────────────────────────────────


@functools.cache
def _doppler_lobe() -> np.ndarray:
    """|DFT| of the zero-padded 256-sample Hamming Doppler window, fftshifted."""
    w_pad = np.zeros(NUM_CHIRPS * PSF_OVERSAMPLE)
    w_pad[:NUM_CHIRPS] = HAMMING_D
    return np.abs(np.fft.fftshift(np.fft.fft(w_pad)))


def compute_doppler_psf_row(b_cont: float) -> np.ndarray:
    """Amplitude weights for all 16 reduced-Doppler bins, for a target at fractional bin b_cont.

    The reduced-Doppler response of the RADIal chain is |DFT| of the 256-sample Hamming
    Doppler window -- the same construction as range_psf_kernel with N = NUM_CHIRPS.
    In bin units the Hamming response does not depend on the transform length, so the
    Doppler and range lobes coincide. (In DDMA the demultiplexer gathers
    rd_spectra[:, (d + o_j) mod 256, :], undoing the per-transmitter Doppler shifts o_j,
    so the transmit offsets do not appear in the response.)

    delta is wrapped to [-8, 8) because the reduced axis is circular with period
    NUM_DOPPLER_BINS (the sensor's unambiguous interval), as in the CUDA rasterizer.

    Amplitude domain (GT = |DFT|, never squared), max-normalised to 1 so SIGNAL_SCALE
    controls the peak value directly -- the same contract as the range kernel.
    """
    amp = _doppler_lobe()
    c, N_pad = len(amp) // 2, len(amp)
    psf = np.zeros(NUM_DOPPLER_BINS)
    for d in range(NUM_DOPPLER_BINS):
        delta = float(d) - b_cont
        delta = (delta + NUM_DOPPLER_BINS / 2) % NUM_DOPPLER_BINS - NUM_DOPPLER_BINS / 2
        idx = int(round(c + delta * PSF_OVERSAMPLE))
        psf[d] = amp[idx] if 0 <= idx < N_pad else 0.0
    peak = psf.max()
    if peak > 0:
        psf /= peak
    return psf


@functools.cache
def doppler_psf_table() -> np.ndarray:
    """[N_FRAC, 16] Doppler PSF (the 256-point Hamming-FFT lobe) at fractional offsets.

    Row i is the response for a target at fractional bin offset i / N_FRAC.
    See compute_doppler_psf_row.
    """
    table = np.zeros((N_FRAC, NUM_DOPPLER_BINS), dtype=np.float32)
    for i in range(N_FRAC):
        table[i] = compute_doppler_psf_row(i / float(N_FRAC))
    return table


# ── Range PSF (|DFT| of the Hamming window) ─────────────────────────────────


@functools.cache
def range_psf_kernel() -> np.ndarray:
    """1D range PSF kernel of PSF_KR taps: |DFT| of the Hamming range-compression window.

    Amplitude domain (GT = |DFT|, never squared); amplitude half-max FWHM ~1.8 bins.
    The DFT is oversampled and sampled at integer offsets to get the discrete kernel.
    Max-normalised to 1, so the peak pixel is SIGNAL_SCALE / r^2.
    """
    N_pad = NUM_RANGE_BINS * PSF_OVERSAMPLE
    w_pad = np.zeros(N_pad)
    w_pad[:NUM_RANGE_BINS] = HAMMING_R
    psf_amp = np.abs(np.fft.fftshift(np.fft.fft(w_pad)))
    half = PSF_KR // 2
    indices = N_pad // 2 + np.arange(-half, half + 1) * PSF_OVERSAMPLE
    kernel = np.maximum(psf_amp[indices], 0.0)
    return (kernel / kernel.max()).astype(np.float64)


# ── Azimuth PSF (amplitude sinc-Hann for 192-element virtual aperture) ───────
# The azimuth beam broadens off boresight (aperture foreshortening): RADIal's
# amplitude half-max FWHM is ~24 bins at 0 deg and ~57 bins at 65 deg, roughly
# 1/cos(az). We model an amplitude |sinc| (not sinc^2), whose FWHM is 1.207 w,
# and broaden w by 1/cos(az).

# Clamp cos(az) so the edge of the +-75 deg FOV does not widen the kernel past its
# support.
_AZ_COS_MIN = math.cos(math.radians(70.0))


def _sinc_hann_1d(k: int, w: float) -> np.ndarray:
    """Hann-windowed |sinc| (amplitude) kernel of length k (odd), first zero at w bins.

    Amplitude half-max FWHM = 1.207 w (the power sinc^2 FWHM is 0.886 w), so
    w = AZ_PSF_W0_BINS = 21.44 gives FWHM ~26 bins, close to RADIal's ~24-bin beam.
    """
    assert k % 2 == 1 and k >= 3
    x = np.arange(k, dtype=np.float64) - (k // 2)
    t = x / max(float(w), 1e-6)
    s = np.abs(np.sinc(t))  # amplitude (not squared)
    hann = np.hanning(k)
    kernel = np.maximum(s * hann, 0.0)
    return (kernel / kernel.max()).astype(np.float64)  # max=1


@functools.cache
def az_psf_kernel(az_deg: int) -> np.ndarray:
    """Azimuth PSF at integer azimuth `az_deg`.

    The real beam broadens off boresight ~1/cos(az) (aperture foreshortening), so the
    first-zero width is w(az) = AZ_PSF_W0_BINS / cos(az), with cos clamped at 70 deg.
    """
    c = max(abs(math.cos(math.radians(az_deg))), _AZ_COS_MIN)
    return _sinc_hann_1d(AZ_PSF_K, AZ_PSF_W0_BINS / c)


# ── Physics helpers ───────────────────────────────────────────────────────────


def compute_radial_velocity(
    p_r: np.ndarray,  # position in radar frame [3]
    v_obj_world: np.ndarray,
    v_ego_world: np.ndarray,
    R_w2c: np.ndarray,
) -> float:
    """Signed radial velocity. Positive = receding (range increasing).

    v_r = dot(R_w2c @ (v_obj - v_ego),  p_r / ||p_r||)

    Range rate is already the RADIal Doppler convention (receding-positive, the
    forward-FFT sign; the renderer matches it with `doppler_sign: -1`).
    """
    r = float(np.linalg.norm(p_r))
    if r < 1e-6:
        return 0.0
    v_rel_r = R_w2c @ (v_obj_world - v_ego_world)
    return float(np.dot(v_rel_r, p_r / r))


def doppler_to_bin_cont(v_doppler: float) -> float:
    """Continuous Doppler bin (DC at bin 0, circular over 16 bins) for a
    receding-positive velocity (real RADIal convention).  bin = (v / DOPPLER_BIN_MPS) mod 16.
      v=0 -> 0,  v=+0.7861 -> 7,  v=-0.8984 -> 8,  v=-0.1123 -> 15.
    """
    return float((v_doppler / DOPPLER_BIN_MPS) % NUM_DOPPLER_BINS)


# ── Background clutter model ──────────────────────────────────────────────────


def generate_background(
    rng: np.random.Generator,
    *,
    range_centers: np.ndarray,
    az_rad: np.ndarray,
    v_ego_sensor: np.ndarray,
) -> np.ndarray:
    """Spatially-correlated diffuse background clutter [D, R, A].

    Real radar clutter is NOT per-bin-independent (that reads as fine-grained TV
    static); the speckle is correlated over a resolution cell (the PSF), and the
    diffuse floor decays with range.  Model:
      1. white Gaussian field -> smoothed over (R, A) by the PSF-scale correlation
         lengths (BG_CORR_R_BINS, BG_CORR_A_BINS) -> renormalised to unit variance,
         so the speckle is coarse/blobby like real clutter, not pixel static;
      2. log-normal level: log10 = BG_LOG10_MEAN + BG_LOG10_STD * correlated_field;
      3. a mild Doppler bump (d_factor) at the wrapped Doppler of static structure
         for this frame's ego velocity, per azimuth;
      4. a gentle far-range rolloff from BG_FAR_START_BIN; the level is otherwise
         flat in range.

    Returns linear amplitude [D, R, A].
    """
    D, R, A = NUM_DOPPLER_BINS, NUM_RANGE_BINS, NUM_AZ_BINS
    corr_r_bins, corr_a_bins = BG_CORR_R_BINS, BG_CORR_A_BINS

    # 1. Correlated white field (smoothed over R, A only; Doppler stays independent).
    # The field is drawn on a grid padded by 4 sigma in R and A, smoothed, then cropped,
    # so every retained bin has full smoothing support. Smoothing the bare field with
    # mode="nearest" would leave edge bins with more variance, which the log-normal map
    # turns into excess power at the azimuth edges (real RADIal edges are slightly dimmer).
    pad_r, pad_a = int(np.ceil(4.0 * corr_r_bins)), int(np.ceil(4.0 * corr_a_bins))
    white = rng.normal(0.0, 1.0, size=(D, R + 2 * pad_r, A + 2 * pad_a))
    field = gaussian_filter(white, sigma=(0.0, corr_r_bins, corr_a_bins), mode="nearest")
    field = field[:, pad_r : pad_r + R, pad_a : pad_a + A]
    field /= field.std() + 1e-9  # restore unit variance after smoothing

    # 2. Log-normal level from the correlated field.
    bg = np.power(10.0, BG_LOG10_MEAN + BG_LOG10_STD * field).astype(np.float64)

    # 3. Mild Doppler dependence: slightly brighter near the static Doppler, i.e. the
    # receding-positive radial velocity -v_ego . u(az) of a stationary point.
    d_centers = doppler_axis_dc_at_bin0()  # [16] DC at bin 0
    v_static = -(v_ego_sensor[0] * np.cos(az_rad) + v_ego_sensor[1] * np.sin(az_rad))  # [A]
    d_dist = np.abs(
        wrap_bins_signed(d_centers[:, None] - v_static[None, :], DOPPLER_PERIOD_MPS)
    )  # [D, A]
    # Real static clutter at highway speed spreads across most Doppler bins, so the
    # off-band floor is high (0.5) with only a slight peak near the static band.
    d_factor = 0.5 + 0.5 * np.exp(-0.5 * (d_dist / (1.5 * DOPPLER_BIN_MPS)) ** 2)  # [D, A]
    bg *= d_factor[:, None, :]

    # 4. Far-range rolloff only. The level is kept roughly flat over the in-scene range
    # so the radar-cloud init's median-relative threshold sits above the clutter.
    r_factor = np.ones(R, dtype=np.float64)
    far_start = BG_FAR_START_BIN
    r_factor[far_start:] = np.exp(
        -(range_centers[far_start:] - range_centers[far_start]) / BG_FAR_DECAY_M
    )
    bg *= r_factor[None, :, None]

    return bg.astype(np.float32)


# ── Spread reflector into RAD tensor ─────────────────────────────────────────


def _spread_reflector(
    rad: np.ndarray,
    amp: float,
    b_cont: float,  # fractional Doppler bin
    r_bin: int,
    az_bin: int,
    psf_az: np.ndarray,  # [kA] azimuth PSF kernel
) -> None:
    """Spread one reflector's amplitude into rad[D, R, A] using the three PSF kernels.

    Pipeline (linear amplitude domain; GT = |beamform|):
      1. Doppler PSF: doppler_psf_table, placed circularly
      2. Range |DFT(Hamming)|, spread over +-PSF_KR//2 bins
      3. Azimuth |sinc| * Hann (1/cos-broadened), +-kA//2 bins
    """
    D, R, A = rad.shape
    psf_r = range_psf_kernel()
    hrR = len(psf_r) // 2
    hrA = len(psf_az) // 2

    # ── Doppler weights: tabulated PSF over all Doppler bins, placed circularly ──
    # b_cont on the 1/N_FRAC grid; a fraction that rounds up to a whole bin carries.
    q = int(round(b_cont * N_FRAC))
    d_weights = np.roll(doppler_psf_table()[q % N_FRAC], (q // N_FRAC) % D)  # [16]

    # ── Range and azimuth taps inside the grid (the kernel is zero-padded) ───
    ri = r_bin + np.arange(-hrR, hrR + 1)  # [kR]
    ai = az_bin + np.arange(-hrA, hrA + 1)  # [kA]
    keep_r = (ri >= 0) & (ri < R)
    keep_a = (ai >= 0) & (ai < A)
    ri, ai = ri[keep_r], ai[keep_a]
    kernel_ra = np.outer(psf_r[keep_r], psf_az[keep_a])  # [kR', kA']

    # ── Accumulate: outer product over D, R, A ───────────────────────────────
    for d in range(D):
        if d_weights[d] < 1e-6:
            continue
        rad[d, ri[:, None], ai[None, :]] += amp * d_weights[d] * kernel_ra


# ── Frame renderer ────────────────────────────────────────────────────────────


def render_frame(
    reflectors: list,
    c2w: np.ndarray,
    v_ego_world: np.ndarray,
    radar_params: dict,
    frame_idx: int,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render one frame: reflectors -> amplitude PSF spread -> background clutter.

    Doppler is raw (NOT ego-compensated) and wrapped circularly. Each reflector snaps
    to the nearest range and azimuth bin centre; its Doppler is placed fractionally.
    The azimuth PSF is chosen per reflector (1/cos broadening) at its integer degree.

    `v_ego_world` is the frame's ego velocity from trajectory.ego_vel_world, the same
    central difference that ego_vel_can.npy and the labels carry, so the trainer's
    measured ego velocity is the one the ground truth was rendered with.
    """
    r_bins = radar_params["range_centers"]  # [512] m
    az_rad = radar_params["az_rad"]  # [751] rad

    rad = np.zeros((NUM_DOPPLER_BINS, NUM_RANGE_BINS, NUM_AZ_BINS), dtype=np.float64)

    R_w2c, t_w2c = invert_c2w(c2w)

    az_max_rad = float(az_rad[-1])  # +75 deg in radians

    # Azimuth beam taper (monostatic gain^2)
    sigma_az = np.deg2rad(BEAM_AZ_DEG) / math.sqrt(2.0 * math.log(2.0))

    for ref in reflectors:
        p_world = reflector_pos(ref, frame_idx, dt)
        v_obj = reflector_vel(ref, frame_idx, dt)

        # Transform to radar frame
        p_r = R_w2c @ p_world + t_w2c

        r = float(np.linalg.norm(p_r))
        if r < RENDER_R_MIN_M or r > RENDER_R_MAX_M:
            continue

        # atan2(+y, x), matching the CUDA projector and RADIal
        az = float(math.atan2(p_r[1], p_r[0]))
        if abs(az) > az_max_rad:
            continue

        # View-dependent specular factor
        wall_normal = ref["wall_normal"]
        specularity = ref["specularity"]
        if wall_normal is not None and specularity > 0.0:
            n_world = np.array(wall_normal, dtype=np.float64)
            n_r = R_w2c @ n_world
            look_from_ref = -p_r / r
            cos_th = max(0.0, float(np.dot(look_from_ref, n_r)))
            spec_factor = cos_th**specularity
        else:
            spec_factor = 1.0

        if spec_factor < 1e-4:
            continue

        # Azimuth beam gain (monostatic: G_tx x G_rx = G^2)
        g = math.exp(-0.5 * (az / sigma_az) ** 2)
        beam_gain = g * g

        # Peak amplitude of this reflector's return
        amp = (
            SIGNAL_SCALE
            * beam_gain
            * spec_factor
            * float(ref["rcs"])
            / max(r, 1.0) ** RANGE_EXPONENT
        )

        if amp < 1e-6:
            continue

        # Doppler: receding-positive radial velocity -> circular bin (real RADIal
        # convention); doppler_to_bin_cont's modulo handles the wrap.
        v_dop = compute_radial_velocity(p_r, v_obj, v_ego_world, R_w2c)
        b_cont = doppler_to_bin_cont(v_dop)

        # Nearest range and azimuth bin centres
        r_bin = int(np.argmin(np.abs(r_bins - r)))
        az_bin = int(np.argmin(np.abs(az_rad - az)))

        _spread_reflector(
            rad, amp, b_cont, r_bin, az_bin, az_psf_kernel(int(round(math.degrees(az))))
        )

    # Background clutter (log-normal, independent per frame)
    bg = generate_background(
        rng, range_centers=r_bins, az_rad=az_rad, v_ego_sensor=R_w2c @ v_ego_world
    )
    rad += bg.astype(np.float64)

    return np.maximum(rad, 0.0).astype(np.float32)
