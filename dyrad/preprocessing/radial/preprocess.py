"""RADIal raw recording -> RAD tensors and poses.

Converts one RADIal raw recording folder (RECORD@...) into:
    <out-dir>/
        rad_tensors/rad_{idx:05d}.npy   float32  [D=16, R=512, A=N_az]  beamformer amplitude
        poses/radar_poses.npy           float64  [N, 4, 4]              GPS translations
        poses/timestamps_us.npy         int64    [N]                    radar frame timestamps (us)
        poses/poses_metadata.json       tesseract_indices (frame id per pose)
        sensor.json                     the sensor configuration (grid, --sensor name)
        label_index_remap.npy           official label `index` -> local frame (-1 = unmapped)

The bin axes are not stored: every consumer computes them from the run config (dyrad.axes).

Signal processing (numpy only):
    ADC int16  ->  build_radar_frame  ->  range FFT  ->  Doppler FFT
    ->  DDMA demux (12 Tx)  ->  AoA beamform (CalibMat)  ->  RAD [D=16, R=512, A=N_az]

Poses: the GPS track only. Each frame's pose holds the GPS position as its translation,
ENU(east, north, 0) from the first fix, with an identity rotation;
`python -m dyrad.preprocessing.radial poses` then builds the radar poses (poses_can/) from
this track and the CAN wheel speed. The GPS fixes carry no usable time stamp, so they are
spread uniformly over the recording by line index and interpolated to the radar frames.

Doppler axis: the stored axis is the DDMA-folded one, bin d = raw 256-bin FFT bin d
(mod 16), bin 0 = DC, DOPPLER_BIN_MPS per bin, wrap period 16 bins = 1.797 m/s.

Sensor configuration (--sensor): `native` is the standard RADIal cube; `coarse` keeps
128 of the 512 ADC samples per chirp and 128 of the 256 chirps (4x coarser range, 2x
coarser Doppler, same azimuth grid and Doppler wrap period), for the sensor-configuration
transfer experiment. The configuration is applied to the time-domain samples before any
transform, so the coarse cube is a different measurement of the same scene, not a
resampled native cube.

Usage:
    python -m dyrad.preprocessing.radial preprocess \\
        --seq-dir  data/radial_raw/RECORD@2020-11-22_12.31.22 \\
        --out-dir  data/radial_processed/seq_31_22 \\
        [--sensor coarse] [--max-frames N]
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from dyrad.constants import DOPPLER_BIN_MPS, EL_IDX, RADIAL_RANGE_MAX_M
from dyrad.domains import Domain, save_rad
from dyrad.labels import REMAP_FILENAME
from dyrad.preprocessing.radial import RADIAL_CALIB, RADIAL_DBREADER

# ───────────────────────────────────────────────────────────────────────────
# Radar parameters (RADIal hardware)
# ───────────────────────────────────────────────────────────────────────────
NUM_SAMPLES = 512  # range samples per chirp
NUM_CHIRPS = 256  # chirps per frame
NUM_RX_CHIP = 4  # Rx antennas per chip
NUM_RX = 16  # total Rx antennas
NUM_TX = 12  # Tx antennas (MIMO)
NUM_VIRTUAL = 192  # NUM_RX x NUM_TX
NUM_REDUCED_D = 16  # unambiguous Doppler bins after MIMO
NUM_CHIRPS_PER_LOOP = 16

# SyncReader tolerance of the RADIal authors' generate_database.py: their labels'
# 'index' column refers to the frame table built with THIS tolerance.
OFFICIAL_SYNC_TOLERANCE_US = 20000
# Tolerance used for our own frame table (keeps more frames).
OUR_SYNC_TOLERANCE_US = 200000

# MIMO Tx-slot offsets: 12 Tx antennas spread over 16-chirp slots [0, 16, 32, ..., 176],
# of which the RADIal chain keeps slot 0 and slots 5.. (12 Tx).
_FULL_OFFSETS = np.arange(0, NUM_REDUCED_D * NUM_CHIRPS_PER_LOOP, NUM_REDUCED_D)  # [16]
TX_OFFSETS = np.concatenate([[_FULL_OFFSETS[0]], _FULL_OFFSETS[5:]])  # [12]


# ───────────────────────────────────────────────────────────────────────────
# Sensor configuration
# ───────────────────────────────────────────────────────────────────────────
# The constants above describe the RADIal hardware as the authors process it. A sensor
# configuration is the same raw ADC turned into a RAD cube from fewer time-domain samples:
#
#   n_samples  kept ADC samples per chirp  -> swept bandwidth -> range resolution
#              (max unambiguous range is unchanged: dr scales as 1/n_samples)
#   n_chirps   kept chirps per frame       -> coherent integration time -> Doppler
#              resolution (the DDMA wrap period is unchanged)
#
# Nothing else about the sensor is varied: same carrier, platform, antenna array and
# illumination.


@dataclass(frozen=True)
class SensorConfig:
    """One sensor configuration: how the raw ADC is turned into a RAD cube."""

    name: str = "native"
    n_samples: int = NUM_SAMPLES
    n_chirps: int = NUM_CHIRPS

    # ── derived ────────────────────────────────────────────────────────────
    @property
    def tx_offsets(self) -> np.ndarray:
        """Tx Doppler slot offsets expressed in THIS configuration's Doppler FFT bins."""
        return (TX_OFFSETS * self.n_chirps) // NUM_CHIRPS

    @property
    def n_reduced_d(self) -> int:
        """Unambiguous Doppler bins = minimum circular gap between Tx slots (16 natively)."""
        o = np.sort(self.tx_offsets % self.n_chirps)
        gaps = np.diff(np.concatenate([o, [o[0] + self.n_chirps]]))
        return int(gaps.min())

    @property
    def n_range_bins(self) -> int:
        return self.n_samples

    @property
    def dr_m(self) -> float:
        """Range bin spacing (m). Max range is fixed by the hardware."""
        return RADIAL_RANGE_MAX_M / self.n_range_bins

    @property
    def doppler_bin_mps(self) -> float:
        """Doppler bin width (m/s). Widens as the CPI shortens."""
        return DOPPLER_BIN_MPS * NUM_CHIRPS / self.n_chirps

    @property
    def unambiguous_mps(self) -> float:
        return self.n_reduced_d * self.doppler_bin_mps

    def validate(self) -> None:
        if not (0 < self.n_samples <= NUM_SAMPLES):
            raise ValueError(f"n_samples must be in (0, {NUM_SAMPLES}], got {self.n_samples}")
        if not (0 < self.n_chirps <= NUM_CHIRPS):
            raise ValueError(f"n_chirps must be in (0, {NUM_CHIRPS}], got {self.n_chirps}")
        bad = (TX_OFFSETS * self.n_chirps) % NUM_CHIRPS
        if np.any(bad != 0):
            raise ValueError(
                f"n_chirps={self.n_chirps} does not keep the Tx Doppler offsets on "
                f"integer bins. Use a multiple of {NUM_CHIRPS // NUM_REDUCED_D}."
            )
        off = self.tx_offsets % self.n_chirps
        if len(np.unique(off)) != len(off):
            raise ValueError(
                f"Tx Doppler offsets collide mod n_chirps ({off.tolist()}): the DDMA "
                f"demux is not invertible for this configuration."
            )

    def describe(self) -> str:
        return (
            f"[Sensor:{self.name}] {self.n_reduced_d}D x {self.n_range_bins}R "
            f"| {NUM_TX}Tx x {NUM_RX}Rx = {NUM_VIRTUAL} virtual "
            f"| dr={self.dr_m:.4f} m/bin | dv={self.doppler_bin_mps:.4f} m/s/bin "
            f"| unambiguous +-{self.unambiguous_mps / 2:.2f} m/s"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_chirps": self.n_chirps,
            "n_tx": NUM_TX,
            "n_rx": NUM_RX,
            "n_virtual": NUM_VIRTUAL,
            "n_range_bins": self.n_range_bins,
            "n_doppler_bins": self.n_reduced_d,
            "range_bin_m": self.dr_m,
            "range_max_m": RADIAL_RANGE_MAX_M,
            "doppler_bin_mps": self.doppler_bin_mps,
            "unambiguous_mps": self.unambiguous_mps,
            "tx_offsets": self.tx_offsets.tolist(),
        }


#: --sensor presets. The sensor name is written to sensor.json and names the transfer-evaluation outputs (`coarse_to_native`).
SENSOR_PRESETS = {
    "native": SensorConfig("native"),
    "coarse": SensorConfig("coarse", n_samples=128, n_chirps=128),
}


# ───────────────────────────────────────────────────────────────────────────
# Signal processing
# ───────────────────────────────────────────────────────────────────────────


def build_radar_frame(
    adc0: np.ndarray, adc1: np.ndarray, adc2: np.ndarray, adc3: np.ndarray
) -> np.ndarray:
    """Decode 4 ADC chip streams -> complex frame [512, 256, 16].

    Each chip yields int16 samples interleaved as real, imag.
    Output: [NUM_SAMPLES, NUM_CHIRPS, NUM_RX]  complex64
    Chip order follows the RADIal reference processing: [frame3, frame0, frame1, frame2].
    """

    def _decode(adc):
        c = adc[0::2].astype(np.float32) + 1j * adc[1::2].astype(np.float32)
        return c.reshape(NUM_SAMPLES, NUM_RX_CHIP, NUM_CHIRPS, order="F").transpose(0, 2, 1)
        # -> [512, 256, 4]

    return np.concatenate(
        [_decode(adc3), _decode(adc0), _decode(adc1), _decode(adc2)], axis=2
    ).astype(np.complex64)
    # -> [512, 256, 16]


def _window(n: int) -> np.ndarray:
    """The Hamming window of the RADIal reference processing, at length n (cached)."""
    w = _WINDOW_CACHE.get(n)
    if w is None:
        w = 0.54 - 0.46 * np.cos(2 * math.pi * np.arange(n) / (n - 1))
        _WINDOW_CACHE[n] = w
    return w


_WINDOW_CACHE: dict = {}


def adc_to_rd_spectra(complex_frame: np.ndarray, sc: SensorConfig = SensorConfig()) -> np.ndarray:
    """Range + Doppler FFTs with Hamming windowing and DC removal.

    Input:  [512, 256, 16]        complex64  (always the full decoded frame)
    Output: [n_samples, n_chirps, 16]  complex64  (Range-Doppler per Rx)

    The sensor configuration is applied here, on the time-domain samples, before any
    transform: truncating ADC samples reduces the swept bandwidth (coarser range) and
    truncating chirps shortens the CPI (coarser Doppler). Cropping after the FFT would
    only resample the native cube, not produce a different measurement.
    """
    frame = complex_frame[: sc.n_samples, : sc.n_chirps, :]
    # Slicing can return a non-contiguous view, which changes numpy's pairwise-summation
    # order inside .mean() below by one float32 ulp. Force the native layout so the
    # output does not depend on the layout (a no-op when already contiguous).
    frame = np.ascontiguousarray(frame)
    frame = frame - frame.mean(axis=(0, 1), keepdims=True)  # DC removal
    frame = frame * _window(sc.n_samples)[:, None, None]  # range window
    rd = np.fft.fft(frame, n=sc.n_range_bins, axis=0)
    rd = rd * _window(sc.n_chirps)[None, :, None]  # Doppler window
    rd = np.fft.fft(rd, n=sc.n_chirps, axis=1)
    return rd.astype(np.complex64)


def rd_to_rad(
    rd_spectra: np.ndarray,
    calib_mat: np.ndarray,
    hamming_win: np.ndarray,
    sc: SensorConfig = SensorConfig(),
) -> np.ndarray:
    """MIMO reorder + AoA beamforming -> RAD tensor.

    For each of the 16 unambiguous Doppler bins, extract the 12 Tx-chirp slots,
    build the 192-virtual-antenna MIMO spectrum, and apply the AoA calibration
    matrix to get the azimuth power spectrum.

    Inputs:
        rd_spectra:  [R, n_chirps, 16]  complex64  (Range-Doppler x Rx)
        calib_mat:   [N_az, 192]        complex64  (AoA beamforming)
        hamming_win: [192]              float64    (Hamming window)
        sc:          the sensor configuration

    Output: [n_reduced_d, R, N_az]  float32  beamformer amplitude |CalibMat @ x|
    """
    N_az = calib_mat.shape[0]
    rad = np.zeros((sc.n_reduced_d, sc.n_range_bins, N_az), dtype=np.float32)
    tx_offs = sc.tx_offsets  # [12]

    for d in range(sc.n_reduced_d):
        # Chirp indices for the Tx slots at reduced Doppler bin d
        chirp_slots = np.remainder(d + tx_offs, sc.n_chirps).astype(int)  # [12]

        # Extract and reshape: [R, 12, 16] -> [R, 192]  (tx-major, the CalibMat column order)
        mimo = rd_spectra[:, chirp_slots, :]  # [R, 12, 16]
        mimo = mimo.reshape(sc.n_range_bins, NUM_VIRTUAL)  # [R, 192]

        # Apply Hamming window and beamform: CalibMat [N_az, V] @ mimo.T [V, R]
        mimo_win = mimo * hamming_win  # [R, 192]
        az_spec = np.abs(calib_mat @ mimo_win.T)  # [N_az, R]  float32

        rad[d] = az_spec.T  # [R, N_az]

    return rad  # [n_reduced_d, R, N_az]


# ───────────────────────────────────────────────────────────────────────────
# GPS parsing (NMEA GGA / RMC)
# ───────────────────────────────────────────────────────────────────────────


def _parse_nmea_latlon(sentence: str):
    """Parse GPGGA or GPRMC sentence -> (lat_deg, lon_deg, alt_m) or None."""
    parts = sentence.strip().split(",")
    try:
        if parts[0] in ("$GPGGA", "$GNGGA"):
            # $GPGGA,time,DDMM.MMMM,N,DDDMM.MMMM,E,fix,sats,hdop,alt,...
            lat_raw, lat_ns = parts[2], parts[3]
            lon_raw, lon_ew = parts[4], parts[5]
            alt = float(parts[9]) if parts[9] else 0.0
        elif parts[0] in ("$GPRMC", "$GNRMC"):
            # $GPRMC,time,status,DDMM.MMMM,N,DDDMM.MMMM,E,...
            lat_raw, lat_ns = parts[3], parts[4]
            lon_raw, lon_ew = parts[5], parts[6]
            alt = 0.0
        else:
            return None

        if not lat_raw or not lon_raw:
            return None

        # DDMM.MMMM -> decimal degrees
        lat_dd = int(float(lat_raw) / 100)
        lat = lat_dd + (float(lat_raw) - lat_dd * 100) / 60.0
        if lat_ns == "S":
            lat = -lat

        lon_dd = int(float(lon_raw) / 100)
        lon = lon_dd + (float(lon_raw) - lon_dd * 100) / 60.0
        if lon_ew == "W":
            lon = -lon

        return lat, lon, alt
    except Exception:
        return None


def wgs84_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """Convert WGS84 (lat, lon, alt) to ENU offset from (lat0, lon0, alt0)."""
    R_EARTH = 6378137.0
    d_lat = math.radians(lat - lat0)
    d_lon = math.radians(lon - lon0)
    cos_lat0 = math.cos(math.radians(lat0))
    east = R_EARTH * cos_lat0 * d_lon
    north = R_EARTH * d_lat
    up = alt - alt0
    return east, north, up


def read_gps_lines(seq_dir: Path):
    """Find the GPS text file in a raw recording folder and return all lines.

    Handles extensions: .gps, .txt, .ascii, or no extension.
    RADIal raw recordings use _gps.ascii naming.
    """
    _GPS_EXTS = {".txt", ".gps", ".ascii", ""}
    gps_files = [
        f for f in seq_dir.iterdir() if "gps" in f.name.lower() and f.suffix.lower() in _GPS_EXTS
    ]

    if not gps_files:
        return []
    gps_file = gps_files[0]
    print(f"[radial] GPS file: {gps_file.name}")
    with open(gps_file, "r", errors="replace") as fd:
        lines = fd.readlines()
    return lines


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seq-dir", required=True, help="raw recording folder (RECORD@...)")
    ap.add_argument(
        "--calib", type=Path, default=RADIAL_CALIB, help="path to CalibrationTable.npy"
    )
    ap.add_argument("--out-dir", required=True, help="output sequence directory")
    ap.add_argument(
        "--az-min",
        type=float,
        default=-75.0,
        help="min azimuth to keep in degrees (default: -75, full CalibMat FOV)",
    )
    ap.add_argument(
        "--az-max",
        type=float,
        default=75.0,
        help="max azimuth to keep in degrees (default: +75, full CalibMat FOV)",
    )
    ap.add_argument(
        "--sensor",
        default="native",
        choices=sorted(SENSOR_PRESETS),
        help="sensor configuration: native, or coarse (128 of 512 ADC samples, 128 of 256 chirps)",
    )
    ap.add_argument(
        "--max-frames", type=int, default=0, help="stop after this many frames (0 = all)"
    )
    args = ap.parse_args()

    # DBReader ships with the RADIal repository clone (see the README).
    sys.path.insert(0, str(RADIAL_DBREADER))
    from DBReader import SyncReader

    seq_dir = Path(args.seq_dir)
    out_dir = Path(args.out_dir)
    calib_path = args.calib

    # ── Sensor configuration ───────────────────────────────────────────────
    sc = SENSOR_PRESETS[args.sensor]
    sc.validate()
    print(f"[radial] {sc.describe()}")

    # ── Load calibration ───────────────────────────────────────────────────
    print(f"[radial] Loading {calib_path}")
    calib = np.load(calib_path, allow_pickle=True).item()
    az_table = calib["Azimuth_table"].astype(np.float64)  # [751] degrees
    CalibFull = calib["Signal"][..., EL_IDX]  # [751, 192] complex64
    hamming = calib["H"][0].astype(np.float64)  # [192]

    # Crop azimuth range (+0.01 deg epsilon handles floating-point edges, e.g. 75.0000000000213)
    az_mask = (az_table >= args.az_min - 0.01) & (az_table <= args.az_max + 0.01)
    az_idxs = np.where(az_mask)[0]  # sorted indices
    az_bins = az_table[az_idxs]  # [N_az] degrees
    CalibMat = CalibFull[az_idxs, :].astype(np.complex64)  # [N_az, 192]
    N_az = len(az_bins)
    print(
        f"[radial] Azimuth crop [{args.az_min}, {args.az_max}] deg -> {N_az} bins "
        f"({az_bins[0]:.2f} to {az_bins[-1]:.2f} deg, res={az_bins[1] - az_bins[0]:.3f} deg/bin)"
    )

    # ── Doppler bin centres (m/s): DDMA as-stored axis ────────────────────
    # Stored tensor bin d = raw 256-bin FFT bin d (mod 16): bin 0 = DC (0 m/s),
    # bins 1..7 positive, bins 8..15 alias to negative velocities (wrap period
    # 16 * DOPPLER_BIN_MPS = 1.797 m/s). The trainer rolls GT by doppler_roll_bins=8 for a
    # monotonic axis.
    d_idx = np.arange(sc.n_reduced_d)
    doppler_bins_mps = (
        np.where(d_idx < sc.n_reduced_d // 2, d_idx, d_idx - sc.n_reduced_d) * sc.doppler_bin_mps
    ).astype(np.float32)
    print(
        f"[radial] Doppler: {sc.n_reduced_d} bins, {doppler_bins_mps.min():.2f} to "
        f"{doppler_bins_mps.max():.2f} m/s  (dv={sc.doppler_bin_mps} m/s/bin, "
        f"bin0=DC, wrap at bin {sc.n_reduced_d // 2})"
    )

    # ── Output directories ─────────────────────────────────────────────────
    rad_dir = out_dir / "rad_tensors"
    pose_dir = out_dir / "poses"
    rad_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    # Declare the sensor configuration next to the data it produced, so downstream code
    # reads it instead of re-deriving it.
    with open(out_dir / "sensor.json", "w") as f:
        json.dump(
            {
                **sc.to_dict(),
                "az_min_deg": float(az_bins[0]),
                "az_max_deg": float(az_bins[-1]),
                "num_az_bins": int(N_az),
                "raw_seq": str(seq_dir.name),
            },
            f,
            indent=2,
        )

    # ── Open DBReader ──────────────────────────────────────────────────────
    print(f"[radial] Opening {seq_dir}")
    db = SyncReader(str(seq_dir), tolerance=OUR_SYNC_TOLERANCE_US, silent=True)

    # ── Drop sync rows referencing a truncated sensor stream ───────────────
    # If a recording is cut mid-write, a sensor's metadata can have one more
    # timestamp than byte-offset entry (e.g. radar_ch1: timestamp=489, offset=488).
    # The sync table matches frames by argmin over the *timestamp* array, so a row
    # can point at an index that is out of range for that sensor's *offset* array
    # -> IndexError in GetSensorData. Prune such rows (only check sensors that use
    # the byte-offset array, i.e. offset length > 0; GPS is line-based, len 0).
    off_len = {s: len(d.get("offset", [])) for s, d in db.dicts.items()}
    keep = [
        n
        for n, row in enumerate(db.table)
        if all(idx < off_len[s] for s, idx in row.items() if off_len.get(s, 0) > 0)
    ]
    if len(keep) < len(db.table):
        print(
            f"[radial] WARNING: dropping {len(db.table) - len(keep)} sync frame(s) referencing "
            f"a truncated sensor stream (out-of-range byte offset)"
        )
        db.table = db.table[np.asarray(keep, dtype=np.int64)]

    N_frames = len(db) if args.max_frames == 0 else min(len(db), args.max_frames)
    print(f"[radial] {len(db)} synchronized frames  (processing {N_frames})")

    # ── Label index remap (official tolerance=20000 table -> ours) ─────────
    # labels_CVPR.csv 'index' refers to the authors' frame table; ours keeps
    # more frames. remap[official_index] = our local frame (-1 = unmapped).
    db_official = SyncReader(str(seq_dir), tolerance=OFFICIAL_SYNC_TOLERANCE_US, silent=True)
    pos_ours = {m["radar_ch3"]: i for i, m in enumerate(db.table)}
    remap = np.array([pos_ours.get(m["radar_ch3"], -1) for m in db_official.table], dtype=np.int64)
    np.save(out_dir / REMAP_FILENAME, remap)
    shifts = remap[remap >= 0] - np.arange(len(remap))[remap >= 0]
    print(
        f"[radial] {REMAP_FILENAME} saved: official N={len(remap)}, "
        f"kept N={len(db)}, shift {shifts.min()}..{shifts.max()}"
    )

    # ── GPS: read all text lines and parse lat/lon ─────────────────────────
    gps_lines = read_gps_lines(seq_dir)
    gps_fixes = []  # list of (line_idx, lat, lon, alt)
    for i, line in enumerate(gps_lines):
        fix = _parse_nmea_latlon(line)
        if fix is not None:
            gps_fixes.append((i, *fix))
    if len(gps_fixes) < 2:
        raise SystemExit(
            f"[radial] {seq_dir.name}: {len(gps_fixes)} valid GPS fix(es); the radar poses "
            f"(`radial poses`) are built along the GPS track, which needs at least 2"
        )
    _, lat0, lon0, alt0 = gps_fixes[0]
    print(f"[radial] GPS: {len(gps_fixes)} valid fixes. Origin: lat={lat0:.6f} lon={lon0:.6f}")

    # ── Radar frame timestamps (from the reader's index, no sensor data read) ─
    radar_ch0 = db.readers["radar_ch0"]
    timestamps_us = np.array(
        [int(radar_ch0.GetTimestamp(db.table[i]["radar_ch0"])) for i in range(N_frames)],
        dtype=np.int64,
    )
    t0_us = timestamps_us[0]
    dt_us = np.diff(timestamps_us)  # [N-1] microseconds between frames
    frame_rate_hz = 1e6 / np.median(dt_us)
    print(
        f"[radial] Frame rate ~ {frame_rate_hz:.2f} Hz  "
        f"(median dt = {np.median(dt_us) / 1e3:.2f} ms)"
    )

    # ── Interpolate GPS fixes to radar timestamps ──────────────────────────
    # GPS timestamps are not exposed by the reader, so the line index is used as a
    # proxy: GPS fixes (~1 Hz) are distributed uniformly over the sequence duration.
    gps_t_frac = np.array([f[0] / max(len(gps_lines) - 1, 1) for f in gps_fixes])
    gps_lat = np.array([f[1] for f in gps_fixes])
    gps_lon = np.array([f[2] for f in gps_fixes])
    gps_alt = np.array([f[3] for f in gps_fixes])
    radar_frac = (timestamps_us - t0_us) / max(timestamps_us[-1] - t0_us, 1)

    lat_interp = np.interp(radar_frac, gps_t_frac, gps_lat)
    lon_interp = np.interp(radar_frac, gps_t_frac, gps_lon)
    alt_interp = np.interp(radar_frac, gps_t_frac, gps_alt)

    # ── GPS poses [N, 4, 4]: ENU translation, identity rotation ────────────
    # GPS mixes GPRMC (no altitude) and GPGGA (real altitude) sentences, so the
    # interpolated up component would alternate between 0 and ~192 m; motion is kept 2D.
    print("[radial] Building GPS poses...")
    poses = np.tile(np.eye(4), (N_frames, 1, 1))
    for i in range(N_frames):
        e, n, _ = wgs84_to_enu(lat_interp[i], lon_interp[i], alt_interp[i], lat0, lon0, alt0)
        poses[i, :3, 3] = [e, n, 0.0]

    np.save(pose_dir / "radar_poses.npy", poses)
    np.save(pose_dir / "timestamps_us.npy", timestamps_us)
    print(
        f"[radial] Saved poses: {poses.shape}  "
        f"trajectory span: {np.linalg.norm(poses[-1, :3, 3] - poses[0, :3, 3]):.1f} m"
    )

    with open(pose_dir / "poses_metadata.json", "w") as f:
        json.dump({"tesseract_indices": list(range(N_frames))}, f, indent=2)
    print(
        f"[radial] Saved poses_metadata.json ({N_frames} frames, "
        f"tesseract_indices = 0..{N_frames - 1})"
    )

    # ── Second pass: RAD tensors ───────────────────────────────────────────
    print(f"[radial] Processing {N_frames} frames...")
    for i in tqdm(range(N_frames), desc="frames"):
        data = db.GetSensorData(i)
        adc0 = data["radar_ch0"]["data"]
        adc1 = data["radar_ch1"]["data"]
        adc2 = data["radar_ch2"]["data"]
        adc3 = data["radar_ch3"]["data"]

        frame = build_radar_frame(adc0, adc1, adc2, adc3)  # [512,256,16]
        rd = adc_to_rd_spectra(frame, sc)  # [R,C,16]
        rad = rd_to_rad(rd, CalibMat, hamming, sc)  # [D,R,N_az]

        rad_path = rad_dir / f"rad_{i:05d}.npy"
        # The on-disk convention every loader assumes: sensor-native and UNSCALED (no
        # ceiling normalization, no log map). Hard check: this is the root sequence
        # everything else is derived from.
        save_rad(
            rad_path,
            rad.astype(np.float32),
            Domain.RAW_POWER,
            note="radial preprocess: RADIal ADC -> RD -> RAD, sensor-native",
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"[radial] Output: {out_dir}")
    print(f"  RAD tensors : {N_frames} x [D={sc.n_reduced_d}, R={sc.n_range_bins}, A={N_az}]")
    print(f"  Poses       : {poses.shape}")
    print(f"  Radar specs : {sc.describe()}")
    print(
        f"    Range  : {sc.n_range_bins} bins, {sc.dr_m:.4f} m/bin "
        f"({sc.n_samples} swept samples), max {RADIAL_RANGE_MAX_M} m"
    )
    print(
        f"    Azimuth: {N_az} bins, {az_bins[1] - az_bins[0]:.3f} deg/bin, "
        f"[{az_bins[0]:.1f}, {az_bins[-1]:.1f}] deg"
    )
    print(
        f"    Doppler: {sc.n_reduced_d} bins, "
        f"[{doppler_bins_mps.min():.2f}, {doppler_bins_mps.max():.2f}] m/s "
        f"(unambiguous +-{sc.unambiguous_mps / 2:.2f})"
    )
    print("[radial] Next steps:")
    print(
        "  1. Prepare a training window: python -m dyrad.preprocessing.radial prepare "
        f"--seq {out_dir.name} --frame-start <F>"
    )
    print("  2. Train: python -m dyrad.train --config configs/radial/<ID>_dyrad.yaml")


if __name__ == "__main__":
    main()
