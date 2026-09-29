"""Estimate per-frame ego velocity from CAN Vehicle_Speed for a RADIal sequence.

CAN Vehicle_Speed (ID=0x3E9) provides wheel odometry at ~75 Hz with ±0.1 km/h
accuracy; it is decoded with the RADIal DBC and interpolated to the radar frame times.

For a non-slipping rigid car, the velocity in sensor/body frame is always forward:
    v_sensor = [speed_mps, 0.0]
It assumes no lateral slip and needs only the CAN speed (no heading estimate).

Output: ego_vel_can.npy  float32 [N, 2] (sensor-frame vx=forward, vy=0)
        index-aligned to radar_poses.npy / timestamps_us.npy.

Usage:
  python -m dyrad.preprocessing.radial ego-velocity \\
    --recording  data/radial_raw/RECORD@2020-11-22_12.31.22 \\
    --timestamps data/radial_processed/seq_31_22/poses/timestamps_us.npy \\
    --out-dir    data/radial_processed/seq_31_22
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dyrad.preprocessing.radial import RADIAL_DBC, RADIAL_DBREADER


# ─────────────────────────────────────────────────────────────────────────────
# CAN parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_rec_can_events(rec_file: Path) -> dict:
    """Parse _events_log.rec → dict suitable for CANReader constructor."""
    labls = [str(i) for i in range(22)]
    df_rec = pd.read_csv(str(rec_file), header=None, names=labls, sep="[-|\\s+]", engine="python")
    df_ev = df_rec.iloc[:, range(1, df_rec.shape[1], 4)]
    df_ev.columns = ["timestamp", "timeofissue", "data_sample", "sensor", "offset", "datasize"]
    df_can = df_ev[df_ev["sensor"] == "can"].reset_index(drop=True)
    return {
        "timestamp": df_can["timestamp"].astype(int).tolist(),
        "timeofissue": df_can["timeofissue"].astype(int).tolist(),
        "sample": df_can["data_sample"].astype(int).tolist(),
        "offset": df_can["offset"].astype(int).tolist(),
        "datasize": df_can["datasize"].astype(int).tolist(),
    }


def extract_can_speed(
    recording_dir: Path,
    dbc_path: Path = RADIAL_DBC,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_us, speed_mps) from CAN Vehicle_Speed at ~75 Hz.

    Timestamps are hardware-clock microseconds, the same clock used by
    poses/timestamps_us.npy, so interpolation is direct.
    """
    import cantools

    dbc = cantools.database.load_file(str(dbc_path))
    SPEED_ID = dbc.get_message_by_name("Vehicle_Speed").frame_id  # 0x3E9

    sys.path.insert(0, str(RADIAL_DBREADER))
    from DBReader import CANReader

    can_file = next(recording_dir.glob("*_can.bin"), None)
    rec_file = next(recording_dir.glob("*_events_log.rec"), None)
    if can_file is None:
        raise RuntimeError(f"No *_can.bin found in {recording_dir}")
    if rec_file is None:
        raise RuntimeError(f"No *_events_log.rec found in {recording_dir}")

    can_dict = _parse_rec_can_events(rec_file)
    can_dict["filename"] = str(can_file)
    can_reader = CANReader(can_dict)

    ts_list: list[float] = []
    spd_list: list[float] = []

    for i in range(len(can_dict["timestamp"])):
        for frame in can_reader.GetData(i):
            if int(frame["ID"]) != SPEED_ID:
                continue
            decoded = dbc.decode_message(int(frame["ID"]), bytes(frame["DATA"]))
            spd_kph = float(decoded["Speed_kph"])
            ts_list.append(float(frame["timestamp"]))
            spd_list.append(spd_kph / 3.6)

    if not ts_list:
        raise RuntimeError("No Vehicle_Speed frames found in CAN stream.")

    ts = np.array(ts_list, dtype=np.float64)
    spd = np.array(spd_list, dtype=np.float32)
    rate = 1e6 / float(np.diff(ts).mean()) if len(ts) > 1 else 0.0
    print(
        f"[ego-velocity] {len(ts)} Vehicle_Speed samples, "
        f"rate≈{rate:.1f} Hz, "
        f"speed=[{spd.min():.2f}, {spd.max():.2f}] m/s"
    )
    return ts, spd


# ─────────────────────────────────────────────────────────────────────────────
# Main method
# ─────────────────────────────────────────────────────────────────────────────


def can_ego_velocity(
    recording_dir: Path,
    timestamps_us: np.ndarray,  # [N] µs, radar frame hardware-clock timestamps
    dbc_path: Path = RADIAL_DBC,
) -> np.ndarray:
    """Return [N, 2] float32 sensor-frame (vx, vy) m/s from CAN wheel speed.

    For a rigid car (Ackermann steering, no wheel slip), the velocity in the
    body/sensor frame is always along the forward axis:
        v_sensor = [speed, 0]

    CAN timestamps are in the same hardware clock as timestamps_us, so
    interpolation is direct (no clock alignment needed).
    """
    ts_can, spd_can = extract_can_speed(recording_dir, dbc_path)

    t_radar = timestamps_us.astype(np.float64)
    spd_interp = np.interp(t_radar, ts_can, spd_can).astype(np.float32)

    v_ego = np.zeros((len(timestamps_us), 2), dtype=np.float32)
    v_ego[:, 0] = spd_interp  # vx = forward speed; vy = 0 (no lateral slip)

    spd_abs = np.abs(spd_interp)
    print(
        f"[ego-velocity] Interpolated to {len(timestamps_us)} radar frames: "
        f"speed=[{spd_abs.min():.2f}, {spd_abs.max():.2f}] m/s "
        f"(mean {spd_abs.mean():.2f} m/s)"
    )
    return v_ego


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--recording",
        required=True,
        type=Path,
        help="Raw RECORD@... directory (contains *_can.bin)",
    )
    p.add_argument(
        "--timestamps",
        required=True,
        type=Path,
        help="poses/timestamps_us.npy for the preprocessed sequence",
    )
    p.add_argument(
        "--out-dir", required=True, type=Path, help="Output directory for ego_vel_can.npy"
    )
    p.add_argument("--dbc", default=RADIAL_DBC, type=Path, help="Path to can_database.dbc")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    timestamps_us = np.load(args.timestamps).astype(np.int64)
    print(f"[ego-velocity] Sequence: {len(timestamps_us)} radar frames")

    v_can = can_ego_velocity(args.recording, timestamps_us, dbc_path=args.dbc)

    out_path = args.out_dir / "ego_vel_can.npy"
    np.save(out_path, v_can)
    print(f"[ego-velocity] Saved {out_path}  shape={v_can.shape}")


if __name__ == "__main__":
    main()
