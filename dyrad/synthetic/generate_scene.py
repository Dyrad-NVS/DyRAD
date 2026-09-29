"""Synthetic RADIal-like sequence generator (command line).

Renders a scene of point reflectors (scene_spec.py, scenes/*.yaml) along an ego
trajectory (trajectory.py) with the synthetic sensor (sensor.py: grid, PSFs, clutter,
frame renderer) and writes a directory laid out exactly like a processed RADIal
sequence: rad_tensors/ ([D=16, R=512, A=751] linear amplitude, DC at Doppler bin 0),
poses/ + poses_can/, range_bins_m.npy, az_bins_deg.npy, doppler_bins_mps.npy (DC at
bin 0), rd_doppler_bins_mps.npy (256-bin (arange-128)*0.1123), ego_vel_can.npy [N,2]
sensor-frame ego velocity, label_index_remap.npy (identity), labels_CVPR.csv and
scene_gt.json (the reflectors). No LiDAR cloud is generated; the method is
initialized from the radar pseudo-LiDAR cloud (dyrad.preprocessing.build_init_cloud).

Off-path views are produced by re-running the generator on the same scene and seed
with --ego-lateral-offset / --ego-yaw-deg (see generate_offpath_views.py). Nothing
view-dependent consumes the RNG, so the reflectors are identical across views and so
is the clutter speckle, index for index; only the ego path changes.

Usage:
    python -m dyrad.synthetic.generate_scene \\
        --out data/synthetic/curve_ramp/base --scene curve_ramp \\
        [--num-frames 40] [--seed 42]
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from dyrad.constants import DOPPLER_BIN_MPS
from dyrad.labels import REMAP_FILENAME
from dyrad.poses import invert_c2w
from dyrad.synthetic import scene_spec
from dyrad.synthetic.scene_spec import reflector_pos, reflector_vel
from dyrad.synthetic.sensor import (
    NUM_CHIRPS,
    NUM_RANGE_BINS,
    compute_radial_velocity,
    make_radial_params,
    render_frame,
)
from dyrad.synthetic.trajectory import compute_ego_vel_sensor, ego_vel_world, generate_ego_poses

# The benchmark's sequence length, frame interval (s) and scene seed.
NUM_FRAMES = 40
DT = 0.1
SEED = 42

# Labels are written for vehicles whose centroid is at 10-90 m and inside the FOV.
LABEL_R_MIN_M = 10.0
LABEL_R_MAX_M = 90.0


# ── Ground truth metadata ─────────────────────────────────────────────────────


def build_scene_gt(reflectors: list, v_ego: float, num_frames: int) -> dict:
    """The scene's world-frame reflectors (with the sequence length, frame interval
    and base ego speed), written as scene_gt.json for benchmark users."""
    ref_list = []
    for ref in reflectors:
        entry = {
            "label": ref["label"],
            "type": ref["type"],
            "p0_world": ref["p0_world"].tolist(),
            "velocity": ref["velocity"].tolist(),
            "rcs": ref["rcs"],
            "wall_normal": ref["wall_normal"],
            "specularity": ref["specularity"],
        }
        if ref["accel"] is not None:
            entry["accel"] = np.asarray(ref["accel"], dtype=np.float64).tolist()
        ref_list.append(entry)

    return {"num_frames": num_frames, "dt": DT, "v_ego": v_ego, "reflectors": ref_list}


# ── Label generation (RADIal labels_CVPR.csv format) ─────────────────────────


def generate_labels_csv(
    reflectors: list, poses: np.ndarray, radar_params: dict, seq_name: str
) -> list:
    """Generate per-frame per-vehicle labels in RADIal labels_CVPR.csv format.

    Columns (the real RADIal CSV's, plus `vehicle_id`):
      index, numSample, x1_pix, y1_pix, x2_pix, y2_pix,
      laser_X_m, laser_Y_m, laser_Z_m,
      radar_X_m, radar_Y_m, radar_R_m, radar_A_deg, radar_D_mps,
      dataset, dataset_index, difficult, vehicle_id
    (RADIal's radar_P_db column is not written.)

    Notes:
      - index = numSample = dataset_index = frame index (single sequence)
      - x1/y1/x2/y2 = 0 (no camera images in synthetic)
      - laser_X/Y/Z = centroid position in radar frame (=sensor frame)
      - radar_D_mps = the 256-bin FFT index of the receding-positive radial
        velocity, like real RADIal (itself aliased beyond +-14.37 m/s)
      - dataset = seq_name; vehicle_id = the car's scene label
      - difficult = 0 (all easy)
      - a vehicle is labelled while its centroid is at LABEL_R_MIN_M-LABEL_R_MAX_M
        and inside the azimuth FOV
    """
    az_max_rad = float(radar_params["az_rad"][-1])

    # Group reflectors by vehicle label — dynamic objects only
    vehicle_groups = {}
    for ref in reflectors:
        if ref["type"] != "dynamic":
            continue
        vehicle_groups.setdefault(ref["label"], []).append(ref)

    rows = []
    v_ego_all = ego_vel_world(np.asarray(poses), DT)
    for frame_idx, c2w in enumerate(poses):
        R_w2c, t_w2c = invert_c2w(c2w)
        v_ego_world = v_ego_all[frame_idx]

        for label, pts in vehicle_groups.items():
            # Compute centroid of this vehicle's points in radar frame at this time
            positions_r = []
            for ref in pts:
                p_world = reflector_pos(ref, frame_idx, DT)
                p_r = R_w2c @ p_world + t_w2c
                positions_r.append(p_r)

            centroid_r = np.mean(positions_r, axis=0)  # [3] in radar frame
            r_m = float(np.linalg.norm(centroid_r))
            if r_m < LABEL_R_MIN_M or r_m > LABEL_R_MAX_M:
                continue

            # atan2(+y, x), as the renderer (see sensor.render_frame)
            az_rad = float(math.atan2(centroid_r[1], centroid_r[0]))
            az_deg = float(math.degrees(az_rad))
            if abs(az_rad) > az_max_rad:
                continue

            # Doppler for vehicle centroid (receding-positive, real RADIal convention).
            v_obj = reflector_vel(pts[0], frame_idx, DT)
            v_dop = compute_radial_velocity(centroid_r, v_obj, v_ego_world, R_w2c)
            # radar_D_mps = 256-bin FFT index of v_dop, like real RADIal labels: it is
            # decoded as rd_doppler_bins_mps[(raw + 128) % 256], so it wraps beyond
            # +-14.37 m/s (the 16-bin tensor wraps at +-0.8984 m/s).
            radar_D_raw = int(round(v_dop / DOPPLER_BIN_MPS)) % NUM_CHIRPS

            rows.append(
                {
                    "index": frame_idx,
                    "numSample": frame_idx,
                    "x1_pix": 0,
                    "y1_pix": 0,
                    "x2_pix": 0,
                    "y2_pix": 0,
                    "laser_X_m": float(centroid_r[0]),
                    "laser_Y_m": float(centroid_r[1]),
                    "laser_Z_m": float(centroid_r[2]),
                    "radar_X_m": float(centroid_r[0]),
                    "radar_Y_m": float(centroid_r[1]),
                    "radar_R_m": r_m,
                    "radar_A_deg": az_deg,
                    "radar_D_mps": radar_D_raw,  # 256-bin FFT index
                    "dataset": seq_name,
                    "dataset_index": frame_idx,
                    "difficult": 0,
                    "vehicle_id": label,
                }
            )

    return rows


def save_labels_csv(out_dir: Path, rows: list) -> None:
    if not rows:
        print("  [Labels] No visible vehicle detections — labels_CVPR.csv will be empty")
    path = out_dir / "labels_CVPR.csv"
    fieldnames = [
        "index",
        "numSample",
        "x1_pix",
        "y1_pix",
        "x2_pix",
        "y2_pix",
        "laser_X_m",
        "laser_Y_m",
        "laser_Z_m",
        "radar_X_m",
        "radar_Y_m",
        "radar_R_m",
        "radar_A_deg",
        "radar_D_mps",
        "dataset",
        "dataset_index",
        "difficult",
        "vehicle_id",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [Labels] {len(rows)} detections → {path}")


# ── Save ─────────────────────────────────────────────────────────────────────


def save_sequence(
    out_dir: Path,
    rad_list: list,
    poses: np.ndarray,
    ego_vel_sensor: np.ndarray,
    scene_gt: dict,
    radar_params: dict,
    label_rows: list,
) -> None:
    rad_dir = out_dir / "rad_tensors"
    pose_dir = out_dir / "poses"
    posec_dir = out_dir / "poses_can"
    rad_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    posec_dir.mkdir(parents=True, exist_ok=True)

    # Save the full range axis (R bins), matching real RADIal sequence dirs; the
    # trainer applies range_crop_first/last as on real data.
    print(f"Saving {len(rad_list)} RAD tensors (full range R={NUM_RANGE_BINS}) → {rad_dir}")
    for i, rad in enumerate(rad_list):
        np.save(rad_dir / f"rad_{i:05d}.npy", rad.astype(np.float32))  # [D, R, A]

    meta = {
        "num_frames": len(rad_list),
        "dt": scene_gt["dt"],
        "v_ego": scene_gt["v_ego"],
        "tesseract_indices": list(range(len(rad_list))),  # required by RadarParser
    }
    # Synthetic poses are exact, so poses/ == poses_can/ (configs read
    # radar_poses_dir -> .../poses_can).
    for d in (pose_dir, posec_dir):
        np.save(d / "radar_poses.npy", poses.astype(np.float64))
        with open(d / "poses_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    # Save the full range axis (b * DR), as real range_bins_m.npy
    np.save(out_dir / "range_bins_m.npy", radar_params["range_centers"].astype(np.float32))
    np.save(out_dir / "az_bins_deg.npy", radar_params["az_deg"].astype(np.float32))
    np.save(out_dir / "doppler_bins_mps.npy", radar_params["doppler_centers"].astype(np.float32))
    np.save(out_dir / "rd_doppler_bins_mps.npy", radar_params["rd_doppler"].astype(np.float64))

    # CAN-style sensor-frame ego velocity (config ego_vel_npy → this file).
    np.save(out_dir / "ego_vel_can.npy", ego_vel_sensor.astype(np.float32))

    # Identity label index remap (synthetic has no SyncReader-tolerance mismatch).
    np.save(out_dir / REMAP_FILENAME, np.arange(len(rad_list), dtype=np.int64))

    with open(out_dir / "scene_gt.json", "w") as f:
        json.dump(scene_gt, f, indent=2)

    save_labels_csv(out_dir, label_rows)

    print(f"  RAD shape          : {rad_list[0].shape}")
    print(f"  Poses shape        : {poses.shape}  (poses/ and poses_can/)")
    print(f"  ego_vel_can.npy    : {ego_vel_sensor.shape}  (sensor-frame m/s)")
    print(f"  Output             : {out_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        required=True,
        help="sequence directory to write (rad_tensors/, poses/, ...)",
    )
    ap.add_argument(
        "--scene",
        required=True,
        help="scene to generate: a spec name under dyrad/synthetic/scenes/ (or a .yaml path)",
    )
    ap.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--ego-lateral-offset",
        type=float,
        default=0.0,
        help="novel-view lateral lane shift in +Y metres (e.g. 1.75 half-lane, 3.5 full)",
    )
    ap.add_argument(
        "--ego-yaw-deg",
        type=float,
        default=0.0,
        help="novel-view sensor yaw (deg); the vehicle still follows the scene's path",
    )
    args = ap.parse_args()

    # The scene spec sets the base ego path (speed, turn rate, longitudinal accel).
    spec = scene_spec.load_spec(args.scene)
    ego_path = scene_spec.ego_params(spec)
    v_ego = ego_path["v_ego"]

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    radar_params = make_radial_params()

    # ── Scene ───────────────────────────────────────────────────────────────
    print(f"Building scene: '{args.scene}'...")
    if (spec.get("description") or "").strip():
        print(f"  {' '.join(spec['description'].split())}")
    reflectors = scene_spec.build_scene(spec, rng, v_ego=v_ego, dt=DT, num_frames=args.num_frames)
    dyn_refs = [r for r in reflectors if r["type"] == "dynamic"]
    stat_refs = [r for r in reflectors if r["type"] == "static"]
    print(f"  Static reflectors  : {len(stat_refs)}")
    print(f"  Dynamic reflectors : {len(dyn_refs)}")

    # ── Ego poses (optionally off-path for novel-view GT) ─────────────────────
    poses = generate_ego_poses(
        args.num_frames,
        DT,
        v_ego,
        lateral_offset=args.ego_lateral_offset,
        yaw_deg=args.ego_yaw_deg,
        yaw_rate_deg_s=ego_path["yaw_rate_deg_s"],
        accel_mps2=ego_path["accel_mps2"],
    )
    if ego_path["yaw_rate_deg_s"] or ego_path["accel_mps2"]:
        print(
            f"\n  [Base path] yaw_rate={ego_path['yaw_rate_deg_s']:+.2f} deg/s  "
            f"accel={ego_path['accel_mps2']:+.2f} m/s^2  "
            f"(scene-defined, not a novel-view offset)"
        )
    if args.ego_lateral_offset or args.ego_yaw_deg:
        print(
            f"\n  [Novel-view trajectory] lateral={args.ego_lateral_offset:+.2f} m  "
            f"yaw={args.ego_yaw_deg:+.1f}°"
        )

    # ── Render all frames ─────────────────────────────────────────────────────
    print(f"\nRendering {args.num_frames} frames...")
    v_ego_all = ego_vel_world(poses, DT)
    rad_list = []
    for i in range(args.num_frames):
        if i % 20 == 0:
            print(f"  frame {i:3d}/{args.num_frames}")
        rad_list.append(render_frame(reflectors, poses[i], v_ego_all[i], radar_params, i, DT, rng))

    # ── Ego velocity (CAN-style, sensor frame) ──────────────────────────────
    ego_vel_sensor = compute_ego_vel_sensor(poses, DT)
    print(
        f"\nEgo velocity (sensor frame): {ego_vel_sensor.shape}  "
        f"frame0 = [{ego_vel_sensor[0, 0]:.2f}, {ego_vel_sensor[0, 1]:.2f}] m/s"
    )

    # ── Labels CSV ───────────────────────────────────────────────────────────
    seq_name = out_dir.name
    print(f"\nGenerating labels_CVPR.csv (sequence '{seq_name}')...")
    label_rows = generate_labels_csv(reflectors, poses, radar_params, seq_name)
    veh_frames = len(set(r["numSample"] for r in label_rows))
    print(f"  {len(label_rows)} vehicle detections across {veh_frames} frames")
    if dyn_refs and not label_rows:
        # A scene with movers but no labels would train without dynamic supervision
        # and score dynamic-region metrics over an empty mask; fail instead.
        raise SystemExit(
            f"[FATAL] scene '{args.scene}' has {len(dyn_refs)} dynamic reflectors but "
            "produced NO labels. Every mover is outside the labelling window "
            "(range 10-90 m or the azimuth FOV) for all frames -- move them closer "
            "to the ego path, or shorten/lengthen the sequence."
        )

    # ── Scene GT & save ───────────────────────────────────────────────────────
    scene_gt = build_scene_gt(reflectors, v_ego, args.num_frames)
    save_sequence(out_dir, rad_list, poses, ego_vel_sensor, scene_gt, radar_params, label_rows)
    print(f"[synthetic] wrote {out_dir}")


if __name__ == "__main__":
    main()
