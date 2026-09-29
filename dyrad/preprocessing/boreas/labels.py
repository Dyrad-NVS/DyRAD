"""Convert Boreas `boreas-objects-v1` tracking labels -> a RADIal-style label CSV.

Input : <boreas-objects>/labels_tracking/<lidar_t_us>.txt
        uuid type l w h x y z rot_y numPoints      (x,y,z in the LIDAR frame)
        (length, width, height; Car medians ~ 4.49 / 2.06 / 1.71 m)
Output: labels_boreas.csv, the label CSV the trainer reads (`object_label_path`).

Association
-----------
Boreas gives a `uuid` that is consistent across frames, so it is emitted as
`vehicle_id`. The trainer skips its nearest-neighbour label tracker whenever that
column exists, so the association here is ground truth.

Only moving objects are emitted (--min-speed)
---------------------------------------------
A labelled clip contains hundreds of tracks of which most are parked cars. Those are
static scene structure and belong in the static reflector cloud, exactly like a
wall; handing each one a rigid track would spend dynamic capacity modelling
things that do not move. Speed is measured in the world frame over the whole
track (median per-step), so it is immune to the ego's own motion.

Frame / azimuth convention
--------------------------
Object position is taken to the radar frame with the calib extrinsic alone,
p_radar = T_radar_lidar @ p_lidar, with no pose involved: label frames are
within ~0.1 s of a radar frame, so the ego moves <1 m, negligible against the
6-75 m label ranges, and this keeps the mapping independent of any pose
convention.

    radar_A_deg = degrees(atan2(y_radar, x_radar))
    laser_X_m, laser_Y_m = R*cos(radar_A_deg), R*sin(radar_A_deg)

The trainer feeds laser_X/Y through the pose c2w to get world positions, so labels, poses
and the tensor azimuth axis must all sit in one frame: the calib radar frame, which is also
the azimuth axis of `boreas convert`.

No Doppler: Navtech does not measure it, so no `radar_D_mps` column is written
(the trainer defaults radar_D_raw to -1).

Run:
    python -m dyrad.preprocessing.boreas labels \\
        --clip-meta data/boreas/_staging/objects_win55_104/clip_metadata.json \\
        --out data/boreas_processed/boreas_obj_win55_104/labels_boreas.csv \\
        --dataset BOREAS-OBJ-WIN55-104
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

from dyrad import ra_partition as RP
from dyrad.preprocessing.boreas.prepare import load_applanix, world_from_lidar


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--boreas-objects",
        default="data/boreas/boreas-objects-v1",
        help="raw download root (labels_tracking/, applanix/, calib/)",
    )
    ap.add_argument(
        "--clip-meta", required=True, help="clip_metadata.json written by `boreas prepare`"
    )
    ap.add_argument("--out", required=True, help="label CSV to write (labels_boreas.csv)")
    ap.add_argument(
        "--dataset",
        required=True,
        help="value of the `dataset` column; must equal the config's seq_name",
    )
    ap.add_argument(
        "--types",
        default="Car",
        help="comma-separated label types to keep (Car,Cyclist,Pedestrian,Misc)",
    )
    ap.add_argument(
        "--min-speed",
        type=float,
        default=1.0,
        help="world-frame median speed [m/s] for a track to count as dynamic",
    )
    ap.add_argument(
        "--min-points",
        type=int,
        default=50,
        help="min median lidar points of a track (drops barely-observed tracks)",
    )
    ap.add_argument(
        "--max-range", type=float, default=80.0, help="max radar range [m] of an emitted label"
    )
    ap.add_argument(
        "--max-dt-us",
        type=float,
        default=1.5e5,
        help="max |t_label - t_radar| for a frame to be associated",
    )
    args = ap.parse_args()

    src = args.boreas_objects
    with open(args.clip_meta) as f:
        meta = json.load(f)
    stamps = [int(t) for t in meta["radar_timestamps_us"]]
    keep_types = {s.strip() for s in args.types.split(",") if s.strip()}

    T_rl = np.loadtxt(os.path.join(src, "calib", "T_radar_lidar.txt"))
    lt, lm = load_applanix(os.path.join(src, "applanix", "lidar_poses.csv"))

    lab_paths = {
        int(os.path.basename(p)[:-4]): p
        for p in glob.glob(os.path.join(src, "labels_tracking", "*.txt"))
    }
    lab_ts = np.asarray(sorted(lab_paths))

    def read(tl):
        out = []
        with open(lab_paths[tl]) as fh:
            lines = fh.readlines()
        for line in lines:
            p = line.split()
            if len(p) < 10 or p[1] not in keep_types:
                continue
            # uuid type l w h x y z rot_y numPoints
            # The order is length, width, height (p[2], p[3], p[4]).
            out.append(
                (
                    p[0],
                    p[1],
                    float(p[5]),
                    float(p[6]),
                    float(p[7]),
                    int(p[9]),
                    float(p[3]),
                    float(p[2]),
                    float(p[8]),
                )
            )
        return out

    # ---- pass 1: world trajectories, to classify moving vs parked -----------
    tracks: dict[str, list] = {}
    assoc = []  # (radar_idx, label_t)
    for k, tr in enumerate(stamps):
        j = int(np.argmin(np.abs(lab_ts - tr)))
        tl = int(lab_ts[j])
        if abs(tl - tr) > args.max_dt_us:
            continue
        assoc.append((k, tl))
        T_wl = world_from_lidar(lt, lm, tl)
        for uuid, ty, x, y, z, npts, _w, _l, _ry in read(tl):
            pw = T_wl @ np.array([x, y, z, 1.0])
            tracks.setdefault(uuid, []).append((tl, pw[0], pw[1], npts, ty))

    moving = {}
    for uuid, v in tracks.items():
        v.sort()
        if len(v) < 3:
            continue
        pos = np.array([(a[1], a[2]) for a in v])
        dt = np.diff([a[0] for a in v]) / 1e6
        step = np.hypot(*np.diff(pos, axis=0).T)
        spd = float(np.median(step / np.maximum(dt, 1e-3)))
        pts = float(np.median([a[3] for a in v]))
        if spd >= args.min_speed and pts >= args.min_points:
            moving[uuid] = spd

    # ---- pass 2: emit rows in the radar frame ------------------------------
    rows = []
    for k, tl in assoc:
        for uuid, ty, x, y, z, npts, box_w, box_l, rot_y in read(tl):
            if uuid not in moving:
                continue
            pr = T_rl @ np.array([x, y, z, 1.0])
            R = float(np.hypot(pr[0], pr[1]))
            if not (0.5 < R <= args.max_range):
                continue
            A = float(np.degrees(np.arctan2(pr[1], pr[0])))
            A = (A + 180.0) % 360.0 - 180.0
            # ── the box's RA footprint, resolved here where the frame is known ──
            # Boreas labels carry real per-object `w`/`l` and `rot_y`, so unlike the
            # RADIal path (camera-bbox width only, length assumed to lie along range)
            # the extent can be computed exactly: the axis-aligned bound of a rotated
            # rectangle, projected onto the radial and cross-range axes. This matters
            # more here than on RADIal: Navtech sees 360 deg, so crossing traffic
            # (length across range) is common.
            yaw_r = rot_y + float(np.arctan2(T_rl[1, 0], T_rl[0, 0]))  # lidar -> radar
            th = yaw_r - float(np.arctan2(pr[1], pr[0]))  # vs the radial dir
            hc, hr = RP.box_half_extents(box_w, box_l, th)
            rows.append(
                {
                    "index": k,
                    "numSample": k,
                    "vehicle_id": uuid,
                    "Annotation": "strong",
                    "laser_X_m": R * np.cos(np.radians(A)),
                    "laser_Y_m": R * np.sin(np.radians(A)),
                    "radar_R_m": R,
                    "radar_A_deg": A,
                    "box_w_m": box_w,
                    "box_l_m": box_l,
                    "box_half_cross_m": hc,
                    "box_half_range_m": hr,
                    "box_yaw_rad": yaw_r,
                    "dataset": args.dataset,
                }
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cols = [
        "index",
        "numSample",
        "vehicle_id",
        "Annotation",
        "laser_X_m",
        "laser_Y_m",
        "radar_R_m",
        "radar_A_deg",
        "box_w_m",
        "box_l_m",
        "box_half_cross_m",
        "box_half_range_m",
        "box_yaw_rad",
        "dataset",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    nf = len({r["index"] for r in rows})
    print(
        f"[labels] {len(tracks)} tracks total -> {len(moving)} dynamic "
        f"(>={args.min_speed} m/s, >={args.min_points} pts)"
    )
    print(f"[labels] {len(rows)} detections over {nf}/{len(stamps)} radar frames -> {args.out}")
    if moving:
        s = np.array(list(moving.values()))
        print(
            f"[labels] track speeds m/s: min {s.min():.1f} median {np.median(s):.1f} max {s.max():.1f}"
        )
        per = np.bincount([r["index"] for r in rows], minlength=len(stamps))
        print(f"[labels] objects per frame: mean {per.mean():.1f} min {per.min()} max {per.max()}")


if __name__ == "__main__":
    main()
