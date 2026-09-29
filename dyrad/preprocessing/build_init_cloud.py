"""Build the static init cloud ("radar pseudo-LiDAR") of a training window from its RAD tensors.

The trainer's `init_type: radar_cloud` path seeds its static reflectors from the [N,3]
point cloud at `init_cloud_path` and its dynamic reflectors from each object's own returns
(paper B.1). This script builds that static cloud from bright Range-Azimuth returns of the
radar itself, so no LiDAR is needed.

Per train frame (frame window from the config, skipping the held-out frames so no
held-out geometry enters the init):
  1. load the [D,R,A] tensor and crop range exactly like the trainer
  2. RA = mean over Doppler -> [R_crop, A]
  3. select every bin with RA > noise_factor * median(RA > 0)
  4. with --partition, drop the bins inside each annotated object's footprint
     (dyrad.ra_partition), so the static cloud samples free space only
  5. (r_bin, az_bin) -> sensor xyz (z=0) -> world via the config's radar poses
Then aggregate across frames and voxel-deduplicate.

Output: `<seq_dir>/radar_cloud_f<S>_<E>_<split>_<vox>m[_free[_psf<k>]].npy`, [N,3] world
xyz, where S..E is the window (E inclusive), `<split>` is `no_every<K>` (every K-th frame
held out) or `full` (no holdout, test_every 0), and `_free_psf<k>` marks a partitioned
cloud (k = car_psf_margin). This is the name the generated run configs expect as
`init_cloud_path`; the `_free*` clouds serve `dyrad` and the ablations, the unpartitioned
cloud serves DyRAD-static, which assumes no annotations exist (the whole frame is free
space). A `.build.json` sidecar records the holdout split and the CLI values (the trainer
refuses a cloud whose split differs from the run's) and, with --partition, a
`.partition.json` records the footprint parameters.

Sensor-frame z=0 is mapped through the pose (flat-road assumption); elevation is
unsupervised in the trainer (num_elevation_bins=1). The mean over Doppler is a lower-variance
statistic than the max. The callers choose --noise-factor: `radial prepare` bisects it to a
target point density, the Boreas chain uses 2.0 and the synthetic scenes a fixed value each.

Usage:
    python -m dyrad.preprocessing.build_init_cloud \\
        --config configs/radial/31_22_dyrad.yaml --noise-factor 2.0 --partition
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dyrad import ra_partition as RP
from dyrad.axes import range_axis_identity, sensor_axes
from dyrad.config import is_held_out, load_config
from dyrad.labels import load_index_remap, read_detections
from dyrad.paths import ROOT
from dyrad.poses import load_poses


def cloud_filename(
    frame_start: int,
    frame_end: int,
    test_every: int,
    voxel_m: float,
    partition: bool,
    psf_margin: float = 0.0,
) -> str:
    """`radar_cloud_f<S>_<E>_<split>_<vox>m[_free[_psf<k>]].npy` (see the module doc).

    `frame_end` is exclusive (the trainer window), so E = frame_end - 1; `<split>` is
    `no_every<K>` for a holdout period K > 0 and `full` for none. The PSF
    dilation changes which cells are free, so a partitioned cloud carries
    `car_psf_margin` (k > 0) in its name. Also used by generate_configs (the run
    configs' `init_cloud_path`).
    """
    vox = ("%gm" % float(voxel_m)).replace(".", "p")
    tail = ""
    if partition:
        tail = "_free"
        k = float(psf_margin)
        if k > 0:
            tail += ("_psf%g" % k).replace(".", "p")
    split = f"no_every{int(test_every)}" if int(test_every) > 0 else "full"
    return f"radar_cloud_f{int(frame_start)}_{int(frame_end) - 1}_{split}_{vox}{tail}.npy"


def load_dets_by_frame(cfg, seq: Path, root: Path) -> dict:
    """{local frame: [detection]} of the config's label CSV, for --partition.

    Uses the trainer's index rule (the authors' `index` remapped through
    label_index_remap.npy, dyrad.labels.read_detections); getting it wrong shifts every
    object by up to ~100 frames and carves free space out of the wrong cells.
    """
    lp = str(cfg.object_label_path)
    if not lp:
        return {}
    q = Path(lp)
    if not q.is_absolute():
        q = root / q
    if not q.is_file():
        print(f"[radar_cloud] WARNING: no label CSV at {q}; --partition has nothing to mask")
        return {}
    return read_detections(q, str(cfg.seq_name), load_index_remap(seq))


def voxel_dedup(pts, voxel):
    if voxel <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts[:, :3] / voxel).astype(np.int64)
    seen = {}
    for i in range(len(pts)):
        k = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
        if k not in seen:
            seen[k] = i
    return pts[np.fromiter(seen.values(), dtype=np.int64, count=len(seen))]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", required=True, help="run config yaml (radar dims, frame window, split)"
    )
    ap.add_argument(
        "--noise-factor",
        type=float,
        required=True,
        help="threshold multiplier: keep bins with RA > factor * median(RA > 0)",
    )
    ap.add_argument(
        "--voxel",
        type=float,
        default=None,
        help="deduplication voxel size (m); default the config's init_cloud_voxel_m, which "
        "also names the cloud file the run configs expect (0.5 m for RADIal and Boreas, "
        "Appendix B.1)",
    )
    ap.add_argument(
        "--partition",
        action="store_true",
        help="sample free space only: drop each annotated object's footprint per frame "
        "(dyrad.ra_partition) and write the `_free*` cloud. DyRAD-static uses the "
        "unpartitioned cloud (no annotations: the whole frame is free space).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.voxel is None:
        args.voxel = float(cfg.init_cloud_voxel_m)
    rad_dir = Path(cfg.rad_tensors_dir)
    seq = rad_dir.parent
    poses = load_poses(str(cfg.radar_poses_dir))[0].astype(np.float64)  # [N,4,4]

    fs = int(cfg.frame_start)
    fe = int(cfg.frame_end)
    fe = len(poses) if fe < 0 else min(fe, len(poses))  # frame_end is EXCLUSIVE
    test_every = int(cfg.test_every)
    test_offset = int(cfg.test_offset)
    rcf, rcl = int(cfg.range_crop_first), int(cfg.range_crop_last)

    # The trainer's float32 bins (make_render_bins): cropped range [R_crop], azimuth in radians
    range_centers, _az_deg, _ = sensor_axes(cfg, doppler=False, dtype=np.float32)
    az_centers = np.deg2rad(_az_deg)  # [A] radians
    cos_az, sin_az = np.cos(az_centers), np.sin(az_centers)
    az_deg = np.degrees(az_centers)

    dets_by_frame, fx = {}, None
    if args.partition:
        dets_by_frame = load_dets_by_frame(cfg, seq, ROOT)
        fx = RP.load_fx(cfg, ROOT)
        n_det = sum(len(v) for v in dets_by_frame.values())
        print(
            f"[radar_cloud] partition: {n_det} detections over {len(dets_by_frame)} frames, fx={fx}"
        )
        if fx is None:
            print(
                f"[radar_cloud] WARNING: no camera calibration; every object falls back "
                f"to car_width_default_m={RP.CAR_WIDTH_DEFAULT_M} m"
            )

    def is_val(i):
        return is_held_out(i, test_every, test_offset)

    chunks, used, skipped_val = [], [], []
    n_cut = 0
    for i in range(fs, fe):
        if is_val(i):
            skipped_val.append(i)
            continue
        f = rad_dir / f"rad_{i:05d}.npy"
        if not f.exists():
            continue
        rad = np.load(f).astype(np.float32)  # [D, R, A]
        R_full = rad.shape[1]
        rad = rad[:, rcf : R_full - rcl, :]  # [D, R_crop, A]
        # Static cloud RA projection: mean over Doppler (RADIal's convention up to 1/D),
        # the same projection the photometric loss fits. This applies to the static path
        # only; dynamic seeds come from each object's own Doppler slice in the trainer,
        # where a mean would dilute a Doppler-concentrated target.
        ra = rad.mean(axis=0)  # [R_crop, A]

        pos = ra[ra > 0]
        if pos.size == 0:
            continue
        thr = args.noise_factor * float(np.median(pos))
        r_idx, a_idx = np.where(ra > thr)

        if args.partition:
            # Keep only bins outside every object's footprint in this frame. Masking per
            # frame removes the object where it actually is, so no ghost trail can form
            # and static structure it merely drove past is kept.
            free = RP.partition_frame(
                dets_by_frame.get(i, []), range_centers, az_deg, fx, cfg
            )
            keep = free[r_idx, a_idx]
            n_cut += int((~keep).sum())
            r_idx, a_idx = r_idx[keep], a_idx[keep]

        if len(r_idx) == 0:
            continue
        r_m = range_centers[r_idx]  # [K]
        # sensor frame: x = cos(az)*r, y = sin(az)*r, z = 0
        p_cam = np.stack(
            [cos_az[a_idx] * r_m, sin_az[a_idx] * r_m, np.zeros_like(r_m)], axis=1
        ).astype(np.float64)  # [K,3]
        c2w = poses[i]
        w = (c2w[:3, :3] @ p_cam.T).T + c2w[:3, 3]  # [K,3] world
        chunks.append(w.astype(np.float32))
        used.append(i)

    if not chunks:
        raise RuntimeError("No radar returns selected: lower --noise-factor or check paths.")
    P = np.concatenate(chunks, axis=0)
    n_raw = len(P)
    P = voxel_dedup(P, args.voxel)

    out = str(
        seq
        / cloud_filename(
            fs,
            fe,
            test_every,
            args.voxel,
            args.partition,
            float(cfg.car_psf_margin),
        )
    )
    np.save(out, P.astype(np.float32))
    if args.partition:
        # sidecar recording the footprint params the cloud was carved with
        Path(out + ".partition.json").write_text(
            json.dumps(RP.identity(cfg), indent=2, sort_keys=True)
        )
    # build sidecar: the holdout split, the range axis the bins were back-projected with and
    # the CLI values that exist in no config. The trainer refuses a cloud whose split or
    # range axis differ from the run's.
    Path(out + ".build.json").write_text(
        json.dumps(
            {
                "test_every": int(test_every),
                "test_offset": int(test_offset),
                "frame_start": int(fs),
                "frame_end": int(fe),
                "frames_used": [int(u) for u in used],
                "val_frames_excluded": [int(v) for v in skipped_val],
                "range_axis": range_axis_identity(cfg),
                "noise_factor": float(args.noise_factor),
                "voxel": float(args.voxel),
                "partition": bool(args.partition),
                "n_raw": int(n_raw),
                "n_points": int(len(P)),
                "config": str(args.config),
            },
            indent=1,
            sort_keys=True,
        )
    )
    z = P[:, 2]
    print(f"[radar_cloud] frames used ({len(used)}): {used[:5]}...{used[-3:]}")
    print(f"[radar_cloud] val frames excluded: {skipped_val}")
    if args.partition:
        print(
            f"[radar_cloud] partition: {n_cut} bins dropped as object footprint "
            f"({100.0 * n_cut / max(n_raw + n_cut, 1):.2f}% of selected bins)"
        )
    print(f"[radar_cloud] {n_raw} raw -> {len(P)} after voxel {args.voxel}m dedup")
    print(f"[radar_cloud] z[min/med/max]={z.min():.2f}/{np.median(z):.2f}/{z.max():.2f}")
    print(f"[radar_cloud] wrote {out}")


if __name__ == "__main__":
    main()
