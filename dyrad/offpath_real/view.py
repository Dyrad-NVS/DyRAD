"""The off-path view writer: one implementation, used by every method variant.

A "view" is a self-contained, re-indexed (0..N-1) sequence dir that the trainer
consumes verbatim:

    <view>/
      rad_tensors/rad_00000.npy ...                      [D, R_full, A] raw power, DC at bin 0
      poses_can/{radar_poses.npy, poses_metadata.json}   c2w for this view, re-indexed 0..N-1
                (+ timestamps_us.npy and frame_slots.npy, the source frames' clock)
      ego_vel_can.npy                                    [N, 2] sensor-frame
      <labels_filename>                                  GT labels reprojected through the shift
      label_index_remap.npy                              identity
      norm.json                                          inherited from the parent sequence
      {range_bins_m, az_bins_deg, doppler_bins_mps, rd_doppler_bins_mps}.npy

`write_view` writes a view definition: `T0` carries the real tensors of the window, `T1`
carries everything except tensors. `derive_view` copies a definition and adds one
variant's rendered tensors. Everything but the tensors (poses, ego velocity,
normalization, axes, labels, metadata) is written only by `write_view`, so two variants
cannot drift onto different windows, normalization ranges or label conventions.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from dyrad.domains import Domain, declare_domain, save_rad
from dyrad.labels import REMAP_FILENAME, load_index_remap
from dyrad.norm import inherit as norm_inherit

from . import labels as view_labels

AXIS_FILES = (
    "range_bins_m.npy",
    "az_bins_deg.npy",
    "doppler_bins_mps.npy",
    "rd_doppler_bins_mps.npy",
)
EGO_VEL_FILE = "ego_vel_can.npy"


def shift_poses(poses: np.ndarray, lateral_m: float) -> np.ndarray:
    """Translate each c2w by `lateral_m` along its sensor-left axis (column 1).

    The rotations are kept, and the view keeps the base path's sensor-frame ego velocity.
    On a curved path the shifted sensor moves at v + omega x d; the omega x d term (at most
    ~0.17 m/s on the paper's windows, 1.5 RADIal Doppler bins) is ignored. The round trip
    stays self-consistent: M0 renders the shifted view and M1 fits it with the same
    velocity, and M1 is scored at T0 with the recorded one.
    """
    out = poses.copy()
    for i in range(len(out)):
        out[i, :3, 3] += lateral_m * out[i, :3, 1]
    return out


def float32_baseline_error(poses_t0, poses_view, lateral_m: float) -> float:
    """Max error in the realized lateral baseline after a float32 round trip.

    The trainer casts poses to float32 on load (`RadarParser.__init__`), so this is
    the error the model sees in the experiment's independent variable. Recorded in
    poses_metadata.json as provenance.
    """
    a = np.asarray(poses_t0, np.float32).astype(np.float64)
    b = np.asarray(poses_view, np.float32).astype(np.float64)
    d = np.linalg.norm(b[:, :3, 3] - a[:, :3, 3], axis=1)
    return float(np.abs(d - lateral_m).max())


def write_timestamps(out, base_seq, fs: int, fe: int) -> None:
    """Carry the source window's clock into the view: its recorded timestamps and, when the
    source has them, its frame slots (re-based to the window's first frame).

    An M1 trained on the view uses them as its time axis (`use_frame_timestamps`, or the
    slots otherwise; see Runner._build_frame_times). The lateral shift does not change
    time, so every view of a window carries the same clock as the source frames it
    re-indexes.
    """
    src = Path(base_seq) / "poses_can" / "timestamps_us.npy"
    ts = np.load(src)
    if len(ts) < fe:
        raise ValueError(f"{src} has {len(ts)} stamps, window needs {fe}")
    np.save(Path(out) / "poses_can" / "timestamps_us.npy", ts[fs:fe])
    slots_src = Path(base_seq) / "poses_can" / "frame_slots.npy"
    if slots_src.exists():
        slots = np.load(slots_src)
        if len(slots) < fe:
            raise ValueError(f"{slots_src} has {len(slots)} slots, window needs {fe}")
        np.save(Path(out) / "poses_can" / "frame_slots.npy", slots[fs:fe] - slots[fs])


def clear(out: Path) -> None:
    """Remove a previous version of this view and recreate `poses_can/`.

    Frames are written re-indexed 0..N-1, so writing a 60-frame window into a
    directory left from a 100-frame one would keep rad_00060..rad_00099: a
    mixed-provenance sequence that still loads and trains.
    """
    for sub in ("rad_tensors", "poses_can"):
        d = out / sub
        if d.exists():
            stale = sorted(d.glob("*.npy"))
            if stale:
                print(f"  [view] clearing {len(stale)} stale file(s) from {d}")
            shutil.rmtree(d)
    (out / "poses_can").mkdir(parents=True)


def write_view(
    out,
    *,
    base_seq,
    frames,
    poses_t0,
    poses_view,
    lateral_m: float,
    content: str,
    range_crop_first: int,
    range_crop_last: int,
    dt: float,
    ego_vel_npy,
    label_csv,
    label_seq_str: str,
    far_range_m: float,
    az_fov_deg: float,
    seq_name: str,
    labels_filename: str,
) -> None:
    """Write one view definition dir.

    frames      global frame indices, in output order (local idx = position)
    poses_t0    full unshifted pose array (global indexing); labels need it
    poses_view  full pose array for this view (global indexing)
    content     "real_gt": copy the source's real tensors of the window (`T0`);
                "definition": write no tensors (`T1`, each variant's `derive_view`
                adds its renders). Recorded in poses_metadata.json.
    """
    if content not in ("real_gt", "definition"):
        raise ValueError(f"content must be 'real_gt' or 'definition', got {content!r}")
    out = Path(out)
    base_seq = Path(base_seq)
    n = len(frames)
    fs, fe = frames[0], frames[-1] + 1
    assert seq_name, "seq_name must come from spec.view_seq_name"

    clear(out)

    # ── tensors ──────────────────────────────────────────────────────────────
    if content == "real_gt":
        src_dir = base_seq / "rad_tensors"
        (out / "rad_tensors").mkdir()
        print(f"  [view] REAL-GT tensors from {src_dir}")
        for li, fi in enumerate(frames):
            src = src_dir / f"rad_{fi:05d}.npy"
            if not src.exists():
                raise SystemExit(f"real GT frame missing: {src}")
            # copied verbatim (never re-encoded)
            shutil.copyfile(src, out / "rad_tensors" / f"rad_{li:05d}.npy")
        declare_domain(
            out / "rad_tensors",
            Domain.RAW_POWER,
            note=f"REAL measured GT, frames {fs}..{fe - 1} re-indexed 0..{n - 1}",
        )

    # ── poses (re-indexed standalone) + provenance ────────────────────────────
    np.save(out / "poses_can" / "radar_poses.npy", poses_view[fs:fe].astype(np.float64))
    meta = {
        "num_frames": n,
        "dt": float(dt),
        "dataset": seq_name,
        "crop_first": int(range_crop_first),
        "crop_last": int(range_crop_last),
        "tesseract_indices": list(range(n)),
        "source_seq": base_seq.name,
        "source_window": [int(fs), int(fe)],
        "lateral_shift_m": float(lateral_m),
        # What the tensors are, recorded at write time so nothing downstream has
        # to infer it from the directory name.
        "content": content,
        "float32_baseline_err_m": float32_baseline_error(poses_t0, poses_view, lateral_m),
    }
    (out / "poses_can" / "poses_metadata.json").write_text(json.dumps(meta, indent=2))
    write_timestamps(out, base_seq, fs, fe)

    # ── ego velocity: the base path's (see shift_poses) ───────────────────────
    ego = np.load(ego_vel_npy).astype(np.float32)
    np.save(out / EGO_VEL_FILE, ego[fs:fe])

    # ── normalization range: inherit the parent's, never recompute ────────────
    # A view is scored against the parent's real GT, so it must use the parent's
    # normalization range. The trainer refuses a sequence with no norm.json.
    norm_inherit(
        out,
        base_seq,
        note=(
            "real-GT reference view"
            if content == "real_gt"
            else f"off-path view, lateral={lateral_m:+g} m"
        ),
    )

    # ── axes + identity label remap ───────────────────────────────────────────
    for fn in AXIS_FILES:
        src = base_seq / fn
        if src.exists():
            shutil.copy2(src, out / fn)
    np.save(out / REMAP_FILENAME, np.arange(n, dtype=np.int64))

    # ── labels ────────────────────────────────────────────────────────────────
    rows = view_labels.reproject_and_write(
        out,
        label_csv,
        filename=labels_filename,
        seq_str=label_seq_str,
        out_seq_name=seq_name,
        poses_t0=poses_t0,
        poses_shifted=poses_view,
        frame_start=fs,
        frame_end=fe,
        far_range_m=far_range_m,
        az_fov_deg=az_fov_deg,
        remap=load_index_remap(base_seq),
    )

    print(f"  [view] {out}  ({n} frames, {len(rows)} labels, content={content})")


def derive_view(out, src_view, tensors, *, rendered_from, labels_filename: str) -> None:
    """A method's own copy of a view definition: same geometry, its own tensors.

    Every method renders the same view definition: the poses, labels, axes, ego
    velocity and normalization range must be identical, or the rows stop being comparable,
    but each method needs its own `rad_tensors/`.

    `tensors` is an iterable of full-size [D, R_full, A] raw-power arrays in frame
    order. The copied `poses_metadata.json` is updated to name the renderer.
    """
    out, src_view = Path(out), Path(src_view)
    if not (src_view / "poses_can/poses_metadata.json").exists():
        raise SystemExit(f"source view definition missing: {src_view}")
    if out.is_symlink():
        # the view may be a symlink to another disk; empty the target, keep the link
        for e in out.resolve().iterdir():
            shutil.rmtree(e) if e.is_dir() and not e.is_symlink() else e.unlink()
    elif out.exists():
        shutil.rmtree(out)
    # Explicit copy of the shared plumbing (everything except the tensors).
    out.mkdir(parents=True, exist_ok=out.is_symlink())
    shutil.copytree(src_view / "poses_can", out / "poses_can")
    for fn in AXIS_FILES:
        if (src_view / fn).exists():
            shutil.copy2(src_view / fn, out / fn)
    # write_view always writes these; a definition without one is broken.
    for fn in (labels_filename, REMAP_FILENAME, "norm.json", EGO_VEL_FILE):
        shutil.copy2(src_view / fn, out / fn)
    (out / "rad_tensors").mkdir()

    n = 0
    for li, arr in enumerate(tensors):
        save_rad(
            out / "rad_tensors" / f"rad_{li:05d}.npy",
            np.asarray(arr, dtype=np.float32),
            Domain.RAW_POWER,
            note=f"derived view, rendered by {rendered_from}",
        )
        n += 1

    mp = out / "poses_can/poses_metadata.json"
    meta = json.loads(mp.read_text())
    if n != meta["num_frames"]:
        raise SystemExit(
            f"derive_view wrote {n} tensors but the view declares {meta['num_frames']} "
            f"frames ({src_view})"
        )
    meta["content"] = "render"
    meta["rendered_from"] = str(rendered_from)
    meta["derived_from_view"] = str(src_view)
    mp.write_text(json.dumps(meta, indent=2))

    print(f"  [view] derived {out} ({n} frames, rendered by {rendered_from})")
