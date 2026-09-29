"""Fetch sensor files of one boreas-objects-v1 time window from the public bucket.

`boreas-objects-v1` is the only Boreas recording with object annotations. The radar frames
of a window (and the tracking labels, which `boreas labels` reads) are fetched here into
the raw layout `<boreas-objects>/<sensor>/<t_us>.<ext>`, which is where
`python -m dyrad.preprocessing.boreas prepare --clip-start/--clip-end` expects them. The two
small per-recording files the chain needs, `calib/T_radar_lidar.txt` and
`applanix/lidar_poses.csv`, are fetched by `fetch_files` (called by `prepare-window`).

Files that are already present are skipped, so it is safe to re-run. Every file with a
timestamp in [--start, --end] is fetched.

    python -m dyrad.preprocessing.boreas fetch --start 1598987711906340 --end 1598987724156947
    python -m dyrad.preprocessing.boreas fetch --sensor labels_tracking \\
        --start 1598987711406340 --end 1598987724656947 --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import urllib.request

BUCKET = "https://boreas.s3.amazonaws.com"
RECORDING = "boreas-objects-v1"
#: the fetched streams and their file extensions (the dataset's own)
SENSORS = {
    "radar": ".png",
    "labels_tracking": ".txt",
}
#: per-recording files needed by `prepare` (poses) and `labels` (extrinsic)
FILES = ["calib/T_radar_lidar.txt", "applanix/lidar_poses.csv"]


def _prefix(sensor: str) -> str:
    return f"{RECORDING}/{sensor}/"


def list_range(t0: int, t_end: int, sensor: str = "radar") -> list[int]:
    """Timestamps of `sensor` in [t0, t_end]."""
    prefix, ext = _prefix(sensor), SENSORS[sensor]
    out: list[int] = []
    after = str(t0 - 1)
    while True:
        url = (
            f"{BUCKET}/?list-type=2&prefix={prefix}&delimiter=%2F"
            f"&max-keys=1000&start-after={prefix}{after}"
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            body = r.read().decode()
        keys = re.findall(r"<Key>([^<]+)</Key>", body)
        if not keys:
            break
        stop = False
        for k in keys:
            t = int(k.split("/")[-1][: -len(ext)])
            if t > t_end:
                stop = True
                break
            out.append(t)
        if stop or not out:
            break
        after = str(out[-1])
    return out


def _download(key: str, dst: str) -> None:
    # .part then rename: a killed download must not leave a truncated file that every
    # later run then treats as present.
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(f"{BUCKET}/{key}", tmp)
    os.replace(tmp, dst)


def fetch_files(root: str) -> None:
    """Fetch the per-recording FILES into `<root>/<file>` (skipped when present)."""
    for f in FILES:
        dst = os.path.join(root, f)
        if os.path.isfile(dst):
            continue
        print(f"[fetch] {RECORDING}/{f} -> {dst}")
        _download(f"{RECORDING}/{f}", dst)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--start", type=int, required=True, help="first timestamp, us")
    ap.add_argument("--end", type=int, required=True, help="last timestamp, us (inclusive)")
    ap.add_argument("--sensor", default="radar", choices=sorted(SENSORS), help="sensor stream")
    ap.add_argument(
        "--out", default=None, help="output dir (default data/boreas/boreas-objects-v1/<sensor>)"
    )
    ap.add_argument("--dry-run", action="store_true", help="list the frames, download nothing")
    a = ap.parse_args()
    out_dir = a.out or f"data/boreas/{RECORDING}/{a.sensor}"

    ts = list_range(a.start, a.end, a.sensor)
    if not ts:
        raise SystemExit(f"bucket returned no {a.sensor} frames in [{a.start}, {a.end}]")
    span = (ts[-1] - ts[0]) / 1e6
    print(f"[fetch] {len(ts)} {a.sensor} frames  {ts[0]} .. {ts[-1]}  ({span:.2f} s)")
    if a.dry_run:
        return

    os.makedirs(out_dir, exist_ok=True)
    got = new = 0
    for i, t in enumerate(ts):
        dst = os.path.join(out_dir, f"{t}{SENSORS[a.sensor]}")
        if os.path.isfile(dst):
            got += 1
            continue
        _download(f"{_prefix(a.sensor)}{t}{SENSORS[a.sensor]}", dst)
        new += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ts)}")
    print(f"[fetch] done: {new} downloaded, {got} already present -> {out_dir}")


if __name__ == "__main__":
    main()
