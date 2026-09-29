"""RADIal CAN dead-reckoning: one frame period per step, two across a dropped frame."""

import json
import sys

import numpy as np

from dyrad.preprocessing.radial import poses


def _straight_sequence(tmp_path, recording: str, n: int = 40, speed: float = 10.0):
    """A sequence dir driving straight along +x at `speed` m/s (GPS one fix per frame)."""
    seq = tmp_path / "seq"
    (seq / "poses").mkdir(parents=True)
    gps = np.tile(np.eye(4), (n, 1, 1))
    gps[:, 0, 3] = np.arange(n) * speed * poses.DT
    np.save(seq / "poses" / "radar_poses.npy", gps)
    (seq / "poses" / "poses_metadata.json").write_text(json.dumps({"sequence": recording}))
    np.save(seq / poses.EGO_VEL, np.tile([speed, 0.0], (n, 1)))
    return seq


def _run(monkeypatch, seq):
    monkeypatch.setattr(sys, "argv", ["poses", "--seq-dir", str(seq)])
    poses.main()
    out = seq / "poses_can"
    return np.load(out / "radar_poses.npy"), np.load(out / "frame_slots.npy")


def test_no_dropped_frame_is_one_period_per_step(tmp_path, monkeypatch):
    c2w, slots = _run(monkeypatch, _straight_sequence(tmp_path, "RECORD@no-drops"))
    np.testing.assert_array_equal(slots, np.arange(len(slots)))
    step = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    np.testing.assert_allclose(step, 10.0 * poses.DT, rtol=1e-5)


def test_dropped_frame_spans_two_periods(tmp_path, monkeypatch):
    recording, fi = next(iter(poses.DROPPED_FRAMES.items()))
    fi = fi[0]
    n = fi + 20
    c2w, slots = _run(monkeypatch, _straight_sequence(tmp_path, recording, n=n))
    assert slots[fi] - slots[fi - 1] == 2
    assert np.array_equal(np.delete(np.diff(slots), fi - 1), np.ones(n - 2, dtype=np.int64))
    step = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    np.testing.assert_allclose(step[fi - 1], 2 * 10.0 * poses.DT, rtol=1e-5)
    np.testing.assert_allclose(np.delete(step, fi - 1), 10.0 * poses.DT, rtol=1e-5)
