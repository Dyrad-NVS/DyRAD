"""Off-path view labels and the object-yaw quaternion."""

import numpy as np

from dyrad.offpath_real.labels import _scaled_bbox


def test_scaled_bbox_keeps_physical_width():
    # An object annotated at 20 m, seen from a view where it sits at 25 m: the pinhole
    # width w_px * R / fx must stay the same.
    row = {"x1_pix": 100, "y1_pix": 50, "x2_pix": 200, "y2_pix": 90}
    box = _scaled_bbox(row, 20.0 / 25.0)
    assert (box["x2_pix"] - box["x1_pix"]) * 25.0 == (200 - 100) * 20.0
    assert (box["x1_pix"] + box["x2_pix"]) / 2 == 150  # centre kept


def test_scaled_bbox_without_camera_box():
    zeros = dict.fromkeys(("x1_pix", "y1_pix", "x2_pix", "y2_pix"), 0)
    assert _scaled_bbox({}, 0.8) == zeros  # Boreas: no camera columns
    assert _scaled_bbox({"x1_pix": 0, "y1_pix": 0, "x2_pix": 0, "y2_pix": 0}, 0.8) == zeros


def test_yaw_quat_matches_rotation_matrix():
    import torch
    from scipy.spatial.transform import Rotation

    from dyrad.trainer.rendering import _yaw_quat

    q_xyzw = Rotation.random(5, random_state=0).as_quat()
    q = torch.tensor(q_xyzw[:, [3, 0, 1, 2]], dtype=torch.float64)  # wxyz
    dtheta = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    got = _yaw_quat(dtheta, q).numpy()
    want = (Rotation.from_euler("z", dtheta.numpy()) * Rotation.from_quat(q_xyzw)).as_matrix()
    assert np.allclose(Rotation.from_quat(got[:, [1, 2, 3, 0]]).as_matrix(), want)
