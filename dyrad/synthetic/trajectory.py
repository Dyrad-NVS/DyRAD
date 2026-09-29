"""Ego trajectory of the synthetic benchmark: c2w poses and the ego velocity.

The base path (speed, constant turn rate, constant longitudinal acceleration) is a
scene property; the off-path views add a lateral lane shift and a sensor yaw.
"""

import math

import numpy as np


def generate_ego_poses(
    num_frames: int,
    dt: float,
    v_ego: float,
    lateral_offset: float = 0.0,
    yaw_deg: float = 0.0,
    yaw_rate_deg_s: float = 0.0,
    accel_mps2: float = 0.0,
) -> np.ndarray:
    """Ego trajectory, with optional off-path novel-view transforms.

    Returns [N, 4, 4] float64 c2w poses (ego-to-world).

    Novel-view offsets (applied on top of whatever base path is driven):
        lateral_offset: +Y shift (m) along the ego's own LEFT axis.  Lane shift
                        (e.g. ±1.75 half-lane, ±3.5 full lane) — a genuine novel
                        viewpoint of the same scene.  On a curved path this
                        follows the heading (column 1 of the c2w rotation), the
                        same way real trajectories are shifted for off-path views.
        yaw_deg:        sensor yaw about +Z (deg).  Semantics: the SENSOR is rotated
                        but the vehicle still follows its path (a look-to-the-side
                        camera), so the world ego-velocity is unchanged and the
                        sensor-frame ego velocity becomes R_c2w^T @ v_world
                        (compute_ego_vel_sensor).

    Base-path shape (scene properties, not view offsets):
        yaw_rate_deg_s: constant turn rate (deg/s).  0 = straight.
        accel_mps2:     constant longitudinal acceleration (m/s²).  0 = constant
                        speed.  Speed is floored at 0 so a decelerating scene
                        cannot reverse.

    With `yaw_rate_deg_s == accel_mps2 == 0` the straight-line path is computed in
    closed form.
    """
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    poses = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))

    if yaw_rate_deg_s == 0.0 and accel_mps2 == 0.0:
        # Straight-line path, closed form.
        for n in range(num_frames):
            poses[n, :3, :3] = Rz
            poses[n, 0, 3] = v_ego * n * dt  # x-translation (vehicle motion)
            poses[n, 1, 3] = lateral_offset  # constant lateral lane shift
        return poses

    # Curved and/or accelerating base path: integrate heading and position, then
    # apply the view offsets in the ego's own frame at each step.
    wz = math.radians(yaw_rate_deg_s)
    pos = np.zeros(3, dtype=np.float64)
    for n in range(num_frames):
        t = n * dt
        heading = wz * t
        ch, sh = math.cos(heading), math.sin(heading)
        R_path = np.array([[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        R = R_path @ Rz  # base heading, then sensor yaw
        poses[n, :3, :3] = R
        # Lateral offset along the path's left axis (column 1 of R_path), not the
        # yawed sensor's — a lane shift is a property of the vehicle, not of the sensor heading.
        poses[n, :3, 3] = pos + R_path[:, 1] * lateral_offset
        # Advance along the current heading at the current speed.
        speed = max(v_ego + accel_mps2 * t, 0.0)
        pos = pos + R_path[:, 0] * (speed * dt)
    return poses


def ego_vel_world(poses: np.ndarray, dt: float) -> np.ndarray:
    """[N, 3] world ego velocity: central FD of position, one-sided at the ends.

    The one definition used by the render (Doppler, clutter), ego_vel_can.npy and the
    labels. generate_ego_poses advances the position along the heading at the start
    of each step, so on a curved path the velocity at frame n points along the heading
    of frame n - 1/2 (a small sensor-frame lateral component, 0.07 m/s on curve_ramp),
    and the one-sided end differences differ from the neighbouring central ones on an
    accelerating path.
    """
    N = len(poses)
    if N < 2:
        return np.zeros((N, 3))
    pos = poses[:, :3, 3].astype(np.float64)
    a = np.maximum(np.arange(N) - 1, 0)
    b = np.minimum(np.arange(N) + 1, N - 1)
    return (pos[b] - pos[a]) / ((b - a)[:, None] * dt)


def compute_ego_vel_sensor(poses: np.ndarray, dt: float) -> np.ndarray:
    """[N, 2] sensor-frame ego velocity — the synthetic analogue of ego_vel_can.npy.

    ego_vel_world rotated into each pose's sensor frame (R_c2w^T @ v_world), xy.
    """
    v_world = ego_vel_world(poses, dt)
    v_sensor = np.einsum("nji,nj->ni", poses[:, :3, :3].astype(np.float64), v_world)
    return v_sensor[:, :2].astype(np.float32)
