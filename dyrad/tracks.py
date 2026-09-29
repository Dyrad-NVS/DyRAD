"""Rigid object tracks (paper Sec. 3.1, Fig. 3): one learned track point per training frame, linear
interpolation between them, a least-squares velocity over neighbouring points, and heading from the
direction of travel.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


class RigidObjectTrack(torch.nn.Module):
    """Per-object rigid track seeded from annotation positions.

    Motion is parametrised directly by the annotation trajectory. Each object ``o``
    has control points ``(t_k, c_k)`` initialised from its annotation detections;
    the xy of ``c_k`` is learnable and z is frozen (elevation is unsupervised: the
    sensor grid has a single elevation bin). Position is piecewise-linear in the
    control points.
    With ``safety_frames > 0`` each track is extended at both ends by a short
    learnable, linearly-extrapolated segment, so a moving object keeps its speed
    for a bounded window past its last annotation before flat-clamping.

    Velocity is analytic (the segment slope, or a least-squares slope over
    ``2*vel_fit_span`` control points), so it cannot drift between control
    points. The rendered Doppler refines the velocities through the control
    points; the annotations fix the absolute alias branch (Doppler is only
    observable modulo the wrap period).

    Interface used by the renderer:
        forward(z, t, obj_idx)  -> dx  [N, 3]   (vz=0)
        velocity(z, t, obj_idx) -> v_t [N, 3]   (vz=0)
    ``z`` is each reflector's position in its object's frame (``means``);
    ``obj_idx`` is the render-time 1-indexed object id (any other value is static:
    zero motion). ``t`` is the absolute frame time (scalar; tensors are reduced to
    their first element — the absolute-time convention gives every reflector the
    same scalar t).
    """

    def __init__(
        self,
        targets,
        device,
        frame_times,
        seed_anchors: Optional[list] = None,
        safety_frames: int = 0,
        dt: float = 0.2,
        vel_fit_span: int = 1,
        skip_knot_frames=None,
        ctrl_interp_every: int = 0,
        rotate_min_travel_m: float = 5.0,
    ):
        super().__init__()
        #: `means` of a dynamic reflector is its position in the object's own frame and
        #: the track applies x = R_j(t)·means + P_j(t) (Config: "object tracks").
        self._safety_frames = int(safety_frames)
        self._dt = float(dt)
        # Rigid-body yaw (see Config.track_rotate_min_travel_m). Per-object enable is
        # decided in the construction loop from the annotation polyline's total travel.
        self._rotate_min_travel = float(rotate_min_travel_m)
        # Heading has no free parameter: it is derived from the control-polyline tangent,
        # so the losses reach the yaw through the control points and it stays
        # data-determined.
        # Interior control points: re-densify the annotation polyline with free points
        # at every ctrl_interp_every-th recorded frame, initialised on the inter-anchor
        # chord. They carry no annotation; only Doppler/photometry can bend the polyline
        # off the chord.
        self._ctrl_interp_every = max(0, int(ctrl_interp_every))
        # Least-squares velocity window (see _interp_vel). 1 = adjacent difference,
        # 2 = 4-point slope.
        self._vel_fit_span = max(1, int(vel_fit_span))
        # Control points only at observed frames: a point on a frame no loss renders has
        # 2 unsupervised DOF. `skip_knot_frames` = global frame indices of the held-out
        # frames.
        self._skip_knot_frames = set(int(f) for f in (skip_knot_frames or []))
        # Per-frame times, global frame index -> seconds (the Runner's time axis).
        self._frame_times = np.asarray(frame_times, dtype=np.float64)
        by_obj: Dict[int, list] = defaultdict(list)
        # Canonical seed anchors (t0, seed_world, obj_id) extend the control points
        # back to the first detection so interp covers early times.
        for t0, pos, oid in seed_anchors or []:
            oid = int(oid)
            by_obj[oid].append((float(t0), np.asarray(pos, dtype=np.float64)))
        for t_sec, pos, oid in targets:
            by_obj[int(oid)].append((float(t_sec), np.asarray(pos, dtype=np.float64)))

        self.ctrl = torch.nn.ParameterDict()  # str(render_idx) -> [M, 2] xy (learnable)
        self._ridx: List[int] = []  # render-time obj_idx values present

        for oid, lst in sorted(by_obj.items()):
            lst = sorted(lst, key=lambda r: r[0])
            ts: List[float] = []
            ps: List[np.ndarray] = []
            for t_sec, p in lst:  # dedup coincident times
                if ts and abs(t_sec - ts[-1]) < 1e-6:
                    continue
                ts.append(t_sec)
                ps.append(p)
            if not ts:
                continue
            # Re-densify: insert chord-interpolated control points so consecutive points
            # are <= ctrl_interp_every frames apart. Inserted before the safety extension
            # so synthetic end anchors stay the outermost points.
            if self._ctrl_interp_every > 0 and len(ts) >= 2:
                ts2: List[float] = []
                ps2: List[np.ndarray] = []
                ft = self._frame_times
                for k in range(len(ts) - 1):
                    ts2.append(ts[k])
                    ps2.append(ps[k])
                    gap = ts[k + 1] - ts[k]
                    # knots at the frame times strictly inside the gap, every
                    # ctrl_interp_every-th frame, skipping held-out frames
                    i0 = int(np.argmin(np.abs(ft - ts[k])))
                    i1 = int(np.argmin(np.abs(ft - ts[k + 1])))
                    for fi in range(i0 + 1, i1):
                        if (fi - i0) % self._ctrl_interp_every:
                            continue
                        if self._skip_knot_frames and fi in self._skip_knot_frames:
                            continue  # held-out frame: no knot here
                        tt = float(ft[fi])
                        if not (ts[k] + 1e-6 < tt < ts[k + 1] - 1e-6):
                            continue
                        a = (tt - ts[k]) / gap
                        ts2.append(tt)
                        ps2.append((1.0 - a) * ps[k] + a * ps[k + 1])
                ts2.append(ts[-1])
                ps2.append(ps[-1])
                ts, ps = ts2, ps2
            _had_safety = False
            # Safety segment: extend the track by safety_frames at each end with a
            # synthetic anchor placed by linear extrapolation of the end segment. Within
            # the extension the object keeps moving; beyond it the interpolation clamps.
            # The synthetic anchors are learnable. Covers a held-out frame that sits just
            # past the last train control point.
            if self._safety_frames > 0 and len(ts) >= 2:
                _m = self._safety_frames * self._dt
                v_end = (ps[-1] - ps[-2]) / max(ts[-1] - ts[-2], 1e-6)
                ts.append(ts[-1] + _m)
                ps.append(ps[-1] + v_end * _m)
                v_start = (ps[1] - ps[0]) / max(ts[1] - ts[0], 1e-6)
                ts.insert(0, ts[0] - _m)
                ps.insert(0, ps[0] + v_start * (-_m))
                _had_safety = True  # indices 0 and -1 are synthetic
            ridx = oid + 1  # render obj_idx is 1-indexed
            t_arr = torch.tensor(ts, dtype=torch.float32, device=device)
            p_arr = torch.tensor(np.stack(ps), dtype=torch.float32, device=device)
            self.register_buffer(f"t_{ridx}", t_arr, persistent=True)
            self.ctrl[str(ridx)] = torch.nn.Parameter(p_arr[:, :2].clone())
            self._ridx.append(ridx)

            # ── rigid-body yaw: per-object gate + reference heading ──────────────
            # `theta_ref` is the heading already baked into the seeded cloud, so the
            # applied rotation is theta(t) - theta_ref, the identity at the seed time.
            # It is taken on the annotation polyline (p_arr) at init, not on the live
            # control points.
            _travel = (
                float(np.linalg.norm(np.diff(np.stack(ps)[:, :2], axis=0), axis=1).sum())
                if len(ps) >= 2
                else 0.0
            )
            _rot_ok = bool(_travel >= self._rotate_min_travel and len(ts) >= 3)
            # persistent=False: re-derived from the labels at every init, so it is not
            # saved with the checkpoint.
            self.register_buffer(
                f"rotok_{ridx}", torch.tensor(bool(_rot_ok), device=device), persistent=False
            )
            if _rot_ok:
                _th0, _tm0 = self._polyline_headings(p_arr[:, :2], t_arr)
                # seed time = the object's first real annotation knot (index 1 when a
                # synthetic safety anchor was prepended).
                _t_seed = float(ts[1] if _had_safety and len(ts) > 2 else ts[0])
                _ref = self._heading_at(_th0, _tm0, _t_seed)[0]
                self.register_buffer(f"thref_{ridx}", _ref.detach().clone(), persistent=False)
                print(
                    f"[Init:tracks] obj {ridx}: yaw on, annotation travel "
                    f"{_travel:.1f} m, theta_ref {float(_ref) * 180 / np.pi:+.1f} deg"
                )
            else:
                print(
                    f"[Init:tracks] obj {ridx}: yaw off, annotation travel "
                    f"{_travel:.1f} m < {self._rotate_min_travel:.1f} m "
                    f"(heading would be noise); pure translator"
                )

    # ── control points ───────────────────────────────────────────────────────
    def _ctrl(self, ridx: int) -> Tensor:
        """Control-point positions [M, 2]; every row is learnable."""
        return self.ctrl[str(ridx)]

    # ── interpolation helpers (differentiable in ctrl; tq is a constant) ──────
    def _interp_pos(self, ridx: int, tq: float) -> Tensor:
        t_arr = getattr(self, f"t_{ridx}")  # [M]
        c = self._ctrl(ridx)  # [M, 2]
        M = t_arr.shape[0]
        if M == 1:
            return c[0]
        j = int(torch.searchsorted(t_arr, torch.tensor(float(tq), device=t_arr.device)).item())
        j = max(0, min(j - 1, M - 2))
        t0, t1 = t_arr[j], t_arr[j + 1]
        # clamp: flat beyond the (safety-extended) span.
        w = ((float(tq) - t0) / (t1 - t0)).clamp(0.0, 1.0)
        return c[j] * (1.0 - w) + c[j + 1] * w

    def _interp_vel(self, ridx: int, tq: float) -> Tensor:
        """Object velocity [2] at time ``tq``, used by the renderer, losses and eval.

        ``vel_fit_span`` k selects the 2k control points symmetric about the segment
        containing ``tq``; the slope is their least-squares fit. k = 1 is the plain
        adjacent difference.

        A two-point slope differentiates control-point position noise, amplifying it
        by 1/dt. For N = 2k equally spaced points the least-squares slope has noise
        (sigma/dt)*sqrt(12/(N(N^2-1))): k = 1 -> 1.414, k = 2 -> 0.447, k = 3 -> 0.239.
        It is unbiased for constant acceleration.

        This is not a regulariser: it changes the map from parameters to the
        predicted observable and adds no penalty term and no DOF.
        """
        t_arr = getattr(self, f"t_{ridx}")
        c = self._ctrl(ridx)
        M = t_arr.shape[0]
        if M == 1:
            return c.new_zeros(2)
        # Zero velocity beyond the (safety-extended) span.
        if float(tq) < float(t_arr[0]) or float(tq) > float(t_arr[-1]):
            return c.new_zeros(2)
        j = int(torch.searchsorted(t_arr, torch.tensor(float(tq), device=t_arr.device)).item())
        j = max(0, min(j - 1, M - 2))

        k = max(1, int(self._vel_fit_span))
        n_pts = 2 * k
        if k == 1 or M < n_pts:
            return (c[j + 1] - c[j]) / (t_arr[j + 1] - t_arr[j])

        # 2k points symmetric about the segment [j, j+1], slid inward at the span
        # ends so the window always holds n_pts (Savitzky-Golay style clamping).
        lo = max(0, min(j + 1 - k, M - n_pts))
        tw = t_arr[lo : lo + n_pts]  # [N] buffer -> constants
        cw = c[lo : lo + n_pts]  # [N, 2] learnable
        dtc = tw - tw.mean()
        denom = (dtc * dtc).sum().clamp(min=1e-12)
        # slope = sum_i w_i * c_i with constant w_i: linear in the control points.
        return ((dtc / denom).unsqueeze(-1) * cw).sum(dim=0)

    # ── rigid-body yaw ───────────────────────────────────────────────────────
    @staticmethod
    def _polyline_headings(c: Tensor, t_arr: Tensor):
        """Per-control-point heading [M], defined at the control-point times.

        Each control point carries (x, y, theta); theta is read off the polyline by
        a central difference:

            theta_k = atan2(c[k+1] - c[k-1])            interior
            theta_0 = atan2(c[1]  - c[0])               one-sided at the ends
            theta_M = atan2(c[M]  - c[M-1])

        Differentiable in the control points, so the Doppler and photometric
        gradients reach the yaw. The central difference places theta at the
        control point and uses a two-segment baseline, halving heading noise
        (which scales as 1/|step|). Interpolating between knots (`_heading_at`)
        keeps theta(t) continuous with an analytic rate.
        """
        # Built with cat so the graph is unambiguous. M == 2 degenerates correctly:
        # the middle slice is empty and both ends take the single segment's heading.
        d = torch.cat(
            [
                (c[1] - c[0]).unsqueeze(0),  # one-sided at the ends
                c[2:] - c[:-2],  # central, spans 2 segments
                (c[-1] - c[-2]).unsqueeze(0),
            ],
            dim=0,
        )  # [M, 2]
        th = torch.atan2(d[:, 1], d[:, 0])  # [M], at the knots
        return th, t_arr

    @staticmethod
    def _wrap_pi(a: Tensor) -> Tensor:
        """Fold an angle difference into (-pi, pi] — the shortest way round."""
        return (a + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _heading_at(th: Tensor, tm: Tensor, tq: float):
        """(theta, omega) at time tq by wrap-safe lerp between control points.

        `tm` is the control-point time array, so theta is anchored at the same times
        as position. Returns the interpolated heading and its analytic rate (the
        lerp's slope). Outside the knot span both clamp to the end knot: flat
        heading, zero yaw rate, matching `_interp_pos`.
        """
        n = th.shape[0]
        if n == 1:
            return th[0], th.new_zeros(())
        # The search and the inside-span test must use the knots' precision. `tm` is
        # float32 while `tq` is a float64 frame time; a knot whose float32 value rounds
        # just below its frame time would otherwise fail `tq <= t1` at the segment's
        # right endpoint and return omega = 0 on that frame. Rounding the query into the
        # knots' dtype makes a knot query land exactly on the boundary.
        tq_t = torch.tensor(float(tq), dtype=tm.dtype, device=tm.device)
        j = int(torch.searchsorted(tm, tq_t).item())
        j = max(0, min(j - 1, n - 2))
        t0, t1 = tm[j], tm[j + 1]
        dth = RigidObjectTrack._wrap_pi(th[j + 1] - th[j])
        span = (t1 - t0).clamp(min=1e-6)
        w = ((float(tq) - t0) / span).clamp(0.0, 1.0)
        theta = th[j] + w * dth
        # zero rate outside the span (w was clamped, so the lerp is flat there)
        inside = (float(tq_t) >= float(t0)) and (float(tq_t) <= float(t1))
        omega = dth / span if inside else th.new_zeros(())
        return theta, omega

    def _yaw(self, ridx: int, tq: float):
        """(dtheta, omega) for object `ridx` at `tq`, relative to its seed heading.

        dtheta is what actually rotates the cloud; omega is the yaw rate the velocity
        term needs. Returns None when this object is a pure translator.
        """
        if not bool(getattr(self, f"rotok_{ridx}").item()):
            return None
        t_arr = getattr(self, f"t_{ridx}")
        if t_arr.shape[0] < 3:
            return None
        th, tm = self._polyline_headings(self._ctrl(ridx), t_arr)
        theta, omega = self._heading_at(th, tm, tq)
        return self._wrap_pi(theta - getattr(self, f"thref_{ridx}")), omega

    # ── the object's pose, as the object-frame render uses it ────────────────
    def tcanon(self, ridx: int) -> float:
        """The object's canonical time: its first real control point.

        Index 1 when a synthetic safety anchor was prepended, else index 0. Derived
        from the knot array rather than stored, so it follows a checkpoint's knots.
        The prepend happens iff `safety_frames > 0` and the object had >= 2 knots,
        which after the extension means M >= 4 (M is never 2 or 3 with safety on).
        """
        M = int(getattr(self, f"t_{ridx}").shape[0])
        i = 1 if (self._safety_frames > 0 and M >= 4) else 0
        return float(getattr(self, f"t_{ridx}")[i])

    @staticmethod
    def _rot2(dtheta: Tensor) -> Tensor:
        """2-D rotation matrix [2,2] for a scalar angle tensor."""
        c, s = torch.cos(dtheta), torch.sin(dtheta)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    @staticmethod
    def _scalar_t(t) -> float:
        if isinstance(t, Tensor):
            return float(t.reshape(-1)[0].item())
        return float(t)

    def forward(self, z: Tensor, t, obj_idx: Optional[Tensor] = None) -> Tensor:
        out = torch.zeros(len(z), 3, device=z.device, dtype=z.dtype)
        if obj_idx is None or not self._ridx:
            return out
        tq = self._scalar_t(t)
        for ridx in self._ridx:
            mask = obj_idx == ridx
            if not bool(mask.any()):
                continue
            # x_i(t) = R_j(t) . means_i + P_j(t), with means_i the reflector's
            # position in the object's frame (origin p0, the object's first
            # train-frame annotation). The caller forms means + dx, so
            #     dx_i = R_j(t) . means_i + P_j(t) - means_i.
            #
            # R_j is relative to the seed heading because the seeds keep world axes,
            # so R_j(t0) = I and x(t0) = means + p0.
            #
            # Computed in float64: with raw UTM coordinates (~4.8e6 m) one float32
            # ULP is ~0.5 m, so only the result is cast back.
            P = self._interp_pos(ridx, tq).to(torch.float64)
            yaw = self._yaw(ridx, tq)
            b = z[mask, :2].to(torch.float64)  # [n, 2]
            if yaw is None:
                out[mask, :2] = P.unsqueeze(0).expand_as(b).to(out.dtype)
            else:
                rot = b @ self._rot2(yaw[0].to(torch.float64)).transpose(0, 1)
                out[mask, :2] = (rot + P.unsqueeze(0) - b).to(out.dtype)
        return out

    def heading(self, t, obj_idx: Tensor) -> Tensor:
        """[N] heading change dtheta_j(t) of each reflector's object relative to its
        seed heading (0 for static reflectors and pure translators), the angle that
        rotates the object frame in `forward`."""
        out = torch.zeros(len(obj_idx), device=obj_idx.device, dtype=torch.float32)
        tq = self._scalar_t(t)
        for ridx in self._ridx:
            mask = obj_idx == ridx
            yaw = self._yaw(ridx, tq) if bool(mask.any()) else None
            if yaw is not None:
                out[mask] = yaw[0].to(out.dtype)
        return out

    def velocity(self, z: Tensor, t, obj_idx: Optional[Tensor] = None) -> Tensor:
        out = torch.zeros(len(z), 3, device=z.device, dtype=z.dtype)
        if obj_idx is None or not self._ridx:
            return out
        tq = self._scalar_t(t)
        for ridx in self._ridx:
            mask = obj_idx == ridx
            if not bool(mask.any()):
                continue
            v_c = self._interp_vel(ridx, tq)  # [2]
            # v_i(t) = Pdot_j(t) + omega_j(t) x (R_j(t) . means_i). The lever arm is
            # the rotated object-frame position. In 2-D, omega_z x (x, y) = omega(-y, x).
            # Same relative R_j as `forward`, so position and velocity are one rigid body.
            # A yawing object's scatterers have different radial velocities across the
            # body (30 deg/s at 2 m is ~1 m/s), which a pure translator cannot represent.
            yaw = self._yaw(ridx, tq)
            if yaw is None:
                out[mask, :2] = v_c.to(out.dtype)
            else:
                dtheta, omega = yaw
                r = z[mask, :2].to(v_c.dtype) @ self._rot2(dtheta).transpose(0, 1)  # [n, 2]
                v_rot = omega * torch.stack([-r[:, 1], r[:, 0]], dim=-1)
                out[mask, :2] = (v_c.unsqueeze(0) + v_rot).to(out.dtype)
        return out
