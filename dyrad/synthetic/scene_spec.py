"""Synthetic scenes: the yaml spec format, the reflector primitives and their motion.

Scenes live in `dyrad/synthetic/scenes/<name>.yaml`; `load_spec` parses a spec and
`build_scene` turns it into a list of world-frame point reflectors with the primitives
below (`car_reflectors`, `barrier_reflectors`, `wall_reflectors`, `pole_reflectors`).
Each reflector is a dict with `label`, `type` ("static" / "dynamic"), `p0_world`,
`velocity`, `accel` (None or [3]), `rcs`, `wall_normal` (None or [3]) and
`specularity`; `reflector_pos` / `reflector_vel` give its state at a frame.

## RNG order

Each element's total RCS is drawn from `rng.uniform(lo, hi)` immediately before
that element is built, and `build_scene()` iterates `elements` in file order, so a
scene is reproducible from its spec and seed. Consequences:

  - Element order in the yaml matters: reordering elements changes every
    downstream jitter draw, so it changes the scene.
  - `rcs:` must be a `[lo, hi]` pair even when lo == hi; a scalar would skip a
    draw and shift everything after it.


## Format

```yaml
description: one line, shown at generation time
ego:
  speed_mps: 31.6         # base ego speed (default 31.6)
  yaw_rate_deg_s: 0.0     # constant turn rate; 0 = straight
  accel_mps2: 0.0         # constant longitudinal accel; 0 = constant speed
elements:
  - kind: barrier         # knee-height posts + rails (guardrail, Jersey barrier)
    label: median_barrier
    y: 10.0
    x_start: -5.0
    x_end: auto           # auto -> v_ego * (num_frames-1) * dt + 100
    rcs: [0.8, 1.2]       # per-post RCS, drawn uniform
    curvature_inv_m: 0.0  # optional 1/R of a curving road, left turn positive:
                          # y(x) = y + x^2 / (2R); 0 = straight (default)
  - kind: wall            # tall specular facade (building)
    label: left_facade
    y: 8.0
    x_start: -5.0
    x_end: auto
    rcs: [2.0, 3.0]       # per-column RCS
    wall_normal_y: -1.0   # facade normal, must face the road: +1 faces +y, -1 faces -y
    height_m: 8.0         # optional (default 8.0)
    spacing_m: 1.5        # optional column spacing (default 1.5)
    specularity: 3.0      # optional (default 3.0)
  - kind: pole            # vertical cylinder (lamp/sign post)
    label: pole_1
    x: 30.0
    y: -7.0
    rcs: [0.5, 0.9]
    height_m: 4.0         # optional (default 4.0)
  - kind: corner          # compact very bright specular point
    label: corner_1
    x: 45.0
    y: -6.0
    rcs: [4.0, 6.0]
  - kind: car             # 102-point car body; static if velocity is all zero
    label: car_overtaking
    x0: 15.0
    y0: 3.7
    velocity: [40.0, 0.0, 0.0]
    rcs: [12.0, 18.0]
    accel: [0.0, 0.35, 0.0]   # optional constant acceleration
```

`kind: car` with zero `velocity` (and no `accel`) is a parked car and is tagged
static by `car_reflectors` itself — there is no separate "parked" kind.

The scene is named by its file stem. Keys outside this format (top level, `ego`, or
per element kind) are rejected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCENES_DIR = Path(__file__).resolve().parent / "scenes"

_REQUIRED = {
    "barrier": ("label", "y", "x_start", "x_end", "rcs"),
    "wall": ("label", "y", "x_start", "x_end", "rcs", "wall_normal_y"),
    "pole": ("label", "x", "y", "rcs"),
    "corner": ("label", "x", "y", "rcs"),
    "car": ("label", "x0", "y0", "velocity", "rcs"),
}
_OPTIONAL = {
    "barrier": ("curvature_inv_m",),
    "wall": ("height_m", "spacing_m", "specularity"),
    "pole": ("height_m",),
    "corner": (),
    "car": ("accel",),
}
_TOP_KEYS = {"description", "ego", "elements"}
_EGO_KEYS = {"speed_mps", "yaw_rate_deg_s", "accel_mps2"}


def scene_names() -> list[str]:
    """Every scene with a yaml spec, sorted."""
    return sorted(p.stem for p in SCENES_DIR.glob("*.yaml"))


def load_spec(name_or_path: str) -> dict:
    """Resolve a scene name (or an explicit path) to its parsed yaml spec."""
    import yaml  # local: the generator has no other yaml dependency

    p = Path(name_or_path)
    if not p.suffix:
        p = SCENES_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"no scene spec at {p}. Available: {', '.join(scene_names()) or '(none)'}"
        )
    spec = yaml.safe_load(p.read_text())
    if not isinstance(spec, dict) or "elements" not in spec:
        raise ValueError(f"{p}: a scene spec needs a top-level 'elements' list")
    unknown = sorted(set(spec) - _TOP_KEYS) + sorted(set(spec.get("ego") or {}) - _EGO_KEYS)
    if unknown:
        raise ValueError(f"{p}: unknown top-level or ego keys {unknown}")
    for i, el in enumerate(spec["elements"]):
        kind = el.get("kind")
        if kind not in _REQUIRED:
            raise ValueError(
                f"{p}: element {i} has kind={kind!r}; "
                f"expected one of {', '.join(sorted(_REQUIRED))}"
            )
        missing = [k for k in _REQUIRED[kind] if k not in el]
        if missing:
            raise ValueError(f"{p}: element {i} ({kind} {el.get('label')!r}) is missing {missing}")
        unknown = sorted(set(el) - {"kind", *_REQUIRED[kind], *_OPTIONAL[kind]})
        if unknown:
            raise ValueError(f"{p}: element {i} ({kind} {el['label']!r}) has unknown keys {unknown}")
        rcs = el["rcs"]
        if not (isinstance(rcs, (list, tuple)) and len(rcs) == 2):
            raise ValueError(
                f"{p}: element {i} ({el['label']!r}) has rcs={rcs!r}; it must be a "
                "[lo, hi] pair — a scalar would skip an RNG draw and desynchronize "
                "every later element (see the RNG-order contract in scene_spec.py)"
            )
    return spec


def ego_params(spec: dict) -> dict:
    """Base-path parameters for `generate_ego_poses`."""
    ego = spec.get("ego") or {}
    return {
        "v_ego": float(ego.get("speed_mps", 31.6)),
        "yaw_rate_deg_s": float(ego.get("yaw_rate_deg_s", 0.0)),
        "accel_mps2": float(ego.get("accel_mps2", 0.0)),
    }


def build_scene(
    spec: dict, rng: np.random.Generator, v_ego: float, dt: float, num_frames: int
) -> list:
    """Turn a spec into a reflector list, consuming `rng` in element order.

    `x_end: auto` resolves to `v_ego * (num_frames - 1) * dt + 100` (the road
    length covered by the sequence plus 100 m).
    """
    x_auto = v_ego * ((num_frames - 1) * dt) + 100.0
    refl: list = []

    def _x(v):
        return x_auto if v == "auto" else float(v)

    for el in spec["elements"]:
        kind = el["kind"]
        lo, hi = float(el["rcs"][0]), float(el["rcs"][1])
        # Draw the RCS first, then build (see "RNG order" above).
        rcs = rng.uniform(lo, hi)

        if kind == "barrier":
            refl += barrier_reflectors(
                el["label"],
                y=float(el["y"]),
                x_start=_x(el["x_start"]),
                x_end=_x(el["x_end"]),
                rcs_per_post=rcs,
                rng=rng,
                curvature_inv_m=float(el.get("curvature_inv_m", 0.0)),
            )
        elif kind == "wall":
            refl += wall_reflectors(
                el["label"],
                y=float(el["y"]),
                x_start=_x(el["x_start"]),
                x_end=_x(el["x_end"]),
                rcs_per_panel=rcs,
                rng=rng,
                wall_normal_y=float(el["wall_normal_y"]),
                height_m=float(el.get("height_m", 8.0)),
                spacing_m=float(el.get("spacing_m", 1.5)),
                specularity=float(el.get("specularity", 3.0)),
            )
        elif kind in ("pole", "corner"):
            refl += pole_reflectors(
                el["label"],
                x=float(el["x"]),
                y=float(el["y"]),
                rcs_total=rcs,
                rng=rng,
                height_m=float(el.get("height_m", 4.0)),
                corner=(kind == "corner"),
            )
        elif kind == "car":
            accel = el.get("accel")
            refl += car_reflectors(
                el["label"],
                float(el["x0"]),
                float(el["y0"]),
                np.asarray(el["velocity"], dtype=np.float64),
                rcs,
                rng,
                accel=None if accel is None else np.asarray(accel, dtype=np.float64),
            )
    return refl


# ── Reflector primitives ─────────────────────────────────────────────────────


def car_reflectors(
    label: str,
    x0: float,
    y0: float,
    velocity: np.ndarray,
    rcs_total: float,
    rng: np.random.Generator,
    accel: np.ndarray | None = None,
) -> list:
    """Generate scattering points for a car body.

    Rear face (licence plate / bumper): specular retroreflector, 35% of the RCS,
    specularity 2, compact in range (7 lateral x 3 heights).

    Body (roof / doors / hood over 3.4 m): diffuse, 65% of the RCS, spread over
    9 positions along the car, giving an extended range profile similar to real
    RADIal car signatures (~5-10 dB below the rear face per range bin).

    The body is laid out along +x and the rear face looks along -x whatever the
    velocity, so a car moving along y (intersection's crossers) travels broadside: a
    benchmark simplification.

    The car is tagged static when both velocity and accel are zero (a parked car).
    """
    pts = []
    moving = velocity.any() or (accel is not None and np.any(accel))
    obj_type = "dynamic" if moving else "static"

    # ── Rear face: specular corner reflector toward radar ────────────────────
    rear_rcs = rcs_total * 0.35
    rear_positions = [
        (x0 + rng.uniform(-0.05, 0.05), y0 + py + rng.uniform(-0.05, 0.05), pz)
        for py in np.linspace(-0.9, 0.9, 7)  # 7 lateral × 3 heights = 21 pts
        for pz in [0.30, 0.75, 1.20]
    ]
    rcs_rear_each = rear_rcs / len(rear_positions)
    for px, py, pz in rear_positions:
        pts.append(
            {
                "label": label,
                "type": obj_type,
                "p0_world": np.array([px, py, pz], dtype=np.float64),
                "velocity": velocity.copy(),
                "accel": None if accel is None else np.asarray(accel, dtype=np.float64),
                "rcs": float(rcs_rear_each * rng.uniform(0.8, 1.2)),
                "wall_normal": [-1.0, 0.0, 0.0],
                "specularity": 2.0,
            }
        )

    # ── Body diffuse: sides, roof edge ───────────────────────────────────────
    body_rcs = rcs_total * 0.65
    body_positions = [
        (x0 + px + rng.uniform(-0.1, 0.1), y0 + py_off + rng.uniform(-0.05, 0.05), pz)
        for px in np.linspace(0.3, 3.7, 9)  # 9 x-positions along car body
        for py_off in [-0.95, 0.0, 0.95]  # 3 lateral
        for pz in [0.35, 0.90, 1.45]
    ]  # bumper, door, roof  (9×3×3=81)
    rcs_body_each = body_rcs / len(body_positions)
    for px, py, pz in body_positions:
        pts.append(
            {
                "label": label,
                "type": obj_type,
                "p0_world": np.array([px, py, pz], dtype=np.float64),
                "velocity": velocity.copy(),
                "accel": None if accel is None else np.asarray(accel, dtype=np.float64),
                "rcs": float(rcs_body_each * rng.uniform(0.5, 1.5)),
                "wall_normal": None,
                "specularity": 0.0,
            }
        )

    return pts


def barrier_reflectors(
    label: str,
    y: float,
    x_start: float,
    x_end: float,
    rcs_per_post: float,
    rng: np.random.Generator,
    curvature_inv_m: float = 0.0,
) -> list:
    """Model a highway guardrail/barrier as discrete posts + horizontal rails.

    In real radar data, guardrails appear as a line of bright spots (posts) connected
    by dimmer rail segments, not as a continuous line of equal-brightness points.
    Posts and rails are omnidirectional (no wall normal), so a barrier looks the same
    from either side.

    curvature_inv_m: 1/R for a barrier that follows a curving road, applied as the
        parabolic approximation y(x) = y + ½·(1/R)·x², with the same 1/R on both
        sides of the road (not concentric arcs). It keeps a curved-path scene's road
        edges beside the road: for R = 286.5 m (curve_ramp) it is within 0.6 m of the
        arc up to x = 100 m and 3.1 m off at 150 m. A left turn is positive, the same
        sense as a positive `yaw_rate_deg_s` in `generate_ego_poses`. 0.0 (default)
        = straight.
    """
    pts = []

    def _y_at(x: float) -> float:
        return y if curvature_inv_m == 0.0 else y + 0.5 * curvature_inv_m * x * x

    # Posts every 3 m: steel cylinders, omnidirectional (no wall_normal)
    for x in np.arange(x_start, x_end, 3.0):
        x_jitter = x + rng.uniform(-0.15, 0.15)
        for pz in [0.25, 0.55, 0.80]:
            pts.append(
                {
                    "label": label,
                    "type": "static",
                    "p0_world": np.array([x_jitter, _y_at(x), pz], dtype=np.float64),
                    "velocity": np.zeros(3, dtype=np.float64),
                    "accel": None,
                    "rcs": float(rcs_per_post * rng.uniform(0.8, 1.2)),
                    "wall_normal": None,
                    "specularity": 0.0,  # cylinder: omnidirectional
                }
            )

    # Rails at 0.75 m spacing: W-beam retroreflective profile, omnidirectional. Rail
    # points with x mod 3 < 0.3 m are left out (a fixed 3 m pattern in world x, not
    # tied to the post positions x_start + 3k).
    rcs_rail = rcs_per_post * 0.20
    for x in np.arange(x_start, x_end, 0.75):
        if x % 3.0 < 0.3:
            continue
        pts.append(
            {
                "label": label,
                "type": "static",
                "p0_world": np.array(
                    [x + rng.uniform(-0.05, 0.05), _y_at(x), 0.45], dtype=np.float64
                ),
                "velocity": np.zeros(3, dtype=np.float64),
                "accel": None,
                "rcs": float(rcs_rail * rng.uniform(0.5, 1.5)),
                "wall_normal": None,
                "specularity": 0.0,  # W-beam: retroreflective, omnidirectional
            }
        )

    return pts


def wall_reflectors(
    label: str,
    y: float,
    x_start: float,
    x_end: float,
    rcs_per_panel: float,
    rng: np.random.Generator,
    wall_normal_y: float = 1.0,
    height_m: float = 8.0,
    spacing_m: float = 1.5,
    specularity: float = 3.0,
) -> list:
    """A tall flat facade (building wall) as a grid of specular panels.

    Unlike `barrier_reflectors` (knee-height posts + rails, omnidirectional),
    a facade is a broad SPECULAR surface: bright at broadside, falling off with
    the view angle, and extending well above the sensor. The return is strong but
    strongly view-dependent, so it changes substantially between the driven path
    and an off-path novel view.

    wall_normal_y is the y component of the facade normal, which must point toward
    the road (ego is near y=0): +1 for a facade on the right (y < 0), -1 on the left.
    """
    pts = []
    wn = [0.0, float(wall_normal_y), 0.0]
    heights = np.arange(0.5, height_m + 1e-9, 1.5)
    rcs_each = rcs_per_panel / len(heights)
    for x in np.arange(x_start, x_end, spacing_m):
        for pz in heights:
            pts.append(
                {
                    "label": label,
                    "type": "static",
                    "p0_world": np.array([x + rng.uniform(-0.08, 0.08), y, pz], dtype=np.float64),
                    "velocity": np.zeros(3, dtype=np.float64),
                    "accel": None,
                    "rcs": float(rcs_each * rng.uniform(0.6, 1.4)),
                    "wall_normal": wn,
                    "specularity": float(specularity),
                }
            )
    return pts


def pole_reflectors(
    label: str,
    x: float,
    y: float,
    rcs_total: float,
    rng: np.random.Generator,
    height_m: float = 4.0,
    corner: bool = False,
) -> list:
    """A vertical pole (lamp post, sign post) or a corner reflector.

    Poles are cylinders, hence omnidirectional, so they stay bright from any view
    and act as sparse landmarks for off-path evaluation. A corner reflector
    (`corner=True`) is a compact, very bright specular point facing back down
    the road.
    """
    pts = []
    if corner:
        pts.append(
            {
                "label": label,
                "type": "static",
                "p0_world": np.array([x, y, 1.0], dtype=np.float64),
                "velocity": np.zeros(3, dtype=np.float64),
                "accel": None,
                "rcs": float(rcs_total * rng.uniform(0.9, 1.1)),
                "wall_normal": [-1.0, 0.0, 0.0],
                "specularity": 4.0,
            }
        )
        return pts
    heights = np.arange(0.4, height_m + 1e-9, 0.8)
    rcs_each = rcs_total / len(heights)
    for pz in heights:
        pts.append(
            {
                "label": label,
                "type": "static",
                "p0_world": np.array(
                    [x + rng.uniform(-0.05, 0.05), y + rng.uniform(-0.05, 0.05), pz],
                    dtype=np.float64,
                ),
                "velocity": np.zeros(3, dtype=np.float64),
                "accel": None,
                "rcs": float(rcs_each * rng.uniform(0.8, 1.2)),
                "wall_normal": None,
                "specularity": 0.0,
            }
        )
    return pts


# ── Reflector motion ─────────────────────────────────────────────────────────


def reflector_pos(ref: dict, frame_idx: int, dt: float) -> np.ndarray:
    """World-frame position [3] at frame_idx: p0 + v·t (+ ½·a·t² when the
    reflector carries a constant "accel" [3])."""
    t = frame_idx * dt
    p = ref["p0_world"] + ref["velocity"] * t
    a = ref["accel"]
    if a is not None:
        p = p + 0.5 * np.asarray(a, dtype=np.float64) * t * t
    return p


def reflector_vel(ref: dict, frame_idx: int, dt: float) -> np.ndarray:
    """World-frame velocity [3] at frame_idx (v + a·t under constant accel)."""
    v = ref["velocity"].copy()
    a = ref["accel"]
    if a is not None:
        v = v + np.asarray(a, dtype=np.float64) * (frame_idx * dt)
    return v
