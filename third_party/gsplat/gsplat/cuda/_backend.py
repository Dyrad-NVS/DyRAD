"""The radar CUDA extension as `_C`: the `gsplat.csrc` module built by `pip install -e ./gsplat`
(see the main README, Installation), or None when it is not built.

Modified from gsplat (Apache-2.0) for DyRAD.
"""

try:
    from gsplat import csrc as _C
except ImportError:
    _C = None

__all__ = ["_C"]
