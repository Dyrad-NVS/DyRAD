"""RADIal preprocessing: `python -m dyrad.preprocessing.radial <command> [options]`.

Each command is a module of this package; `<command> --help` shows its options.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "pipeline": (
        "pipeline",
        "process a raw recording end to end (RAD tensors, poses, ego velocity)",
    ),
    "preprocess": (
        "preprocess",
        "raw recording -> RAD tensors, GPS poses, sensor.json, label_index_remap.npy",
    ),
    "ego-velocity": ("ego_velocity", "CAN wheel speed -> ego_vel_can.npy"),
    "poses": ("poses", "CAN dead-reckoned radar poses (poses_can/)"),
    "prepare": (
        "prepare_sequence",
        "prepare a training window: stub, configs, norm.json, init clouds",
    ),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        width = max(len(c) for c in COMMANDS)
        print(__doc__.strip() + "\n\ncommands:")
        for name, (_, help_) in COMMANDS.items():
            print(f"  {name:<{width}}  {help_}")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    module = importlib.import_module(f"dyrad.preprocessing.radial.{COMMANDS[argv[0]][0]}")
    sys.argv = [f"dyrad.preprocessing.radial {argv[0]}"] + argv[1:]
    return module.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
