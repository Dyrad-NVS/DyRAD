"""Boreas preprocessing: `python -m dyrad.preprocessing.boreas <command> [options]`.

Each command is a module of this package; `<command> --help` shows its options.
"""

from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "fetch": ("fetch", "download the radar frames (or labels) of a time window"),
    "prepare": ("prepare", "stage a labelled clip: poses, sensor.yaml"),
    "convert": ("convert", "staged clip -> RAD tensors, poses, ego velocity"),
    "labels": ("labels", "object boxes -> labels_boreas.csv"),
    "recentre": ("recentre", "recentre a window on its mean position (float32-safe poses)"),
    "prepare-window": (
        "prepare_window",
        "run the whole chain for one of the paper's windows (sparse2, sparse4, win55_104)",
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
    module = importlib.import_module(f"dyrad.preprocessing.boreas.{COMMANDS[argv[0]][0]}")
    sys.argv = [f"dyrad.preprocessing.boreas {argv[0]}"] + argv[1:]
    return module.main() or 0


if __name__ == "__main__":
    raise SystemExit(main())
