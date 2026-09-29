"""Training entry point: fit DyRAD to one sequence.

    python -m dyrad.train --config configs/radial/31_22_dyrad.yaml

The model and training loop live in dyrad.trainer (`dyrad.trainer.runner.Runner`).
"""

import argparse

from dyrad.config import load_config
from dyrad.trainer.runner import Runner


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", required=True, help="run config, e.g. configs/radial/31_22_dyrad.yaml"
    )
    args = parser.parse_args()
    Runner(load_config(args.config)).train()


if __name__ == "__main__":
    main()
