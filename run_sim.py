import argparse
import json
from pathlib import Path

from modules.simulator.runner import run_from_config


def load_config(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run SimLKAS simulation via config")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/demo.json",
        help="Path to configuration JSON (default: configs/demo.json)",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    run_from_config(config)


if __name__ == "__main__":
    main()
