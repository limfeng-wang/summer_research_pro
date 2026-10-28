"""Command-line entry points for the standalone research release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "run-hierarchical":
        from dental_ai.run import main as run

        return run(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run-hierarchical", help="Run annotation; append --help for inference options")
    validate = commands.add_parser("validate-data", help="Verify a separately supplied private dataset")
    validate.add_argument("--dataset", type=Path, required=True)
    models = commands.add_parser("check-models", help="Check configured local model directories")
    models.add_argument("--config", default="configs/model_stack.yaml")
    models.add_argument("--models-root", default="models")
    args = parser.parse_args(arguments)
    if args.command == "validate-data":
        validator = Path(__file__).resolve().parents[1] / "data/publication_dataset/validate.py"
        return subprocess.call([sys.executable, str(validator), "--dataset", str(args.dataset)])
    if args.command == "check-models":
        from dental_ai.model_config import check_model_paths, load_model_stack_config

        report = check_model_paths(load_model_stack_config(args.config), args.models_root)
        print(json.dumps(report, indent=2))
        return 0 if all(item["has_config_json"] for item in report.values()) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
