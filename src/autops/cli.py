"""The single AUTOPS command-line surface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from autops.config import expand_coordinate, parse_overrides
from autops.core.runner import ExperimentRunner


def _seed_values(value: str) -> list[int]:
    if ":" not in value:
        return [int(part) for part in value.split(",") if part]
    start, end = (int(part) for part in value.split(":", 1))
    if end < start:
        raise argparse.ArgumentTypeError("seed range end must be >= start")
    return list(range(start, end + 1))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="autops")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one matrix coordinate")
    run.add_argument("coordinate")
    run.add_argument("--episodes", type=int, default=1)
    run.add_argument("--steps", type=int)
    run.add_argument("--seeds", type=_seed_values)
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--no-orekit", action="store_true")
    commands.add_parser("train", help="train a world model or probes")
    commands.add_parser("sweep", help="run an applicable matrix slice")
    commands.add_parser("export", help="export mission traces through the shared schema")
    commands.add_parser("board", help="build the unified static results board")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command != "run":
        raise SystemExit(f"{args.command!r} is not wired yet")
    seeds = args.seeds
    episodes = len(seeds) if seeds is not None else args.episodes
    overrides = parse_overrides(args.set)
    spec = expand_coordinate(
        args.coordinate,
        episodes=episodes,
        steps=args.steps,
        seeds=seeds,
        overrides=overrides,
    )
    result = ExperimentRunner(spec, prefer_orekit=not args.no_orekit).run()
    print(json.dumps({"experiment": spec.name, "metrics": result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
