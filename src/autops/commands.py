"""Implementation of the single ``autops`` command surface."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autops.board.generator import build_manifest_board
from autops.config import asset_root, expand_coordinate, parse_overrides, runtime_root
from autops.core.exporter import export_traces
from autops.core.probe_audit import audit_probe_decodability
from autops.core.runner import ExperimentRunner
from autops.core.workflows import (
    evaluate_lewm_cem,
    fit_planner_artifact,
    run_sweep,
    train_world_model,
)


def seed_values(value: str) -> list[int]:
    if ":" not in value:
        try:
            seeds = [int(part) for part in value.split(",") if part]
        except ValueError as exc:
            raise argparse.ArgumentTypeError("seeds must be integers") from exc
        if not seeds:
            raise argparse.ArgumentTypeError("at least one seed is required")
        return seeds
    try:
        start, end = (int(part) for part in value.split(":", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed range must be START:END") from exc
    if end < start:
        raise argparse.ArgumentTypeError("seed range end must be >= start")
    return list(range(start, end + 1))


def _run_arguments(command: argparse.ArgumentParser, *, coordinate: bool = True) -> None:
    if coordinate:
        command.add_argument("coordinate")
    command.add_argument("--episodes", type=int, default=1)
    command.add_argument("--steps", type=int)
    command.add_argument("--seeds", type=seed_values)
    command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    command.add_argument("--no-orekit", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="autops")
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one matrix coordinate")
    _run_arguments(run)

    sweep = commands.add_parser("sweep", help="run an applicable matrix slice")
    sweep.add_argument("mission", choices=("eventsat", "ssa"))
    _run_arguments(sweep, coordinate=False)
    sweep.add_argument("--organisation")
    sweep.add_argument("--paradigm")
    sweep.add_argument("--representation")
    sweep.add_argument("--dry-run", action="store_true")

    export = commands.add_parser(
        "export", help="export one or more coordinates through the shared trace schema"
    )
    _run_arguments(export, coordinate=False)
    export.add_argument(
        "coordinates",
        nargs="+",
        help="compatible coordinates; --episodes and --seeds apply to each coordinate",
    )
    export.add_argument("--output", type=Path)

    train = commands.add_parser("train", help="train LeWM or fit probes/artifact")
    training = train.add_subparsers(dest="training_command", required=True)
    wm = training.add_parser("wm", help="train the canonical LeWM checkpoint")
    wm.add_argument("trace", type=Path)
    wm.add_argument("--output", type=Path, required=True)
    wm.add_argument("--max-steps", type=int, default=150_000)
    wm.add_argument("--batch-size", type=int, default=64)
    wm.add_argument("--device", default="cpu")
    wm.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "space-world-models"),
    )
    wm.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    wm.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    probes = training.add_parser("probes", help="fit probes and write a planner artifact")
    probes.add_argument("trace", type=Path)
    probes.add_argument("--checkpoint", type=Path, required=True)
    probes.add_argument("--output", type=Path, required=True)
    probes.add_argument("--ridge", type=float)
    probes.add_argument("--device", default="cpu")
    probes.add_argument("--seed", type=int, default=3072)
    evaluate = training.add_parser(
        "evaluate", help="evaluate artifact CEM on held-out trace contexts"
    )
    evaluate.add_argument("trace", type=Path)
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--mission-mode", default="science")
    evaluate.add_argument("--max-contexts", type=int, default=32)
    audit = training.add_parser("audit", help="compare linear and nonlinear frozen-feature probes")
    audit.add_argument("trace", type=Path)
    audit.add_argument("--checkpoint", type=Path, required=True)
    audit.add_argument("--features", choices=("latents", "obs"), default="latents")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--window", type=int, default=1)
    audit.add_argument("--mlp-epochs", type=int, default=100)
    audit.add_argument("--hidden", default="256,128")
    audit.add_argument("--device", default="cpu")
    audit.add_argument("--seed", type=int, default=3072)
    audit.add_argument("--ridge", type=float, default=1e-3)
    audit.add_argument("--learning-rate", type=float, default=1e-3)
    audit.add_argument("--weight-decay", type=float, default=1e-4)
    audit.add_argument("--validation-episodes", type=int, default=3)

    board = commands.add_parser("board", help="build the unified static results board")
    board.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/papers/paper_a.yaml"),
    )
    board.add_argument("--output", type=Path, default=Path("boards/index.html"))
    board.add_argument("--title", default="AUTOPS results")
    return root


def _experiment_spec(args: argparse.Namespace, coordinate: str | None = None):
    seeds = args.seeds
    episodes = len(seeds) if seeds is not None else args.episodes
    return expand_coordinate(
        args.coordinate if coordinate is None else coordinate,
        episodes=episodes,
        steps=args.steps,
        seeds=seeds,
        overrides=parse_overrides(args.set),
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    spec = _experiment_spec(args)
    result = ExperimentRunner(spec, prefer_orekit=not args.no_orekit).run()
    return {"experiment": spec.name, "metrics": result["metrics"]}


def _sweep(args: argparse.Namespace) -> dict[str, Any]:
    seeds = args.seeds
    episodes = len(seeds) if seeds is not None else args.episodes
    return run_sweep(
        args.mission,
        episodes=episodes,
        steps=args.steps,
        seeds=seeds,
        overrides=parse_overrides(args.set),
        organisation=args.organisation,
        paradigm=args.paradigm,
        representation=args.representation,
        prefer_orekit=not args.no_orekit,
        dry_run=args.dry_run,
    )


def _export(args: argparse.Namespace) -> dict[str, Any]:
    specs = [_experiment_spec(args, coordinate) for coordinate in args.coordinates]
    destination = export_traces(specs, args.output, prefer_orekit=not args.no_orekit)
    if len(specs) == 1:
        return {"experiment": specs[0].name, "trace": str(destination)}
    return {
        "experiments": [spec.name for spec in specs],
        "episodes_per_coordinate": specs[0].episodes,
        "trace": str(destination),
    }


def _train(args: argparse.Namespace) -> dict[str, Any]:
    if args.training_command == "wm":
        return train_world_model(
            args.trace,
            args.output,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            device=args.device,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_name=args.wandb_name,
        )
    if args.training_command == "probes":
        return fit_planner_artifact(
            args.trace,
            args.checkpoint,
            args.output,
            device=args.device,
            ridge=args.ridge,
            seed=args.seed,
        )
    if args.training_command == "evaluate":
        return evaluate_lewm_cem(
            args.trace,
            args.artifact,
            args.output,
            device=args.device,
            mission_mode=args.mission_mode,
            max_contexts=args.max_contexts,
        )
    try:
        hidden = tuple(int(width) for width in args.hidden.split(",") if width)
    except ValueError as exc:
        raise ValueError("--hidden must be a comma-separated list of integers") from exc
    return audit_probe_decodability(
        args.trace,
        checkpoint_path=args.checkpoint,
        features=args.features,
        output=args.output,
        feature_window=args.window,
        mlp_epochs=args.mlp_epochs,
        hidden=hidden,
        device=args.device,
        seed=args.seed,
        ridge=args.ridge,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        validation_episodes=args.validation_episodes,
    )


def _board(args: argparse.Namespace) -> dict[str, Any]:
    root = runtime_root()
    manifest = args.manifest if args.manifest.is_absolute() else asset_root() / args.manifest
    output = args.output if args.output.is_absolute() else root / args.output
    return {"board": str(build_manifest_board(manifest, root, output, title=args.title))}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    handlers = {
        "run": _run,
        "sweep": _sweep,
        "export": _export,
        "train": _train,
        "board": _board,
    }
    result = handlers[args.command](args)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "parser", "seed_values"]
