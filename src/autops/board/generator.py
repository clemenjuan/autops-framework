"""Build the one public board from completed, provenance-bearing result documents."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from autops.board.evidence import validate_result_document
from autops.board.manifest import verified_result_paths
from autops.board.model import LeWMEvidence


@dataclass(frozen=True)
class BoardRun:
    coordinate: str
    organisation: str
    paradigm: str
    representation: str
    episodes: int
    metrics: dict[str, float]
    metric_names: dict[str, str]
    source: Path
    config_sha256: str
    variant: str
    lewm: LeWMEvidence | None


def _flatten_config(value: dict[str, object], prefix: str = "") -> list[str]:
    items: list[str] = []
    for key, raw in sorted(value.items()):
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(raw, dict):
            items.extend(_flatten_config(raw, name))
        elif isinstance(raw, (str, int, float, bool)) or raw is None:
            items.append(f"{name}={raw}")
    return items


def _variant(experiment: dict[str, object], lewm: LeWMEvidence | None) -> str:
    if lewm is not None:
        treatment = lewm.treatment
        return (
            f"mode={treatment.mission_mode} · "
            f"hold/H={treatment.plan_hold}/{treatment.horizon} · "
            f"CEM={treatment.samples}/{treatment.elites}x{treatment.iterations} · "
            f"{treatment.onboard_compute_w:g} W · "
            f"artifact={treatment.artifact_sha256[:8]} · "
            f"trace={treatment.trace_sha256[:8]} · "
            f"checkpoint={treatment.checkpoint_sha256[:8]}"
        )
    config = experiment.get("representation_config", {})
    if not isinstance(config, dict):
        return "default"
    items = _flatten_config(config)
    if len(items) > 6:
        items = [*items[:6], "…"]
    return " · ".join(items) if items else "default"


def load_completed_run(path: str | Path) -> BoardRun:
    """Read one result and reject partial or presentation-only data."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    evidence = validate_result_document(payload, source)
    experiment = evidence.experiment
    representation = experiment.get("representation") or "+".join(
        str(experiment.get(name, ""))
        for name in ("onboard_representation", "ground_representation")
    )
    return BoardRun(
        coordinate=str(experiment.get("coordinate", experiment.get("name", ""))),
        organisation=str(experiment.get("organisation", "")),
        paradigm=str(experiment.get("paradigm", "")),
        representation=str(representation),
        episodes=evidence.episodes,
        metrics=evidence.metrics,
        metric_names=evidence.metric_names,
        source=source,
        variant=_variant(experiment, evidence.lewm),
        lewm=evidence.lewm,
        config_sha256=evidence.config_sha256,
    )


def _runs_from_paths(paths: list[Path]) -> list[BoardRun]:
    runs = [load_completed_run(path) for path in paths]
    if not runs:
        raise ValueError("no completed results were selected")
    identities = [(run.coordinate, run.config_sha256) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("board input contains a duplicate experiment configuration")
    return sorted(runs, key=lambda run: run.coordinate)


def discover_runs(results_root: str | Path) -> list[BoardRun]:
    """Load completed results for non-paper exploratory boards."""

    root = Path(results_root)
    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    return _runs_from_paths(paths)


def discover_manifest_runs(manifest_path: str | Path, runtime_root: str | Path) -> list[BoardRun]:
    """Load only approved, identity-verified paper results."""

    paths = verified_result_paths(manifest_path, runtime_root)
    return _runs_from_paths(paths)


def _cell(value: str, *, numeric: bool = False, title: str | None = None) -> str:
    class_name = ' class="numeric"' if numeric else ""
    title_attribute = f' title="{html.escape(title, quote=True)}"' if title else ""
    return f"<td{class_name}{title_attribute}>{html.escape(value)}</td>"


def _representation_cell(run: BoardRun, *, detailed: bool) -> str:
    name = html.escape(run.representation)
    if not detailed:
        return f'<td class="rep">{name}</td>'
    variant = html.escape(f"{run.variant} · cfg={run.config_sha256[:12]}")
    title = html.escape(f"config SHA-256: {run.config_sha256}", quote=True)
    return (
        f'<td class="rep" title="{title}"><div class="rep-name">{name}</div>'
        f'<div class="rep-variant">{variant}</div></td>'
    )


def _render_table(runs: list[BoardRun]) -> tuple[str, str]:
    metric_ids = sorted({name for run in runs for name in run.metrics})
    metric_labels = {
        metric_id: label
        for run in runs
        for metric_id, label in run.metric_names.items()
        if metric_id in metric_ids
    }
    fixed_headers = "".join(
        f"<th>{html.escape(name)}</th>"
        for name in ("coordinate", "organisation", "paradigm", "representation", "n")
    )
    metric_headers = "".join(
        f'<th title="{html.escape(metric_labels.get(metric_id, ""), quote=True)}">'
        f"{html.escape(metric_id)}</th>"
        for metric_id in metric_ids
    )
    header_html = fixed_headers + metric_headers
    rows: list[str] = []
    coordinate_counts = {
        coordinate: sum(item.coordinate == coordinate for item in runs)
        for coordinate in {item.coordinate for item in runs}
    }
    for run in runs:
        detailed = run.lewm is not None or coordinate_counts[run.coordinate] > 1
        values = [
            _cell(run.coordinate),
            _cell(run.organisation),
            _cell(run.paradigm),
            _representation_cell(run, detailed=detailed),
            _cell(str(run.episodes), numeric=True),
        ]
        values.extend(
            _cell(f"{run.metrics[name]:.6g}", numeric=True) if name in run.metrics else _cell("—")
            for name in metric_ids
        )
        rows.append("<tr>" + "".join(values) + "</tr>")
    return header_html, "\n".join(rows)


def _write_board(
    runs: list[BoardRun],
    output: str | Path,
    *,
    title: str = "AUTOPS results",
) -> Path:
    headers, rows = _render_table(runs)
    template = (
        Path(__file__).with_name("templates").joinpath("index.html").read_text(encoding="utf-8")
    )
    rendered = (
        template.replace("{{TITLE}}", html.escape(title))
        .replace("{{GENERATED_AT}}", datetime.now(UTC).isoformat(timespec="seconds"))
        .replace("{{RUN_COUNT}}", str(len(runs)))
        .replace("{{HEADERS}}", headers)
        .replace("{{ROWS}}", rows)
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(destination)
    return destination


def build_board(
    results_root: str | Path,
    output: str | Path,
    *,
    title: str = "AUTOPS results",
) -> Path:
    """Render a non-paper exploratory board from completed result files."""

    return _write_board(discover_runs(results_root), output, title=title)


def build_manifest_board(
    manifest_path: str | Path,
    runtime_root: str | Path,
    output: str | Path,
    *,
    title: str = "AUTOPS results",
) -> Path:
    """Render a paper board only from approved manifest identities."""

    return _write_board(
        discover_manifest_runs(manifest_path, runtime_root),
        output,
        title=title,
    )


__all__ = [
    "BoardRun",
    "build_board",
    "build_manifest_board",
    "discover_manifest_runs",
    "discover_runs",
    "load_completed_run",
]
