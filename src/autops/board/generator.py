"""Build the one public board from completed, provenance-bearing result documents."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from autops.board.evidence import validate_result_document
from autops.board.model import LeWMEvidence


@dataclass(frozen=True)
class BoardRun:
    coordinate: str
    organisation: str
    paradigm: str
    representation: str
    episodes: int
    metrics: dict[str, float]
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
        source=source,
        variant=_variant(experiment, evidence.lewm),
        lewm=evidence.lewm,
        config_sha256=evidence.config_sha256,
    )


def discover_runs(results_root: str | Path) -> list[BoardRun]:
    """Load every canonical ``results.json`` beneath a runtime result root."""

    root = Path(results_root)
    paths = [root] if root.is_file() else sorted(root.rglob("results.json"))
    runs = [load_completed_run(path) for path in paths]
    if not runs:
        raise ValueError(f"no completed results found under {root}")
    identities = [(run.coordinate, run.config_sha256) for run in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("board input contains a duplicate experiment configuration")
    return sorted(runs, key=lambda run: run.coordinate)


def _cell(value: str, *, numeric: bool = False, title: str | None = None) -> str:
    class_name = ' class="numeric"' if numeric else ""
    title_attribute = f' title="{html.escape(title, quote=True)}"' if title else ""
    return f"<td{class_name}{title_attribute}>{html.escape(value)}</td>"


def _render_table(runs: list[BoardRun]) -> tuple[str, str]:
    metric_names = sorted({name for run in runs for name in run.metrics})
    headers = ["coordinate", "organisation", "paradigm", "representation", "n", *metric_names]
    header_html = "".join(f"<th>{html.escape(name)}</th>" for name in headers)
    rows: list[str] = []
    coordinate_counts = {
        coordinate: sum(item.coordinate == coordinate for item in runs)
        for coordinate in {item.coordinate for item in runs}
    }
    for run in runs:
        detailed = run.lewm is not None or coordinate_counts[run.coordinate] > 1
        representation = run.representation
        title = None
        if detailed:
            representation += f" · {run.variant} · cfg={run.config_sha256[:12]}"
            title = f"config SHA-256: {run.config_sha256}"
        values = [
            _cell(run.coordinate),
            _cell(run.organisation),
            _cell(run.paradigm),
            _cell(representation, title=title),
            _cell(str(run.episodes), numeric=True),
        ]
        values.extend(
            _cell(f"{run.metrics[name]:.6g}", numeric=True) if name in run.metrics else _cell("—")
            for name in metric_names
        )
        rows.append("<tr>" + "".join(values) + "</tr>")
    return header_html, "\n".join(rows)


def build_board(
    results_root: str | Path,
    output: str | Path,
    *,
    title: str = "AUTOPS results",
) -> Path:
    """Render an auditable static board; empty and incomplete inputs fail closed."""

    runs = discover_runs(results_root)
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


__all__ = ["BoardRun", "build_board", "discover_runs", "load_completed_run"]
