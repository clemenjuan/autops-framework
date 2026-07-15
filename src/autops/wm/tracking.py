"""Required W&B tracking for command-driven world-model training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def require_wandb() -> Any:
    """Import W&B only when the world-model training workflow is invoked."""

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - exercised without the wm extra
        raise RuntimeError(
            "world-model training requires W&B; install the 'wm' extra"
        ) from exc
    return wandb


@dataclass
class WandbTrainingRun:
    """Small fail-closed wrapper around one W&B training run."""

    _wandb: Any
    _run: Any
    project: str

    @classmethod
    def start(
        cls,
        *,
        project: str,
        entity: str | None,
        name: str,
        config: Mapping[str, Any],
        trace_path: str | Path,
        trace_metadata: Mapping[str, Any],
    ) -> WandbTrainingRun:
        selected_project = project.strip()
        if not selected_project:
            raise ValueError("W&B project must be non-empty")
        wandb = require_wandb()
        run = wandb.init(
            project=selected_project,
            entity=entity or None,
            name=name,
            job_type="train-lewm",
            config=dict(config),
            tags=("autops", "lewm"),
            settings=wandb.Settings(save_code=False),
        )
        if run is None:
            raise RuntimeError("W&B did not initialize a training run")
        tracker = cls(wandb, run, selected_project)
        try:
            tracker._log_trace(trace_path, trace_metadata)
        except BaseException:
            run.finish(exit_code=1)
            raise
        return tracker

    @property
    def run_id(self) -> str:
        return str(self._run.id)

    @property
    def url(self) -> str:
        return str(getattr(self._run, "url", "") or "")

    def _log_trace(
        self, trace_path: str | Path, trace_metadata: Mapping[str, Any]
    ) -> None:
        digest = str(trace_metadata["trace_sha256"])
        artifact = self._wandb.Artifact(
            name=f"eventsat-trace-{digest[:12]}",
            type="dataset",
            metadata=dict(trace_metadata),
        )
        artifact.add_file(str(Path(trace_path)), name="trace.npz")
        self._run.log_artifact(artifact)

    def log_validation(self, step: int, metrics: Mapping[str, float]) -> None:
        self._run.log(dict(metrics), step=step)

    def log_checkpoint(
        self, checkpoint_path: str | Path, metadata: Mapping[str, Any]
    ) -> None:
        artifact = self._wandb.Artifact(
            name=f"eventsat-lewm-{self.run_id}",
            type="model",
            metadata=dict(metadata),
        )
        artifact.add_file(str(Path(checkpoint_path)), name="model.pt")
        self._run.log_artifact(artifact)
        for key, value in metadata.items():
            self._run.summary[key] = value

    def finish(self, *, exit_code: int) -> None:
        self._run.finish(exit_code=exit_code)


__all__ = ["WandbTrainingRun", "require_wandb"]
