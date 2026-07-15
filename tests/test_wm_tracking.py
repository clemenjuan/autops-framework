from __future__ import annotations

from pathlib import Path

import autops.wm.tracking as tracking


class _Artifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path, *, name):
        self.files.append((path, name))


class _Run:
    id = "run123"
    url = "https://example.invalid/run123"

    def __init__(self):
        self.artifacts = []
        self.logs = []
        self.summary = {}
        self.exit_code = None

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def log(self, metrics, *, step):
        self.logs.append((step, metrics))

    def finish(self, *, exit_code):
        self.exit_code = exit_code


class _Wandb:
    Artifact = _Artifact

    class Settings:
        def __init__(self, **kwargs):
            self.values = kwargs

    def __init__(self):
        self.run = _Run()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def test_wandb_training_run_logs_trace_metrics_and_checkpoint(tmp_path: Path, monkeypatch) -> None:
    fake = _Wandb()
    monkeypatch.setattr(tracking, "require_wandb", lambda: fake)
    trace = tmp_path / "trace.npz"
    checkpoint = tmp_path / "model.pt"
    trace.write_bytes(b"trace")
    checkpoint.write_bytes(b"model")

    run = tracking.WandbTrainingRun.start(
        project="space-world-models",
        entity=None,
        name="test-run",
        config={"training": {"max_steps": 2}},
        trace_path=trace,
        trace_metadata={"trace_sha256": "a" * 64},
    )
    run.log_validation(2, {"validation/loss": 0.5})
    run.log_checkpoint(checkpoint, {"best_validation_loss": 0.5})
    run.finish(exit_code=0)

    assert fake.init_kwargs["project"] == "space-world-models"
    assert fake.init_kwargs["settings"].values == {"save_code": False}
    assert [(item.type, item.files[0][1]) for item in fake.run.artifacts] == [
        ("dataset", "trace.npz"),
        ("model", "model.pt"),
    ]
    assert fake.run.logs == [(2, {"validation/loss": 0.5})]
    assert fake.run.summary["best_validation_loss"] == 0.5
    assert fake.run.exit_code == 0
