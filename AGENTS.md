# Conduct

1. This is a top-university research repository. Code quality is part of the research output: write concise, elegant, best-practice Python, and keep only components that serve the papers or the operations-architecture comparison.
2. No duplicate or useless functions. Search before writing. There is one definition per concept: one trace schema, artifact contract, CEM search, board generator, and environment per mission.
3. Ask before structural changes: new dependencies or top-level modules; changes to organisation, representation, paradigm, metrics, physics, prompts, or the planner-artifact contract. Focused work inside the approved structure does not need permission.
4. No trash files. Do not commit one-off scripts, generated configs, notebooks, caches, run artifacts, or scratch output. Prefer improving an existing file.
5. Boards contain only completed, decision-relevant results. Ask before adding panels, columns, or derived metrics. Always show sample size; never publish placeholder rows.
6. Report honestly. Separate bugs from findings, retain negative results privately, and identify the run behind every quantitative claim.
7. This repository is public. `research/` is private and ignored; never cite or expose its contents in code, public docs, commits, or boards. Never track hostnames, IPs, endpoints, credentials, or personal paths. Put machine details in ignored `CLAUDE.local.md`.
8. Ground modelling and algorithm choices in literature verified through the owner's Zotero collection or a DOI. Put durable rationale and citations in tracked code/docs. Flag missing references; never invent them.

# Repository contract

- Package: `autops`; Python 3.11+; package management and commands use `uv run`.
- Experiments are matrix coordinates expanded from `configs/matrix.yaml`, never generated per-cell YAML.
- One `ExperimentSpec` in `autops.config`; no legacy names or compatibility aliases.
- Representation plugins implement `encode_observation`, `select_action`, optional `update`, and `last_rationale`; decorator registration is discovered automatically.
- `rl` and `hrl` are reserved, deferred cells. Keep Gymnasium-compatible observation/action seams, but ship no RLlib, Torch-RL, or stand-ins.
- The world model is an ordinary representation. Training and deployment share the schema, artifact contract, and CEM implementation in `autops.wm`.
- Fixed memory is the fairness invariant. Writable online memory is deferred behind the interface.
- Prefer files below 500 lines and functions below 80 lines; split by concern before exceeding them.
- Keep source and tests Ruff-clean. Commit `uv.lock`; every document cited by code must be tracked.

# Scientific invariants

- The axes are Organisation × Representation × Operational paradigm. EventSat fixes `sas`; SSA exercises `sas`, `cmas`, `dmas`, `hmas`, and `imas`.
- EventSat documents the canonical 32-cell design, while only real implementations run. `lewm-cem` is a first-class additional representation.
- Preserve all M-01…M-14 definitions, paired episode seeds, mission prompts, and scenario parameters.
- EventSat uses a 60 s step, 400 km / 97.4° SSO, launch-lottery RAAN/ArgP/TA, a 70 Wh battery, 50 kbps effective S-band, a three-pool data pipeline, 135 s settling, and environment-enforced anomaly safe mode.
- Jetson compute power is added only to modes where the Jetson would otherwise be off; never double-count modes in `jetson_active_modes`.
- Orekit Eckstein-Hechler J2 is the preferred orbital backend; the deterministic-seeded simplified backend is the documented fallback.
- LeWM uses a JEPA-style objective, embed dimension 192, history 3, degenerate-target probe checks, affine `W,b` probes, and artifact `target_std` scale normalization by default.
- LLM prompts are precision invariants. The decisive agentic prompt, prompt-folded echo telemetry, what-if-only tools, and `max_agentic_steps=3` prevent thinking-model spirals.

# Working practices

- Run `uv sync --extra dev` for local development; add `--extra orbital`, `--extra llm`, or `--extra wm` only when needed.
- Live LLM access comes only from `OLLAMA_HOST` or `OPENAI_API_KEY`; tests always use the deterministic mock.
- LLM responses use a SHA256 cache under ignored runtime storage. A cache hit reports no new provider latency or token usage.
- Orekit requires Java 17 and `orekit-data.zip` at repository root.
- Use small conventional commits on `main`, with tests green at every commit.
- Before finishing, run tests and Ruff, remove generated caches, and review staged files for private paths or material.
