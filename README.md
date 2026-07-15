# AUTOPS Framework

AUTOPS is a controlled research testbed for comparing satellite **operations architectures**: who schedules, where decisions run, and which decision substrate they use. Mission goals and planning objectives are inputs, not design variables. The framework does not claim to optimize mission concepts or satellite hardware.

An experiment is a coordinate in three axes:

- organisation: `sas`, `cmas`, `dmas`, `hmas`, or `imas`;
- representation: rules, LLM single-shot/agentic variants, hybrid safety-shielded LLM variants, or LeWM-CEM planning;
- operational paradigm: conventional ground (`conventional`), autonomous ground (`ag`), autonomous onboard (`ao`), or dual-core autonomous hybrid (`ah`).

EventSat is the single-spacecraft scheduling benchmark. SSA is the constellation custody benchmark with sensing- and link-gated information flow. `rl` and `hrl` names are reserved for a later RLlib/PPO contribution; this repository intentionally contains no placeholder policy or RL dependency.

The design follows established spacecraft-operations practice and controlled autonomy comparisons, including Sellmaier et al., *Spacecraft Operations* (2022, [DOI](https://doi.org/10.1007/978-3-030-88593-9)) and Castano et al., “Operations for Autonomous Spacecraft” (2022, [DOI](https://doi.org/10.1109/AERO53065.2022.9843352)). LLM representations build on spacecraft-operator prompting and bounded language-agent architectures ([Rodriguez-Fernandez et al. 2024](https://doi.org/10.48550/arXiv.2404.00413); [Sumers et al. 2024](https://doi.org/10.48550/arXiv.2309.02427); [Yao et al. 2023](https://doi.org/10.48550/arXiv.2210.03629)).

## Quick start

```bash
uv sync --extra dev
uv run autops run eventsat/sas/ag/symb --episodes 1
uv run autops sweep eventsat --paradigm ag --representation symb --episodes 10 --seeds 42:51
uv run autops export eventsat/sas/ao/symb --episodes 2 --steps 64
uv run autops export eventsat/sas/ao/symb eventsat/sas/ag/symb --episodes 2 --steps 64
uv run autops board --manifest configs/papers/paper_a.yaml
uv run pytest
```

Experiments expand at runtime from `configs/matrix.yaml` and the mission configuration. Per-run changes use `--episodes`, `--seeds`, `--steps`, and `--set key=value`; no per-cell configuration files are generated.

Passing multiple compatible coordinates to `export` creates one mixed-policy corpus. `--episodes` and `--seeds` apply independently to every coordinate; the trace records each source coordinate, scientific configuration hash, source revision/dirty state, actual orbital backend, timestep, episode count, and seed sequence without local paths.

Install `--extra orbital` to use Orekit's Eckstein-Hechler J2 propagation. The source repository and built wheel already include `orekit-data.zip`; source checkouts keep it at the repository root. Without Java/Orekit, AUTOPS uses its documented seeded fallback. Install `--extra llm` for live providers or use the deterministic mock in tests. Install `--extra wm` for LeWM training and learned planning.

The learned-planning workflow ends with `uv run autops train evaluate TRACE --artifact PLANNER.json --output EVAL.json`. Evaluation uses the same categorical CEM and latent candidate scorer as closed-loop control, restricted to deterministic contexts from checkpoint-held-out episodes.

See [framework.md](docs/framework.md) for the complete matrix and metric contract, and [architecture.md](docs/architecture.md) for package boundaries and extension seams.
