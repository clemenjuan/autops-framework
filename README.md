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
uv run autops sweep eventsat --episodes 10 --seeds 42:51
uv run autops export eventsat/sas/ao/symb --episodes 2 --steps 64
uv run autops board
uv run pytest
```

Experiments expand at runtime from `configs/matrix.yaml` and the mission configuration. Per-run changes use `--episodes`, `--seeds`, `--steps`, and `--set key=value`; no per-cell configuration files are generated.

Install `--extra orbital` to use Orekit's Eckstein-Hechler J2 propagation and place `orekit-data.zip` in the repository root. Without Java/Orekit, AUTOPS uses its documented seeded fallback. Install `--extra llm` for live providers or use the deterministic mock in tests. Install `--extra wm` for LeWM training and learned planning.

See [framework.md](docs/framework.md) for the complete matrix and metric contract, and [architecture.md](docs/architecture.md) for package boundaries and extension seams.
