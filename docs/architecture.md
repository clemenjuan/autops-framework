# Architecture

AUTOPS has one implementation for each scientific contract: one experiment spec, one
environment per mission, one world-model trace schema, one planner-artifact schema, one
categorical CEM search, and one results board. This makes the comparison boundary
inspectable: missions own truth, paradigms own authority and delay, organisations own
multi-satellite allocation, and representations own decision substrate.

## Runtime flow

```text
matrix coordinate + mission YAML + overrides
                  │
                  ▼
          frozen ExperimentSpec
                  │
         ┌────────┴────────┐
         ▼                 ▼
   mission environment   representation plugin
         │                 │
         └──── paradigm / organisation ────┐
                                           ▼
                                  requested action(s)
                                           │
                                           ▼
                            physics and safety resolution
                                           │
                      observation, reward, metrics, trace row
                                           │
                                           ▼
                           provenance-bearing results.json
```

The environment is the only source of physical state. Ground paradigms receive delayed
resource/health state while deterministic almanac timing can be refreshed at a contact
boundary. An RF transfer still requires physical contact. AO decides locally; AH has
independent onboard and ground cores with explicit promotion and arbitration. These
authority boundaries reflect the operational allocation problem described by Castano
et al. [1], rather than treating autonomy as a single scalar level.

Onboard LLM representations use the same AO boundary as other planners. A local model
produces one immediate action and a bounded schedule, the adapter executes held actions
for `plan_hold`, and only a new planning event invokes the model again. The hybrid LLM
variants recheck every held action against fresh telemetry before execution; the mission
environment still applies the final physical and safety resolution. Measured inference
time is carried as `planner_active_s`, while the mission power model converts that time
to incremental energy only in modes where the Jetson base load is otherwise absent.

## Package boundaries

| Package | Owns | Must not own |
|---|---|---|
| `autops.config` | coordinate parsing, matrix applicability, one Pydantic spec | factories, legacy aliases, generated configs |
| `autops.core` | plugin registry, lifecycle, public-safe provenance, RL-ready types | mission physics or model-specific planning |
| `autops.missions` | EventSat and SSA truth, transitions, rewards, metrics | provider clients or paradigm delay |
| `autops.orbital` | Orekit/fallback propagation, eclipse/access, link and ISL budgets | scheduling policy |
| `autops.paradigms` | CG/AG/AO/AH authority, latency, stale-view semantics | physics constants |
| `autops.organisations` | SAS/CMAS/DMAS/HMAS/IMAS task and knowledge allocation | sensing/link truth |
| `autops.representations` | symbolic, LLM/hybrid, and LeWM-CEM plugins | runner special cases |
| `autops.llm` | environment-selected clients, SHA-256 cache, deterministic mock, prompts | mission-state mutation |
| `autops.wm` | trace/dataset, JEPA model/training, probes, artifact, CEM | duplicate mission adapters |
| `autops.board` | validation and static rendering of completed results | metric recomputation or placeholder data |

Optional dependencies are lazy. Importing core AUTOPS does not import Torch, start a
JVM, or contact an LLM endpoint. `orbital`, `llm`, and `wm` extras activate those
surfaces; the empty `rl` extra reserves the future contribution without pulling in an
RL stack.

## Representation plugin seam

A representation implements four small operations:

```python
encode_observation(observation)
select_action(context)
update(transition)       # optional
last_rationale           # optional property value
```

Plugins register by mission, token, and role (`onboard`, `ground`, or `any`). Discovery
walks the built-in representation package, the requested mission package only, and the
`autops.representations` entry-point group. The runner does not maintain an import list.
`SpaceSpec` exposes the observation/action boundary a later Gymnasium/RLlib adapter can
use without adding Gymnasium to the base install.

## World-model path

The trace row stores the pre-transition observation/state and requested one-hot action;
reward, resolved action, and forced flag describe that row's transition. The next row is
the resulting observation. EventSat uses 25 observations, 25 state attributes, and 7
actions. SSA adds a satellite axis and uses its canonical 6-action order. NPZ files are
pickle-free and carry names, axes, episode IDs, seeds, and a versioned schema.

Dataset windows never cross episode boundaries. LeWM uses a 192-dimensional embedding
and history 3 with a JEPA-style action-conditioned objective. Probes are affine
(`W`, `b`) and store target means/standard deviations plus degenerate-target labels.
The relocatable planner artifact contains those probes, normalisation, action names,
relative checkpoint path, and CEM parameters. Both evaluation and closed-loop planning
call the same CEM function. This representation follows the world-model control pattern
demonstrated by Hafner et al. [2]; the exact AUTOPS contract is deliberately narrower
and auditable.

Before scoring or elite selection, every sampled EventSat sequence is projected across
the complete horizon by `autops.wm.guidance.project_executable_candidates`. It mirrors
settling, health and battery constraints and calls the atomic transitions in
`autops.missions.eventsat.transitions` for observe, compress, detect, CAN transfer, and
physical-contact downlink. CEM therefore returns executable requested commands, and the
learned scorer, pipeline shaping, and analytical oracle consume the identical projected
candidate bank. Held-action fallbacks remain as a runtime guard and report their own
repair rate.

The `analytical-cem` reference [3, 4, 5] replaces only the latent rollout/readout with
canonical terminal attributes computed from that projection. Its exogenous contact and sunlight
arrays are generated by the environment's active orbital backend (Orekit for paper
runs); action-dependent power, settling, storage, and byte-pipeline dynamics remain the
same shared functions used by the truth environment. This isolates propagation model
quality without changing the optimizer or executable candidates.

Terminal affine remains the deployed readout until selection-level evidence justifies a
change. `autops.wm.scoring.candidate_selection_metrics` compares terminal-affine,
windowed-affine, and MLP scores on one shared bank using top-elite overlap and analytical
regret; probe R²/AUC alone is not deployment evidence.

Planner compute uses an event model rather than a full-step surrogate:

\[
E_{event}=P_{active}t_{plan}+E_{boot}+P_{idle}t_{idle}.
\]

`planner_active_s` is measured around CEM or local LLM execution. The power and
boot/idle terms are declared under `power` in the mission configuration and are labelled
`assumed` unless replaced by hardware evidence. Incremental planner energy is zero in
modes whose base load already includes the Jetson. Board-level INA3221 rails cannot
replace the scalar model unless a non-overlapping total-input boundary is established;
the current hardware results therefore retain every exposed rail separately.

## Commands and runtime data

```bash
uv run autops run COORDINATE [--episodes N] [--seeds A:B] [--set key=value]
uv run autops sweep MISSION [filters]
uv run autops export COORDINATE [COORDINATE ...] [run options]
uv run autops train wm ...
uv run autops train probes ...
uv run autops train evaluate TRACE --artifact PLANNER.json --output EVAL.json
uv run autops train audit ...
uv run autops board [--manifest PATH] [--output PATH]
```

`run` emits an append-only, content-addressed result JSON. `sweep` expands applicable matrix coordinates;
it does not create YAML. `export` writes the shared trace contract for either mission;
with multiple compatible coordinates, episode and seed options apply per coordinate and
the output retains source hashes/revisions, dirty state, actual orbital backend, and canonical concatenated episode IDs.
`train wm` consumes that trace and writes a checkpoint; `train probes` emits a relocatable
planner artifact. `train evaluate` verifies the trace, artifact, and checkpoint hashes,
then runs the deployed categorical CEM and shared latent scorer on deterministic contexts
from held-out episodes. Its portable JSON records plans, learned and realised scores,
attribute calibration, aggregate evidence, and source/runtime provenance without input
paths. `train audit` compares linear and nonlinear frozen-feature readouts. The paper-facing
`board` reads only approved identities from the selected paper manifest (Paper B by
default) and verifies the result ID, commit, configuration, and checkpoint hashes.
Diagnostic entries remain preserved but excluded. Board generation fails closed on an
empty approval list or on incomplete, non-finite, duplicate, mismatched, or
provenance-free results. Mission utility and M-01…M-14 belong on that results board;
hardware latency, rail energy, and thermal evidence use the separate, provenance-bearing
[Jetson planner evidence](jetson-benchmark.md) table rather than inventing mission
metrics.

`train wm` also requires a W&B run. Tracking records optimizer/validation metrics,
the public-safe model, data, and source contracts, and content-addressed dataset and
checkpoint artifacts. Authentication stays outside the repository.

Runtime outputs are relative to the invoking working directory, or to `AUTOPS_ROOT`
when it is set. Immutable packaged assets are resolved independently of runtime output.
Tracked source and public documentation contain no service endpoint or credential.

## Extension checklist

1. Add a real representation plugin; never mark an unimplemented token runnable.
2. Declare applicability once in `configs/matrix.yaml`.
3. Keep mission action ordering stable in the shared trace schema.
4. Test behavior at the plugin boundary and through one end-to-end coordinate.
5. Add no dependency, axis token, metric, physics change, prompt change, or artifact
   field without an explicit framework decision.

## References

1. R. Castano et al., “Operations for Autonomous Spacecraft,” 2022.
   [doi:10.1109/AERO53065.2022.9843352](https://doi.org/10.1109/AERO53065.2022.9843352)
2. D. Hafner et al., “Mastering Diverse Domains through World Models,” 2023.
   [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
3. R. Y. Rubinstein and D. P. Kroese, *The Cross-Entropy Method: A Unified Approach
   to Combinatorial Optimization, Monte-Carlo Simulation and Machine Learning*,
   Springer, 2004.
   [doi:10.1007/978-1-4757-4321-0](https://doi.org/10.1007/978-1-4757-4321-0)
4. H. Bharadhwaj, K. Xie, and F. Shkurti, “Model-Predictive Control via Cross-Entropy
   and Gradient-Based Optimization,” L4DC, 2020.
   [paper](https://people.eecs.berkeley.edu/~brecht/l4dc2020/papers/bharadhwaj20.pdf)
5. B. Amos and D. Yarats, “The Differentiable Cross-Entropy Method,” ICML, 2020.
   [PMLR v119](https://proceedings.mlr.press/v119/amos20a/amos20a.pdf)
