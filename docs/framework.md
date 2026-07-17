# AUTOPS framework specification

AUTOPS is a controlled testbed for satellite scheduling and operations architectures.
The mission objective is held fixed. An operations system, **O**, is a coordinate on
three independent axes: organisation, decision representation, and operational
paradigm. This scope follows the separation between mission operations, onboard
autonomy, and ground authority described by Sellmaier et al. [1] and Castano et al.
[2]. AUTOPS does not compare spacecraft designs, mission concepts, or objective
functions.

## Axes

| Axis | Tokens | Meaning |
|---|---|---|
| Organisation | `sas`, `cmas`, `dmas`, `hmas`, `imas` | single agent; centralised, decentralised, hierarchical, or independent multi-agent scheduling |
| Representation | `symb`, `llm-s`, `llm-a`, `hllm-s`, `hllm-a`, `analytical-cem`, `lewm-cem` | rules; LLM single-shot/agentic; symbolically shielded LLM; analytical or learned latent CEM MPC |
| Paradigm | `conventional`, `ag`, `ao`, `ah` | conventional ground, autonomous ground, autonomous onboard, or dual-core hybrid authority |

The LLM distinction is operational rather than rhetorical: `llm-s` makes one bounded
decision, whereas `llm-a` may perform at most three tool-mediated reasoning steps
before a decisive action. This bounded agent loop is informed by ReAct [3] and by
language-agent architecture surveys [4]. Hybrid variants use the identical model
response but pass it through deterministic mission grounding and safety constraints.

`lewm-cem` is a representation, not a special runner mode. It encodes observations,
predicts action-conditioned latent futures, decodes audited attributes through affine
probes, and performs categorical cross-entropy-method search. Learned world-model
planning is motivated by Hafner et al. [5] and the LeWorldModel formulation [6].

`analytical-cem` is its matched model-based reference, in the sampling-based
CEM-MPC tradition [7, 8, 9]. It uses the same CEM, candidate projection,
scalarisation, guidance, plan-hold, and safety controls, but scores terminal
attributes from the exact EventSat transitions. Contact and sunlight
lookahead come from the environment's active orbit backend, so Orekit propagation is
computed once as the authoritative almanac rather than duplicated per candidate.

`rl` and `hrl` are reserved names, with `implemented: false` in `matrix.yaml`. They
cannot be expanded or run. A future PPO/RLlib contribution can implement the common
representation protocol and Gymnasium-style `SpaceSpec` seam without changing any
coordinate. AUTOPS ships no RL library, neural policy, or symbolic stand-in.

## EventSat matrix

The canonical EventSat design contains 32 cells. The definition is retained even while
the real RL cells are deferred:

| Paradigm | Onboard representations | Ground representations | Cells |
|---|---|---|---:|
| `conventional` | — | `symb` | 1 |
| `ag` | — | `symb`, `rl`, `hrl`, `llm-s`, `llm-a`, `hllm-s`, `hllm-a` | 7 |
| `ao` | `symb`, `rl`, `hrl` | — | 3 |
| `ah` | `symb`, `rl`, `hrl` | the seven AG representations | 21 |

The canonical total is `1 + 7 + 3 + (3 × 7) = 32`. Runtime applicability is narrower:

- `conventional`: ground `symb`;
- `ag`: ground `symb`, `llm-s`, `llm-a`, `hllm-s`, or `hllm-a`;
- `ao`: onboard `symb`, `analytical-cem`, or `lewm-cem`;
- `ah`: onboard `symb` or `lewm-cem`, paired with any runnable AG ground representation.

The two CEM references extend the current runnable study but do not alter the historical
32-cell definition. EventSat uses `sas`; SSA exercises all five organisation tokens
with the symbolic AO representation at constellation sizes 20 and 100.

Coordinates are expanded at runtime from `configs/matrix.yaml` and one mission file:

```text
eventsat/sas/ag/symb
eventsat/sas/ah/lewm-cem/hllm-a
ssa/dmas/ao/symb
```

There are no generated cell YAMLs and no compatibility aliases. A coordinate that is
unknown, inapplicable, reserved, or unimplemented fails validation before simulation.

## Mission contracts

EventSat models a 6U event-camera spacecraft in a 400 km sun-synchronous orbit, one
ground station, 60-second control steps, a 70 Wh battery, 16.8 W effective peak solar
generation, a 50 kbps effective S-band downlink, and the three-pool raw/compressed/OBC
pipeline. A 135-second slew is deliberately floored to two nonproductive steps. The
optional orbital backend uses Orekit Eckstein-Hechler J2 propagation; the seeded
fallback preserves eclipse/contact structure when Java is unavailable.

SSA models detect-gated object records, local knowledge, physical ISL transport, and
ground-delivered custody. The six actions are `charging`, `communication`,
`payload_observe`, `payload_detect`, `isl_share`, and `safe`. Utility comes from fresh
records delivered to ground, not omniscient simulator state. Organisation policies may
change allocation and information routing, but never sensing, link, or target truth.

## EventSat metrics

The result document always reports the same 14 metric identifiers. Episode means are
reported for every metric; M-09 is computed across paired episodes.

| ID | Name | Definition |
|---|---|---|
| M-01 | utility | weighted, duration-scaled observation/downlink objective minus anomaly penalty |
| M-02 | mean AoI | mean seconds since the last productive downlink |
| M-03 | peak AoI | maximum seconds since the last productive downlink |
| M-04 | recovery | mean control steps from anomaly arrival to nominal, non-safe operation |
| M-05 | safety-override rate | fraction of satellite-steps resolved to safety `safe` |
| M-06 | resource efficiency | M-01 divided by gross energy consumed in Wh |
| M-07 | decision latency | mean latency over permitted inference decisions |
| M-08 | explainability | fraction of inference decisions carrying a rationale |
| M-09 | robustness | sample standard deviation of episode utility divided by mean utility |
| M-10 | scale efficiency | per-satellite utility relative to a configured single-satellite baseline |
| M-11 | downlink efficiency | delivered MB divided by physically achievable contact capacity |
| M-12 | value of information | delivered raw-equivalent MB divided by raw MB captured |
| M-13 | constraint violations | non-safety forced actions divided by satellite-steps |
| M-14 | commanding effort | mode changes plus weighted manual anomaly interventions per simulated day |

Every board row retains the sample count. Metrics are computed by the mission collector,
never reconstructed by board code.

## Fairness and reproducibility

- Every comparison uses `FixedMemory`; writable online learning is outside scope.
- Launch-lottery and anomaly streams use paired episode seeds.
- Physics, objectives, prompts, metric definitions, and trace action order are shared
  across applicable cells.
- The world-model split is episode-disjoint. Normalisation is fitted on training
  episodes only. Degenerate probe targets are named explicitly.
- Results carry a configuration SHA-256, source revision, Python version, and package
  versions, but no hostname, username, endpoint, or absolute path.
- Runtime results, artifacts, boards, logs, and private research material are ignored.

## References

1. M. Sellmaier et al., *Spacecraft Operations*, Springer, 2022.
   [doi:10.1007/978-3-030-88593-9](https://doi.org/10.1007/978-3-030-88593-9)
2. R. Castano et al., “Operations for Autonomous Spacecraft,” IEEE Aerospace
   Conference, 2022.
   [doi:10.1109/AERO53065.2022.9843352](https://doi.org/10.1109/AERO53065.2022.9843352)
3. S. Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models,” 2023.
   [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
4. T. R. Sumers et al., “Cognitive Architectures for Language Agents,” 2024.
   [arXiv:2309.02427](https://arxiv.org/abs/2309.02427)
5. D. Hafner et al., “Mastering Diverse Domains through World Models,” 2023.
   [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
6. “LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels,” 2026.
   [arXiv:2603.19312](https://arxiv.org/abs/2603.19312)
7. R. Y. Rubinstein, “Optimization of Computer Simulation Models with Rare Events,”
   European Journal of Operational Research, 99:89–112, 1997.
   [doi:10.1016/S0377-2217(96)00385-2](https://doi.org/10.1016/S0377-2217(96)00385-2)
8. H. Bharadhwaj, K. Xie, and F. Shkurti, “Model-Predictive Control via Cross-Entropy
   and Gradient-Based Optimization,” L4DC, 2020.
   [paper](https://people.eecs.berkeley.edu/~brecht/l4dc2020/papers/bharadhwaj20.pdf)
9. B. Amos and D. Yarats, “The Differentiable Cross-Entropy Method,” ICML, 2020.
   [PMLR v119](https://proceedings.mlr.press/v119/amos20a/amos20a.pdf)
