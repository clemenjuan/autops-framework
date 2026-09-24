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
boundary. Onboard learned planning sees only the onboard information boundary below. An RF transfer still requires physical contact. AO decides locally; AH has
independent onboard and ground cores with explicit promotion and arbitration. These
authority boundaries reflect the operational allocation problem described by Castano
et al. [1], rather than treating autonomy as a single scalar level.

Onboard LLM representations use the same AO boundary as other planners. A local model
produces one immediate action and a bounded schedule, the adapter executes held actions
for `plan_hold`, and only a new planning event invokes the model again. The hybrid LLM
variants recheck every held action against fresh telemetry before execution; the mission
environment still applies the final physical and safety resolution. A planning event
powers the Jetson for its decision step; the mission power model charges that energy only
in modes where the Jetson base load is otherwise absent. Measured inference time is a
diagnostic and never enters the energy budget.

## Package boundaries

| Package | Owns | Must not own |
|---|---|---|
| `autops.config` | coordinate parsing, matrix applicability, one Pydantic spec | factories, legacy aliases, generated configs |
| `autops.core` | plugin registry, lifecycle, public-safe provenance, RL-ready types | mission physics or model-specific planning |
| `autops.missions` | EventSat and SSA truth, transitions, rewards, metrics | provider clients or paradigm delay |
| `autops.orbital` | Orekit/fallback propagation, eclipse/access, link and ISL budgets | scheduling policy |
| `autops.paradigms` | CG/AG/AO/AH authority, latency, stale-view semantics | physics constants |
| `autops.organisations` | SAS/CMAS/DMAS/HMAS/IMAS task and knowledge allocation | sensing/link truth |
| `autops.representations` | symbolic, RL, LLM/hybrid, and LeWM-CEM plugins | runner special cases |
| `autops.rl` | RLlib bridge, mission adapters, PPO trainer, checkpoint manifests, shaping | mission physics or a second reward |
| `autops.llm` | environment-selected clients, SHA-256 cache, deterministic mock, prompts | mission-state mutation |
| `autops.wm` | trace/dataset, JEPA model/training, probes, artifact, CEM | duplicate mission adapters |
| `autops.board` | validation and static rendering of completed results | metric recomputation or placeholder data |

Optional dependencies are lazy. Importing core AUTOPS does not import Torch, start a
JVM, or contact an LLM endpoint. `orbital`, `llm`, and `wm` extras activate those
surfaces, and the `rl` extra adds RLlib, Gymnasium, Torch, and W&B for `autops train rl`.
Deploying an `rl` checkpoint restores only the trained policy, so no Ray workers start.

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
use without adding Gymnasium to the base install. EventSat observation bounds are per
input and depend on the mission power model, so `observation_space(power)` in
`autops.missions.eventsat.observation` derives them instead of a class constant.

## EventSat information boundary

A decision record at step `t` contains only information the spacecraft could hold by
`t`. `autops.missions.eventsat.observation` owns that boundary and its vector encoding;
`autops.wm.schema.EVENTSAT_OBSERVATIONS` fixes the 45 input names and order.

| Group | Inputs | Encoding |
|---|---|---|
| Navigation | Earth-fixed (ITRF, IERS 2010) position and coordinate velocity | km / 7000; km s⁻¹ / 8 |
| Present geometry | Sun unit vector in the same frame; station elevation; station visible; in sunlight | unit vector; sin(elevation); flags |
| Resources and health | battery state of charge; OBC, total Jetson, and compressed Jetson fills; health nominal | fraction; log(1+stored products) / log(1+capacity in products); flag |
| Payload processing | unprocessed and undetected product counts; compression and detection progress | log(1+n) / log(1+pool capacity in products); fraction of job duration |
| Attitude and command feedback | settling remaining; last command forced to safe or to charging; last action accepted; executed mode; attitude target | fraction of settling time; flags; two one-hot blocks |
| Last-interval outcome | product captured; compression and detection completed; bytes moved to the OBC; bytes downlinked; net platform energy | fraction of one product or of one step's link capacity; energy / one step of peak solar generation |

Navigation is an ideal GNSS position-velocity-time fix sampled every step from the same
Orekit propagation as the episode [6, 7]. It is not a receiver model: no noise, delay,
outage, or fix age is simulated. The seeded fallback has no state-derived orbit, reports
the fix as invalid, and encodes zeros; it is unsuitable for navigation-learning corpora.
Sun direction and station elevation stand for onboard-computable ephemeris and known
station geometry. Station visibility and sunlight are instantaneous at the step start;
the illumination flag is geometric, not a measured power estimate.

The following are deliberately not inputs:

- elapsed-time surrogates: nominal orbital phase, episode progress, and cumulative
  totals. The launch lottery draws each episode's start, to the minute, uniformly over
  one year from the configured epoch, so the date and time of day that reach the model
  through the Earth-fixed Sun direction no longer tell it how far into an episode it is.
  With one shared start, the Sun's declination rose identically in every episode and
  acted as an episode clock;
- future-event information: pass and eclipse countdowns, remaining pass duration,
  next-pass and whole-episode transfer capacity, and contact or sunlight arrays;
- declared constants such as capacities, rates, job durations, and station coordinates,
  which are identical in every trace row and instead scale the inputs and parameterize
  the planner;
- the previous requested command, which the model already receives as its action input;
- values that are ideal constants in this simulator (navigation validity and age).

The net energy input excludes the declared planner charge. Rule-based training
collectors never plan, so a planner-energy input would be constant in training and unseen
at deployment; the battery state of charge still carries the drain, and neither CEM
scorer forecasts its own future compute cost.

Storage fills count stored products on a log scale, so the first product is resolved
instead of appearing as ~10⁻⁵ of Jetson capacity. The OBC and compressed-Jetson fills
count compressed products; the total Jetson fill counts raw products over raw plus
compressed bytes, the quantity that observation admission checks against capacity. A
raw-only fill would duplicate the unprocessed product count and is not an input.
A step of a multi-step compression or detection job with a product to work on is an
accepted action; only its completing step can be rejected, by the product's transition.

Values without a simulated sensor or subsystem model, such as voltages, temperatures,
attitude quaternions, pointing error, link lock, or fault diagnoses, are not invented.

Onboard `lewm-cem`, `symb`, and the four onboard LLM planners decide from this view.
The onboard rules are reactive: they communicate while the station is visible and have
no pass or remaining-capacity forecast. Onboard LLM prompts state that no pass or
eclipse forecast exists and report present station visibility, elevation, sunlight,
settling, and the last interval's outcome. Their what-if tools treat communication
before visibility as feasible antenna prepointing without transfer and scale value by
one step of link capacity. Their symbolic shield admits communication with OBC data
waiting, matching the CEM mask. Ground roles legitimately keep the pass almanac, and
`analytical-cem` keeps it as an oracle (see below).

The symbolic ground scheduler caps science production by the next pass. It sizes new
products against that pass's deliverable capacity (`future_pass_capacity_mb`) minus data
already staged, with at least one product per gap while capacity exceeds it. One
compressed product (≈1.84 MB) is about one average pass (≈1.8 MB), so the baseline plans
roughly one product per gap. The scheduler never reads the battery; the cap is what keeps
it from over-producing, draining the battery, or duplicating products planned from the
stale ground view. Removing the cap alone would let only the gap's time budget limit
production (about five products per one-orbit gap).

Plan B, not implemented, would size production against the remaining-episode capacity
(`remaining_achievable_downlink_mb`) only together with two further changes: a simulated
battery that charges first below 0.40 (to 0.55), processes only at or above 0.35, and
observes only above 0.60 with the OBC below 80%, using `advance_projected_battery` and the
almanac's `planning_sunlight`; and planning from fresh telemetry after a pass's first
housekeeping downlink, which changes when AG, CG, and AH ground cores plan. Adopting it
requires a paired AG/CG/AH-symb comparison of M-01, M-05, and M-13.

Trace state rows are privileged simulator labels for probe targets and evaluation, never
decision inputs. They include the future pass and eclipse countdowns, which are censored
as `-1` when no later event lies within the recorded episode. Contact labels distinguish
instantaneous station visibility, physical contact seconds during the coming step, and
the settling-lead `contact_window_active` used by the communication-opportunity target.

## World-model path

The trace row stores the pre-transition observation/state and requested one-hot action;
reward, resolved action, and forced flag describe that row's transition. The next row is
the resulting observation. EventSat trace v4 uses the 45 observations above, 25 state
labels, and 7 actions. SSA adds a satellite axis and uses its canonical 6-action order.
NPZ files are pickle-free and carry names, axes, episode IDs, seeds, and a versioned
schema. Older traces, checkpoints, and planner artifacts are rejected: a new observation
contract requires re-export, retraining, and probe refitting, never padding or relabeling.

Training and validation split launch seeds, not episode indices: every policy's
realization of one physical episode stays on one side, and checkpoint v5 records the
per-episode seeds that define its split. With unique seeds this equals an episode shuffle.
Dataset windows never cross episode boundaries. LeWM uses a 192-dimensional embedding
and history 3 with a JEPA-style action-conditioned objective. Probes are affine
(`W`, `b`) and store target means/standard deviations plus degenerate-target labels.
Probe targets v3 distinguish stocks from flows. Battery and storage margins, the incoming
override, health, and contact opportunity are stocks of the labelled state. Downlink,
science, and detection progress are flows: the amount completed in the interval ending at
the record, which the onboard record reports as its last-interval outcome. Cumulative
totals are not inputs, so their level cannot be read from a latent. The planner reads
every predicted latent, takes stocks from the terminal one, and sums flows along the
rollout, as physics probes decode per-step state increments from imagined latents [8] and
world-model planners sum per-step reward predictions [2, 9]. Preset weights are divided by
an objective scale stored with the probes: a stock's own spread over the training episodes,
and for a flow the spread of the cumulative total it accumulates. Dividing a flow by the
spread of single steps would make one rare observation step outweigh the battery by an
order of magnitude; in a two-day smoke run even the analytical oracle then observed
greedily and starved the downlink pipeline.
The relocatable planner artifact contains those probes, normalisation, action names,
relative checkpoint path, CEM parameters, and all policy controls (reserve thresholds,
reflexes, guidance, and shaping). The v6 artifact binds these settings at probe-fit time;
explicit representation overrides remain part of the experiment configuration. Evaluation
executes the same runner and representation as closed-loop planning. This representation follows the world-model control pattern
demonstrated by Hafner et al. [2]; the exact AUTOPS contract is deliberately narrower
and auditable.

Before scoring or elite selection, every sampled EventSat sequence is projected across
the complete horizon by `autops.wm.guidance.project_executable_candidates`. It shares
safety resolution, settling, and battery calculations with mission truth and calls the atomic transitions in
`autops.missions.eventsat.transitions` for observe, compress, detect, CAN transfer, and
physical-contact downlink. CEM therefore returns executable requested commands, and the
selected scorer and pipeline shaping consume the same projected candidate bank within
each search. Separate analytical and learned runs adapt their later candidate banks to
their own scores; identical seeds do not imply identical banks throughout a run. Held-action fallbacks remain as a runtime guard and report their own
repair rate.

The two CEM leaves have different information. `lewm-cem` receives only the onboard
view: its mask, projection, guidance, pipeline seed, and shaping know present station
visibility but no future contact, hold current sunlight constant, and may command
communication before visibility for prepointing, while the environment still gates every
transfer on physical contact. The station-visibility downlink reflex is common to both.

The `analytical-cem` reference [3, 4, 5] replaces only the latent rollout/readout with
canonical attributes computed from that projection: terminal stocks and the exact horizon
increments of the flows. It is a forecast oracle:
it receives the exact contact and sunlight arrays that the learned leaf must infer and
uses the same transitions as the truth environment, so its results are an upper bound,
not an onboard-realizable peer. Its exogenous contact and sunlight
arrays are generated by the environment's active orbital backend (Orekit for paper
runs). Execution and trace export size these arrays from the effective CEM horizon,
including the terminal settling margin; explicit forecasts that are too short are
rejected rather than padded with invented contact or sunlight. Ground almanac refreshes
advance these deterministic arrays along with the planning clock. Action-dependent power, settling, storage, and byte-pipeline dynamics remain the
same shared functions used by the truth environment. The projection is conditional on the supplied almanac and current health: it does not
sample future anomalies or predict anomaly clearance. It also excludes future planner
compute events, whose declared cost is charged by the actual environment. These
assumptions must accompany an analytical-reference claim.

Candidate repair, physical settling, and environment safety overrides are distinct.
A slew fixes its target when it starts; commands issued while settling, including a
policy-requested `safe`, are dropped rather than queued or used to redirect the slew.
Only mandatory safe mode (anomaly or critical battery) preempts and cancels pending
settling, in both truth and candidate projection. Only safety resolution contributes to the projected forced flag. Probe targets
align that flag and the flows with the incoming transition: the labels for state `s_t` come
from `a_(t-1)`, and reset has no override and zero flows. Contact opportunity refers to the
physical or settling-lead contact window. It remains a fitted probe but carries no weight in
the planning presets: contact is exogenous, so it cannot rank one decision's candidates, and
a learned readout's dependence on commands could only be spurious. M-01…M-14 are unchanged. Changing an artifact
or checkpoint version string cannot migrate fitted weights.

Affine readouts remain deployed until selection-level evidence justifies a change.
`autops.wm.scoring.candidate_selection_metrics` compares scorers on one shared bank using
top-elite overlap and analytical regret; probe R²/AUC alone is not deployment evidence.

A planning event powers the Jetson for the whole decision step in which it plans:

\[
E_{event}=P_{active}\,\Delta t+E_{boot}+P_{idle}t_{idle},
\]

with the 60 s step `Δt`. This is an accounting bound that makes the charge independent
of the host running the simulation; measured CEM or LLM time is kept only as a diagnostic.
The power and boot/idle terms are declared under `power` in the mission configuration and
are labelled `assumed` unless replaced by hardware evidence. Incremental planner energy is
zero in modes whose base load already includes the Jetson and in safe mode, which keeps
the payload computer off. Board-level INA3221 rails cannot
replace the scalar model unless a non-overlapping total-input boundary is established;
the current hardware results therefore retain every exposed rail separately.

## SSA prototype boundary

SSA reads its defaults from `configs/missions/ssa.yaml`; direct environment construction
and matrix runs use the same nested battery, solar, ground-station, mode, and target keys.
The implemented orbital model is Keplerian two-body propagation. Orekit/J2 constellation
propagation and CTDE world models remain future work. The declared 20/100-satellite
coordinates do not imply completed scale-validation evidence.

Ground record transfers obey the X-band byte budget. The custody upper bound permits
instantaneous global sensing/delivery at every active pass step, then measures freshness
at the same end-of-step clock as achieved utility. It relaxes processing, pointing, and
relay delays, so it is an optimistic bound. Organisation memories retain the information
actually delivered to their decision loops, excluding metric-only global truth.

`autops.organisations` holds one organisation layer for every decision substrate, the
agentic framework's contract. Each organisation declares its agents, each agent's
actuation scope (a disjoint cover of the constellation) and largest observation scope,
the satellite that hosts it, and optionally the logical agent graph that authorises
inter-satellite links. The runner's decision loops and the RLlib bridge call the same
`distribute_observation` and `collect_actions`. Under the canonical `physical` link
gating an agent hosted on one satellite sees and commands another only over a published
ISL pair; an unreachable satellite keeps its last received command and its staleness is
reported. `logical` gating makes every link available, as in the agentic organisations.

| Token | Agents | Commands | Sees |
|---|---|---|---|
| `sas` | `central_agent`, not hosted | every satellite | every satellite |
| `imas` | `sat_agent_i` on satellite i | its satellite | its satellite |
| `dmas` | `sat_agent_i` on satellite i | its satellite | its satellite (`peer_view: local`); with `linked`, also linked neighbours |
| `cmas` | `mission_manager` on the first satellite | every satellite over links | satellites linked to its host |
| `hmas` | `cluster_agent_i` on its cluster's first satellite | its cluster over links | cluster members linked to its host |

Clusters are contiguous in satellite index (`branching_factor`, `num_clusters`, or an
explicit `clusters` partition). Strictly local DMAS peers learn only through physical
`isl_share`; each view marks `has_isl_peer` when an authorised peer is reachable, and the
rule-based policy treats that as coordination. The `linked` view keeps the earlier
idealised neighbour telemetry, counted as one message per neighbour and step; a future
bandwidth/failure study must define and validate those message costs explicitly.

ISL sharing snapshots every sharer's knowledge and custody buffer before any transfer,
plans record relays from the snapshots, and commits both afterwards, so an estimate or
record travels at most one hop per step. A received estimate keeps its own acquisition
age, and a repeated or inferior message changes nothing; a satellite's own
re-observation still refreshes it. Estimates rank by track quality, custody records by
recency, each with a stable tie-break. Only organisation-authorised, idle, physically
reachable satellites receive.

## Reinforcement-learning path

`autops.rl` is the agentic framework's RLlib stack behind the base structure. The
RLlib bridge exposes one agent per organisation agent: it distributes the observation
through the organisation layer, encodes each view with the mission's adapter (for
EventSat, the onboard vector above, so `rl` sees exactly what `lewm-cem` sees), decodes
and shields the chosen modes, collects them through the organisation, and steps the
canonical environment. The runner deploys the same adapter and shield, and a parity
test drives both paths with identical actions. Training episodes draw launch seeds from
10^6 upwards, keeping the small paired evaluation seeds unseen. The reward is the
mission's; optional potential-based pipeline shaping is added only inside the bridge.
An EventSat AH coordinate trains the onboard policy inside the hybrid paradigm: the
coordinate's ground planner plans at contacts, and the paradigm's `arbitrate` applies
plan promotion and override to the shielded onboard modes before the environment steps,
so the policy learns under the authority it is later evaluated with.
SSA agents follow the organisation layer: SAS and CMAS agents command every satellite,
HMAS agents their cluster, and IMAS and DMAS agents their own satellite, each observing a
fixed scope in which a satellite hidden by the channel encodes as zeros. The SSA
environment publishes pass and eclipse countdowns only for `rl` coordinates, and
declares the catalog size, custody tau, and orbital period in every record so each view
carries its own normalisation. Agents sharing a policy must have identical spaces;
unequal HMAS clusters use `policy_sharing: independent_per_agent`.
Every checkpoint, including intermediate `step_<sampled steps>` snapshots, carries a
manifest with the observation schema and names, per-policy spaces, recipe, and
public-safe provenance; loading rejects any other contract, and results record the
deployed policy's identity. Mock-policy results are never boardable.

## Commands and runtime data

```bash
uv run autops run COORDINATE [--episodes N] [--seeds A:B] [--set key=value]
uv run autops sweep MISSION [filters]
uv run autops export COORDINATE [COORDINATE ...] [run options]
uv run autops train wm ...
uv run autops train rl COORDINATE --output DIR [--recipe-set key=value]
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
then runs complete AO episodes through `ExperimentRunner` on validation seeds. Masks,
projection, shaping, guidance, warm starts, held actions, reflexes, and compute-energy
accounting follow the deployed path. `--max-episodes` limits seed count; `--set` uses
ordinary matrix overrides. These are new trajectories under the recorded mission
configuration, not reconstructions of historical trace contexts. Evaluation v2 records
per-episode mission metrics and actual planner diagnostics without local input paths.
The split informed checkpoint selection and is explicitly labeled as validation.
`train audit` reads frozen latents and their controls (raw record, optionally stacked
frames; an untrained encoder of the same architecture; elapsed time alone) with the same
affine and MLP heads. Heads are fitted on the checkpoint's training episodes and scored on
its validation episodes or, with `--test-trace`, on untouched seeds; test seeds that occur
in the training trace are rejected. The elapsed-time control bounds what an episode
clock explains; with drawn start times the Sun vector no longer provides one. `train selection` replays logged episodes from their launch seeds, whose
records must re-encode to the logged observations exactly, and at sampled decision points
scores one bank of requested command sequences with the deployed learned planner and the
analytical oracle, each under its own information. It reports top-elite overlap, oracle
regret, and best-candidate rate against a random-scorer chance level, overall and split by
proximity to contact. `train forecast` rolls the model out under logged commands and
compares the frozen readouts of every predicted latent with the labelled future: stocks at
each step and flows as sums since the decision. The references are the readout of the
encoded true future record, which isolates the readout's own error, persistence, and the
planner's analytical projection of the same commands from the onboard view without
mission-policy repair. It reports RMSE and skill against the variance across contexts.
`train counterfactual` rolls one set of command sequences (each mode held, plus random
sequences) through the model from the logged history and through copies of the replayed
simulator. It reports how much the exogenous contact-opportunity readout moves with the
commands, how the model's response to commands (relative to holding charging) matches the
simulator's, and the error on steps the simulator dropped while settling or overrode for
safety. `train events` predicts the next pass start and duration and the next eclipse
entry and exit at the same held-out contexts with an onboard recurrence (last observed
event plus nominal orbital periods), an Eckstein-Hechler propagation of the onboard
navigation fix with the known station, affine readouts from the latent and from the raw
record, and thresholded visibility and sunlight readouts along the model's rollout.
Censored events are excluded and times keep the 60 s step resolution. The paper-facing
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
6. NASA Small Spacecraft Systems Virtual Institute, “State-of-the-Art of Small
   Spacecraft Technology: Guidance, Navigation, and Control.”
   [nasa.gov](https://www.nasa.gov/smallsat-institute/sst-soa/guidance-navigation-and-control/)
7. A. P. M. Chiaradia, H. K. Kuga, and A. F. B. A. Prado, “Onboard and Real-Time
   Artificial Satellite Orbit Determination Using GPS,” *Mathematical Problems in
   Engineering*, 2013. [doi:10.1155/2013/530516](https://doi.org/10.1155/2013/530516)
8. Z. Li, C. Ren, P. Wang, and X. Sun, “Orbit-Planner: Towards Latent World Models for
   On-Orbit Obstacle Avoidance of Satellite Agents,” preprint, 2026.
   [doi:10.48550/arXiv.2608.16651](https://doi.org/10.48550/arXiv.2608.16651)
9. N. Hansen, H. Su, and X. Wang, “TD-MPC2: Scalable, Robust World Models for
   Continuous Control,” ICLR, 2024. [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)
