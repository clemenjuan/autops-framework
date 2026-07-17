# Jetson planner evidence

This page separates compute evidence from mission-effect evidence. The matched Jetson
benchmark is candidate Paper A evidence for planner latency, throughput, rail energy,
memory, and thermal behaviour. Paper B utility and M-01…M-14 remain result-document
metrics selected by `configs/papers/paper_b.yaml`; no utility row is inferred from a
hardware benchmark or a one-call smoke test.

## Matched CEM benchmark

The diagnostic run `paper-a-balanced-aggregate` compared the deployed LeWM-CEM scorer
on CUDA with the canonical analytical scorer on CPU. Both methods used the same
categorical CEM (`H=12`, 256 samples, 32 elites, four iterations, `plan_hold=12`),
executable-candidate projection, mission scalarisation, and four trace-bound contexts.
Each method ran in five counterbalanced blocks with five warm-ups, a matched loaded-idle
baseline, and at least 100 planning events or 30 seconds of workload per block.

| Planner | Blocks | Planning events | Median latency | p95 latency | Rollouts/s | First-event median | Initialisation median | Peak temperature |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Analytical CEM, CPU | 5 | 775 | 188.8 ms | 219.6 ms | 5,276 | 193.0 ms | 22.8 ms | 56.75 °C |
| LeWM-CEM, CUDA | 5 | 500 | 879.8 ms | 913.0 ms | 1,159 | 2.055 s | 3.710 s | 58.812 °C |

Under this deployed configuration, analytical CEM was 4.55 times faster by pooled mean
end-to-end planning latency. The CUDA planner's peak Torch allocation was 43,658,240
bytes. All five CUDA blocks passed the production closed-loop check; in each block it
recorded two planning events, 16 held steps, six contact reflexes, and no held-action
repairs.

The board exposes no verified total-input rail. The rails below are retained separately
and **must not be summed**, because their overlap is not established. Incremental energy
subtracts the matched loaded-idle baseline.

| Planner | Rail | Idle power | Workload power | Incremental energy/event |
|---|---|---:|---:|---:|
| Analytical CEM, CPU | `VDD_CPU_CV` | 0.534 W | 2.626 W | 0.406 J |
| Analytical CEM, CPU | `VDD_GPU_SOC` | 2.277 W | 3.213 W | 0.182 J |
| Analytical CEM, CPU | `VIN_SYS_5V0` | 4.164 W | 4.677 W | 0.100 J |
| Analytical CEM, CPU | `VDDQ_VDD2_1V8AO` | 0.514 W | 0.932 W | 0.081 J |
| LeWM-CEM, CUDA | `VDD_CPU_CV` | 0.460 W | 2.465 W | 1.771 J |
| LeWM-CEM, CUDA | `VDD_GPU_SOC` | 3.049 W | 12.784 W | 8.598 J |
| LeWM-CEM, CUDA | `VIN_SYS_5V0` | 4.300 W | 6.106 W | 1.596 J |
| LeWM-CEM, CUDA | `VDDQ_VDD2_1V8AO` | 0.470 W | 1.663 W | 1.054 J |

No thermal-throttling signature was observed: the maximum temperature was 58.812 °C,
below the lowest exposed 70 °C trip, while the workload reached the configured GPU and
EMC maxima. NVML and explicit current/power throttle counters were unavailable, so
current- or power-limit throttling is not fully observable. The optional LeWM CPU path
failed closed on non-finite recursive-rollout attributes and is excluded from the
comparison.

### Provenance

| Item | Identity |
|---|---|
| Source revision | `de3c13fca6ab5f39cda5495e346a5f88ea6d23ec` (clean) |
| Aggregate manifest SHA-256 | `bebc821291151cbd5b25acf3fbf7b8f804b0598eb44155b5f23dd65c4b60874f` |
| Aggregate summary SHA-256 | `0c89d59407501121d1618f3fce5ddfd85602315ee4223f5b4991742d6658b66a` |
| Harness SHA-256 | `5bbc61b5eebde37a7a87ed52fab23cdd48eb86784f8680fb8c747efec2f40fa2` |
| Artifact SHA-256 | `8c39f1f77b5383d5cb6d78f88ca8a0211e8c71cc78415bf802d3dbab51835edf` |
| Trace SHA-256 | `d10c167a6006786f0933b266312567e896b88f208ade0353af6621354f3a725c` |
| Checkpoint SHA-256 | `9a580b05f42b998a4e76d674d6df229de3d13add3808c4fca4cf5b10ca41c2bf` |
| Checkpoint size | 13,902,865 bytes |

The fixture contexts were exact trace rows for nominal charging, active payload
pipeline, pre-contact, and active contact. The trace does not contain transition
counters or a full almanac; contact seconds were reconstructed from consecutive
remaining-pass durations and `transition_steps_remaining` was set to zero. Candidate
banks diverge after the first CEM iteration because scorer-dependent elites update later
proposals, so decision equivalence is not claimed.

## Local Nemotron feasibility smoke

A separate live smoke used Nemotron-3-Nano-4B Q4_K_M through Ollama for
`eventsat/sas/ao/llm-s`. It completed one local planning call followed by two held
actions. The measured active planning time was 16.386 seconds. The reported 0.03186 Wh
is **modelled**, not rail-measured: it is the configured 7 W incremental compute model
times the measured latency. This establishes local AO plumbing and plan-hold execution,
but a single call is neither a performance distribution nor Paper B utility evidence.
