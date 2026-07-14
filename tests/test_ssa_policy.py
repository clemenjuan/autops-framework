"""Action-space and local symbolic-policy contracts for SSA."""

from __future__ import annotations

from autops.core.types import DecisionContext
from autops.missions.ssa.policy import SSA_ACTION_SPACE, SSA_MODES, RuleBasedSSA


def _satellite(**updates: object) -> dict[str, object]:
    state: dict[str, object] = {
        "battery_soc": 0.9,
        "health": "nominal",
        "ground_pass_active": False,
        "storage_used_fraction": 0.0,
        "jetson_raw_mb": 0.0,
        "jetson_capacity_mb": 249_036.8,
        "observation_size_mb": 2_016.0,
        "unprocessed_batches": 0,
        "undelivered_records": 0,
        "undelivered_record_age_steps": 0,
        "predicted_in_fov": [],
        "ground_view": {},
    }
    state.update(updates)
    return state


def _context(satellites: dict[str, dict[str, object]]) -> DecisionContext:
    return DecisionContext(
        state={"satellites": satellites},
        observation={},
        memory=None,
        step=0,
        role="onboard",
    )


def test_six_mode_order_is_stable_for_encoded_actions_and_traces() -> None:
    assert SSA_MODES == (
        "charging",
        "communication",
        "payload_observe",
        "payload_detect",
        "isl_share",
        "safe",
    )
    assert SSA_ACTION_SPACE.labels == SSA_MODES
    assert SSA_ACTION_SPACE.low == 0
    assert SSA_ACTION_SPACE.high == 5


def test_record_age_relay_preempts_detection_backlog_at_tau_over_eight() -> None:
    policy = RuleBasedSSA({"custody_tau_steps": 4_320})
    action = policy.select_action(
        _context(
            {
                "sat_0": _satellite(
                    unprocessed_batches=9,
                    undelivered_records=1,
                    undelivered_record_age_steps=540,
                ),
                "sat_1": _satellite(battery_soc=0.2),
            }
        )
    )
    assert action["sat_0"] == {"mode": "isl_share"}
    assert "stale custody record" in (policy.last_rationale or "")


def test_single_satellite_policy_never_selects_isl_relay() -> None:
    policy = RuleBasedSSA({"satellite_id": "sat_0", "custody_tau_steps": 4_320})
    action = policy.select_action(
        _context(
            {
                "sat_0": _satellite(
                    unprocessed_batches=9,
                    undelivered_records=1,
                    undelivered_record_age_steps=4_320,
                ),
                "sat_1": _satellite(),
            }
        )
    )
    assert action == {"sat_0": {"mode": "payload_detect"}}


def test_policy_downlinks_only_during_a_physical_pass() -> None:
    policy = RuleBasedSSA()
    no_pass = policy.select_action(_context({"sat_0": _satellite(undelivered_records=1)}))
    in_pass = policy.select_action(
        _context(
            {
                "sat_0": _satellite(
                    undelivered_records=1,
                    ground_pass_active=True,
                )
            }
        )
    )
    assert no_pass["sat_0"]["mode"] != "communication"
    assert in_pass["sat_0"] == {"mode": "communication"}


def test_encoded_observation_contains_only_policy_visible_fields() -> None:
    policy = RuleBasedSSA()
    encoded = policy.encode_observation(
        {
            "satellites": {
                "sat_0": {
                    **_satellite(),
                    "position_km": [7_000.0, 0.0, 0.0],
                    "private_truth": {"future_detection": True},
                }
            },
            "global": {"full_target_truth": ["rso_0"]},
        }
    )
    satellite = encoded["satellites"]["sat_0"]
    assert "position_km" not in satellite
    assert "private_truth" not in satellite
    assert "global" not in encoded
