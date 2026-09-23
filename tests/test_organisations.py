from __future__ import annotations

import pytest

from autops.organisations.ssa import create_organisation, scope_observation


def observation(*, linked: bool = True) -> dict:
    satellites = {
        f"sat_{index}": {
            "battery_soc": 0.9,
            "health": "nominal",
            "ground_pass_active": False,
            "storage_used_fraction": 0.0,
            "unprocessed_batches": 0,
            "undelivered_records": 0,
            "predicted_in_fov": ["rso_0"],
            "ground_view": {},
        }
        for index in range(3)
    }
    return {
        "step": 0,
        "satellites": satellites,
        "global": {
            "max_steps": 10,
            "isl_feasible_pairs": [["sat_0", "sat_1"]] if linked else [],
            "ground_pass_active": {satellite_id: False for satellite_id in satellites},
            "ssa_custody_utility": 0.75,
        },
        "tasks": [],
    }


def test_scoping_removes_metric_truth_and_remote_satellites() -> None:
    scoped = scope_observation(observation(), ["sat_0", "sat_1"])
    assert set(scoped["satellites"]) == {"sat_0", "sat_1"}
    assert "ssa_custody_utility" not in scoped["global"]
    assert scoped["global"]["isl_feasible_pairs"] == [["sat_0", "sat_1"]]


@pytest.mark.parametrize("token", ["sas", "cmas", "dmas", "hmas", "imas"])
def test_every_organisation_returns_one_action_per_satellite(token: str) -> None:
    state = observation()
    controller = create_organisation(token)
    controller.reset(4, state)
    actions = controller.act(state)
    assert set(actions) == set(state["satellites"])
    assert all("mode" in action for action in actions.values())


def test_centralised_disconnected_member_holds_last_command() -> None:
    connected = observation(linked=True)
    controller = create_organisation("cmas")
    controller.reset(1, connected)
    first = controller.act(connected)
    disconnected = observation(linked=False)
    second = controller.act(disconnected)
    assert second["sat_1"] == first["sat_1"]
    assert second["sat_2"] == {"mode": "charging"}
    assert controller.metrics()["mean_command_staleness"] > 0.0


def test_hmas_branching_one_is_finite_and_independent() -> None:
    state = observation()
    controller = create_organisation("hmas", {"branching_factor": 1})
    controller.reset(2, state)
    assert controller.metrics()["num_clusters"] == 3.0
    assert set(controller.act(state)) == set(state["satellites"])


def test_unknown_organisation_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown organisation"):
        create_organisation("unknown")


def test_decentralised_memory_cannot_recover_remote_truth_after_step() -> None:
    state = observation(linked=False)
    controller = create_organisation("imas")
    controller.reset(4, state)
    controller.act(state)
    next_state = observation(linked=True)
    next_state["satellites"]["sat_0"]["battery_soc"] = 0.1
    controller.after_step({"global_truth": "unavailable"}, next_state)
    record = controller.memories["sat_0"].recent(1)[0]
    remembered = record["observation"]
    assert set(remembered["satellites"]) == {"sat_0"}
    assert remembered["satellites"]["sat_0"]["battery_soc"] == 0.9
    assert "ssa_custody_utility" not in remembered["global"]
    assert "info" not in record
    assert controller.memories["sat_0"].recent(0) == ()


def test_organisations_create_decision_loops_through_the_plugin_registry(monkeypatch) -> None:
    from autops.core import plugin

    created: list[tuple[str, str, str]] = []
    original = plugin.create_representation

    def recording(mission: str, token: str, role: str, config: dict | None = None):
        created.append((mission, token, role))
        return original(mission, token, role, config)

    monkeypatch.setattr("autops.organisations.ssa.create_representation", recording)
    controller = create_organisation("imas", {"representation": "symb"})
    observation = {"step": 0, "satellites": {"sat_0": {}, "sat_1": {}}, "global": {}}
    controller.reset(0, observation)
    controller.act(observation)
    assert created == [("ssa", "symb", "onboard")] * 2
