from __future__ import annotations

import pytest

from autops.organisations import (
    AgentAction,
    Channel,
    create_organisation,
    scope_observation,
)
from autops.organisations.topologies import ORGANISATIONS


def observation(*, linked: bool = True, size: int = 3) -> dict:
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
        for index in range(size)
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
    loops = create_organisation(token)
    loops.reset(4, state)
    actions = loops.act(state)
    assert set(actions) == set(state["satellites"])
    assert all("mode" in action for action in actions.values())


def test_centralised_disconnected_member_holds_last_command() -> None:
    loops = create_organisation("cmas")
    loops.reset(1, observation(linked=True))
    first = loops.act(observation(linked=True))
    second = loops.act(observation(linked=False))
    assert second["sat_1"] == first["sat_1"]
    assert second["sat_2"] == {"mode": "charging"}
    assert loops.metrics()["mean_command_staleness"] > 0.0


def test_logical_gating_delivers_to_unlinked_members() -> None:
    loops = create_organisation("cmas", {"link_gating": "logical"})
    loops.reset(1, observation(linked=False))
    loops.act(observation(linked=False))
    assert loops.metrics()["mean_command_staleness"] == 0.0


def test_hmas_branching_one_is_finite_and_independent() -> None:
    state = observation()
    loops = create_organisation("hmas", {"branching_factor": 1})
    loops.reset(2, state)
    assert loops.metrics()["num_clusters"] == 3.0
    assert set(loops.act(state)) == set(state["satellites"])


def test_hmas_groups_satellites_by_index_not_by_text_order() -> None:
    organisation = ORGANISATIONS["hmas"]({"branching_factor": 10})
    organisation.initialize([f"sat_{index}" for index in range(20)])
    assert organisation.clusters == [
        [f"sat_{index}" for index in range(10)],
        [f"sat_{index}" for index in range(10, 20)],
    ]
    near_equal = ORGANISATIONS["hmas"]({"num_clusters": 3})
    near_equal.initialize([f"sat_{index}" for index in range(7)])
    assert [len(cluster) for cluster in near_equal.clusters] == [3, 2, 2]


def test_strict_local_dmas_sees_only_itself_and_marks_reachable_peers() -> None:
    organisation = ORGANISATIONS["dmas"]()
    organisation.initialize(["sat_0", "sat_1", "sat_2"])
    state = observation(linked=True)
    views = organisation.distribute_observation(state, organisation.channel(state))
    view = views["sat_agent_0"].local_state["full_observation"]
    assert set(view["satellites"]) == {"sat_0"}
    assert view["satellites"]["sat_0"]["has_isl_peer"]
    isolated = views["sat_agent_2"].local_state["full_observation"]["satellites"]["sat_2"]
    assert not isolated["has_isl_peer"]
    assert "has_isl_peer" not in state["satellites"]["sat_0"]
    assert organisation.metrics()["coordination_messages"] == 0.0


def test_linked_dmas_view_is_the_idealised_neighbour_telemetry() -> None:
    organisation = ORGANISATIONS["dmas"]({"peer_view": "linked"})
    organisation.initialize(["sat_0", "sat_1", "sat_2"])
    state = observation(linked=True)
    views = organisation.distribute_observation(state, organisation.channel(state))
    assert set(views["sat_agent_0"].local_state["full_observation"]["satellites"]) == {
        "sat_0",
        "sat_1",
    }
    assert organisation.metrics()["coordination_messages"] == 2.0


def test_commands_for_unknown_satellites_are_rejected() -> None:
    organisation = ORGANISATIONS["imas"]()
    organisation.initialize(["sat_0", "sat_1"])
    plans = {"sat_agent_0": AgentAction("sat_agent_0", {"sat_9": {"mode": "charging"}})}
    with pytest.raises(ValueError, match="unknown satellites"):
        organisation.collect_actions(plans, Channel(None))


def test_unknown_organisation_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown organisation"):
        create_organisation("unknown")


def test_decentralised_memory_cannot_recover_remote_truth_after_step() -> None:
    state = observation(linked=False)
    loops = create_organisation("imas")
    loops.reset(4, state)
    loops.act(state)
    next_state = observation(linked=True)
    next_state["satellites"]["sat_0"]["battery_soc"] = 0.1
    loops.after_step({"global_truth": "unavailable"}, next_state)
    record = loops.memories["sat_agent_0"].recent(1)[0]
    remembered = record["observation"]
    assert set(remembered["satellites"]) == {"sat_0"}
    assert remembered["satellites"]["sat_0"]["battery_soc"] == 0.9
    assert "ssa_custody_utility" not in remembered["global"]
    assert "info" not in record
    assert loops.memories["sat_agent_0"].recent(0) == ()


def test_organisations_create_decision_loops_through_the_plugin_registry(monkeypatch) -> None:
    from autops.core import plugin

    created: list[tuple[str, str, str]] = []
    original = plugin.create_representation

    def recording(mission: str, token: str, role: str, config: dict | None = None):
        created.append((mission, token, role))
        return original(mission, token, role, config)

    monkeypatch.setattr("autops.organisations.loops.create_representation", recording)
    loops = create_organisation("imas", {"representation": "symb"})
    state = {"step": 0, "satellites": {"sat_0": {}, "sat_1": {}}, "global": {}}
    loops.reset(0, state)
    loops.act(state)
    assert created == [("ssa", "symb", "onboard")] * 2


@pytest.mark.parametrize(
    ("token", "links"),
    [("imas", set()), ("dmas", {("sat_0", "sat_1"), ("sat_1", "sat_0")})],
)
def test_only_organisations_with_a_channel_authorise_isl_links(token, links) -> None:
    from autops.organisations.base import bind_communication_topology

    class Recorder:
        links: object = "unset"

        def configure_communication_links(self, value: object) -> None:
            self.links = value

    organisation = ORGANISATIONS[token]()
    organisation.initialize(["sat_0", "sat_1"])
    recorder = Recorder()
    bind_communication_topology(organisation, recorder)
    assert recorder.links == links
    assert organisation.authorized_destinations("sat_0") == sorted(
        destination for source, destination in links if source == "sat_0"
    )
