"""Single-shot and bounded-agentic EventSat schedule representations."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, ClassVar

from autops.core.plugin import Representation, register
from autops.core.types import DecisionContext, SpaceSpec
from autops.llm.agentic_prompts import (
    AGENTIC_SCHEDULE_SYSTEM_PROMPT,
    format_forced_schedule_prompt,
    format_schedule_planning_prompt,
    format_schedule_tool_result_prompt,
)
from autops.llm.client import LLMClient
from autops.llm.llm_prompts import SCHEDULE_SYSTEM_PROMPT, format_schedule_prompt
from autops.llm.onboard_prompts import (
    ONBOARD_AGENTIC_SCHEDULE_SYSTEM_PROMPT,
    ONBOARD_SCHEDULE_SYSTEM_PROMPT,
    format_forced_onboard_schedule_prompt,
    format_onboard_schedule_prompt,
    format_onboard_tool_result_prompt,
)
from autops.llm.tools import execute_tool
from autops.missions.eventsat.physics import MODES, encode_vectors
from autops.paradigms.base import expand_schedule
from autops.wm.cem import CEMConfig

_VALID_MODES = frozenset(MODES)
_OPERATIONAL_MODES = frozenset(
    {"payload_observe", "payload_compress", "payload_detect", "payload_send"}
)


def _json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating fences and bounded leading model chatter."""

    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("```")
        ).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as original:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise original
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def _decision(value: dict[str, Any]) -> dict[str, Any] | None:
    nested = value.get("decision")
    if isinstance(nested, dict):
        return nested
    return value if value.get("schedule") is not None else None


def _merge_schedule(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for block in blocks:
        if merged and merged[-1]["mode"] == block["mode"]:
            merged[-1]["steps"] += block["steps"]
        else:
            merged.append(dict(block))
    return merged


class LLMSchedulePlanner(Representation):
    """Shared schedule parser, bounded LLM loop, and optional symbolic shield."""

    observation_space = SpaceSpec((25,), "float32", 0.0, 1.0)
    action_space = SpaceSpec((7,), "int64", 0, 1, MODES)
    symbolic_grounding: ClassVar[bool] = False
    agentic: ClassVar[bool] = False
    role: ClassVar[str] = "ground"
    allow_scheduled_communication: ClassVar[bool] = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.client = LLMClient(self.config)
        self.max_retries = max(0, int(self.config.get("llm_parse_retries", 2)))
        # Echo tools are in the prompt. Three model turns bound the what-if loop;
        # one forced answer-extraction call may follow if the model never decides.
        self.max_agentic_steps = max(1, int(self.config.get("max_agentic_steps", 3)))
        self._tool_calls = 0
        self._grounding_overrides = 0
        self.plan_hold = max(1, int(self.config.get("plan_hold", CEMConfig().plan_hold)))
        self._held_modes: list[str] = []
        self._planning_events = 0
        self._held_action_steps = 0
        self._planning_latency_s = 0.0

    def encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        _, _, raw = encode_vectors(observation)
        raw["ground_pass_active"] = bool(raw.get("contact_window_active", False))
        raw["step"] = int(observation.get("step", 0))
        raw.setdefault("step_duration_s", 60.0)
        raw.setdefault("battery_min_soc", 0.20)
        raw.setdefault("compression_time_factor", 2.0)
        raw.setdefault("detection_steps", 5.0)
        raw["estimated_gap_steps"] = max(
            1,
            int(
                raw.get(
                    "planning_gap_steps",
                    raw.get("following_gap_steps", raw.get("orbital_period_steps", 92)),
                )
            ),
        )
        return raw

    def select_action(self, context: DecisionContext) -> dict[str, Any]:
        state = context.state
        if not state:
            raise RuntimeError(f"LLM {self.role} planner received no EventSat state")
        if self.role == "onboard":
            return self._select_onboard(state)
        mode, schedule = self._plan(state, max(1, int(state.get("estimated_gap_steps", 92))))
        return {"eventsat_0": {"mode": mode}, "schedule": schedule}

    def _select_onboard(self, state: dict[str, Any]) -> dict[str, Any]:
        if self._held_modes:
            mode = self._held_modes.pop(0)
            if self.symbolic_grounding:
                mode = self._ground_held_mode(mode, state)
            self._held_action_steps += 1
            self._last_rationale = f"Executed held {self.__class__.__name__} action"
            return self._onboard_action(mode, planned=False)

        remaining_steps = max(0, self.plan_hold - 1)
        started = perf_counter()
        mode, schedule = self._plan(state, max(1, remaining_steps))
        elapsed = max(0.0, perf_counter() - started)
        self._held_modes = expand_schedule(schedule)[:remaining_steps]
        self._planning_events += 1
        self._planning_latency_s += elapsed
        return self._onboard_action(mode, planned=True, planner_active_s=elapsed)

    def _plan(self, state: dict[str, Any], schedule_steps: int) -> tuple[str, list[dict[str, Any]]]:
        errors: list[str] = []
        for _ in range(self.max_retries + 1):
            try:
                payload, trace = (
                    self._agentic_payload(state, schedule_steps)
                    if self.agentic
                    else self._single_shot_payload(state, schedule_steps)
                )
                mode = self._immediate_mode(payload, state)
                schedule = self._schedule(payload.get("schedule"), schedule_steps, state)
                rationale = str(payload.get("rationale", "")).strip()
                prefix = "Agentic" if self.agentic else "Single-shot"
                self._last_rationale = f"{prefix} LLM {self.role} plan{trace}: {rationale}"
                return mode, schedule
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        detail = errors[-1] if errors else "no valid decision"
        raise RuntimeError(
            "LLM substrate integrity violation: no valid immediate mode and schedule "
            f"after {self.max_retries + 1} attempts ({detail})"
        )

    def _single_shot_payload(
        self, state: dict[str, Any], gap_steps: int
    ) -> tuple[dict[str, Any], str]:
        system_prompt = (
            ONBOARD_SCHEDULE_SYSTEM_PROMPT if self.role == "onboard" else SCHEDULE_SYSTEM_PROMPT
        )
        user_prompt = (
            format_onboard_schedule_prompt(state, gap_steps)
            if self.role == "onboard"
            else format_schedule_prompt(state, gap_steps)
        )
        raw = self.client.generate(
            system_prompt,
            user_prompt,
            json_mode=True,
        )
        parsed = _decision(_json_object(raw))
        if parsed is None:
            raise ValueError("response contains no schedule decision")
        return parsed, ""

    def _agentic_payload(self, state: dict[str, Any], gap_steps: int) -> tuple[dict[str, Any], str]:
        context: list[dict[str, Any]] = []
        system_prompt = (
            ONBOARD_AGENTIC_SCHEDULE_SYSTEM_PROMPT
            if self.role == "onboard"
            else AGENTIC_SCHEDULE_SYSTEM_PROMPT
        )
        user_prompt = (
            format_onboard_schedule_prompt(state, gap_steps)
            if self.role == "onboard"
            else format_schedule_planning_prompt(state, gap_steps)
        )
        raw = self.client.generate(
            system_prompt,
            user_prompt,
            json_mode=True,
        )
        parsed = _json_object(raw)
        turns = 1
        decision = _decision(parsed)
        tool_call = parsed.get("tool_call")
        context.append({"step": "plan", "content": parsed.get("plan", "")})

        while decision is None and isinstance(tool_call, dict) and turns < self.max_agentic_steps:
            name = str(tool_call.get("name", ""))
            args = tool_call.get("args", {})
            if not isinstance(args, dict):
                args = {}
            result = execute_tool(name, args, state)
            self._tool_calls += 1
            context.append({"step": "tool", "name": name, "result": result})
            user_prompt = (
                format_onboard_tool_result_prompt(name, result, context, gap_steps)
                if self.role == "onboard"
                else format_schedule_tool_result_prompt(name, result, context, gap_steps)
            )
            raw = self.client.generate(
                system_prompt,
                user_prompt,
                json_mode=True,
            )
            parsed = _json_object(raw)
            turns += 1
            context.append({"step": "reflect", "content": parsed.get("reflection", "")})
            decision = _decision(parsed)
            tool_call = parsed.get("tool_call")

        if decision is None:
            user_prompt = (
                format_forced_onboard_schedule_prompt(gap_steps)
                if self.role == "onboard"
                else format_forced_schedule_prompt(context, gap_steps)
            )
            raw = self.client.generate(
                system_prompt,
                user_prompt,
                json_mode=True,
            )
            decision = _decision(_json_object(raw))
        if decision is None:
            raise ValueError("agentic loop ended without a schedule decision")
        names = [str(item.get("name")) for item in context if item.get("step") == "tool"]
        trace = f" via tools [{', '.join(names)}]" if names else ""
        return decision, trace

    def _immediate_mode(self, payload: dict[str, Any], state: dict[str, Any]) -> str:
        mode = payload.get("mode", payload.get("pass_mode"))
        if mode not in _VALID_MODES:
            raise ValueError(f"invalid contact mode {mode!r}")
        if not self.symbolic_grounding:
            return str(mode)
        grounded = str(mode)
        unhealthy = state.get("health_status", "nominal") != "nominal"
        battery_critical = float(state.get("battery_soc", 0.5)) <= float(
            state.get("battery_min_soc", 0.20)
        )
        if unhealthy or battery_critical:
            grounded = "safe"
        elif grounded == "communication" and not state.get("ground_pass_active", False):
            grounded = "charging"
        if grounded != mode:
            self._grounding_overrides += 1
        return grounded

    def _schedule(self, raw: Any, gap_steps: int, state: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            raise TypeError("schedule must be a list")
        blocks: list[dict[str, Any]] = []
        total = 0
        allowed = (
            _VALID_MODES - {"communication"}
            if self.symbolic_grounding and not self.allow_scheduled_communication
            else _VALID_MODES
        )
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                mode, duration = entry[0], entry[1]
            elif isinstance(entry, dict):
                mode, duration = entry.get("mode"), entry.get("steps", entry.get("duration"))
            else:
                continue
            if mode not in allowed or isinstance(duration, bool):
                continue
            try:
                steps = int(duration)
            except (TypeError, ValueError):
                continue
            if steps < 1:
                continue
            if self.symbolic_grounding:
                steps = min(steps, gap_steps - total)
            if steps < 1:
                break
            blocks.append({"mode": str(mode), "steps": steps})
            total += steps
            if self.symbolic_grounding and total >= gap_steps:
                break
        if not blocks:
            raise ValueError("schedule contains no valid positive-duration block")
        if self.symbolic_grounding and total < gap_steps:
            blocks.append({"mode": "charging", "steps": gap_steps - total})
        return self._shield(_merge_schedule(blocks), state)

    def _shield(self, blocks: list[dict[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.symbolic_grounding:
            return blocks
        soc = float(state.get("battery_soc", 0.5))
        floor = float(self.config.get("llm_operations_soc_floor", 0.35))
        obc = float(state.get("obc_data_mb", 0.0))
        capacity = max(float(state.get("storage_capacity_mb", 4096.0)), 1e-12)
        shielded: list[dict[str, Any]] = []
        for block in blocks:
            mode = block["mode"]
            if mode in _OPERATIONAL_MODES and (
                soc < floor or (mode == "payload_observe" and obc >= 0.8 * capacity)
            ):
                mode = "charging"
                self._grounding_overrides += 1
            shielded.append({"mode": mode, "steps": block["steps"]})
        return _merge_schedule(shielded)

    def _ground_held_mode(self, mode: str, state: dict[str, Any]) -> str:
        """Apply the current-state symbolic guard before executing a held action."""

        immediate = self._immediate_mode({"mode": mode}, state)
        return str(self._shield([{"mode": immediate, "steps": 1}], state)[0]["mode"])

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = {
            **self.client.metrics(),
            "agentic_tool_calls": float(self._tool_calls),
            "grounding_overrides": float(self._grounding_overrides),
            "llm_provenance": self.client.provenance(),
        }
        if self.role == "onboard":
            diagnostics.update(
                {
                    "plan_hold": self.plan_hold,
                    "planning_events": self._planning_events,
                    "held_action_steps": self._held_action_steps,
                    "planning_latency_total_s": self._planning_latency_s,
                    "planning_latency_mean_s": (
                        self._planning_latency_s / self._planning_events
                        if self._planning_events
                        else 0.0
                    ),
                }
            )
        return diagnostics

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._tool_calls = 0
        self._grounding_overrides = 0
        self._held_modes.clear()
        self._planning_events = 0
        self._held_action_steps = 0
        self._planning_latency_s = 0.0

    def _onboard_action(
        self,
        mode: str,
        *,
        planned: bool,
        planner_active_s: float = 0.0,
    ) -> dict[str, Any]:
        action: dict[str, Any] = {"mode": mode, "jetson_planned": planned}
        if planned:
            action["planner_active_s"] = max(0.0, planner_active_s)
        return {"eventsat_0": action}


@register("llm-s", mission="eventsat", role="ground")
class EventSatLLMSingleShot(LLMSchedulePlanner):
    """Pure LLM single-shot ground planner; safety remains prompt/environment-owned."""


@register("hllm-s", mission="eventsat", role="ground")
class EventSatHybridLLMSingleShot(LLMSchedulePlanner):
    """Single-shot LLM with symbolic schedule-format and safety grounding."""

    symbolic_grounding = True


@register("llm-a", mission="eventsat", role="ground")
class EventSatLLMAgentic(LLMSchedulePlanner):
    """Pure LLM bounded Plan-Tool-Reflect-Decide ground planner."""

    agentic = True


@register("hllm-a", mission="eventsat", role="ground")
class EventSatHybridLLMAgentic(LLMSchedulePlanner):
    """Bounded agentic LLM ground planner with a symbolic safety shield."""

    symbolic_grounding = True
    agentic = True


class LLMOnboardSchedulePlanner(LLMSchedulePlanner):
    """Fresh-telemetry onboard scheduler with bounded plan-and-hold execution."""

    role = "onboard"
    allow_scheduled_communication = True


@register("llm-s", mission="eventsat", role="onboard")
class EventSatOnboardLLMSingleShot(LLMOnboardSchedulePlanner):
    """Pure single-shot onboard LLM schedule planner."""


@register("hllm-s", mission="eventsat", role="onboard")
class EventSatOnboardHybridLLMSingleShot(LLMOnboardSchedulePlanner):
    """Single-shot onboard LLM planner with symbolic schedule grounding."""

    symbolic_grounding = True


@register("llm-a", mission="eventsat", role="onboard")
class EventSatOnboardLLMAgentic(LLMOnboardSchedulePlanner):
    """Bounded agentic onboard LLM schedule planner."""

    agentic = True


@register("hllm-a", mission="eventsat", role="onboard")
class EventSatOnboardHybridLLMAgentic(LLMOnboardSchedulePlanner):
    """Bounded agentic onboard LLM planner with symbolic grounding."""

    symbolic_grounding = True
    agentic = True
