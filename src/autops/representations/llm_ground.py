"""Single-shot and bounded-agentic EventSat ground-planner representations."""

from __future__ import annotations

import json
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
from autops.llm.tools import execute_tool
from autops.missions.eventsat.physics import MODES, encode_vectors

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


class LLMGroundPlanner(Representation):
    """Shared parser, substrate-integrity checks, and optional symbolic shield."""

    observation_space = SpaceSpec((25,), "float32", 0.0, 1.0)
    action_space = SpaceSpec((7,), "int64", 0, 1, MODES)
    symbolic_grounding: ClassVar[bool] = False
    agentic: ClassVar[bool] = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.client = LLMClient(self.config)
        self.max_retries = max(0, int(self.config.get("llm_parse_retries", 2)))
        # Echo tools are in the prompt. Three model turns bound the what-if loop;
        # one forced answer-extraction call may follow if the model never decides.
        self.max_agentic_steps = max(1, int(self.config.get("max_agentic_steps", 3)))
        self._tool_calls = 0
        self._grounding_overrides = 0

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
            raise RuntimeError("LLM ground planner received no EventSat state")
        gap_steps = max(1, int(state.get("estimated_gap_steps", 92)))
        errors: list[str] = []
        for _ in range(self.max_retries + 1):
            try:
                payload, trace = (
                    self._agentic_payload(state, gap_steps)
                    if self.agentic
                    else self._single_shot_payload(state, gap_steps)
                )
                mode = self._contact_mode(payload, state)
                schedule = self._schedule(payload.get("schedule"), gap_steps, state)
                rationale = str(payload.get("rationale", "")).strip()
                prefix = "Agentic" if self.agentic else "Single-shot"
                self._last_rationale = f"{prefix} LLM ground plan{trace}: {rationale}"
                return {"eventsat_0": {"mode": mode}, "schedule": schedule}
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        detail = errors[-1] if errors else "no valid decision"
        raise RuntimeError(
            "LLM substrate integrity violation: no valid contact mode and schedule "
            f"after {self.max_retries + 1} attempts ({detail})"
        )

    def _single_shot_payload(
        self, state: dict[str, Any], gap_steps: int
    ) -> tuple[dict[str, Any], str]:
        raw = self.client.generate(
            SCHEDULE_SYSTEM_PROMPT,
            format_schedule_prompt(state, gap_steps),
            json_mode=True,
        )
        parsed = _decision(_json_object(raw))
        if parsed is None:
            raise ValueError("response contains no schedule decision")
        return parsed, ""

    def _agentic_payload(self, state: dict[str, Any], gap_steps: int) -> tuple[dict[str, Any], str]:
        context: list[dict[str, Any]] = []
        raw = self.client.generate(
            AGENTIC_SCHEDULE_SYSTEM_PROMPT,
            format_schedule_planning_prompt(state, gap_steps),
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
            raw = self.client.generate(
                AGENTIC_SCHEDULE_SYSTEM_PROMPT,
                format_schedule_tool_result_prompt(name, result, context, gap_steps),
                json_mode=True,
            )
            parsed = _json_object(raw)
            turns += 1
            context.append({"step": "reflect", "content": parsed.get("reflection", "")})
            decision = _decision(parsed)
            tool_call = parsed.get("tool_call")

        if decision is None:
            raw = self.client.generate(
                AGENTIC_SCHEDULE_SYSTEM_PROMPT,
                format_forced_schedule_prompt(context, gap_steps),
                json_mode=True,
            )
            decision = _decision(_json_object(raw))
        if decision is None:
            raise ValueError("agentic loop ended without a schedule decision")
        names = [str(item.get("name")) for item in context if item.get("step") == "tool"]
        trace = f" via tools [{', '.join(names)}]" if names else ""
        return decision, trace

    def _contact_mode(self, payload: dict[str, Any], state: dict[str, Any]) -> str:
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
        allowed = _VALID_MODES - {"communication"} if self.symbolic_grounding else _VALID_MODES
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

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self.client.metrics(),
            "agentic_tool_calls": float(self._tool_calls),
            "grounding_overrides": float(self._grounding_overrides),
            "llm_provenance": self.client.provenance(),
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._tool_calls = 0
        self._grounding_overrides = 0


@register("llm-s", mission="eventsat", role="ground")
class EventSatLLMSingleShot(LLMGroundPlanner):
    """Pure LLM single-shot ground planner; safety remains prompt/environment-owned."""


@register("hllm-s", mission="eventsat", role="ground")
class EventSatHybridLLMSingleShot(LLMGroundPlanner):
    """Single-shot LLM with symbolic schedule-format and safety grounding."""

    symbolic_grounding = True


@register("llm-a", mission="eventsat", role="ground")
class EventSatLLMAgentic(LLMGroundPlanner):
    """Pure LLM bounded Plan-Tool-Reflect-Decide ground planner."""

    agentic = True


@register("hllm-a", mission="eventsat", role="ground")
class EventSatHybridLLMAgentic(LLMGroundPlanner):
    """Bounded agentic LLM ground planner with a symbolic safety shield."""

    symbolic_grounding = True
    agentic = True
