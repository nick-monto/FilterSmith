from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass

from agent_framework.ollama import OllamaChatClient

from ..models import FilterPlan, FilterStep


@dataclass
class PlannerConfig:
    model: str = "qwen3.5:9b"
    host: str = "http://localhost:11434"


def _log(message: str) -> None:
    print(f"[filtersmith] {message}")


class PromptPlanner:
    """Creates a DSP plan from user intent using Agent Framework providers."""

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()
        self.agent = self._build_planner_agent()

    def _build_planner_agent(self):
        client = OllamaChatClient(
            model_id=self.config.model,
            host=self.config.host,
        )

        return client.as_agent(
            name="PromptPlanner",
            instructions="You are a DSP planning agent for vocal FX. Return JSON only.",
        )

    def create_plan(self, user_prompt: str) -> FilterPlan:
        try:
            generated = self._generate_json_plan(user_prompt)
            return self._coerce_plan(generated)
        except Exception as exc:
            _log(f"Planner generation failed ({type(exc).__name__}); using heuristic fallback plan")
            return self._fallback_plan(user_prompt)

    def _generate_json_plan(self, user_prompt: str) -> dict:
        response_text = self._generate_plan_text(user_prompt)
        return self._parse_json_plan(response_text)

    def _parse_json_plan(self, response_text: str) -> dict:
        text = response_text.strip()

        # Fast path for well-formed JSON responses.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Common pattern: JSON wrapped in markdown fences.
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass

        # Fallback: scan for the first balanced JSON object.
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : idx + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except Exception:
                            break
            start = text.find("{", start + 1)

        raise json.JSONDecodeError("Could not locate valid JSON object in planner response", text, 0)

    def _generate_plan_text(self, user_prompt: str) -> str:
        schema_hint = {
            "style": "robotic_vintage",
            "rationale": "short explanation",
            "steps": [
                {"type": "bandpass", "params": {"low_hz": 300, "high_hz": 2600, "order": 4}},
                {"type": "bitcrush", "params": {"bits": 6}},
                {"type": "ring_mod", "params": {"freq_hz": 75, "mix": 0.35}},
                {"type": "normalize", "params": {}},
            ],
        }

        prompt = (
            "You are a DSP planning agent for vocal FX. Return JSON only. "
            "Allowed filter step types: lowpass, highpass, bandpass, bandstop, bitcrush, ring_mod, distortion, echo, normalize. "
            "No markdown, no prose.\n"
            f"User request: {user_prompt}\n"
            f"Example schema: {json.dumps(schema_hint)}"
        )

        return self._generate_with_agent_framework(prompt)

    def _generate_with_agent_framework(self, prompt: str) -> str:
        _log(f"Calling Agent Framework planner (model='{self.config.model}', host='{self.config.host}')")
        return asyncio.run(self._run_agent_prompt(self.agent, prompt)).strip()

    async def _run_agent_prompt(self, agent, prompt: str) -> str:
        response = await agent.run(prompt)
        if isinstance(response, str):
            return response

        # Normalize likely response object shapes across framework versions.
        for attr in ("text", "output_text", "content"):
            value = getattr(response, attr, None)
            if isinstance(value, str) and value.strip():
                return value

        return str(response)

    def _coerce_plan(self, data: dict) -> FilterPlan:
        steps = []
        for step in data.get("steps", []):
            step_type = step.get("type", "normalize")
            params = step.get("params", {}) if isinstance(step.get("params", {}), dict) else {}
            steps.append(FilterStep(type=step_type, params=params))

        if not steps:
            steps = [FilterStep(type="normalize", params={})]

        return FilterPlan(
            style=str(data.get("style", "custom")),
            rationale=str(data.get("rationale", "Generated from prompt.")),
            steps=steps,
        )

    def _fallback_plan(self, user_prompt: str) -> FilterPlan:
        text = user_prompt.lower()
        if "robot" in text or "cyborg" in text:
            return FilterPlan(
                style="robotic_vintage",
                rationale="Band-limit speech and add ring modulation and quantization for retro robot character.",
                steps=[
                    FilterStep(type="highpass", params={"cutoff_hz": 180, "order": 3}),
                    FilterStep(type="bandpass", params={"low_hz": 300, "high_hz": 2600, "order": 4}),
                    FilterStep(type="ring_mod", params={"freq_hz": 72, "mix": 0.35}),
                    FilterStep(type="bitcrush", params={"bits": 7}),
                    FilterStep(type="normalize", params={}),
                ],
            )

        if "telephone" in text or "radio" in text:
            return FilterPlan(
                style="lofi_comms",
                rationale="Narrow bandpass plus slight distortion recreates old communication systems.",
                steps=[
                    FilterStep(type="bandpass", params={"low_hz": 300, "high_hz": 3000, "order": 4}),
                    FilterStep(type="distortion", params={"drive": 1.25}),
                    FilterStep(type="normalize", params={}),
                ],
            )

        return FilterPlan(
            style="enhanced_clean",
            rationale="Default enhancement keeps voice intelligible while adding mild character.",
            steps=[
                FilterStep(type="highpass", params={"cutoff_hz": 120, "order": 3}),
                FilterStep(type="normalize", params={}),
            ],
        )
