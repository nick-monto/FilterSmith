from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydub import AudioSegment

from ..dsp import apply_filter_plan, plan_to_dict
from ..models import AudioContext, FilterPlan
from .planner import PlannerConfig, PromptPlanner


@dataclass
class CoordinatorConfig:
    planner: PlannerConfig = field(default_factory=PlannerConfig)


class FilterCoordinator:
    """
    Coordinates planning and DSP execution.

    If Microsoft Agent Framework is installed and preferred, this class can be
    used as the central orchestrator while delegating plan generation and DSP
    execution to separate agents/tools.
    """

    def __init__(self, config: CoordinatorConfig | None = None):
        self.config = config or CoordinatorConfig()
        self.planner = PromptPlanner(self.config.planner)

    def process_prompt(self, audio: AudioSegment, user_prompt: str) -> tuple[AudioSegment, FilterPlan, AudioContext]:
        # Normalize source format to 16-bit PCM for predictable DSP processing.
        normalized = audio.set_sample_width(2)
        ctx = AudioContext(
            sample_rate=normalized.frame_rate,
            channels=normalized.channels,
            sample_width=normalized.sample_width,
            frame_count=int(normalized.frame_count()),
        )
        plan = self.planner.create_plan(user_prompt)
        output = apply_filter_plan(normalized, plan)
        return output, plan, ctx

    def process_plan(self, audio: AudioSegment, plan: FilterPlan) -> tuple[AudioSegment, AudioContext]:
        # Reuse the same normalization/context flow used in prompt-based processing.
        normalized = audio.set_sample_width(2)
        ctx = AudioContext(
            sample_rate=normalized.frame_rate,
            channels=normalized.channels,
            sample_width=normalized.sample_width,
            frame_count=int(normalized.frame_count()),
        )
        output = apply_filter_plan(normalized, plan)
        return output, ctx

    def inspect_plan(self, user_prompt: str) -> dict[str, Any]:
        plan = self.planner.create_plan(user_prompt)
        return plan_to_dict(plan)
