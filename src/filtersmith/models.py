from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FilterType = Literal[
    "lowpass",
    "highpass",
    "bandpass",
    "bandstop",
    "bitcrush",
    "ring_mod",
    "distortion",
    "echo",
    "normalize",
]


@dataclass
class FilterStep:
    """One DSP action requested by the planner."""

    type: FilterType
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterPlan:
    """A full plan for transforming the audio signal."""

    style: str
    rationale: str
    steps: list[FilterStep]


@dataclass
class AudioContext:
    """Information extracted from the loaded file used by agents and DSP."""

    sample_rate: int
    channels: int
    sample_width: int
    frame_count: int
