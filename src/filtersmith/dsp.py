from __future__ import annotations

from dataclasses import asdict
from typing import Literal, cast

import numpy as np
from pydub import AudioSegment
from scipy import signal

from .models import FilterPlan, FilterStep, FilterType

_INT16_MAX = np.iinfo(np.int16).max
_VALID_FILTER_TYPES: set[FilterType] = {
    "lowpass",
    "highpass",
    "bandpass",
    "bandstop",
    "bitcrush",
    "ring_mod",
    "distortion",
    "echo",
    "normalize",
}


def _log(message: str) -> None:
    print(f"[filtersmith] {message}")


def audiosegment_to_array(seg: AudioSegment) -> tuple[np.ndarray, int, int, int]:
    """Return float32 array in range [-1, 1], sample rate, channels, sample width."""
    sample_width = seg.sample_width
    channels = seg.channels
    sample_rate = seg.frame_rate

    samples = np.array(seg.get_array_of_samples())
    if channels > 1:
        samples = samples.reshape((-1, channels))
    else:
        samples = samples.reshape((-1, 1))

    max_val = float(1 << (8 * sample_width - 1))
    normalized = samples.astype(np.float32) / max_val
    return normalized, sample_rate, channels, sample_width


def array_to_audiosegment(
    arr: np.ndarray, sample_rate: int, channels: int, sample_width: int
) -> AudioSegment:
    clipped = np.clip(arr, -1.0, 1.0)
    max_val = float(1 << (8 * sample_width - 1)) - 1.0
    int_data = (clipped * max_val).astype(np.int16)

    if channels > 1:
        interleaved = int_data.reshape((-1,)).tolist()
    else:
        interleaved = int_data[:, 0].tolist()

    return AudioSegment(
        data=np.array(interleaved, dtype=np.int16).tobytes(),
        sample_width=2,
        frame_rate=sample_rate,
        channels=channels,
    )


def _butter_filter(
    audio: np.ndarray,
    sample_rate: int,
    step_type: Literal["lowpass", "highpass", "bandpass", "bandstop"],
    cutoff: float | tuple[float, float],
    order: int = 4,
) -> np.ndarray:
    nyquist = sample_rate / 2.0

    if step_type in {"lowpass", "highpass"}:
        if isinstance(cutoff, tuple):
            cutoff_value = cutoff[0]
        else:
            cutoff_value = cutoff
        w: float | tuple[float, float] = float(cutoff_value) / nyquist
    else:
        if isinstance(cutoff, tuple):
            lo, hi = cutoff
        else:
            lo = hi = cutoff
        w = (float(lo) / nyquist, float(hi) / nyquist)

    # Stubs for scipy can include alternate return shapes; we always request
    # transfer-function coefficients and use the filtered array output path.
    b, a = cast(tuple[np.ndarray, np.ndarray], signal.butter(order, w, btype=step_type, output="ba"))
    return cast(np.ndarray, signal.lfilter(b, a, audio, axis=0))


def _apply_ring_mod(audio: np.ndarray, sample_rate: int, freq: float, mix: float) -> np.ndarray:
    t = np.arange(audio.shape[0]) / sample_rate
    carrier = np.sin(2.0 * np.pi * freq * t).reshape((-1, 1))
    wet = audio * carrier
    return (1.0 - mix) * audio + mix * wet


def _apply_bitcrush(audio: np.ndarray, bits: int) -> np.ndarray:
    levels = float(2**bits)
    return np.round(audio * levels) / levels


def _apply_distortion(audio: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(drive * audio)


def _apply_echo(audio: np.ndarray, sample_rate: int, delay_ms: int, decay: float) -> np.ndarray:
    delay_samples = int(sample_rate * (delay_ms / 1000.0))
    if delay_samples <= 0:
        return audio
    out = np.copy(audio)
    out[delay_samples:] += decay * audio[:-delay_samples]
    return np.clip(out, -1.0, 1.0)


def _apply_step(audio: np.ndarray, sample_rate: int, step: FilterStep) -> np.ndarray:
    p = step.params
    if step.type == "lowpass":
        return _butter_filter(audio, sample_rate, "lowpass", p.get("cutoff_hz", 1800.0), p.get("order", 4))
    if step.type == "highpass":
        return _butter_filter(audio, sample_rate, "highpass", p.get("cutoff_hz", 120.0), p.get("order", 4))
    if step.type == "bandpass":
        return _butter_filter(
            audio,
            sample_rate,
            "bandpass",
            (p.get("low_hz", 250.0), p.get("high_hz", 3200.0)),
            p.get("order", 4),
        )
    if step.type == "bandstop":
        return _butter_filter(
            audio,
            sample_rate,
            "bandstop",
            (p.get("low_hz", 300.0), p.get("high_hz", 1800.0)),
            p.get("order", 4),
        )
    if step.type == "ring_mod":
        return _apply_ring_mod(audio, sample_rate, p.get("freq_hz", 70.0), p.get("mix", 0.4))
    if step.type == "bitcrush":
        return _apply_bitcrush(audio, p.get("bits", 6))
    if step.type == "distortion":
        return _apply_distortion(audio, p.get("drive", 1.8))
    if step.type == "echo":
        return _apply_echo(audio, sample_rate, p.get("delay_ms", 60), p.get("decay", 0.25))
    if step.type == "normalize":
        peak = np.max(np.abs(audio))
        if peak < 1e-9:
            return audio
        return audio / peak * 0.95
    return audio


def apply_filter_plan(seg: AudioSegment, plan: FilterPlan) -> AudioSegment:
    audio, sr, channels, sw = audiosegment_to_array(seg)
    out = audio
    _log(f"Applying DSP plan style='{plan.style}' with {len(plan.steps)} steps")
    for step in plan.steps:
        _log(f"Applying step '{step.type}' with params={step.params}")
        out = _apply_step(out, sr, step)
    _log("DSP plan application complete")
    return array_to_audiosegment(out, sr, channels, sw)


def plan_to_dict(plan: FilterPlan) -> dict:
    return {
        "style": plan.style,
        "rationale": plan.rationale,
        "steps": [asdict(step) for step in plan.steps],
    }


def plan_from_dict(data: dict) -> FilterPlan:
    steps_data = data.get("steps", [])
    steps: list[FilterStep] = []
    for raw_step in steps_data:
        if not isinstance(raw_step, dict):
            continue
        step_type_raw = str(raw_step.get("type", "normalize"))
        step_type: FilterType = step_type_raw if step_type_raw in _VALID_FILTER_TYPES else "normalize"
        params_raw = raw_step.get("params", {})
        params = params_raw if isinstance(params_raw, dict) else {}
        steps.append(FilterStep(type=step_type, params=params))

    if not steps:
        steps = [FilterStep(type="normalize", params={})]

    return FilterPlan(
        style=str(data.get("style", "custom")),
        rationale=str(data.get("rationale", "Loaded from saved profile.")),
        steps=steps,
    )
