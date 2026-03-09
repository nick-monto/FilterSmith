from __future__ import annotations
import argparse
import json
from pathlib import Path
from pydub import AudioSegment
from .agents.coordinator import CoordinatorConfig, FilterCoordinator
from .agents.planner import PlannerConfig
from .dsp import plan_from_dict, plan_to_dict


SUPPORTED_INPUT_EXTENSIONS = {
    "wav",
    "mp3",
    "m4a",
    "aac",
    "flac",
    "ogg",
    "opus",
    "wma",
    "aif",
    "aiff",
    "alac",
    "mp4",
    "webm",
}


def _log(message: str) -> None:
    print(f"[filtersmith] {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt-driven voice modulation filters")
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to input audio file. Supported extensions: "
            "wav, mp3, m4a, aac, flac, ogg, opus, wma, aif, aiff, alac, mp4, webm"
        ),
    )
    parser.add_argument("--output", required=True, help="Path to output audio file (.wav is always written)")
    parser.add_argument("--prompt", help="Desired voice effect prompt")
    parser.add_argument(
        "--profile",
        help="Path to a saved filter profile JSON. If provided, prompt generation is skipped.",
    )
    parser.add_argument(
        "--save-profile",
        help="Path to write generated filter profile JSON for reuse.",
    )
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model",
        default="qwen3.5:9b",
        help="Ollama model name (alias: --model-name)",
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL")
    parser.add_argument("--show-plan", action="store_true", help="Print generated filter plan")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    _log(f"Starting run with input='{input_path}' output='{output_path}'")

    input_ext = input_path.suffix.lower().removeprefix(".")
    if input_ext not in SUPPORTED_INPUT_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise SystemExit(f"Unsupported input extension '.{input_ext}'. Supported: {allowed}")
    _log(f"Input extension '.{input_ext}' accepted")

    audio = AudioSegment.from_file(input_path)
    _log(
        "Loaded input audio "
        f"(sr={audio.frame_rate}, channels={audio.channels}, sample_width={audio.sample_width})"
    )

    if output_path.suffix.lower() != ".wav":
        output_path = output_path.with_suffix(".wav")
        _log(f"Adjusted output path to WAV: '{output_path}'")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.prompt and not args.profile:
        raise SystemExit("Provide --prompt or --profile")

    coordinator = FilterCoordinator(
        CoordinatorConfig(
            planner=PlannerConfig(
                model=args.model,
                host=args.ollama_host,
            )
        )
    )

    if args.profile:
        _log(f"Loading filter profile from '{args.profile}'")
        profile_path = Path(args.profile)
        profile_data = json.loads(profile_path.read_text(encoding="utf-8"))
        plan = plan_from_dict(profile_data)
        _log(f"Loaded profile style='{plan.style}' with {len(plan.steps)} steps")
        processed, _ctx = coordinator.process_plan(audio, plan)
    else:
        _log(f"Generating filter plan from prompt using model '{args.model}'")
        processed, plan, _ctx = coordinator.process_prompt(audio, args.prompt)
        _log(f"Generated plan style='{plan.style}' with {len(plan.steps)} steps")

    if args.save_profile:
        save_profile_path = Path(args.save_profile)
        save_profile_path.parent.mkdir(parents=True, exist_ok=True)
        save_profile_path.write_text(json.dumps(plan_to_dict(plan), indent=2), encoding="utf-8")
        _log(f"Saved filter profile to '{save_profile_path}'")

    processed.export(output_path, format="wav")
    _log(f"Wrote processed output WAV to '{output_path}'")

    if args.show_plan:
        _log("Printing generated/loaded plan")
        print(json.dumps(plan_to_dict(plan), indent=2))


if __name__ == "__main__":
    main()
