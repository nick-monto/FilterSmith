# FilterSmith

Prompt-driven voice modulation using Agent Framework with local Ollama, multi-agent orchestration, and DSP filtering with `scipy`.

## What this project does

You provide:
- an audio file
- a natural-language prompt like: `Make me sound like a robot from the past`

Input formats supported by CLI include: `wav`, `mp3`, `m4a`, `aac`, `flac`, `ogg`, `opus`, `wma`, `aif`, `aiff`, `alac`, `mp4`, and `webm`.

The system:
- uses a planner agent backed by `OllamaChatClient` to convert intent into a structured filter plan
- coordinates planning + execution through a coordinator agent layer
- applies signal processing steps (`scipy`) to audio loaded with `pydub`
- always writes output as `.wav`

## Architecture

- `PromptPlanner` (`src/filtersmith/agents/planner.py`): prompt -> filter plan (JSON schema), using `OllamaChatClient(...).as_agent(...).run(...)`
- `FilterCoordinator` (`src/filtersmith/agents/coordinator.py`): orchestration between planner and DSP executor
- DSP engine (`src/filtersmith/dsp.py`): executes filter chains on PCM arrays
- MAF adapter (`src/filtersmith/agents/maf_adapter.py`): compatibility shell for Microsoft Agent Framework

## Supported DSP steps

- `lowpass`, `highpass`, `bandpass`, `bandstop`
- `bitcrush`, `ring_mod`, `distortion`, `echo`, `normalize`

## Requirements

- Python 3.10+
- `ffmpeg` installed on system path (required by `pydub`)
- Ollama running locally

## Setup

```bash
cd /home/nmonto/FilterSmith
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Ollama install, model pull, and serve (linux):

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:9b
ollama serve

#when done
systemctl stop ollama
```

## Usage

```bash
filtersmith-forge \
  --input input.wav \
  --output output_robot.wav \
  --prompt "Make me sound like a robot from the 1960s." \
  --model-name "qwen3.5:9b" \
  --show-plan
```

Equivalent module invocation:

```bash
python -m filtersmith.cli --help
```

Input can be any supported format, and output is always WAV:

```bash
filtersmith-forge \
  --input input.mp3 \
  --output output_from_mp3.wav \
  --prompt "Make me sound like an old radio announcer."
```

Save a generated filter profile as reusable JSON:

```bash
filtersmith-forge \
  --input input.wav \
  --output output_robot.wav \
  --prompt "Make me sound like a robot from the 1960s." \
  --save-profile profiles/robotic_vintage.json \
  --show-plan
```

Reuse a previously saved filter profile:

```bash
filtersmith-forge \
  --input input.wav \
  --output output_reused.wav \
  --profile profiles/robotic_vintage.json
```

## Notes on Microsoft Agent Framework

The planner uses `OllamaChatClient(model_id=..., host=...)` from Agent Framework.

- Use `--model-name` (or `--model`) to select an Ollama model.
- Use `--ollama-host` to target a non-default Ollama endpoint.
- If Agent Framework calls fail due to API/version differences, planner falls back to a built-in heuristic DSP plan so CLI behavior remains usable.

## Next enhancements

- Add multi-pass chains (parallel dry/wet routing and blend controls)
- Add objective metrics to auto-tune filter parameters
- Add a library of style presets with semantic retrieval
