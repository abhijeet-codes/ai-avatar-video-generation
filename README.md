# Avatar Studio

Avatar Studio is a local AI video workspace for Apple Silicon. It generates
speech, lip-synced avatar videos, two-speaker podcast videos, narrated slide
videos, and reusable slide presenter assets without a cloud API in the default
workflow.

## Documentation

| Document | Use it for |
| --- | --- |
| [doc/setup.md](doc/setup.md) | Installation, user manual, dashboard guide, folders, validation, and troubleshooting. |
| [doc/pipeline.md](doc/pipeline.md) | Current pipeline architecture, app flows, engine map, artifacts, and config reference. |

## What It Builds

| Workflow | Input | Output |
| --- | --- | --- |
| TTS | Text | Local WAV audio from Kokoro or Bark. |
| Voice Studio | Reference audio | Saved MLX voice profile and generated/converted speech. |
| Audio Lipsync | Avatar + uploaded audio | Lip-synced MP4. |
| Text Lipsync | Avatar + script | Final captioned avatar MP4. |
| Podcast | Two avatars + script or audio tracks | Two-speaker podcast MP4. |
| Slide Narrator | PDF + narration JSON | Narrated slideshow MP4. |
| Slide Presenter | PDF + JSON + avatar | Per-slide presenter clips, composites, and optional export MP4. |

## Current Stack

| Area | Implementation |
| --- | --- |
| App | Gradio dashboard in `app/dashboard.py`. |
| Package | Python source under `src/avatarpipeline/`. |
| TTS | Kokoro, Bark, and MLX/Qwen voice tools. |
| Lip-sync | MuseTalk 1.5 by default; optional SadTalker 256 px and SadTalker HD. |
| Enhancement | CodeFormer, GFPGAN, or passthrough depending on local availability. |
| Captions | faster-whisper SRT generation, burned in with FFmpeg. |
| Video assembly | FFmpeg H.264/AAC outputs for 9:16, 16:9, and 1:1. |

## Quick Start

Install the system tools first:

```bash
brew install python@3.10 git ffmpeg uv espeak-ng
```

Set up the project and MuseTalk:

```bash
bash install/setup.sh
source .venv/bin/activate
```

Start the dashboard:

```bash
python scripts/run_dashboard.py
```

The app opens on `http://127.0.0.1:7860` or the next free port.

Useful dashboard launch options:

```bash
python scripts/run_dashboard.py --port 7861
python scripts/run_dashboard.py --no-browser
python scripts/run_dashboard.py --host 0.0.0.0
python scripts/run_dashboard.py --share
```

## Required Local Paths

| Path | Purpose |
| --- | --- |
| `.venv/` | Project Python environment. |
| `~/MuseTalk/` | MuseTalk checkout and model files used by the default lip-sync engine. |
| `~/MuseTalk/musetalk-env/` | MuseTalk Python environment. |
| `~/SadTalker/` | Optional SadTalker checkout for SadTalker engines. |
| `data/avatars/avatar.png` | Default avatar for CLI runs. |
| `data/output/` | Generated videos. |
| `data/presentations/` | Slide presenter projects and exports. |
| `data/voices/` | Saved MLX voice profiles. |

## Running From the CLI

Use the CLI for repeatable single-avatar text-to-video runs:

```bash
python scripts/run_pipeline.py \
  --script "Hello, this is Avatar Studio." \
  --voice af_heart \
  --orientation 9:16 \
  --engine musetalk \
  --out data/output/example.mp4
```

Fast preview run:

```bash
python scripts/run_pipeline.py \
  --script "Quick local test." \
  --no-enhance \
  --no-captions
```

List Kokoro voices:

```bash
python scripts/run_pipeline.py --list-voices
```

CLI engine choices are `musetalk`, `sadtalker`, and `sadtalker_hd`.

## Project Layout

```text
ai-avatar-video-generation/
  app/                         Gradio dashboard
  assets/                      Logo and favicon
  configs/                     Runtime settings
  data/                        Local runtime data, outputs, and temporary files
  doc/                         Setup manual and pipeline reference
  install/                     Setup scripts for external model environments
  scripts/                     Dashboard, CLI, smoke test, maintenance entry points
  src/avatarpipeline/          Core package, engines, pipelines, postprocess helpers
  tests/                       Unit and integration tests
  tools/                       Standalone helper scripts
```

## Configuration

Edit `configs/settings.yaml` when you want stable defaults for repeated runs.
The main settings are:

| Setting | Purpose |
| --- | --- |
| `musetalk_dir` | Location of the MuseTalk checkout. |
| `sadtalker_dir` | Location of the optional SadTalker checkout. |
| `lipsync.default_engine` | Default engine key. |
| `musetalk.default_batch_size` | MuseTalk throughput/memory tradeoff. |
| `musetalk.default_bbox_shift` | MuseTalk lip-region adjustment. |
| `tts.default_voice` | Default Kokoro voice. |
| `bark.model_id` | Bark model variant. |

Most generated files are intentionally written under `data/` and are not part
of the source package.

## Validation

Run the lightweight test suite:

```bash
python -m pytest tests/unit tests/integration -v
```

Run a real smoke test after models are installed:

```bash
bash scripts/smoke_test.sh --no-enhance --no-captions
```

The smoke test creates a short MP4 and validates video stream, audio stream,
duration, and file size.

## Notes

The current product path uses MuseTalk and SadTalker for lip-sync. Older
LatentSync documentation was removed because it no longer describes the active
dashboard or CLI pipeline.

License: Apache 2.0. See [LICENSE](LICENSE).
