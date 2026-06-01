# Avatar Studio Setup Manual

This manual explains what needs to be installed, where each dependency lives,
and how to operate the local Avatar Studio app after setup. It is written as a
user manual: use it to understand each setup area and when you need it.

## Supported Environment

Avatar Studio is built for a local macOS workstation with Apple Silicon. The
project uses Python 3.10, FFmpeg, Gradio, local text-to-speech models, and
external lip-sync model folders under your home directory.

Expected baseline:

| Area | Utility |
| --- | --- |
| macOS on Apple Silicon | Runs local ML workloads through PyTorch MPS and MLX. |
| Homebrew | Installs Python, FFmpeg, uv, Git, and espeak-ng. |
| Python 3.10 | Required by the project and the external model environments. |
| FFmpeg and FFprobe | Audio conversion, video assembly, captions, and validation. |
| espeak-ng | Phoneme backend used by Kokoro TTS. |
| uv | Creates virtual environments and installs Python dependencies. |

## Repository Setup

Start in the repository root:

`/Users/abhijeetanand/Projects/Personal/02_Github_Projects/ai-avatar-video-generation`

The main setup script is `install/setup.sh`. It creates the project virtual
environment at `.venv`, installs the package in editable mode, clones MuseTalk
into `~/MuseTalk`, creates the MuseTalk virtual environment, downloads MuseTalk
model weights when available, and applies Apple Silicon patches to MuseTalk.

After setup, the project should have these local runtime folders:

| Path | Utility |
| --- | --- |
| `.venv/` | Python environment for the dashboard, CLI, tests, and local library. |
| `~/MuseTalk/` | External MuseTalk repository and model weights. |
| `~/MuseTalk/musetalk-env/` | Dedicated MuseTalk Python environment. |
| `data/avatars/` | Saved avatars and active `avatar.png`. |
| `data/audio/` | Generated TTS, uploaded audio conversions, and master audio. |
| `data/output/` | Final MP4 outputs and intermediate video outputs. |
| `data/captions/` | Generated SRT captions. |
| `data/presentations/` | Persistent slide presenter projects. |
| `data/voices/` | Saved MLX voice profiles and reference audio. |
| `data/temp/` | Temporary pipeline scratch files. |

## Optional Model Setup

MuseTalk is the default lip-sync engine and is installed by the main setup
script. SadTalker is optional. Install it only when you want the SadTalker 256
px or SadTalker HD engines in the dashboard.

SadTalker should live at `~/SadTalker` with its own Python 3.10 virtual
environment named `sadtalker-env`. Its model files should be downloaded using
the SadTalker project instructions. Avatar Studio reads the location from
`configs/settings.yaml`.

LatentSync installer and cleanup scripts still exist as legacy utilities, but
the current dashboard and CLI use MuseTalk and SadTalker. Do not run the
LatentSync installer unless you are intentionally experimenting outside the
active app paths.

## Environment File

The `.env` file is optional for the default local workflow. Keep it untracked.
Use it only for tokens or keys needed by optional downloads or experiments.

Common variables:

| Variable | Utility |
| --- | --- |
| `HF_TOKEN` | Allows Hugging Face downloads that require authentication. |
| `ELEVENLABS_KEY` | Legacy/optional setting in the UI; the default workflow uses local TTS. |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Lets PyTorch fall back to CPU for unsupported MPS ops. |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | Reduces MPS memory pressure on Apple Silicon. |

The launch scripts set the MPS variables automatically.

## Configuration

The main configuration file is `configs/settings.yaml`.

Use it to control:

| Section | Utility |
| --- | --- |
| `musetalk_dir` | Points to the MuseTalk checkout. |
| `sadtalker_dir` | Points to the optional SadTalker checkout. |
| `avatar_path` | Default avatar image used by the CLI pipeline. |
| `default_orientation` | Default output aspect ratio. |
| `lipsync` | Default lip-sync engine and SadTalker expression intensity. |
| `musetalk` | MuseTalk FPS, batch size, bbox shift, and float16 setting. |
| `tts` | Default Kokoro voice, speed, and language code. |
| `bark` | Bark model and default speaker preset. |

Most users should change settings through the dashboard first. Edit the YAML
directly when you want stable defaults for repeated CLI runs.

## Starting the Dashboard

The dashboard entry point is `scripts/run_dashboard.py`. It starts Gradio,
finds an available port beginning at `7860`, opens the browser by default, and
loads the UI from `app/dashboard.py`.

Useful launch choices:

| Choice | Utility |
| --- | --- |
| Default launch | Opens the app locally at `http://127.0.0.1:7860` or the next free port. |
| Custom port | Use when another Gradio app is already using the default port. |
| No browser | Use for server-style launches where you want to open the URL yourself. |
| LAN host | Bind to `0.0.0.0` only when you want another device on your network to connect. |
| Share link | Uses Gradio sharing; avoid for private assets unless you understand the exposure. |

## Dashboard Guide

| Tab | Utility |
| --- | --- |
| TTS | Generate local speech from text with Kokoro or Bark. |
| Voice Studio | Save reference voices and use MLX for text-to-voice or voice-to-voice conversion. |
| Audio Lipsync | Upload existing audio and an avatar, then generate a lip-synced video. |
| Text Lipsync | Full text-to-video path: TTS, audio prep, lip-sync, enhancement, composite, captions, encode. |
| Podcast | Create two-speaker videos from a marked script or from separate speaker audio tracks. |
| Slide Narrator | Turn a PDF plus narration JSON into a narrated slideshow video. |
| Slide Presenter | Build reusable per-slide presenter assets with a lip-synced presenter overlay. |
| Merge Videos | Concatenate selected MP4 files from `data/output`. |
| Export Presentation | Merge the newest slide presenter composites into a presentation export. |
| Settings | Stores optional settings such as legacy API keys. |

## Avatar Inputs

For the CLI pipeline, place the active avatar at `data/avatars/avatar.png`.
For the dashboard, upload an avatar from the Avatar area in the relevant tab.
The app preserves the original image framing where possible and stores saved
avatars in `data/avatars/`.

Use clear, front-facing images. MuseTalk prepares avatars into a square model
input internally, but the app keeps the source avatar available for reuse.

## Voice Inputs

Kokoro is the default text-to-speech engine. It supports English voices plus
Japanese presets configured in the app.

Bark is available for expressive multilingual speech and non-verbal cues. Its
first run downloads a larger model and uses more memory.

MLX Voice Studio is used for local voice profiles. It stores each saved voice
under `data/voices/<voice-slug>/` with the reference audio and metadata. The
slide narrator and slide presenter can use saved MLX voices or built-in Qwen
preset voices for Japanese narration.

## PDF and JSON Inputs

Slide Narrator and Slide Presenter require a PDF and a narration JSON file.
The validator accepts a flexible schema, but the canonical shape is a top-level
`slides` list where each item maps narration to a slide number.

Each slide entry can include:

| Field | Utility |
| --- | --- |
| `slide_number` | Maps the entry to a PDF page. |
| `narration` | Spoken text for that slide. |
| `duration_seconds` or `display_seconds` | Minimum time to hold the slide. |
| `pause_seconds` | Silence after that slide. |

Slide Presenter also accepts a project tag, slide selection, optional logo,
presenter avatar, output mode, and background color.

## CLI Usage

The CLI entry point is `scripts/run_pipeline.py`. It runs the single-avatar
text-to-video pipeline from the terminal.

Use the CLI when you want repeatable batch runs, fast smoke tests, or a simple
scriptable path without the dashboard. Use the dashboard when you need to
manage avatars, voices, podcasts, slide assets, or generated outputs visually.

## Validation

Use these validation layers after setup:

| Check | Utility |
| --- | --- |
| Import tests | Confirms the local package, dashboard, and helpers import correctly. |
| Unit tests | Exercises config, media helpers, narration validation, voice metadata, and assembly helpers. |
| Fake-engine integration test | Confirms the avatar pipeline orchestration works without heavy model inference. |
| Smoke test | Runs a real end-to-end output path and checks the produced MP4. |

The full smoke test can be slow because lip-sync and caption models are real
ML workloads. Use disabled enhancement and captions for a faster first check.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Kokoro import or voice errors | Confirm `.venv` is active and `espeak-ng` is installed. |
| FFmpeg errors | Confirm `ffmpeg` and `ffprobe` are available in your shell. |
| MuseTalk not found | Confirm `~/MuseTalk` and `~/MuseTalk/musetalk-env/bin/python` exist. |
| SadTalker not found | Install SadTalker only if you selected a SadTalker engine. |
| MPS memory errors | Close GPU-heavy apps, use preview mode, skip enhancement, or use shorter inputs. |
| Captions are slow | faster-whisper runs on CPU int8 on Apple Silicon. |
| Port already in use | Launch the dashboard on a different port. |
| Slide validation fails | Confirm PDF page count matches JSON entries and slide numbers are sequential. |
