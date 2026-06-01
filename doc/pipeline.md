# Avatar Studio Pipeline Reference

This reference describes the active app and pipeline implementation in this
repository. It reflects the current code paths under `app/`, `scripts/`, and
`src/avatarpipeline/`.

## Product Surface

Avatar Studio is a local Gradio app for generating audio, lip-synced avatar
videos, two-speaker podcast videos, narrated slide videos, and slide presenter
exports. The app is launched by `scripts/run_dashboard.py` and defined in
`app/dashboard.py`.

Main work surfaces:

| Surface | Pipeline |
| --- | --- |
| TTS | Text -> Kokoro or Bark audio. |
| Voice Studio | Reference audio -> saved MLX voice -> generated or converted speech. |
| Audio Lipsync | Uploaded audio + avatar -> lip-synced MP4. |
| Text Lipsync | Text -> speech -> lip-sync -> final MP4. |
| Podcast | Speaker script or audio tracks -> two-avatar podcast MP4. |
| Slide Narrator | PDF + JSON -> narrated slideshow MP4. |
| Slide Presenter | PDF + JSON + avatar -> reusable presenter clips and slide composites. |

## Single-Avatar Video Pipeline

The full avatar pipeline has seven stages:

```text
Text or uploaded audio
  -> voice synthesis
  -> audio normalization
  -> lip-sync generation
  -> optional face enhancement
  -> background composite and optional music
  -> optional captions
  -> final H.264/AAC encode
```

### Stage 1: Voice Synthesis

Code paths:

| Engine | Code | Utility |
| --- | --- | --- |
| Kokoro | `src/avatarpipeline/engines/tts/kokoro.py` | Fast local TTS and default app voice engine. |
| Bark | `src/avatarpipeline/engines/tts/bark.py` | Expressive multilingual TTS with speaker presets. |
| MLX | `src/avatarpipeline/engines/tts/mlx.py` | Saved reference voices, voice conversion, and Japanese Qwen presets. |

Kokoro and Bark are available in the text-to-video dashboard flow. MLX is used
by Voice Studio and by the slide narrator/presenter Japanese narration paths.

All engines produce WAV audio. The app stores generated audio in `data/audio/`.

### Stage 2: Audio Normalization

Code path: `src/avatarpipeline/core/media.py`

Audio is normalized to 16 kHz mono PCM WAV before lip-sync. Uploaded audio is
converted with FFmpeg. TTS output is converted through each engine's
`convert_to_16k` helper or the shared `normalize_to_16k_mono` helper.

### Stage 3: Lip-Sync Generation

Active engines:

| Engine key | Dashboard label | Code | Utility |
| --- | --- | --- | --- |
| `musetalk` | MuseTalk 1.5 | `src/avatarpipeline/engines/lipsync/musetalk.py` | Default lip-sync engine. Runs the external `~/MuseTalk` checkout. |
| `sadtalker` | SadTalker 256px | `src/avatarpipeline/engines/lipsync/sadtalker.py` | Optional SadTalker preset. |
| `sadtalker_hd` | SadTalker HD | `src/avatarpipeline/engines/lipsync/sadtalker.py` | Optional 512 px SadTalker preset with GFPGAN enhancer flag. |

The registry in `src/avatarpipeline/engines/__init__.py` maps engine names to
implementation classes. The CLI uses `--engine` with the same engine keys.

MuseTalk requires `~/MuseTalk/musetalk-env/bin/python` and its model files.
SadTalker requires `~/SadTalker/sadtalker-env/bin/python` and SadTalker model
files.

### Stage 4: Face Enhancement

Code path: `src/avatarpipeline/postprocess/enhancer.py`

Enhancement is optional. The backend is detected at runtime:

| Priority | Backend | Utility |
| --- | --- | --- |
| 1 | CodeFormer through ComfyUI reactor node | Best available local restorer when installed. |
| 2 | GFPGAN in the project environment | Direct Python backend when available. |
| 3 | GFPGAN through SadTalker env | Uses `tools/gfpgan_runner.py` without installing GFPGAN in `.venv`. |
| 4 | Passthrough | Copies or reassembles without restoration. |

Enhancement extracts frames to `data/temp/enhance_frames/`, processes each
frame, reassembles video, and muxes the original audio back in.

### Stage 5: Composite and Music

Code path: `src/avatarpipeline/postprocess/assembler.py`

The assembler places the avatar video into a target canvas:

| Orientation | Resolution | Use |
| --- | --- | --- |
| `9:16` | 1080x1920 | Shorts, Reels, TikTok. |
| `16:9` | 1920x1080 | YouTube, presentations. |
| `1:1` | 1080x1080 | Square social posts. |

Background choices are solid black, solid white, blurred video, or a custom
image. Optional music is mixed with FFmpeg at the selected volume.

### Stage 6: Captions

Code path: `src/avatarpipeline/postprocess/captions.py`

Captions are generated with faster-whisper and saved as SRT files in
`data/captions/`. The caption model runs on CPU int8 because faster-whisper
does not use MPS on Apple Silicon.

### Stage 7: Final Encode

Code path: `src/avatarpipeline/postprocess/assembler.py`

The final encode uses FFmpeg H.264 video, AAC audio, and `+faststart`.
Captions are burned in when enabled. Final app outputs are written under
`data/output/`.

## CLI Pipeline

The command-line pipeline is implemented in
`src/avatarpipeline/pipelines/avatar.py` and exposed by
`scripts/run_pipeline.py`.

The CLI always uses Kokoro for text-to-speech, then uses the selected lip-sync
engine, optional face enhancement, optional captions, and final assembly. It
expects the active avatar at `data/avatars/avatar.png`.

Use the CLI for repeatable single-avatar generation. Use the dashboard for
voice management, uploaded audio flows, podcasts, slide workflows, and output
management.

## Dashboard Pipeline

The dashboard implements streaming progress wrappers around the same lower
level engines. It adds:

| Capability | Utility |
| --- | --- |
| Avatar gallery | Saves uploads to `data/avatars/` and updates `avatar.png`. |
| Voice previews | Generates small Kokoro preview WAV files. |
| Audio-only generation | Saves TTS output without running lip-sync. |
| Uploaded audio path | Skips TTS and starts at audio normalization. |
| Preview mode | Skips slow enhancement and captions in text/audio lip-sync flows. |
| Merge utilities | Concatenates MP4 outputs with FFmpeg. |
| Presentation export | Merges slide presenter composites into `exports/`. |

## Podcast Pipeline

Code path: `src/avatarpipeline/pipelines/podcast.py`

The podcast pipeline supports two input modes:

| Mode | Flow |
| --- | --- |
| Script | Parse `[Speaker]: text`, generate per-segment Kokoro TTS, build per-speaker tracks, lip-sync each avatar, compose final video. |
| Upload Audio | Normalize two uploaded speaker tracks, mix a master audio track, lip-sync each avatar, compose final video. |

Composition options:

| Layout | Utility |
| --- | --- |
| Sequential Active Speaker | Cuts between speaker A and B based on the timeline. |
| Split Screen | Shows both speakers together. |
| Focus Speaker A | Speaker A is primary with speaker B as picture-in-picture. |
| Focus Speaker B | Speaker B is primary with speaker A as picture-in-picture. |

Overlay presets are FFmpeg filters such as warm color, vignette, grain, cool
tone, and soft blur. A custom transparent image overlay can also be applied.

## Slide Narrator Pipeline

Code path: `src/avatarpipeline/pipelines/narration.py`

Input is a PDF plus narration JSON. The pipeline is:

```text
Validate PDF/JSON
  -> generate per-slide narration audio
  -> build one master audio track
  -> render PDF pages to PNG
  -> encode slideshow video in one FFmpeg pass
```

Validation is handled by `src/avatarpipeline/pipelines/_validate.py`. It checks
page count, slide number sequence, slide number range, and empty narration
warnings. PDF rendering is handled by `src/avatarpipeline/pipelines/_slide_pdf.py`
with PyMuPDF.

The narration JSON accepts flexible names for slide number, narration text,
display duration, and pause duration. The normalized form is stored internally
as `slides` with `slide_number`, `narration`, `display_seconds`, and
`pause_seconds`.

## Slide Presenter Pipeline

Code path: `src/avatarpipeline/pipelines/presenter.py`

Slide Presenter is designed for repeatable presentation work. It persists
source files, generated audio, presenter lip-sync clips, slide renders,
composites, exports, and a manifest under:

`data/presentations/<project-tag>/`

Pipeline:

```text
Validate PDF/JSON
  -> copy source PDF, JSON, logo, and avatar
  -> generate or reuse per-slide narration audio
  -> generate or reuse per-slide presenter lip-sync video
  -> build selected master audio
  -> render or reuse numbered slide PNGs
  -> composite presenter over each selected slide
  -> optionally concatenate composites into one export
```

Reuse is based on hashes of the PDF, avatar, narration, selected voice,
lip-sync settings, layout version, and logo. This lets a project rerun without
regenerating unchanged heavy assets.

The current presenter layout is `consulting_left_presenter_v4`: 1920x1080,
top header, optional logo, bounded slide area, and presenter anchored on the
left.

## Data and Artifact Map

| Path | Producer | Contents |
| --- | --- | --- |
| `data/avatars/` | Dashboard and user setup | Active avatar, saved avatars, podcast avatars. |
| `data/audio/` | TTS and pipeline steps | Speech WAVs, previews, converted audio. |
| `data/captions/` | Caption generator | SRT files. |
| `data/output/` | Avatar, podcast, merge flows | Final and intermediate MP4 files. |
| `data/presentations/` | Slide presenter | Persistent project assets and exports. |
| `data/voices/` | MLX Voice Studio | Saved reference voices and metadata. |
| `data/temp/` | Pipelines | Scratch files that can be deleted between runs. |

## Configuration Map

Primary config: `configs/settings.yaml`

| Key | Used by |
| --- | --- |
| `musetalk_dir` | MuseTalk engine wrapper. |
| `sadtalker_dir` | SadTalker engine wrapper. |
| `avatar_path` | CLI default avatar path. |
| `default_fps` | Pipeline defaults. |
| `default_orientation` | Pipeline defaults. |
| `lipsync.default_engine` | Config loader and defaults. |
| `lipsync.expression_scale` | SadTalker expression default. |
| `musetalk.*` | MuseTalk FPS, batch size, bbox shift, float16 setting. |
| `tts.*` | Kokoro default voice, speed, and language code. |
| `bark.*` | Bark model and default speaker. |

The typed loader is `src/avatarpipeline/core/config.py`. Shared media helpers
are in `src/avatarpipeline/core/media.py`.

## Legacy Notes

Older documentation described LatentSync as the default engine. That is no
longer the active app path. The current dashboard and CLI expose MuseTalk and
SadTalker. `install/install_latentsync.sh` and `scripts/free_memory.sh` remain
as legacy maintenance scripts, but the current product documentation focuses on
the active implementation.
