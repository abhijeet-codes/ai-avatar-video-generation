"""
avatarpipeline.engines.tts.bark — Text-to-speech via suno/bark (HuggingFace).

Model variants
--------------
suno/bark        Full model  ~5 GB download  ~8 GB RAM  ← recommended for M4 Pro 24 GB
suno/bark-small  Small model ~2 GB download  ~4 GB RAM  lighter alternative

On this system (Apple M4 Pro, 24 GB unified memory) the full suno/bark model
is loaded onto the MPS device for hardware-accelerated inference.

Speaker presets follow the pattern "v2/<lang>_speaker_<n>", e.g.:
  "v2/en_speaker_6"   English, speaker 6
  "v2/ja_speaker_3"   Japanese, speaker 3
Bark also supports inline non-verbal cues such as [laughs], [sighs], etc.

HuggingFace credentials are read from HF_TOKEN in the project .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

import numpy as np
import soundfile as sf
from loguru import logger

from avatarpipeline import AUDIO_DIR
from avatarpipeline.core.media import normalize_to_16k_mono


class BarkVoiceGenerator:
    """Text-to-speech using suno/bark via HuggingFace Transformers.

    Recommended model for Apple M4 Pro (24 GB): ``suno/bark`` (full quality).
    Uses the MPS device for Apple Silicon acceleration.

    Usage::

        tts = BarkVoiceGenerator()                    # full suno/bark model
        tts = BarkVoiceGenerator("suno/bark-small")   # lighter alternative
        path = tts.generate("Hello world", voice="en_speaker_6")
    """

    # Recommended model for this system (M4 Pro 24 GB)
    MODEL_ID: ClassVar[str] = "suno/bark"
    SAMPLE_RATE: ClassVar[int] = 24_000  # Bark audio codec native sample rate

    # Curated preset map: friendly key → HuggingFace voice preset string
    VOICES: ClassVar[dict[str, str]] = {
        # ── English ────────────────────────────────────────────────────
        "en_speaker_0": "v2/en_speaker_0",
        "en_speaker_1": "v2/en_speaker_1",
        "en_speaker_2": "v2/en_speaker_2",
        "en_speaker_3": "v2/en_speaker_3",
        "en_speaker_4": "v2/en_speaker_4",
        "en_speaker_5": "v2/en_speaker_5",
        "en_speaker_6": "v2/en_speaker_6",  # clear, neutral male — good default
        "en_speaker_7": "v2/en_speaker_7",
        "en_speaker_8": "v2/en_speaker_8",
        "en_speaker_9": "v2/en_speaker_9",
        # ── Japanese ───────────────────────────────────────────────────
        "ja_speaker_0": "v2/ja_speaker_0",
        "ja_speaker_1": "v2/ja_speaker_1",
        "ja_speaker_2": "v2/ja_speaker_2",
        "ja_speaker_3": "v2/ja_speaker_3",
        "ja_speaker_4": "v2/ja_speaker_4",
        "ja_speaker_5": "v2/ja_speaker_5",
        "ja_speaker_6": "v2/ja_speaker_6",
        "ja_speaker_7": "v2/ja_speaker_7",
        # ── Chinese ────────────────────────────────────────────────────
        "zh_speaker_0": "v2/zh_speaker_0",
        "zh_speaker_1": "v2/zh_speaker_1",
        "zh_speaker_2": "v2/zh_speaker_2",
        "zh_speaker_3": "v2/zh_speaker_3",
        "zh_speaker_4": "v2/zh_speaker_4",
        "zh_speaker_5": "v2/zh_speaker_5",
        "zh_speaker_6": "v2/zh_speaker_6",
        "zh_speaker_7": "v2/zh_speaker_7",
        "zh_speaker_8": "v2/zh_speaker_8",
        "zh_speaker_9": "v2/zh_speaker_9",
        # ── German ─────────────────────────────────────────────────────
        "de_speaker_0": "v2/de_speaker_0",
        "de_speaker_1": "v2/de_speaker_1",
        "de_speaker_2": "v2/de_speaker_2",
        "de_speaker_3": "v2/de_speaker_3",
        # ── French ─────────────────────────────────────────────────────
        "fr_speaker_0": "v2/fr_speaker_0",
        "fr_speaker_1": "v2/fr_speaker_1",
        "fr_speaker_2": "v2/fr_speaker_2",
        "fr_speaker_3": "v2/fr_speaker_3",
        "fr_speaker_4": "v2/fr_speaker_4",
        "fr_speaker_5": "v2/fr_speaker_5",
        # ── Spanish ────────────────────────────────────────────────────
        "es_speaker_0": "v2/es_speaker_0",
        "es_speaker_1": "v2/es_speaker_1",
        "es_speaker_2": "v2/es_speaker_2",
        "es_speaker_3": "v2/es_speaker_3",
        "es_speaker_4": "v2/es_speaker_4",
        "es_speaker_5": "v2/es_speaker_5",
        # ── Hindi ──────────────────────────────────────────────────────
        "hi_speaker_0": "v2/hi_speaker_0",
        "hi_speaker_1": "v2/hi_speaker_1",
        "hi_speaker_2": "v2/hi_speaker_2",
        "hi_speaker_3": "v2/hi_speaker_3",
        # ── Korean ─────────────────────────────────────────────────────
        "ko_speaker_0": "v2/ko_speaker_0",
        "ko_speaker_1": "v2/ko_speaker_1",
        "ko_speaker_2": "v2/ko_speaker_2",
        "ko_speaker_3": "v2/ko_speaker_3",
        # ── Portuguese ─────────────────────────────────────────────────
        "pt_speaker_0": "v2/pt_speaker_0",
        "pt_speaker_1": "v2/pt_speaker_1",
        "pt_speaker_2": "v2/pt_speaker_2",
        "pt_speaker_3": "v2/pt_speaker_3",
        # ── Russian ────────────────────────────────────────────────────
        "ru_speaker_0": "v2/ru_speaker_0",
        "ru_speaker_1": "v2/ru_speaker_1",
        "ru_speaker_2": "v2/ru_speaker_2",
        "ru_speaker_3": "v2/ru_speaker_3",
    }

    DEFAULT_VOICE: ClassVar[str] = "en_speaker_6"

    def __init__(self, model_id: str | None = None) -> None:
        """Initialise the Bark engine (model loaded lazily on first generate call).

        Args:
            model_id: HuggingFace model ID. Defaults to ``suno/bark``.
        """
        self.model_id = model_id or self.MODEL_ID
        self._processor = None
        self._model = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hf_token() -> str | None:
        """Return HF_TOKEN from environment, loading the project .env first."""
        try:
            from dotenv import load_dotenv
            from avatarpipeline import ROOT
            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            pass
        return os.environ.get("HF_TOKEN")

    def _get_device(self) -> str:
        if self._device is None:
            import torch
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"
            logger.info(f"Bark device: {self._device}")
        return self._device

    def _load_model(self) -> None:
        """Lazy-load the BarkProcessor and BarkModel on the first call."""
        if self._model is not None:
            return

        try:
            from transformers import BarkModel, BarkProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed. Run:\n"
                "  uv pip install transformers>=4.36.0"
            ) from exc

        token = self._load_hf_token()
        device = self._get_device()

        logger.info(f"Loading BarkProcessor from {self.model_id}...")
        self._processor = BarkProcessor.from_pretrained(
            self.model_id,
            token=token,
        )

        logger.info(f"Loading BarkModel from {self.model_id} onto {device}...")
        import torch
        self._model = BarkModel.from_pretrained(
            self.model_id,
            token=token,
            torch_dtype=torch.float32,  # MPS requires float32
        ).to(device)
        self._model.eval()

        logger.info("Bark model ready.")

    # ------------------------------------------------------------------
    # Public API  (satisfies TtsEngine protocol)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Chunking helpers (Bark caps output at ~13 s per inference call)
    # ------------------------------------------------------------------

    _MAX_CHUNK_CHARS: ClassVar[int] = 220  # ~13 s at average speaking rate

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split *text* into sentence-level chunks suitable for Bark.

        Splits on ``.  !  ?`` followed by whitespace or end-of-string.
        Chunks longer than ``_MAX_CHUNK_CHARS`` are further split at commas
        or semicolons so no single chunk exceeds the limit.
        """
        import re

        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks: list[str] = []
        for sentence in raw:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= BarkVoiceGenerator._MAX_CHUNK_CHARS:
                chunks.append(sentence)
            else:
                # Split long sentence at commas / semicolons
                sub_parts = re.split(r"(?<=[,;])\s+", sentence)
                current = ""
                for part in sub_parts:
                    if len(current) + len(part) + 1 <= BarkVoiceGenerator._MAX_CHUNK_CHARS:
                        current = (current + " " + part).strip()
                    else:
                        if current:
                            chunks.append(current)
                        # Hard-split any part that is still too long
                        while len(part) > BarkVoiceGenerator._MAX_CHUNK_CHARS:
                            chunks.append(part[: BarkVoiceGenerator._MAX_CHUNK_CHARS])
                            part = part[BarkVoiceGenerator._MAX_CHUNK_CHARS :]
                        current = part
                if current:
                    chunks.append(current)
        return chunks or [text]

    def _generate_chunk(self, text: str, voice_preset: str) -> np.ndarray:
        """Run a single Bark inference call and return a 1-D float32 array."""
        import torch

        inputs = self._processor(
            text,
            voice_preset=voice_preset,
            return_tensors="pt",
        ).to(self._get_device())

        with torch.inference_mode():
            audio_array = self._model.generate(**inputs)

        return audio_array.cpu().numpy().squeeze()

    def generate(
        self,
        text: str,
        voice: str | None = None,
        out_path: str | None = None,
    ) -> str:
        """Generate speech from *text* and save as a 16 kHz mono WAV.

        Long text is automatically split into sentence-level chunks (Bark
        caps output at ~13 s per inference call) and the chunks are
        concatenated into a single WAV.

        Bark supports inline non-verbal cues in square brackets, e.g.:
            "Hello! [laughs] How are you today?"

        Args:
            text:     The script to synthesise (any length).
            voice:    Voice key from VOICES dict or a raw preset like
                      ``"v2/en_speaker_6"``.  Defaults to ``en_speaker_6``.
            out_path: Destination WAV path. Defaults to data/audio/output.wav.

        Returns:
            Absolute path to the 16 kHz mono WAV file.
        """
        self._load_model()

        voice_key = voice or self.DEFAULT_VOICE
        voice_preset = self.VOICES.get(voice_key, voice_key)

        dest = Path(out_path) if out_path else AUDIO_DIR / "output.wav"
        if not dest.is_absolute():
            from avatarpipeline import ROOT
            dest = ROOT / dest
        dest.parent.mkdir(parents=True, exist_ok=True)

        chunks = self._split_sentences(text)
        logger.info(
            f"Bark TTS: {len(text)} chars split into {len(chunks)} chunk(s), "
            f"preset={voice_preset}"
        )

        # Small silence (0.2 s) between sentences for natural pacing
        silence = np.zeros(int(self.SAMPLE_RATE * 0.20), dtype=np.float32)

        audio_parts: list[np.ndarray] = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"  Chunk {i}/{len(chunks)}: {len(chunk)} chars")
            part = self._generate_chunk(chunk, voice_preset)
            audio_parts.append(part.astype(np.float32))
            if i < len(chunks):
                audio_parts.append(silence)

        audio_np = np.concatenate(audio_parts) if audio_parts else np.zeros(1, dtype=np.float32)

        raw_path = dest.with_suffix(".24k.wav")
        sf.write(str(raw_path), audio_np, self.SAMPLE_RATE)

        final = self.convert_to_16k(str(raw_path), str(dest))
        raw_path.unlink(missing_ok=True)
        return final

    def convert_to_16k(self, input_path: str, output_wav: str) -> str:
        """Resample *input_path* to 16 kHz mono WAV.

        Returns:
            Absolute path to the converted file.
        """
        normalize_to_16k_mono(input_path, output_wav)
        logger.info(f"16 kHz WAV saved: {output_wav}")
        return os.path.abspath(output_wav)

    def list_voices(self) -> list[str]:
        """Return all available voice preset keys."""
        return list(self.VOICES.keys())

    def estimate_cost(self, text: str) -> float:
        """Bark is fully local — cost is always $0."""
        return 0.0

    # ------------------------------------------------------------------
    # One-shot download helper
    # ------------------------------------------------------------------

    @classmethod
    def download_model(cls, model_id: str | None = None) -> str:
        """Pre-download model weights to the local HuggingFace cache.

        Reads HF_TOKEN from the project .env file automatically.

        Args:
            model_id: HuggingFace repo ID. Defaults to ``suno/bark``.

        Returns:
            Local snapshot directory path.
        """
        from huggingface_hub import snapshot_download

        try:
            from dotenv import load_dotenv
            from avatarpipeline import ROOT
            load_dotenv(ROOT / ".env", override=False)
        except ImportError:
            pass

        token = os.environ.get("HF_TOKEN")
        target = model_id or cls.MODEL_ID
        logger.info(f"Downloading {target} from HuggingFace…")
        path = snapshot_download(repo_id=target, token=token)
        logger.info(f"Model cached at: {path}")
        return path
