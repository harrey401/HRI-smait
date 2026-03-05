"""NVIDIA NeMo Parakeet TDT 0.6B v2 ASR wrapper."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from smait.core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    """Result from ASR transcription."""
    text: str
    confidence: float
    word_timestamps: list
    latency_ms: float
    start_time: float = 0.0
    end_time: float = 0.0


class ParakeetASR:
    """NVIDIA NeMo Parakeet TDT 0.6B v2.

    - Load model once at init (GPU, ~2GB VRAM)
    - transcribe(audio) -> TranscriptResult
    - Input: clean separated audio from Dolphin (16kHz mono float32)
    - Output: TranscriptResult(text, confidence, word_timestamps, latency_ms)

    NeMo 2.0 quirks:
    - Returns tuple (text, timestamps), not just text
    - Disable CUDA graphs on Blackwell (sm_120) via NEMO_DISABLE_CUDA_GRAPHS=1
    """

    def __init__(self, config: Config) -> None:
        self._config = config.asr
        self._model = None
        self._available = False

    async def init_model(self) -> None:
        """Load Parakeet TDT model from NeMo/HuggingFace."""
        # Disable CUDA graphs for Blackwell GPUs (sm_120)
        os.environ.setdefault("NEMO_DISABLE_CUDA_GRAPHS", "1")

        logger.info("Loading Parakeet TDT model: %s", self._config.model)
        try:
            import nemo.collections.asr as nemo_asr  # type: ignore[import-not-found]

            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self._config.model,
            )
            self._model.eval()
            self._available = True
            logger.info("Parakeet TDT loaded")
        except ImportError:
            logger.warning(
                "NeMo not installed. ASR will be unavailable. "
                "Install: pip install nemo_toolkit[asr]"
            )
        except Exception:
            logger.exception("Failed to load Parakeet model")

    @property
    def available(self) -> bool:
        return self._available

    def transcribe(self, audio: np.ndarray, start_time: float = 0.0, end_time: float = 0.0) -> Optional[TranscriptResult]:
        """Transcribe a clean audio segment.

        Args:
            audio: float32 mono audio at 16kHz
            start_time: Segment start time (for logging)
            end_time: Segment end time (for logging)

        Returns:
            TranscriptResult or None if ASR is unavailable.
        """
        if not self._available or self._model is None:
            return None

        t0 = time.monotonic()

        try:
            # NeMo 2.0 transcribe API
            # Parakeet expects: list of numpy arrays or file paths
            # Returns: tuple of (texts, timestamps) for TDT models
            result = self._model.transcribe([audio])

            # Handle NeMo 2.0 tuple return (Issue: NeMo returns (text, timestamps))
            if isinstance(result, tuple):
                texts = result[0]
                word_ts = result[1] if len(result) > 1 else []
            elif isinstance(result, list):
                texts = result
                word_ts = []
            else:
                texts = [str(result)]
                word_ts = []

            text = texts[0] if texts else ""
            timestamps = word_ts[0] if word_ts and len(word_ts) > 0 else []

            # Compute confidence from logprobs if available
            confidence = self._compute_confidence()

            latency = (time.monotonic() - t0) * 1000

            logger.info("ASR: '%s' (conf=%.2f, %.1fms)", text, confidence, latency)

            return TranscriptResult(
                text=text.strip(),
                confidence=confidence,
                word_timestamps=timestamps if isinstance(timestamps, list) else [],
                latency_ms=latency,
                start_time=start_time,
                end_time=end_time,
            )

        except Exception:
            logger.exception("ASR transcription failed")
            return None

    def _compute_confidence(self) -> float:
        """Compute confidence score from model's last inference.

        NeMo doesn't directly expose per-utterance confidence in all modes.
        We use a heuristic based on available logprobs.
        """
        try:
            if hasattr(self._model, "last_logprobs"):
                logprobs = self._model.last_logprobs
                if logprobs is not None:
                    import torch
                    avg_logprob = torch.mean(logprobs).item()
                    # Convert log probability to 0-1 confidence
                    return min(1.0, max(0.0, 1.0 + avg_logprob / 10.0))
        except Exception:
            pass
        # Default moderate confidence when we can't compute
        return 0.65
