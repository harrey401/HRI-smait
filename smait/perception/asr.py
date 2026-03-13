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

            # Disable CUDA graphs for Blackwell (sm_120) compatibility —
            # NeMo's TDT decoder uses cu_call() which returns different tuple
            # sizes on Blackwell's CUDA driver, causing unpacking errors.
            try:
                from omegaconf import OmegaConf
                decoding_cfg = OmegaConf.create({
                    "strategy": "greedy",
                    "greedy": {
                        "max_symbols": 10,
                        "preserve_alignments": False,
                        "confidence_cfg": {"preserve_frame_confidence": False},
                    },
                })
                self._model.change_decoding_strategy(decoding_cfg)
                logger.info("Parakeet TDT loaded (greedy decoding, CUDA graphs disabled)")
            except Exception:
                logger.warning("Could not change decoding strategy, using default")

            self._available = True
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

            raw = texts[0] if texts else ""
            # NeMo may return a Hypothesis object instead of a plain string
            text = raw.text if hasattr(raw, "text") else str(raw)
            timestamps = word_ts[0] if word_ts and len(word_ts) > 0 else []

            # Extract confidence from NeMo hypotheses (with fallback to 0.65)
            confidence = self._extract_confidence(result)

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

    def _extract_confidence(self, model_output) -> float:
        """Extract utterance confidence from NeMo model output.

        Attempts to read ``word_confidence`` from a NeMo ``Hypothesis`` object
        (returned when ``return_hypotheses=True``).  Falls back to ``score``
        if available, and ultimately returns 0.65 when neither is present.

        Args:
            model_output: Raw output from ``self._model.transcribe()``.
                May be a tuple, list, or a single value.

        Returns:
            Confidence score in [0, 1].
        """
        try:
            if isinstance(model_output, tuple):
                hyps = model_output[0]
            else:
                hyps = model_output

            # Unwrap list/sequence to get the first hypothesis object
            if isinstance(hyps, (list, tuple)):
                if not hyps:
                    return 0.65
                first = hyps[0]
                # If it's a plain string, no hypothesis object available
                if isinstance(first, str):
                    return 0.65
                hyp = first
            else:
                # model_output is itself a hypothesis object
                hyp = hyps

            word_confidence = getattr(hyp, "word_confidence", None)
            if word_confidence is not None and len(word_confidence) > 0:
                return float(min(word_confidence))
            score = getattr(hyp, "score", None)
            if score is not None:
                return float(score)
        except Exception:
            pass

        return 0.65

    def _compute_confidence(self) -> float:
        """Deprecated: use _extract_confidence() instead.

        Kept for backward compatibility. Returns 0.65 fallback.
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
