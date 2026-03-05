"""Dolphin AV-TSE: lip video + multi-channel audio -> clean speech."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from smait.core.config import Config
from smait.core.events import EventBus, EventType
from smait.perception.lip_extractor import LipROI

logger = logging.getLogger(__name__)


@dataclass
class SeparationResult:
    """Output from Dolphin AV-TSE."""
    separated_audio: np.ndarray    # Clean speech, 16kHz mono float32
    separation_confidence: float   # SNR improvement estimate
    latency_ms: float
    used_multichannel: bool


class DolphinSeparator:
    """Dolphin Audio-Visual Target Speaker Extraction.

    This is the core innovation — it simultaneously:
    1. Identifies who is speaking (via lip-audio correlation)
    2. Separates their voice from the mix (source separation)

    Inputs:
    - audio: raw 4-channel PCM segment (or single-channel CAE fallback)
    - lip_frames: sequence of mouth ROI crops for the target face
    - target_face_id: which face to extract speech for

    Process:
    1. DP-LipCoder: lip video -> discrete semantic tokens via vector quantization
    2. Multi-channel audio encoder: computes IPD across 4 mic channels -> spatial embedding
    3. Global-Local Attention (GLA) separator
    4. Visual tokens guide audio decoder

    Output:
    - separated_audio: clean speech of target speaker (16kHz mono)
    - separation_confidence: SNR improvement estimate

    Fallback: If no raw 4-channel audio available, Dolphin works with
    single-channel audio + visual features.
    """

    def __init__(self, config: Config, event_bus: EventBus) -> None:
        self._config = config.separation
        self._event_bus = event_bus
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[torch.nn.Module] = None
        self._available = False

    async def init_model(self) -> None:
        """Load Dolphin AV-TSE model."""
        logger.info("Loading Dolphin AV-TSE model...")
        try:
            # Dolphin is loaded from the JusperLee/Dolphin repository
            # The model expects:
            #   - audio: (batch, channels, samples) for multi-channel
            #   - video: (batch, frames, H, W, C) lip ROI sequence
            from dolphin import DolphinModel  # type: ignore[import-not-found]

            self._model = DolphinModel.from_pretrained("JusperLee/Dolphin")
            self._model = self._model.to(self._device)
            self._model.eval()
            self._available = True
            logger.info("Dolphin AV-TSE loaded on %s", self._device)
        except ImportError:
            logger.warning(
                "Dolphin not installed. Speech separation will pass through CAE audio. "
                "Install from: github.com/JusperLee/Dolphin"
            )
        except Exception:
            logger.exception("Failed to load Dolphin model")

    @property
    def available(self) -> bool:
        return self._available

    async def separate(
        self,
        audio: np.ndarray,
        lip_frames: list[LipROI],
        channels: int = 4,
    ) -> SeparationResult:
        """Run audio-visual target speaker extraction.

        Args:
            audio: Raw PCM int16 audio (interleaved multi-channel or single-channel)
            lip_frames: Sequence of lip ROI crops for the target speaker
            channels: Number of audio channels (4 for raw, 1 for CAE fallback)

        Returns:
            SeparationResult with clean separated audio.
        """
        start = time.monotonic()

        if not self._available or self._model is None:
            # Fallback: return audio as-is (CAE-processed is already somewhat clean)
            return self._passthrough(audio, channels, start)

        try:
            return await self._run_dolphin(audio, lip_frames, channels, start)
        except Exception:
            logger.exception("Dolphin separation failed, using passthrough")
            return self._passthrough(audio, channels, start)

    async def _run_dolphin(
        self,
        audio: np.ndarray,
        lip_frames: list[LipROI],
        channels: int,
        start: float,
    ) -> SeparationResult:
        """Run the Dolphin model."""
        # Prepare audio tensor
        audio_float = audio.astype(np.float32) / 32768.0

        if channels > 1:
            # Reshape interleaved to (channels, samples)
            n_samples = len(audio_float) // channels
            audio_tensor = torch.from_numpy(
                audio_float[:n_samples * channels].reshape(n_samples, channels).T
            ).unsqueeze(0).to(self._device)
            used_multi = True
        else:
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0).unsqueeze(0).to(self._device)
            used_multi = False

        # Prepare lip video tensor
        if lip_frames:
            lip_images = np.stack([roi.image for roi in lip_frames], axis=0)
            # Normalize to [0, 1]
            lip_tensor = torch.from_numpy(lip_images).float() / 255.0
            lip_tensor = lip_tensor.unsqueeze(0).to(self._device)  # (1, T, H, W, C)
        else:
            # No lip frames — create dummy (Dolphin can work audio-only with reduced quality)
            lip_tensor = None

        # Run model
        with torch.no_grad():
            if lip_tensor is not None:
                output = self._model(audio_tensor, lip_tensor)
            else:
                output = self._model(audio_tensor)

        # Extract separated audio
        if isinstance(output, tuple):
            separated = output[0]
            confidence = output[1].item() if len(output) > 1 else 0.8
        else:
            separated = output
            confidence = 0.8

        separated_np = separated.squeeze().cpu().numpy()

        latency = (time.monotonic() - start) * 1000
        logger.info("Dolphin separation: %.1fms, confidence=%.2f, multichannel=%s",
                     latency, confidence, used_multi)

        return SeparationResult(
            separated_audio=separated_np,
            separation_confidence=confidence,
            latency_ms=latency,
            used_multichannel=used_multi,
        )

    def _passthrough(
        self,
        audio: np.ndarray,
        channels: int,
        start: float,
    ) -> SeparationResult:
        """Passthrough fallback: just convert to mono float32."""
        audio_float = audio.astype(np.float32) / 32768.0

        if channels > 1:
            # Mix down to mono by averaging channels
            n_samples = len(audio_float) // channels
            reshaped = audio_float[:n_samples * channels].reshape(n_samples, channels)
            mono = reshaped.mean(axis=1)
        else:
            mono = audio_float

        latency = (time.monotonic() - start) * 1000

        return SeparationResult(
            separated_audio=mono,
            separation_confidence=0.0,  # No separation performed
            latency_ms=latency,
            used_multichannel=False,
        )
