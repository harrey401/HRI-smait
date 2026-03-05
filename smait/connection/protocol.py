"""Frame types, message schemas, serialization for Jackie ↔ PC protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class FrameType(IntEnum):
    """Binary frame type byte prefix."""
    AUDIO_CAE = 0x01       # CAE-processed single-channel audio (Jackie → PC)
    VIDEO = 0x02           # JPEG video frame (Jackie → PC)
    AUDIO_RAW = 0x03       # Raw 4-channel audio (Jackie → PC)
    CONTROL = 0x04         # Control frame (reserved)
    TTS_AUDIO = 0x05       # TTS audio from PC (PC → Jackie)


class MessageSchema:
    """JSON text frame message constructors."""

    # --- Outbound: PC → Jackie ---

    @staticmethod
    def state(state: str, robot_status: str) -> str:
        """State update for Jackie UI.

        state: "idle" | "engaged"
        robot_status: "listening" | "thinking" | "speaking"
        """
        return json.dumps({
            "type": "state",
            "state": state,
            "robot_status": robot_status,
        })

    @staticmethod
    def transcript(text: str, speaker: str) -> str:
        """Transcript for Jackie chat display.

        speaker: "user" | "robot"
        """
        return json.dumps({
            "type": "transcript",
            "text": text,
            "speaker": speaker,
        })

    @staticmethod
    def tts_text(text: str) -> str:
        """Fallback: send text for Android TTS if PC TTS fails."""
        return json.dumps({"type": "tts", "text": text})

    @staticmethod
    def tts_control(action: str) -> str:
        """Mic gating signal.

        action: "start" | "end"
        """
        return json.dumps({"type": "tts_control", "action": action})

    @staticmethod
    def response(text: str) -> str:
        """Full response text (adds to chat + triggers TTS)."""
        return json.dumps({"type": "response", "text": text})

    # --- Inbound: Jackie → PC (parsing) ---

    @staticmethod
    def parse_text_message(raw: str) -> dict[str, Any]:
        """Parse an incoming JSON text frame from Jackie."""
        return json.loads(raw)


@dataclass
class BinaryFrame:
    """A parsed binary frame from Jackie."""
    frame_type: FrameType
    payload: bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> BinaryFrame:
        """Parse a binary frame: first byte is type, rest is payload."""
        if len(data) < 2:
            raise ValueError(f"Binary frame too short: {len(data)} bytes")
        frame_type = FrameType(data[0])
        return cls(frame_type=frame_type, payload=data[1:])

    @staticmethod
    def pack(frame_type: FrameType, payload: bytes) -> bytes:
        """Pack a frame type byte + payload into a binary frame."""
        return bytes([frame_type]) + payload
