"""Unit tests for BinaryFrame protocol — TTS 0x05 frame encoding."""

import pytest

from smait.connection.protocol import BinaryFrame, FrameType


class TestBinaryFrameTTSAudio:
    """Tests for BinaryFrame TTS_AUDIO (0x05) frame encoding and parsing."""

    def test_tts_audio_frame_encoding(self):
        """BinaryFrame.pack(FrameType.TTS_AUDIO, payload) produces 0x05 prefix + payload."""
        pcm = b'\x01\x02\x03'
        result = BinaryFrame.pack(FrameType.TTS_AUDIO, pcm)
        assert result == bytes([0x05, 0x01, 0x02, 0x03])

    def test_tts_audio_frame_roundtrip(self):
        """from_bytes(pack(TTS_AUDIO, payload)) reconstructs frame_type and payload."""
        payload = b'\xDE\xAD\xBE\xEF' * 25  # 100 bytes of PCM data
        packed = BinaryFrame.pack(FrameType.TTS_AUDIO, payload)
        frame = BinaryFrame.from_bytes(packed)
        assert frame.frame_type == FrameType.TTS_AUDIO
        assert frame.payload == payload

    def test_binary_frame_too_short(self):
        """BinaryFrame.from_bytes raises ValueError when data is too short (< 2 bytes)."""
        with pytest.raises(ValueError):
            BinaryFrame.from_bytes(b'\x05')

    def test_tts_audio_frame_type_value(self):
        """FrameType.TTS_AUDIO has integer value 0x05."""
        assert int(FrameType.TTS_AUDIO) == 0x05

    def test_tts_audio_empty_payload(self):
        """BinaryFrame.pack with empty payload produces single 0x05 byte."""
        result = BinaryFrame.pack(FrameType.TTS_AUDIO, b'')
        assert result == bytes([0x05])

    def test_binary_frame_too_short_empty(self):
        """BinaryFrame.from_bytes raises ValueError on empty bytes."""
        with pytest.raises(ValueError):
            BinaryFrame.from_bytes(b'')
