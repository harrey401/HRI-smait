"""
SMAIT HRI System v2.0 - Engagement Detection

Determines if a person intends to interact with the robot using:
1. Proximity - Person is close enough (face size in frame)
2. Attention - Person is facing the robot (head pose)
3. Greeting - Person uses greeting words

This prevents the robot from responding to:
- People walking by while talking
- Background conversations
- TV/audio from other sources
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Set

from smait.core.events import FaceDetection, BoundingBox
from smait.core.config import get_config


# Greeting words that indicate intent to interact (STRICT - must be actual greetings)
GREETING_WORDS = {
    "hi", "hello", "hey", "excuse me", "pardon",
    "good morning", "good afternoon", "good evening",
    "yo", "sup", "whats up", "what's up"
}

# Farewell words that indicate intent to end interaction
FAREWELL_WORDS = {
    "bye", "goodbye", "good bye", "see you", "see ya",
    "later", "take care", "have a good day", "have a nice day",
    "thanks bye", "thank you bye", "that's all", "thats all",
    "i'm done", "im done", "nothing else", "no thanks"
}


@dataclass
class EngagementResult:
    """Result of engagement detection"""
    is_engaged: bool
    proximity_ok: bool
    attention_ok: bool
    greeting_detected: bool
    reason: str

    # Details
    face_area: int = 0
    head_yaw: float = 0.0  # degrees, 0 = facing camera
    head_pitch: float = 0.0


class EngagementDetector:
    """
    Detects if a person wants to interact with the robot.

    Requirements for new session:
    - Person is close enough (face area > threshold)
    - Person is facing the robot (head yaw < threshold)
    - Person uses greeting or question directed at robot

    Once engaged, only proximity + attention needed to continue.
    """

    # MediaPipe landmark indices for head pose estimation
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291

    def __init__(
        self,
        min_face_area: int = 5000,  # Minimum face bbox area in pixels (lowered — was 15000, too strict up close)
        max_head_yaw: float = 40.0,  # Max yaw angle to consider "facing" (degrees, relaxed from 35)
        max_head_pitch: float = 35.0,  # Max pitch angle (relaxed from 30)
        require_greeting_for_new: bool = False,  # No greeting required — engage naturally
    ):
        self.config = get_config()
        self.min_face_area = min_face_area
        self.max_head_yaw = max_head_yaw
        self.max_head_pitch = max_head_pitch
        self.require_greeting_for_new = require_greeting_for_new

        print(f"[ENGAGEMENT] Initialized (min_area={min_face_area}, max_yaw={max_head_yaw}°)")

    def check_engagement(
        self,
        face: FaceDetection,
        transcript: Optional[str] = None,
        is_new_session: bool = True
    ) -> EngagementResult:
        """
        Check if a person is engaged with the robot.

        Args:
            face: The detected face
            transcript: The spoken text (for greeting detection)
            is_new_session: True if this would start a new session

        Returns:
            EngagementResult with engagement status
        """
        # Check proximity (face size)
        face_area = face.bbox.area
        proximity_ok = face_area >= self.min_face_area

        # Check attention (head pose)
        head_yaw, head_pitch = self._estimate_head_pose(face)
        attention_ok = abs(head_yaw) <= self.max_head_yaw and abs(head_pitch) <= self.max_head_pitch

        # Check greeting (only for new sessions)
        greeting_detected = False
        if transcript and is_new_session:
            greeting_detected = self._detect_greeting(transcript)

        # Determine engagement
        if is_new_session:
            # New session: need proximity + attention + greeting
            if self.require_greeting_for_new:
                is_engaged = proximity_ok and attention_ok and greeting_detected
            else:
                is_engaged = proximity_ok and attention_ok
        else:
            # Existing session: just need proximity + attention
            is_engaged = proximity_ok and attention_ok

        # Build reason string
        reasons = []
        if not proximity_ok:
            reasons.append(f"too_far(area={face_area})")
        if not attention_ok:
            reasons.append(f"not_facing(yaw={head_yaw:.0f}°)")
        if is_new_session and self.require_greeting_for_new and not greeting_detected:
            reasons.append("no_greeting")

        reason = ", ".join(reasons) if reasons else "engaged"

        return EngagementResult(
            is_engaged=is_engaged,
            proximity_ok=proximity_ok,
            attention_ok=attention_ok,
            greeting_detected=greeting_detected,
            reason=reason,
            face_area=face_area,
            head_yaw=head_yaw,
            head_pitch=head_pitch
        )

    def _estimate_head_pose(self, face: FaceDetection) -> tuple:
        """
        Estimate head yaw and pitch from landmarks.

        Uses the relative positions of eyes, nose, and mouth to estimate
        head orientation. Returns (yaw, pitch) in degrees.

        Yaw: negative = looking left, positive = looking right
        Pitch: negative = looking down, positive = looking up
        """
        if face.landmarks is None or len(face.landmarks) < 300:
            # No landmarks, assume facing forward
            return 0.0, 0.0

        try:
            landmarks = face.landmarks

            # Get key points
            nose = landmarks[self.NOSE_TIP][:2]
            left_eye = landmarks[self.LEFT_EYE_OUTER][:2]
            right_eye = landmarks[self.RIGHT_EYE_OUTER][:2]
            chin = landmarks[self.CHIN][:2]
            left_mouth = landmarks[self.LEFT_MOUTH][:2]
            right_mouth = landmarks[self.RIGHT_MOUTH][:2]

            # Eye center
            eye_center = (left_eye + right_eye) / 2

            # Mouth center
            mouth_center = (left_mouth + right_mouth) / 2

            # Face center (between eyes and mouth)
            face_center = (eye_center + mouth_center) / 2

            # Estimate YAW from nose position relative to face center
            # If nose is left of center, person is looking right (positive yaw)
            eye_width = np.linalg.norm(right_eye - left_eye)
            nose_offset = nose[0] - face_center[0]
            yaw = (nose_offset / (eye_width / 2)) * 45  # Scale to ~45 degrees max

            # Estimate PITCH from nose-to-eye-center vs nose-to-chin ratio
            nose_to_eyes = np.linalg.norm(eye_center - nose)
            nose_to_chin = np.linalg.norm(chin - nose)

            # Normal ratio is about 0.7-0.8 (nose closer to eyes)
            ratio = nose_to_eyes / (nose_to_chin + 0.001)
            pitch = (ratio - 0.75) * 60  # Scale deviation to degrees

            return float(np.clip(yaw, -90, 90)), float(np.clip(pitch, -90, 90))

        except Exception as e:
            if self.config.debug:
                print(f"[ENGAGEMENT] Head pose error: {e}")
            return 0.0, 0.0

    def _detect_greeting(self, transcript: str) -> bool:
        """
        Detect if transcript contains a greeting to start interaction.

        STRICT: Only actual greeting words trigger session start.
        Questions like "what time is it" do NOT count as greetings.
        """
        text = transcript.lower().strip()

        # Check for greeting words only
        for greeting in GREETING_WORDS:
            if greeting in text:
                return True

        return False

    def detect_farewell(self, transcript: str) -> bool:
        """
        Detect if transcript contains a farewell to end interaction.
        """
        text = transcript.lower().strip()

        # Check for farewell words
        for farewell in FAREWELL_WORDS:
            if farewell in text:
                return True

        return False

    def check_proximity(self, face: FaceDetection) -> bool:
        """Quick check if face is close enough"""
        return face.bbox.area >= self.min_face_area

    def check_attention(self, face: FaceDetection) -> bool:
        """Quick check if person is facing robot"""
        yaw, pitch = self._estimate_head_pose(face)
        return abs(yaw) <= self.max_head_yaw and abs(pitch) <= self.max_head_pitch
