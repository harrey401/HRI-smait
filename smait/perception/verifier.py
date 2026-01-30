"""
SMAIT HRI System v2.0 - Speaker Verifier
Combines face tracking, ASD, and session management.
Determines if detected speech came from the target user.
"""

import time
import threading
from typing import Optional, List, Tuple, Dict
from enum import Enum, auto
import numpy as np
import cv2

from smait.core.config import get_config
from smait.core.events import (
    FaceDetection, ActiveSpeakerResult, TranscriptResult,
    VerifyResult, VerifyOutput, SessionState
)
from smait.perception.face_tracker import FaceTracker
from smait.perception.asd import ActiveSpeakerDetector
from smait.perception.engagement import EngagementDetector


class SpeakerVerifier:
    """
    Verifies that detected speech came from the target user.
    
    Flow:
    1. Track all visible faces
    2. Detect which face is speaking (ASD)
    3. Lock onto primary user when speech + face match
    4. Accept/reject speech based on verification
    
    IMPORTANT: Uses temporal buffering to handle audio-visual delay.
    Speech verification checks if a face WAS speaking during the audio,
    not just if they're speaking NOW.
    """
    
    def __init__(self):
        self.config = get_config()
        
        # Components
        self.face_tracker = FaceTracker(max_faces=self.config.vision.max_faces)
        self.asd = ActiveSpeakerDetector()
        self.engagement = EngagementDetector()
        
        # Session state
        self.state = SessionState.IDLE
        self.target_user_id: Optional[int] = None
        self.session_start: Optional[float] = None
        self.last_activity: Optional[float] = None
        self.turn_count: int = 0
        
        # Latest frame results
        self.current_faces: List[FaceDetection] = []
        self.current_asd_results: List[ActiveSpeakerResult] = []
        self.current_frame: Optional[np.ndarray] = None
        
        # === TEMPORAL BUFFER for audio-visual sync ===
        # Stores recent ASD results to check historical speaking activity
        self._asd_history: List[Tuple[float, List[ActiveSpeakerResult]]] = []
        self._asd_history_duration = 5.0  # Keep 5 seconds of history
        self._speech_segment_start: Optional[float] = None  # Track when current speech started

        # Farewell handling - end session after robot responds to goodbye
        self._pending_session_end: bool = False

        self.lock = threading.Lock()
        
        print(f"[VERIFIER] Initialized (ASD backend: {self.asd.backend_name})")
    
    def mark_speech_start(self):
        """Call this when VAD detects speech start"""
        self._speech_segment_start = time.time()
        if self.config.debug:
            print(f"[VERIFIER] Speech segment started at {self._speech_segment_start:.2f}")
    
    def mark_speech_end(self):
        """Call this when VAD detects speech end"""
        if self.config.debug and self._speech_segment_start:
            duration = time.time() - self._speech_segment_start
            print(f"[VERIFIER] Speech segment ended (duration: {duration:.2f}s)")
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a video frame.
        
        Returns dict with:
        - faces: List of detected faces
        - asd_results: List of ASD results
        - primary_speaker_id: ID of the speaking face (if any)
        - frame: Annotated frame (if show_video enabled)
        - status: Current session status string
        """
        timestamp = time.time()
        
        # Track faces
        faces, annotated_frame = self.face_tracker.process_frame(frame)
        
        # Get recent audio for ASD (if available)
        # In the full system, this would come from the audio pipeline
        audio = None
        
        # Run ASD
        asd_results, primary_speaker_id = self.asd.process_faces(faces, audio)
        
        with self.lock:
            self.current_faces = faces
            self.current_asd_results = asd_results
            self.current_frame = annotated_frame
            
            # === Store ASD results in history buffer ===
            self._asd_history.append((timestamp, asd_results.copy()))
            
            # Prune old history
            cutoff = timestamp - self._asd_history_duration
            self._asd_history = [(t, r) for t, r in self._asd_history if t > cutoff]
        
        # Update session state based on faces
        self._update_session_state(faces, asd_results, primary_speaker_id, timestamp)
        
        # Draw additional info on frame
        if annotated_frame is not None:
            self._draw_session_info(annotated_frame, asd_results, primary_speaker_id)
        
        return {
            'faces': faces,
            'asd_results': asd_results,
            'primary_speaker_id': primary_speaker_id,
            'frame': annotated_frame,
            'status': self._get_status_string(),
            'state': self.state,
            'target_user_id': self.target_user_id
        }
    
    def verify_speech(self, transcript: TranscriptResult) -> VerifyOutput:
        """
        Verify that a transcript came from the target user.
        
        Uses TEMPORAL BUFFERING to check if a face was speaking
        during the audio segment, not just at the current moment.
        
        Args:
            transcript: The transcription result to verify
        
        Returns:
            VerifyOutput with accept/reject decision
        """
        with self.lock:
            faces = self.current_faces
            asd_history = self._asd_history.copy()
        
        # No faces visible
        if not faces:
            return VerifyOutput(
                result=VerifyResult.NO_FACE,
                text="",
                confidence=0.0,
                reason="no_face_visible"
            )
        
        # === TEMPORAL CHECK: Who was speaking during the audio? ===
        # Determine the time window to check
        now = time.time()
        
        # Use speech segment timestamps if available, otherwise estimate
        if self._speech_segment_start is not None:
            window_start = self._speech_segment_start
            window_end = now
        else:
            # Estimate: assume speech was 0.5-3 seconds ago
            window_start = now - 3.0
            window_end = now - 0.3  # Small buffer for processing delay
        
        # Find who was speaking during the audio window
        speaking_scores = self._get_speaking_scores_in_window(asd_history, window_start, window_end)
        
        # Only print debug for actual verification attempts (not spam)
        # Debug output moved to end of function for cleaner flow
        
        # Reset speech segment tracking
        self._speech_segment_start = None
        
        if not speaking_scores:
            # No one was detected speaking during the audio window
            # This means ASD didn't see any mouth movement during speech
            
            # REJECT - trust the ASD detection
            # The audio might be from someone off-camera, background noise, or TV
            return VerifyOutput(
                result=VerifyResult.REJECT,
                text=transcript.text,
                confidence=0.0,
                reason="no_visual_speech_detected"
            )
        
        # Get the face with highest speaking score during the window
        best_speaker_id = max(speaking_scores, key=speaking_scores.get)
        best_score = speaking_scores[best_speaker_id]

        # Get the face object for engagement check
        best_face = next((f for f in faces if f.track_id == best_speaker_id), None)

        # Session management
        if self.state == SessionState.IDLE or self.state == SessionState.DETECTING:
            # NEW SESSION: Check engagement (proximity + attention + greeting)
            if best_face:
                engagement = self.engagement.check_engagement(
                    face=best_face,
                    transcript=transcript.text,
                    is_new_session=True
                )

                if not engagement.is_engaged:
                    # Not engaged - reject
                    if self.config.debug:
                        print(f"[ENGAGEMENT] Not engaged: {engagement.reason}")
                    return VerifyOutput(
                        result=VerifyResult.REJECT,
                        text=transcript.text,
                        confidence=best_score,
                        reason=f"not_engaged:{engagement.reason}",
                        face_id=best_speaker_id,
                        asd_score=best_score
                    )

            # Engaged - start new session
            self._start_session(best_speaker_id)

            return VerifyOutput(
                result=VerifyResult.ACCEPT,
                text=transcript.text,
                confidence=best_score,
                reason="new_session_started",
                face_id=best_speaker_id,
                asd_score=best_score
            )
        
        # Active session - verify it's the same user
        if self.target_user_id is not None:
            if best_speaker_id == self.target_user_id:
                # Same user - check if they're saying goodbye
                if self.engagement.detect_farewell(transcript.text):
                    if self.config.debug:
                        print(f"[SESSION] Farewell detected: \"{transcript.text}\"")

                    # Accept the farewell message, then end session
                    result = VerifyOutput(
                        result=VerifyResult.ACCEPT,
                        text=transcript.text,
                        confidence=best_score,
                        reason="farewell_ending_session",
                        face_id=best_speaker_id,
                        asd_score=best_score
                    )

                    # Mark for session end after response
                    self._pending_session_end = True
                    return result

                # Same user, accept
                self.last_activity = time.time()
                self.turn_count += 1

                return VerifyOutput(
                    result=VerifyResult.ACCEPT,
                    text=transcript.text,
                    confidence=best_score,
                    reason="verified_user",
                    face_id=best_speaker_id,
                    asd_score=best_score
                )
            else:
                # Different person speaking
                # Check if target user also spoke (might be overlapping speech)
                if self.target_user_id in speaking_scores:
                    target_score = speaking_scores[self.target_user_id]
                    if target_score > 0.3:  # Target also spoke
                        self.last_activity = time.time()
                        self.turn_count += 1
                        
                        return VerifyOutput(
                            result=VerifyResult.ACCEPT,
                            text=transcript.text,
                            confidence=target_score,
                            reason="target_also_speaking",
                            face_id=self.target_user_id,
                            asd_score=target_score
                        )
                
                # Reject - different speaker
                return VerifyOutput(
                    result=VerifyResult.REJECT,
                    text=transcript.text,
                    confidence=best_score,
                    reason="different_speaker",
                    face_id=best_speaker_id,
                    asd_score=best_score
                )
        
        # Fallback - accept with the detected speaker
        return VerifyOutput(
            result=VerifyResult.ACCEPT,
            text=transcript.text,
            confidence=best_score,
            reason="fallback_accept",
            face_id=best_speaker_id,
            asd_score=best_score
        )
    
    def _get_speaking_scores_in_window(
        self,
        asd_history: List[Tuple[float, List[ActiveSpeakerResult]]],
        window_start: float,
        window_end: float
    ) -> Dict[int, float]:
        """
        Get speaking scores for each face during a time window.
        
        Returns dict of {track_id: average_speaking_probability}
        """
        from collections import defaultdict
        
        # Collect all speaking probabilities per face in the window
        face_probs = defaultdict(list)
        
        for timestamp, results in asd_history:
            if window_start <= timestamp <= window_end:
                for result in results:
                    if result.is_speaking or result.probability > 0.3:
                        face_probs[result.track_id].append(result.probability)
        
        # Calculate average speaking score for each face
        speaking_scores = {}
        for track_id, probs in face_probs.items():
            if probs:
                # Use a weighted score: higher if spoke more often AND with higher confidence
                avg_prob = sum(probs) / len(probs)
                speaking_ratio = len(probs) / max(len(asd_history), 1)
                speaking_scores[track_id] = avg_prob * (0.5 + 0.5 * speaking_ratio)
        
        return speaking_scores
    
    def _start_session(self, user_id: int):
        """Start a new interaction session"""
        self.state = SessionState.ENGAGED
        self.target_user_id = user_id
        self.session_start = time.time()
        self.last_activity = time.time()
        self.turn_count = 0
        
        if self.config.debug:
            print(f"[SESSION] Started with user {user_id}")
    
    def end_session(self):
        """End the current session"""
        old_user = self.target_user_id
        
        self.state = SessionState.IDLE
        self.target_user_id = None
        self.session_start = None
        self.last_activity = None
        self.turn_count = 0
        
        if self.config.debug:
            print(f"[SESSION] Ended (was user {old_user})")
    
    def check_timeout(self) -> bool:
        """Check if session should timeout"""
        if self.state != SessionState.ENGAGED:
            return False

        if self.last_activity is None:
            return False

        elapsed = time.time() - self.last_activity
        if elapsed > self.config.session.timeout_seconds:
            if self.config.debug:
                print(f"[SESSION] Timeout after {elapsed:.0f}s")
            self.end_session()
            return True

        return False

    def check_pending_session_end(self) -> bool:
        """
        Check if session should end after farewell was processed.
        Call this after robot finishes responding to a farewell.
        Returns True if session was ended.
        """
        if self._pending_session_end:
            self._pending_session_end = False
            if self.config.debug:
                print(f"[SESSION] Ending after farewell")
            self.end_session()
            return True
        return False
    
    def _update_session_state(
        self,
        faces: List[FaceDetection],
        asd_results: List[ActiveSpeakerResult],
        primary_speaker_id: Optional[int],
        timestamp: float
    ):
        """Update session state based on current observations"""
        
        visible_ids = {f.track_id for f in faces}
        
        # Check if target user is still visible
        if self.target_user_id is not None:
            if self.target_user_id not in visible_ids:
                # User not visible
                if self.state == SessionState.ENGAGED:
                    self.state = SessionState.PAUSED
                    self._pause_start = timestamp
                    if self.config.debug:
                        print(f"[SESSION] User {self.target_user_id} not visible, pausing")
                
                # Check if paused too long - end session
                elif self.state == SessionState.PAUSED:
                    pause_duration = timestamp - getattr(self, '_pause_start', timestamp)
                    max_pause = self.config.session.face_lost_grace_seconds
                    
                    if pause_duration > max_pause:
                        if self.config.debug:
                            print(f"[SESSION] User {self.target_user_id} lost for {pause_duration:.1f}s, ending session")
                        self.end_session()
            else:
                # User visible again
                if self.state == SessionState.PAUSED:
                    self.state = SessionState.ENGAGED
                    if self.config.debug:
                        print(f"[SESSION] User {self.target_user_id} visible again")
        
        # Start detecting if no session and faces visible
        if self.state == SessionState.IDLE and faces:
            self.state = SessionState.DETECTING
            self.state = SessionState.DETECTING
    
    def _get_status_string(self) -> str:
        """Get human-readable status string"""
        if self.state == SessionState.IDLE:
            return "Waiting for user"
        elif self.state == SessionState.DETECTING:
            return "Detecting speaker..."
        elif self.state == SessionState.ENGAGED:
            return f"Engaged (User {self.target_user_id}, {self.turn_count} turns)"
        elif self.state == SessionState.PAUSED:
            return f"Paused (User {self.target_user_id} not visible)"
        else:
            return "Unknown"
    
    def _draw_session_info(
        self,
        frame: np.ndarray,
        asd_results: List[ActiveSpeakerResult],
        primary_speaker_id: Optional[int]
    ):
        """Draw session and ASD info on frame"""
        h, w = frame.shape[:2]
        
        # Draw ASD results on each face
        for result in asd_results:
            # Find the corresponding face
            face = next((f for f in self.current_faces if f.track_id == result.track_id), None)
            if face is None:
                continue
            
            bbox = face.bbox
            
            # Check head motion (for debugging)
            head_motion = 0.0
            if hasattr(self.asd.backend, '_calculate_head_motion'):
                head_motion = self.asd.backend._calculate_head_motion(face.track_id, face.timestamp)
            
            # Speaking indicator
            if result.is_speaking:
                color = (0, 255, 0)  # Green
                status = f"SPEAKING (vel={result.lip_movement:.2f})"
            elif head_motion > 50:  # Head moving
                color = (0, 165, 255)  # Orange - head moving
                status = f"HEAD MOVING ({head_motion:.0f}px/s)"
            else:
                color = (128, 128, 128)  # Gray
                status = f"Silent (vel={result.lip_movement:.2f})"
            
            # Highlight target user
            if face.track_id == self.target_user_id:
                cv2.rectangle(frame, (bbox.x1-3, bbox.y1-3), (bbox.x2+3, bbox.y2+3), (255, 0, 0), 3)
            
            # Speaking status below face
            cv2.putText(
                frame,
                status,
                (bbox.x1, bbox.y2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Show head motion speed (small text)
            if head_motion > 0:
                cv2.putText(
                    frame,
                    f"Motion: {head_motion:.0f}px/s",
                    (bbox.x1, bbox.y2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1
                )
        
        # Session status at bottom
        status_text = self._get_status_string()
        
        if self.state == SessionState.ENGAGED:
            color = (0, 255, 0)
        elif self.state == SessionState.PAUSED:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)
        
        cv2.putText(
            frame,
            f"Session: {status_text}",
            (10, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Timeout countdown
        if self.state == SessionState.ENGAGED and self.last_activity:
            remaining = self.config.session.timeout_seconds - (time.time() - self.last_activity)
            if remaining > 0:
                cv2.putText(
                    frame,
                    f"Timeout in: {remaining:.0f}s",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        # ASD backend indicator
        cv2.putText(
            frame,
            f"ASD: {self.asd.backend_name}",
            (w - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1
        )
    
    def cleanup(self):
        """Release resources"""
        self.face_tracker.cleanup()
