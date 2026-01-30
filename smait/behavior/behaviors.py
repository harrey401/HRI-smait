"""
SMAIT HRI System v2.0 - Behavior Implementations

Additional behaviors for the HRI behavior tree.
These can be composed with the core behaviors for extended functionality.
"""

import time
import re
from typing import Optional, List
import py_trees
from py_trees import common, behaviour

from smait.core.config import get_config
from smait.core.events import SessionState


# Re-export core behaviors for convenience
from smait.behavior.tree import (
    get_blackboard,
    TrackFaces,
    ListenForSpeech,
    VerifySpeaker,
    GenerateResponse,
    Backchannel,
    CheckTimeout,
    WaitForUser,
)


class SemanticEndOfTurn(behaviour.Behaviour):
    """
    Predict end-of-turn using semantic analysis.
    
    Uses OpenAI to predict if the user has finished their turn,
    enabling earlier response preparation.
    
    This reduces perceived latency by starting response generation
    before the VAD confirms silence.
    """
    
    def __init__(self, name: str = "SemanticEndOfTurn"):
        super().__init__(name)
        self.bb = get_blackboard()
        self.config = get_config()
        
        # Prediction state
        self._last_partial = ""
        self._predicted_complete = False
        self._confidence = 0.0
        
        # Simple pattern-based triggers (fallback if no LLM)
        self.completion_patterns = [
            r'[.?!]$',           # Sentence-ending punctuation
            r'\?$',              # Question
            r'thank you\.?$',    # Thanks
            r"that'?s all\.?$",  # Explicit completion
            r'please\.?$',       # Request ending
        ]
        
        self.continuation_patterns = [
            r'\b(and|but|so|or|because)\s*$',  # Conjunctions
            r',\s*$',                           # Trailing comma
            r'\.\.\.$',                         # Ellipsis
            r'\b(um|uh|er)\s*$',               # Hesitation
        ]
    
    def update(self) -> common.Status:
        """Check if turn appears complete"""
        
        # Get current partial transcript
        partial = self._get_partial_transcript()
        
        if not partial:
            self._predicted_complete = False
            return common.Status.RUNNING
        
        # Check for completion
        self._confidence = self._predict_completion(partial)
        self._predicted_complete = self._confidence > 0.7
        
        if self._predicted_complete:
            return common.Status.SUCCESS
        
        return common.Status.RUNNING
    
    def _get_partial_transcript(self) -> str:
        """Get current partial transcript"""
        # Would come from streaming ASR
        return self._last_partial
    
    def _predict_completion(self, text: str) -> float:
        """
        Predict probability that turn is complete.
        
        Uses pattern matching as baseline, could use LLM for better accuracy.
        """
        text = text.strip().lower()
        
        if not text:
            return 0.0
        
        # Check continuation patterns (turn NOT complete)
        for pattern in self.continuation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 0.2
        
        # Check completion patterns
        for pattern in self.completion_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return 0.85
        
        # Length heuristic - longer sentences more likely complete
        words = len(text.split())
        length_factor = min(words / 10, 1.0) * 0.3
        
        # Default moderate confidence for other cases
        return 0.5 + length_factor
    
    async def predict_with_llm(self, text: str) -> float:
        """
        Use OpenAI to predict turn completion.
        
        More accurate but adds latency - use sparingly.
        """
        try:
            import openai
            
            client = openai.AsyncOpenAI()
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You predict if a speaker has finished their turn. "
                            "Respond with only a number 0-100 indicating confidence "
                            "that the turn is complete. Consider: sentence structure, "
                            "punctuation, common ending phrases, trailing conjunctions."
                        )
                    },
                    {
                        "role": "user",
                        "content": f'Is this turn complete? "{text}"'
                    }
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            # Parse response
            try:
                confidence = int(response.choices[0].message.content.strip()) / 100
                return min(max(confidence, 0.0), 1.0)
            except:
                return 0.5
                
        except Exception as e:
            if self.config.debug:
                print(f"[BT] LLM prediction error: {e}")
            return self._predict_completion(text)
    
    @property
    def is_complete(self) -> bool:
        """Whether turn is predicted complete"""
        return self._predicted_complete
    
    @property
    def confidence(self) -> float:
        """Confidence in completion prediction"""
        return self._confidence


class PrepareResponse(behaviour.Behaviour):
    """
    Prepare response early based on turn prediction.
    
    Works with SemanticEndOfTurn to start response generation
    before the user finishes speaking, reducing latency.
    """
    
    def __init__(self, name: str = "PrepareResponse"):
        super().__init__(name)
        self.bb = get_blackboard()
        self._preparing = False
        self._prepared_response = None
        self._prepared_for_text = ""
    
    def update(self) -> common.Status:
        # Check if we should prepare a response
        if not self._should_prepare():
            return common.Status.RUNNING
        
        # Start preparation if not already
        if not self._preparing:
            self._start_preparation()
        
        # Check if preparation is complete
        if self._prepared_response is not None:
            return common.Status.SUCCESS
        
        return common.Status.RUNNING
    
    def _should_prepare(self) -> bool:
        """Check if conditions are right for early preparation"""
        # Need partial transcript
        # Would integrate with SemanticEndOfTurn
        return False
    
    def _start_preparation(self):
        """Start async response preparation"""
        self._preparing = True
        # Would start async LLM call here
    
    def get_prepared_response(self):
        """Get the prepared response if ready"""
        return self._prepared_response


class DetectEngagementLoss(behaviour.Behaviour):
    """
    Detect when user is disengaging from the conversation.
    
    Signs of disengagement:
    - Looking away for extended time
    - Walking away
    - Starting conversation with someone else
    """
    
    def __init__(self, name: str = "DetectEngagementLoss"):
        super().__init__(name)
        self.bb = get_blackboard()
        self.config = get_config()
        
        self._user_not_visible_since: Optional[float] = None
        self._user_not_speaking_since: Optional[float] = None
    
    def update(self) -> common.Status:
        if self.bb.target_user_id is None:
            return common.Status.FAILURE
        
        now = time.time()
        
        # Check if target user is visible
        target_visible = any(
            f.track_id == self.bb.target_user_id 
            for f in self.bb.current_faces
        )
        
        if not target_visible:
            if self._user_not_visible_since is None:
                self._user_not_visible_since = now
            
            # Check if lost for too long
            elapsed = now - self._user_not_visible_since
            if elapsed > self.config.session.face_lost_grace_seconds:
                return common.Status.SUCCESS  # Engagement lost
        else:
            self._user_not_visible_since = None
        
        return common.Status.FAILURE  # Still engaged


class SafetyInterrupt(behaviour.Behaviour):
    """
    Safety interrupt behavior - highest priority.
    
    Checks for:
    - Emergency stop signals
    - Low battery (for mobile robots)
    - Collision warnings
    """
    
    def __init__(self, name: str = "SafetyInterrupt"):
        super().__init__(name)
        self.bb = get_blackboard()
        
        # Safety flags (would be set by external systems)
        self.emergency_stop = False
        self.low_battery = False
        self.collision_warning = False
    
    def update(self) -> common.Status:
        if self.emergency_stop:
            print("[SAFETY] Emergency stop activated!")
            return common.Status.SUCCESS
        
        if self.collision_warning:
            print("[SAFETY] Collision warning!")
            return common.Status.SUCCESS
        
        if self.low_battery:
            print("[SAFETY] Low battery warning")
            # Might want to gracefully end conversation
        
        return common.Status.FAILURE


class VisualBackchannel(behaviour.Behaviour):
    """
    Visual backchanneling - nodding behavior.
    
    Less intrusive than audio backchannels.
    Triggered by backchannel-relevant spaces in speech.
    """
    
    def __init__(self, name: str = "VisualBackchannel"):
        super().__init__(name)
        self.bb = get_blackboard()
        
        self.last_nod = 0.0
        self.nod_interval = 4.0  # Minimum seconds between nods
        self.nod_duration = 0.5  # Duration of nod animation
        
        self._nodding = False
        self._nod_start = 0.0
    
    def update(self) -> common.Status:
        now = time.time()
        
        # Check if currently nodding
        if self._nodding:
            if now - self._nod_start > self.nod_duration:
                self._nodding = False
            return common.Status.RUNNING
        
        # Check if we should nod
        if self._should_nod():
            self._trigger_nod()
        
        return common.Status.RUNNING
    
    def _should_nod(self) -> bool:
        """Determine if we should nod now"""
        now = time.time()
        
        # Not during our own speech
        if self.bb.response_ready:
            return False
        
        # Not too frequently
        if now - self.last_nod < self.nod_interval:
            return False
        
        # Only when user is speaking
        if self.bb.session_state != SessionState.ENGAGED:
            return False
        
        # Check if user is actively speaking
        speaking = any(
            r.is_speaking and r.track_id == self.bb.target_user_id
            for r in self.bb.current_asd_results
        )
        
        return speaking
    
    def _trigger_nod(self):
        """Trigger a nod animation"""
        self._nodding = True
        self._nod_start = time.time()
        self.last_nod = time.time()
        
        # Would send nod command to robot here
        if self.bb.verifier and hasattr(self.bb.verifier, 'trigger_nod'):
            self.bb.verifier.trigger_nod()


class AudioBackchannel(behaviour.Behaviour):
    """
    Audio backchanneling - "mm-hmm", "uh-huh" sounds.
    
    More intrusive than visual, use sparingly.
    Requires very low latency playback to avoid interrupting user.
    """
    
    def __init__(self, name: str = "AudioBackchannel"):
        super().__init__(name)
        self.bb = get_blackboard()
        
        self.last_audio_bc = 0.0
        self.audio_bc_interval = 8.0  # Less frequent than visual
        
        # Pre-cached audio files (would be loaded in production)
        self.audio_files = {
            'mmhmm': 'assets/audio/mmhmm.wav',
            'uhuh': 'assets/audio/uhuh.wav',
            'okay': 'assets/audio/okay.wav',
        }
    
    def update(self) -> common.Status:
        # Audio backchanneling requires very careful timing
        # For now, just return RUNNING
        return common.Status.RUNNING
    
    def _play_backchannel(self, sound: str = 'mmhmm'):
        """Play a backchannel sound"""
        # Would play pre-cached audio with minimal latency
        pass
