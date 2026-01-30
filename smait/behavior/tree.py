"""
SMAIT HRI System v2.0 - Behavior Tree Structure

Main behavior tree for human-robot interaction.
Uses py_trees for composable, parallel behaviors.

Tree Structure:
```
Root (Selector)
├── Safety (Priority)
│   └── CheckTimeout
├── Interaction (Sequence)  
│   ├── Parallel [success_on_one]
│   │   ├── TrackFaces (Running)
│   │   ├── ListenForSpeech (Running → Success on speech)
│   │   └── Backchannel (Running)
│   └── ProcessSpeech (Sequence)
│       ├── VerifySpeaker
│       ├── GenerateResponse
│       └── UpdateSession
└── Idle
    └── WaitForUser
```
"""

import time
from typing import Optional, Dict, Any
import py_trees
from py_trees import common, composites, decorators, behaviour

from smait.core.config import get_config
from smait.core.events import SessionState


class HRIBlackboard:
    """
    Shared state for behavior tree nodes.
    
    This replaces direct component references with a centralized
    blackboard pattern for cleaner data flow.
    """
    
    def __init__(self):
        # Components (set externally)
        self.audio_pipeline = None
        self.transcriber = None
        self.verifier = None
        self.dialogue = None
        
        # Current state
        self.session_state = SessionState.IDLE
        self.target_user_id: Optional[int] = None
        self.last_activity: Optional[float] = None
        
        # Current frame data
        self.current_faces = []
        self.current_asd_results = []
        self.primary_speaker_id: Optional[int] = None
        
        # Speech data
        self.pending_transcript = None
        self.pending_verification = None
        self.pending_response = None
        
        # Flags
        self.speech_detected = False
        self.response_ready = False
        
        # Metrics
        self.turn_count = 0
        self.session_start: Optional[float] = None


# Global blackboard instance
_blackboard: Optional[HRIBlackboard] = None


def get_blackboard() -> HRIBlackboard:
    """Get or create the global blackboard"""
    global _blackboard
    if _blackboard is None:
        _blackboard = HRIBlackboard()
    return _blackboard


def set_blackboard(bb: HRIBlackboard):
    """Set the global blackboard"""
    global _blackboard
    _blackboard = bb


class TrackFaces(behaviour.Behaviour):
    """
    Continuously track faces in the video stream.
    Always returns RUNNING (background task).
    """
    
    def __init__(self, name: str = "TrackFaces"):
        super().__init__(name)
        self.bb = get_blackboard()
    
    def update(self) -> common.Status:
        # Face tracking happens in video thread
        # This behavior just monitors the state
        
        if self.bb.verifier is None:
            return common.Status.FAILURE
        
        # Update blackboard with current face state
        self.bb.current_faces = self.bb.verifier.current_faces
        self.bb.current_asd_results = self.bb.verifier.current_asd_results
        
        # Always running - continuous task
        return common.Status.RUNNING


class ListenForSpeech(behaviour.Behaviour):
    """
    Listen for speech input.
    Returns SUCCESS when speech is detected and verified, RUNNING otherwise.
    """
    
    def __init__(self, name: str = "ListenForSpeech"):
        super().__init__(name)
        self.bb = get_blackboard()
    
    def update(self) -> common.Status:
        # Check if there's a pending transcript
        if self.bb.pending_transcript is not None:
            # We have speech to process
            self.bb.speech_detected = True
            return common.Status.SUCCESS
        
        # Still listening
        return common.Status.RUNNING
    
    def terminate(self, new_status: common.Status):
        """Called when behavior completes or is interrupted"""
        pass  # Keep speech_detected flag for VerifySpeaker


class VerifySpeaker(behaviour.Behaviour):
    """
    Verify that detected speech came from the target user.
    Uses the SpeakerVerifier with temporal ASD buffering.
    """
    
    def __init__(self, name: str = "VerifySpeaker"):
        super().__init__(name)
        self.bb = get_blackboard()
        self.config = get_config()
    
    def update(self) -> common.Status:
        if self.bb.verifier is None or self.bb.pending_transcript is None:
            return common.Status.FAILURE
        
        # Verify the speech using the verifier (which checks ASD history)
        from smait.core.events import VerifyResult
        
        result = self.bb.verifier.verify_speech(self.bb.pending_transcript)
        self.bb.pending_verification = result
        
        if result.result == VerifyResult.ACCEPT:
            # Update session state
            self.bb.session_state = SessionState.ENGAGED
            self.bb.last_activity = time.time()
            
            if self.bb.target_user_id is None:
                self.bb.target_user_id = result.face_id
                self.bb.session_start = time.time()
            
            return common.Status.SUCCESS
        
        elif result.result == VerifyResult.REJECT:
            # Clear pending transcript - rejected
            self.bb.pending_transcript = None
            self.bb.speech_detected = False
            return common.Status.FAILURE
        
        elif result.result == VerifyResult.NO_FACE:
            # No face visible
            self.bb.pending_transcript = None
            self.bb.speech_detected = False
            return common.Status.FAILURE
        
        else:
            # Uncertain - treat as success but with lower confidence
            return common.Status.SUCCESS


class GenerateResponse(behaviour.Behaviour):
    """
    Generate a response using the dialogue manager.
    
    Note: BT behaviors run synchronously, so we use the sync version.
    For true async, the response would be prepared in background.
    """
    
    def __init__(self, name: str = "GenerateResponse"):
        super().__init__(name)
        self.bb = get_blackboard()
        self.config = get_config()
    
    def update(self) -> common.Status:
        if self.bb.dialogue is None:
            return common.Status.FAILURE
        
        if self.bb.pending_verification is None:
            return common.Status.FAILURE
        
        text = self.bb.pending_verification.text
        
        if not text:
            # Clear state and fail
            self.bb.pending_transcript = None
            self.bb.pending_verification = None
            return common.Status.FAILURE
        
        try:
            # Get confidence from verification
            confidence = self.bb.pending_verification.confidence
            
            # Generate response (sync version)
            response = self.bb.dialogue.ask(text, confidence)
            
            self.bb.pending_response = response
            self.bb.turn_count += 1
            self.bb.response_ready = True
            self.bb.last_activity = time.time()
            
            # Clear pending data
            self.bb.pending_transcript = None
            self.bb.pending_verification = None
            self.bb.speech_detected = False
            
            return common.Status.SUCCESS
            
        except Exception as e:
            print(f"[BT] Response generation error: {e}")
            # Clear state on error
            self.bb.pending_transcript = None
            self.bb.pending_verification = None
            self.bb.speech_detected = False
            return common.Status.FAILURE


class Backchannel(behaviour.Behaviour):
    """
    Provide backchanneling (nodding indication) during user speech.
    Always returns RUNNING (background task).
    
    Backchanneling signals:
    - Visual: Changes status display when backchannel triggers
    - Audio: Could play "mm-hmm" sounds (not implemented)
    
    Triggers based on:
    - Speech duration (backchannel after 2+ seconds)
    - User is actively speaking (ASD says so)
    """
    
    def __init__(self, name: str = "Backchannel"):
        super().__init__(name)
        self.bb = get_blackboard()
        self.config = get_config()
        
        self.last_backchannel = 0.0
        self.min_interval = 3.0  # Minimum seconds between backchannels
        self.speech_start_time: Optional[float] = None
        self.backchannel_after_seconds = 2.0  # Backchannel after this much speech
        
        # State
        self.backchannel_active = False
        self.backchannel_count = 0
    
    def update(self) -> common.Status:
        now = time.time()
        
        # Check if user is currently speaking
        user_speaking = self._is_user_speaking()
        
        if user_speaking:
            # Track speech start
            if self.speech_start_time is None:
                self.speech_start_time = now
            
            # Check if we should backchannel
            speech_duration = now - self.speech_start_time
            time_since_last = now - self.last_backchannel
            
            if (speech_duration > self.backchannel_after_seconds and 
                time_since_last > self.min_interval):
                self._trigger_backchannel()
        else:
            # User stopped speaking
            self.speech_start_time = None
            self.backchannel_active = False
        
        return common.Status.RUNNING
    
    def _is_user_speaking(self) -> bool:
        """Check if the target user is currently speaking"""
        if self.bb.target_user_id is None:
            return False
        
        for result in self.bb.current_asd_results:
            if result.track_id == self.bb.target_user_id and result.is_speaking:
                return True
        
        return False
    
    def _trigger_backchannel(self):
        """Trigger a backchannel response"""
        self.last_backchannel = time.time()
        self.backchannel_active = True
        self.backchannel_count += 1
        
        # This is useful feedback - keep it
        print(f"[BACKCHANNEL] *nod*")


class CheckTimeout(behaviour.Behaviour):
    """
    Check for session timeout.
    Returns SUCCESS if timeout occurred, FAILURE otherwise.
    """
    
    def __init__(self, name: str = "CheckTimeout"):
        super().__init__(name)
        self.bb = get_blackboard()
        self.config = get_config()
    
    def update(self) -> common.Status:
        if self.bb.session_state != SessionState.ENGAGED:
            return common.Status.FAILURE
        
        if self.bb.last_activity is None:
            return common.Status.FAILURE
        
        elapsed = time.time() - self.bb.last_activity
        
        if elapsed > self.config.session.timeout_seconds:
            # Timeout - end session
            self._end_session()
            return common.Status.SUCCESS
        
        return common.Status.FAILURE
    
    def _end_session(self):
        """End the current session"""
        print(f"[BT] Session timeout after {self.config.session.timeout_seconds}s")
        
        self.bb.session_state = SessionState.IDLE
        self.bb.target_user_id = None
        self.bb.last_activity = None
        self.bb.turn_count = 0
        self.bb.session_start = None
        
        # Reset dialogue memory
        if self.bb.dialogue:
            self.bb.dialogue.reset_session()


class WaitForUser(behaviour.Behaviour):
    """
    Idle behavior - wait for a user to appear.
    """
    
    def __init__(self, name: str = "WaitForUser"):
        super().__init__(name)
        self.bb = get_blackboard()
    
    def update(self) -> common.Status:
        # Check if any faces are visible
        if self.bb.current_faces:
            self.bb.session_state = SessionState.DETECTING
            return common.Status.SUCCESS
        
        # Keep waiting
        return common.Status.RUNNING


def create_hri_tree() -> py_trees.trees.BehaviourTree:
    """
    Create the main HRI behavior tree.
    
    Structure:
    - Safety checks (timeout) have highest priority
    - Interaction subtree handles speech + response
    - Idle subtree handles waiting for user
    """
    
    # Create blackboard
    bb = get_blackboard()
    
    # === Safety subtree ===
    safety = decorators.FailureIsRunning(
        name="Safety",
        child=CheckTimeout("CheckTimeout")
    )
    
    # === Parallel perception ===
    # These run continuously in parallel
    parallel_perception = composites.Parallel(
        name="Perception",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne()
    )
    parallel_perception.add_children([
        TrackFaces("TrackFaces"),
        ListenForSpeech("ListenForSpeech"),
        Backchannel("Backchannel"),
    ])
    
    # === Process speech sequence ===
    process_speech = composites.Sequence(
        name="ProcessSpeech",
        memory=True
    )
    process_speech.add_children([
        VerifySpeaker("VerifySpeaker"),
        GenerateResponse("GenerateResponse"),
    ])
    
    # === Interaction subtree ===
    interaction = composites.Sequence(
        name="Interaction",
        memory=True
    )
    interaction.add_children([
        parallel_perception,
        process_speech,
    ])
    
    # === Idle subtree ===
    idle = WaitForUser("WaitForUser")
    
    # === Root selector ===
    root = composites.Selector(
        name="HRI_Root",
        memory=False
    )
    root.add_children([
        safety,
        interaction,
        idle,
    ])
    
    # Create tree
    tree = py_trees.trees.BehaviourTree(root)
    
    return tree


class HRIBehaviorTree:
    """
    High-level wrapper for the HRI behavior tree.
    
    Provides a simpler interface for the main system to use,
    handling setup, ticking, and shutdown.
    """
    
    def __init__(self):
        self.config = get_config()
        self.bb = get_blackboard()
        self.tree: Optional[py_trees.trees.BehaviourTree] = None
        self._running = False
    
    def setup(
        self,
        audio_pipeline=None,
        transcriber=None,
        verifier=None,
        dialogue=None
    ):
        """
        Set up the behavior tree with component references.
        """
        # Set component references on blackboard
        self.bb.audio_pipeline = audio_pipeline
        self.bb.transcriber = transcriber
        self.bb.verifier = verifier
        self.bb.dialogue = dialogue
        
        # Create tree
        self.tree = create_hri_tree()
        self.tree.setup(timeout=5.0)
        
        print("[BT] Behavior tree initialized")
    
    def tick(self):
        """
        Execute one tick of the behavior tree.
        Should be called at regular intervals (e.g., 30 Hz).
        """
        if self.tree is None:
            return
        
        self.tree.tick()
    
    def feed_transcript(self, transcript):
        """Feed a transcript result to the behavior tree"""
        self.bb.pending_transcript = transcript
    
    def get_response(self):
        """Get pending response if ready"""
        if self.bb.response_ready:
            response = self.bb.pending_response
            self.bb.pending_response = None
            self.bb.response_ready = False
            return response
        return None
    
    @property
    def session_state(self) -> SessionState:
        """Current session state"""
        return self.bb.session_state
    
    @property
    def target_user_id(self) -> Optional[int]:
        """Current target user ID"""
        return self.bb.target_user_id
    
    def shutdown(self):
        """Shutdown the behavior tree"""
        if self.tree:
            self.tree.shutdown()
        print("[BT] Behavior tree shutdown")
