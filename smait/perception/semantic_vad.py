"""
SMAIT HRI System v2.0 - Semantic Voice Activity Detection

Predicts turn completion using semantic analysis of partial transcripts.
Reduces latency by 50%+ compared to silence-based timeout.

Instead of waiting for VAD to confirm silence (500-1000ms), we analyze
the text to predict when the user is done speaking.

References:
- "Semantic VAD" approach from conversational AI research
- Reduces perceived latency by enabling early response preparation
"""

import asyncio
import re
import time
from typing import Optional, Callable, List
from dataclasses import dataclass

from smait.core.config import get_config


@dataclass
class TurnPrediction:
    """Result of turn completion prediction"""
    is_complete: bool
    confidence: float
    reason: str
    partial_text: str
    timestamp: float


class SemanticVAD:
    """
    Semantic Voice Activity Detection.
    
    Predicts when a user has finished their turn based on:
    1. Sentence structure (complete vs incomplete)
    2. Punctuation patterns
    3. Common ending/continuation phrases
    4. Optional LLM analysis for complex cases
    
    Usage:
        vad = SemanticVAD()
        
        # Feed partial transcripts as they arrive
        for partial in streaming_transcripts:
            prediction = vad.predict(partial)
            
            if prediction.is_complete and prediction.confidence > 0.8:
                # Start preparing response early!
                prepare_response(partial)
    """
    
    def __init__(self, use_llm: bool = True):
        self.config = get_config()
        self.use_llm = use_llm
        
        # Tracking state
        self._last_text = ""
        self._last_prediction = None
        self._stable_since: Optional[float] = None
        self._min_stability_ms = 200  # Text must be stable for this long
        
        # Pattern-based detection (fast, no API calls)
        self.completion_patterns = [
            # Strong completion signals
            (r'[.!?]\s*$', 0.9, "sentence_end"),
            (r'\?\s*$', 0.95, "question"),
            
            # Explicit completion phrases
            (r'(?:thank\s*you|thanks)\.?\s*$', 0.9, "thanks"),
            (r"(?:that'?s?\s+all|that'?s?\s+it)\.?\s*$", 0.95, "explicit_end"),
            (r'(?:please|okay|alright)\.?\s*$', 0.7, "closing_word"),
            
            # Natural endings
            (r'(?:bye|goodbye|see\s+you)\.?\s*$', 0.95, "farewell"),
            (r'(?:got\s+it|understood|makes\s+sense)\.?\s*$', 0.8, "acknowledgment"),
        ]
        
        self.continuation_patterns = [
            # Strong continuation signals
            (r'\b(?:and|but|so|or|because|since|although)\s*$', 0.2, "conjunction"),
            (r',\s*$', 0.3, "comma"),
            (r'\.{3}\s*$', 0.2, "ellipsis"),
            
            # Hesitation markers
            (r'\b(?:um|uh|er|ah|like)\s*$', 0.25, "hesitation"),
            
            # Incomplete phrases
            (r'\b(?:the|a|an|my|your|this|that)\s*$', 0.15, "determiner"),
            (r'\b(?:is|are|was|were|will|would|can|could)\s*$', 0.2, "verb_incomplete"),
            (r'\b(?:to|for|with|about|from)\s*$', 0.15, "preposition"),
        ]
        
        # LLM prompt for complex cases
        self.llm_prompt = """You are a turn-taking predictor for conversational AI. 
Analyze if the speaker has finished their turn.

Consider:
- Is the sentence grammatically complete?
- Are there trailing conjunctions suggesting more to come?
- Is there hesitation or ellipsis?
- Does it end with a complete thought?

Respond with ONLY a number 0-100 indicating confidence the turn is COMPLETE.
0 = definitely continuing, 100 = definitely complete.

Speaker said: "{text}"

Confidence (0-100):"""
    
    def predict(self, partial_text: str) -> TurnPrediction:
        """
        Predict if the current partial transcript represents a complete turn.
        
        Args:
            partial_text: Current partial transcript from streaming ASR
        
        Returns:
            TurnPrediction with completion status and confidence
        """
        now = time.time()
        text = partial_text.strip()
        
        if not text:
            return TurnPrediction(
                is_complete=False,
                confidence=0.0,
                reason="empty",
                partial_text=text,
                timestamp=now
            )
        
        # Check text stability (avoid predicting on rapidly changing text)
        if text != self._last_text:
            self._last_text = text
            self._stable_since = now
            return TurnPrediction(
                is_complete=False,
                confidence=0.0,
                reason="text_changing",
                partial_text=text,
                timestamp=now
            )
        
        # Require minimum stability
        if self._stable_since and (now - self._stable_since) * 1000 < self._min_stability_ms:
            return TurnPrediction(
                is_complete=False,
                confidence=0.0,
                reason="stabilizing",
                partial_text=text,
                timestamp=now
            )
        
        # Run pattern-based prediction
        confidence, reason = self._pattern_predict(text)
        
        # Determine if complete
        is_complete = confidence > 0.7
        
        prediction = TurnPrediction(
            is_complete=is_complete,
            confidence=confidence,
            reason=reason,
            partial_text=text,
            timestamp=now
        )
        
        self._last_prediction = prediction
        return prediction
    
    def _pattern_predict(self, text: str) -> tuple[float, str]:
        """
        Pattern-based prediction (fast, no API calls).
        
        Returns:
            Tuple of (confidence, reason)
        """
        text_lower = text.lower().strip()
        
        # Check continuation patterns first (they override completion)
        for pattern, conf_penalty, reason in self.continuation_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return (conf_penalty, f"continuation_{reason}")
        
        # Check completion patterns
        for pattern, confidence, reason in self.completion_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return (confidence, f"completion_{reason}")
        
        # Heuristics for uncertain cases
        words = text.split()
        word_count = len(words)
        
        # Very short utterances need explicit completion
        if word_count < 3:
            return (0.4, "too_short")
        
        # Longer utterances more likely complete
        length_bonus = min(word_count / 15, 0.3)
        
        # Check for complete sentence structure (rough heuristic)
        has_verb = bool(re.search(
            r'\b(?:is|are|was|were|have|has|had|do|does|did|will|would|can|could|'
            r'want|need|think|know|see|go|get|make|take|come|give|find|tell|'
            r'like|love|hate|feel|believe|understand|remember)\b',
            text_lower
        ))
        
        if has_verb:
            return (0.5 + length_bonus, "has_verb")
        
        return (0.4 + length_bonus, "uncertain")
    
    async def predict_with_llm(self, text: str) -> TurnPrediction:
        """
        LLM-based prediction for complex cases.
        
        More accurate but adds ~200-500ms latency.
        Use when pattern-based confidence is in uncertain range (0.4-0.7).
        """
        if not self.use_llm:
            return self.predict(text)
        
        try:
            import openai
            
            client = openai.AsyncOpenAI()
            
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": self.llm_prompt.format(text=text)
                    }
                ],
                max_tokens=5,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            try:
                confidence = int(re.search(r'\d+', response_text).group()) / 100
                confidence = min(max(confidence, 0.0), 1.0)
            except:
                # Fallback to pattern-based
                confidence, _ = self._pattern_predict(text)
            
            return TurnPrediction(
                is_complete=confidence > 0.7,
                confidence=confidence,
                reason="llm_prediction",
                partial_text=text,
                timestamp=time.time()
            )
            
        except Exception as e:
            if self.config.debug:
                print(f"[SVAD] LLM error: {e}")
            return self.predict(text)
    
    def reset(self):
        """Reset internal state"""
        self._last_text = ""
        self._last_prediction = None
        self._stable_since = None


class TurnTakingManager:
    """
    High-level turn-taking manager.
    
    Coordinates between VAD, ASR, and response generation
    to minimize latency while maintaining natural conversation flow.
    """
    
    def __init__(
        self,
        on_turn_complete: Optional[Callable[[str], None]] = None,
        on_early_prediction: Optional[Callable[[str, float], None]] = None
    ):
        self.config = get_config()
        self.semantic_vad = SemanticVAD(use_llm=True)
        
        # Callbacks
        self._on_turn_complete = on_turn_complete
        self._on_early_prediction = on_early_prediction
        
        # State
        self._current_text = ""
        self._response_preparing = False
        self._early_prediction_made = False
    
    def process_partial(self, partial_text: str) -> Optional[TurnPrediction]:
        """
        Process a partial transcript.
        
        Called whenever streaming ASR emits a new partial result.
        Returns prediction if turn appears complete.
        """
        self._current_text = partial_text
        
        # Get prediction
        prediction = self.semantic_vad.predict(partial_text)
        
        # High confidence completion - trigger early response
        if prediction.is_complete and prediction.confidence > 0.8:
            if not self._early_prediction_made:
                self._early_prediction_made = True
                
                if self._on_early_prediction:
                    self._on_early_prediction(partial_text, prediction.confidence)
        
        return prediction
    
    def confirm_end_of_turn(self, final_text: str):
        """
        Called when VAD confirms end of speech.
        
        At this point, if we made an early prediction, the response
        should already be preparing/ready.
        """
        self._current_text = final_text
        
        if self._on_turn_complete:
            self._on_turn_complete(final_text)
        
        # Reset for next turn
        self._early_prediction_made = False
        self.semantic_vad.reset()
    
    def cancel_turn(self):
        """Cancel current turn (e.g., user interrupted)"""
        self._current_text = ""
        self._early_prediction_made = False
        self.semantic_vad.reset()


# Convenience function
def create_turn_taking_manager(**kwargs) -> TurnTakingManager:
    """Create a turn-taking manager with optional callbacks"""
    return TurnTakingManager(**kwargs)
