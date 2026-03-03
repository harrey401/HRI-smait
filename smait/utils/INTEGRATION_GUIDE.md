# Data Logger Integration Guide
_For wiring `data_logger.py` into the SMAIT HRI system_

## Quick Start

### 1. Initialize logger in `main.py` → `HRISystem.__init__`

```python
from smait.utils.data_logger import init_data_logger, get_data_logger, TurnData, EngagementInfo
```

In `__init__`:
```python
self.data_logger = init_data_logger(output_dir="logs/march2", save_audio=True)
```

### 2. Log session start — in `_proactive_greet()` and `verifier._start_session()` calls

When a session starts via proactive greeting (`_video_loop` → `_proactive_greet`):
```python
if self.data_logger:
    from smait.utils.data_logger import EngagementInfo
    eng = EngagementInfo(
        proximity_m=0.0,  # TODO: compute from face area
        head_yaw_deg=face.head_yaw if hasattr(face, 'head_yaw') else 0,
        greeting_type="proactive",
        face_area=face.bbox.area,
        attention_ok=engagement.attention_ok,
        proximity_ok=engagement.proximity_ok,
    )
    self.data_logger.start_session(engagement=eng, proactive_greeting=True)
```

When session starts from speech (in `_process_transcript` after ACCEPT + new_session_started):
```python
if self.data_logger and verification.reason == "new_session_started":
    eng = EngagementInfo(greeting_type="reactive")
    self.data_logger.start_session(engagement=eng, proactive_greeting=False)
```

### 3. Log turns — in `_process_transcript()` after robot responds

After `response = await self.dialogue.ask_async(...)`:
```python
if self.data_logger:
    turn = TurnData(
        user_text=text,
        asr_confidence=transcript.confidence,
        asr_latency_ms=0,  # already tracked separately
        segment_duration_ms=(transcript.end_time - transcript.start_time) * 1000 if transcript.start_time else 0,
        asd_score=verification.asd_score if hasattr(verification, 'asd_score') else 0,
        verification_result=verification.result.name,
        verification_reason=verification.reason,
        concurrent_faces=len(self.verifier.current_faces),
        robot_text=response.text,
        llm_latency_ms=response.latency_ms,
        llm_model=response.model if hasattr(response, 'model') else "",
        total_response_time_ms=response_time,
        error_recovery="repeat" in response.text.lower() or "say that again" in response.text.lower(),
    )
    self.data_logger.log_turn(turn)
```

### 4. Save audio segments — in `Transcriber._process_loop()`

In `transcriber.py` `_process_loop`, before transcribing:
```python
from smait.utils.data_logger import get_data_logger
logger = get_data_logger()
if logger:
    logger.save_audio_segment(segment.audio, label="pre_asr")
```

### 5. Log rejections — in `_process_transcript()` REJECT/NO_FACE paths

```python
if self.data_logger:
    self.data_logger.log_rejected_transcript(
        text=text,
        reason=verification.reason,
        asd_score=verification.asd_score if hasattr(verification, 'asd_score') else 0,
        is_phantom=False,
        is_wrong_speaker=(verification.reason == "different_speaker"),
    )
```

### 6. Log filtered phantoms — in hallucination filter section

When a filler/hallucination is dropped:
```python
if self.data_logger:
    self.data_logger.log_rejected_transcript(
        text=text, reason="hallucination_filter", is_phantom=True
    )
```

### 7. Log session end — in timeout, farewell, and face_lost handlers

```python
if self.data_logger:
    self.data_logger.end_session(reason="farewell")  # or "timeout" or "face_lost"
```

### 8. Wire CAE status — in `network_source.py` `_handle_command`

In the `cae_status` handler:
```python
from smait.utils.data_logger import get_data_logger
logger = get_data_logger()
if logger:
    logger.update_cae_status(
        aec=cmd.get('aec', False),
        beamforming=cmd.get('beamforming', False),
        noise_suppression=cmd.get('noise_suppression', False)
    )
```

### 9. Wire DOA angle — in `network_source.py` DOA handlers

```python
from smait.utils.data_logger import get_data_logger
logger = get_data_logger()
if logger:
    logger.log_doa_angle(angle)
```

### 10. Post-event analysis

```python
from smait.utils.data_logger import DataLogger
stats = DataLogger.analyze_event("logs/march2")
print(json.dumps(stats, indent=2))
```

## Config Changes to Make (in config.py or env)

```
silence_duration_ms: 500 → 1000
```

## Code Changes to Make

### verifier.py — Fix threshold mismatch
In `_get_speaking_scores_in_window`:
```python
# OLD:
if result.is_speaking or result.probability > 0.3:
# NEW:
if result.is_speaking or result.probability > 0.2:
```

### verifier.py — Add single-user fast path (optional)
In `verify_speech`, after `speaking_scores` is empty:
```python
# If only one face visible and VAD confirmed speech, accept (single-user fast path)
if len(faces) == 1 and self.state == SessionState.ENGAGED:
    only_face = faces[0]
    if only_face.track_id == self.target_user_id:
        return VerifyOutput(
            result=VerifyResult.ACCEPT,
            text=transcript.text,
            confidence=0.5,
            reason="single_user_fast_path",
            face_id=only_face.track_id,
            asd_score=0.0
        )
```
