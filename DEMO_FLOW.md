# SMAIT Demo Flow — March 3, SJSU Library
_Audience: City of San Jose officials (non-technical)_

## Before visitors arrive (~30 min early)
1. Boot: `conda activate smait && python run_jackie.py`
2. Confirm: `[DATA-LOG] Session logger ready`, camera feed showing, no crash loops
3. Quick test: stand in front, say a sentence, confirm full loop (greet → ASR → response → TTS)
4. Check Jackie's tablet UI
5. If broken, you have 30 min. If unfixable, know your fallback (bottom of doc)

## The Demo (~2-3 min per group)

### Phase 1 — The Hook (10 sec)
Don't explain first. Let Jackie proactively greet visitors as they approach. That's the wow moment.

### Phase 2 — Let Them Talk (30-60 sec)
Step back. Let visitors ask Jackie anything. Stay quiet, let the interaction happen naturally.

### Phase 3 — Your Narration (60 sec)
After they've had their moment, 3 punchy facts max:
- "She detects who's speaking by watching mouth movements — not just audio"
- "Everything runs locally on this PC — no cloud, no internet"
- "Same kind of speech recognition NVIDIA built for autonomous systems"

### Phase 4 — Stress Test (optional, if engaged)
"Want to try something cool? Have two people talk at once."
Shows ASD works — she tracks who's speaking.

## Failure Recovery

| What breaks | What you do |
|---|---|
| No greeting | Walk up, say "Hey Jackie" to trigger reactive session |
| ASR garbles | "She's still learning noisy rooms — that's one of our research problems" |
| Freeze/crash | Auto-restarts (up to 10x). Say "give her a second" |
| Robotic TTS | "We're upgrading her voice next — Android default for now" |
| Total down | Show camera feed + architecture on laptop. Talk through design. |

## Rules
- Don't apologize for bugs → frame as "active research problems"
- Don't read slides → live demo, not presentation
- Don't explain code → city officials, not engineers
- Don't touch terminal while visitors watch
