# Jackie Robot - Hardware Reference

## Key Specs for SMAIT HRI
- **CPU:** RK3588 (Android 12) — 2GB RAM, 16GB ROM
- **Mic Array:** Linear 4-mic array, effective within 3m
- **Speakers:** 2 loudspeakers (front body)
- **Face Camera:** 2MP, 0.5-3m range, >99% recognition, <1s
- **Surveillance Cameras:** 4x 2MP with IR night vision (top of head)
- **Display:** 15.6" 1080p touchscreen
- **LIDAR:** Single-line laser (SLAM/nav)
- **Connectivity:** WiFi + 4G
- **Battery:** 8hr working, auto-dock charging
- **Movement:** Differential + mecanum wheels, 0-0.8m/s

## Architecture Notes
- Robot runs Android 12 on RK3588
- PC connects via WiFi (WebSocket over static IP)
- Audio: 4-mic linear array → can do beamforming/DOA
- Speakers on front body → echo from speakers can reach mic array
- Face camera at 0.5-3m effective range matches our engagement distance

## iFLYTEK CAE SDK (4-MIC)
The robot has an iFLYTEK CAE (Computer Audition Engine) SDK that provides:
- **Hardware AEC** (Acoustic Echo Cancellation) — handles robot-hearing-itself at hardware level
- **Beamforming** — directional audio from 4-mic linear array
- **DOA** (Direction of Arrival) — know which direction speech comes from
- **Noise suppression** — background voice filtering
- **Wake word detection** — built-in "hello" detection (via res_ivw_model.bin)
- Audio format: 8 channels at 16kHz, 16-bit (4 raw mics + processed channels)
- MicType configs: 2mic, 4mic, 4mic-32bit, 6mic

## Key Insight
The Android app should use the CAE SDK to pre-process audio BEFORE sending
to PC. This gives us hardware-level echo cancellation, beamforming, and noise
suppression — solving issues #1, #2, and #6 at the source instead of in software.

Current software mute/unmute on PC side is a good fallback but the real fix
is leveraging the CAE SDK on the robot.
