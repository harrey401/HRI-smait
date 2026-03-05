# CAE SDK Integration Guide for Jackie Android App

## Overview
Replace raw `AudioRecord` with iFLYTEK CAE SDK for:
- ✅ Hardware AEC (echo cancellation)
- ✅ 4-mic beamforming
- ✅ Noise suppression
- ✅ DOA (Direction of Arrival)

## Prerequisites
1. CAE SDK files from `Hardware SDK/CAEDemoAIUI-4 MIC/`
2. Resource files on Jackie's SD card:
   - `/sdcard/cae/res_cae_model.bin`
   - `/sdcard/cae/res_ivw_model.bin`
   - `/sdcard/cae/hlw.param`
   - `/sdcard/cae/hlw.ini`
3. Native libs: `libcae.so` (ARM64)

## Step 1: Add SDK to Android Project
1. Copy the iFLYTEK `.aar` or `.jar` + `.so` files into `app/libs/`
2. In `build.gradle`:
```groovy
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.aar', '*.jar'])
}
```
3. In `jniLibs/arm64-v8a/`, place the native `.so` files

## Step 2: Replace AudioRecord with CAE

In MainActivity.kt, replace `startAudioCapture()`:

```kotlin
// OLD: Raw mic
audioRecord = AudioRecord(
    MediaRecorder.AudioSource.MIC,
    SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT,
    bufferSize * 2
)

// NEW: CAE SDK (processed audio)
import com.iflytek.iflyos.cae.CAE
import com.iflytek.iflyos.cae.ICAEListener

private var caeOperator: CaeOperator? = null

private fun startAudioCaptureWithCAE() {
    caeOperator = CaeOperator()
    // CAE outputs processed single-channel audio via callback
    caeOperator?.setAudioCallback { processedAudio ->
        // This is already AEC'd, beamformed, noise-suppressed
        sendAudioChunk(processedAudio)
    }
    caeOperator?.setDOACallback { angle ->
        // Send DOA to PC
        sendDOAAngle(angle)
    }
    caeOperator?.start()
}

private fun sendDOAAngle(angle: Float) {
    val json = JSONObject()
    json.put("type", "doa")
    json.put("angle", angle)
    webSocket?.send(json.toString())
}
```

## Step 3: CAE Configuration
The CAE SDK uses 8 channels (4 raw mics + 4 processed):
- Channels 0-3: Raw microphone data
- Channel 4: Beamformed output (USE THIS)
- Channel 5: AEC reference
- Channels 6-7: Reserved

Set `MicType.FMic32Bit` for the 4-mic 32-bit configuration.

## Step 4: Test
1. Build and deploy to Jackie
2. Connect to PC via WebSocket
3. Verify:
   - No echo when robot speaks
   - Background voices filtered
   - DOA angle updating in PC logs

## Fallback
If CAE SDK causes issues, the app still works with raw AudioRecord.
The PC side handles both protocols.

## Reference
See full CAE demo code at:
`docs/hardware-sdk/CAEDemoAIUI-4 MIC/CAEDemoAIUI/app/src/main/java/com/voice/caePk/CaeOperator.java`
