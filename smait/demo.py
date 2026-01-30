"""
SMAIT HRI System v2.0 - Demo Script
Quick demos for testing individual components.
"""

import asyncio
import sys
import time
import numpy as np


def demo_audio_pipeline():
    """Test audio capture and VAD"""
    print("\n" + "="*60)
    print("DEMO: Audio Pipeline + VAD")
    print("="*60)
    print("Speak to test voice activity detection...")
    print("Press Ctrl+C to stop\n")
    
    from smait.sensors.audio_pipeline import AudioPipeline
    
    pipeline = AudioPipeline()
    
    def on_vad(is_speech, prob):
        bar = "█" * int(prob * 30)
        status = "SPEECH" if is_speech else "      "
        print(f"\r[{status}] {prob:.2f} {bar:<30}", end="", flush=True)
    
    pipeline.set_vad_callback(on_vad)
    pipeline.start()
    
    try:
        # Run async loop
        async def run():
            async for segment in pipeline.speech_segments():
                duration = segment.duration
                print(f"\n[SEGMENT] Duration: {duration:.2f}s, Samples: {len(segment.audio)}")
        
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        print("\n\nDemo complete.")


def demo_transcription():
    """Test ASR transcription"""
    print("\n" + "="*60)
    print("DEMO: Speech Recognition (faster-whisper)")
    print("="*60)
    print("Speak to test transcription...")
    print("Press Ctrl+C to stop\n")
    
    from smait.sensors.audio_pipeline import AudioPipeline
    from smait.perception.transcriber import Transcriber
    
    pipeline = AudioPipeline()
    transcriber = Transcriber(pipeline)
    
    def on_transcript(result):
        print(f"[ASR] \"{result.text}\" (confidence: {result.confidence:.2f})")
    
    transcriber.set_final_callback(on_transcript)
    pipeline.start()
    
    try:
        async def run():
            await transcriber.start()
            while True:
                await asyncio.sleep(0.1)
        
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    finally:
        asyncio.run(transcriber.stop())
        pipeline.stop()
        print("\n\nDemo complete.")


def demo_dialogue():
    """Test dialogue manager"""
    print("\n" + "="*60)
    print("DEMO: Dialogue Manager")
    print("="*60)
    print("Type messages to test LLM responses.")
    print("Type 'quit' to exit\n")
    
    from smait.dialogue.manager import DialogueManager
    
    dialogue = DialogueManager()
    
    while True:
        try:
            user_input = input("\n[YOU] ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            response = dialogue.ask(user_input)
            print(f"[ROBOT] {response.text}")
            print(f"        (latency: {response.latency_ms:.0f}ms)")
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\n\nDemo complete.")


def demo_vision():
    """Test face tracking and ASD"""
    print("\n" + "="*60)
    print("DEMO: Vision (Face Tracking + Active Speaker Detection)")
    print("="*60)
    print("Look at the camera. Green box = face detected.")
    print("Speak to see speaking detection.")
    print("Press ESC to stop\n")
    
    import cv2
    from smait.perception.verifier import SpeakerVerifier
    from smait.sensors.sources import CameraSource
    
    # Initialize
    camera = CameraSource(device=0)
    verifier = SpeakerVerifier()
    
    try:
        camera.start()
        cv2.namedWindow("SMAIT Vision Demo", cv2.WINDOW_NORMAL)
        
        fps_counter = []
        
        while True:
            start = time.time()
            
            ret, frame = camera.read()
            if not ret or frame is None:
                continue
            
            # Process frame
            result = verifier.process_frame(frame)
            
            # Calculate FPS
            fps_counter.append(time.time() - start)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            fps = 1.0 / (sum(fps_counter) / len(fps_counter))
            
            # Draw FPS
            if result['frame'] is not None:
                cv2.putText(
                    result['frame'],
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                cv2.imshow("SMAIT Vision Demo", result['frame'])
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        verifier.cleanup()
        cv2.destroyAllWindows()
        print("\n\nDemo complete.")


def demo_full_system():
    """Run the full HRI system"""
    print("\n" + "="*60)
    print("DEMO: Full HRI System")
    print("="*60)
    
    from smait.main import main
    asyncio.run(main())


def run():
    """Run demo based on command line argument"""
    demos = {
        'audio': demo_audio_pipeline,
        'asr': demo_transcription,
        'dialogue': demo_dialogue,
        'vision': demo_vision,
        'full': demo_full_system,
    }
    
    if len(sys.argv) < 2:
        print("SMAIT HRI System v2.0 - Demo Script")
        print("\nUsage: python -m smait.demo <demo_name>")
        print("\nAvailable demos:")
        print("  audio    - Test audio capture and VAD")
        print("  asr      - Test speech recognition")
        print("  dialogue - Test dialogue manager (text input)")
        print("  vision   - Test face tracking + speaker detection")
        print("  full     - Run full HRI system")
        print("\nExample: python -m smait.demo vision")
        return
    
    demo_name = sys.argv[1].lower()
    
    if demo_name not in demos:
        print(f"Unknown demo: {demo_name}")
        print(f"Available: {', '.join(demos.keys())}")
        return
    
    demos[demo_name]()


if __name__ == "__main__":
    run()
