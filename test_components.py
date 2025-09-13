#!/usr/bin/env python3
"""
Test script to verify all components work without GUI
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera_handler import CameraHandler, VideoFileHandler
from diff_engine import DifferenceEngine
from vlm_processor import VLMProcessor
import numpy as np

def test_camera_handler():
    print("Testing Camera Handler...")
    try:
        camera = CameraHandler(0)
        if camera.is_opened:
            print("✅ Camera opened successfully")
            frame = camera.get_frame()
            if frame is not None:
                print(f"✅ Got frame: {frame.shape}")
            else:
                print("⚠️  Camera opened but no frame received")
            camera.release()
        else:
            print("⚠️  No camera available (expected in headless environment)")
    except Exception as e:
        print(f"⚠️  Camera test failed: {e}")

def test_video_file_handler():
    print("\nTesting Video File Handler...")
    try:
        # Test with dummy path (will fail but should handle gracefully)
        handler = VideoFileHandler("nonexistent.mp4")
        print(f"✅ Video handler created, opened: {handler.is_opened}")
    except Exception as e:
        print(f"⚠️  Video handler test failed: {e}")

def test_diff_engine():
    print("\nTesting Difference Engine...")
    try:
        engine = DifferenceEngine()
        print("✅ Difference engine created")

        # Create dummy frames
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        diff_map, highlights = engine.compute_difference(frame1, frame2)
        print(f"✅ Difference computed: {len(highlights)} highlights")

        viz_frame = engine.visualize_difference(frame1, diff_map, highlights)
        print(f"✅ Visualization created: {viz_frame.shape}")

    except Exception as e:
        print(f"❌ Difference engine test failed: {e}")

def test_vlm_processor():
    print("\nTesting VLM Processor...")
    try:
        vlm = VLMProcessor()
        status = vlm.get_model_status()
        print(f"✅ VLM processor created")
        print(f"   Status: {status}")

        if status['available']:
            print("✅ VLM model is available and connected")
        else:
            print("⚠️  VLM model not available (may need server running)")

    except Exception as e:
        print(f"⚠️  VLM processor test failed: {e}")

def main():
    print("Video-Diff Component Tests")
    print("=" * 40)

    test_camera_handler()
    test_video_file_handler()
    test_diff_engine()
    test_vlm_processor()

    print("\n" + "=" * 40)
    print("Component tests completed!")
    print("\nTo run the full application:")
    print("1. Start local model server (if using local mode):")
    print("   python -m llama_cpp.server --model models/LFM2-VL-450M-Q4_0.gguf --port 8000")
    print("2. Run the application:")
    print("   python src/main.py")

if __name__ == "__main__":
    main()