#!/usr/bin/env python3
"""
Debug script to test controller initialization
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("=== Controller Debug Test ===")

try:
    print("1. Testing imports...")
    from gui import VideoDiffApp
    print("   ✅ GUI imported")

    from camera_handler import CameraHandler, VideoFileHandler
    print("   ✅ Camera handlers imported")

    from diff_engine import DifferenceEngine
    print("   ✅ Diff engine imported")

    from vlm_processor import VLMProcessor
    print("   ✅ VLM processor imported")

    print("\n2. Testing GUI creation...")
    gui = VideoDiffApp()
    print("   ✅ GUI created successfully")
    print(f"   GUI processing_mode: {gui.processing_mode}")
    print(f"   GUI toggle_detection method: {gui.toggle_detection}")

    print("\n3. Testing method override...")
    def test_override():
        print("   TEST OVERRIDE CALLED!")
        return "override_success"

    original_method = gui.toggle_detection
    print(f"   Original method: {original_method}")

    gui.toggle_detection = test_override
    print(f"   New method: {gui.toggle_detection}")

    result = gui.toggle_detection()
    print(f"   Override test result: {result}")

    print("\n4. Testing button command override...")
    gui.start_button.configure(command=test_override)
    button_command = gui.start_button.cget('command')
    print(f"   Button command: {button_command}")

    print("\n✅ Controller debug test completed successfully!")

except Exception as e:
    print(f"❌ Controller debug test failed: {e}")
    import traceback
    traceback.print_exc()