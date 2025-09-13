#!/usr/bin/env python3
"""
Test GUI integration by simulating button clicks
"""
import sys
import os
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import VideoDiffController

def test_controller_integration():
    print("=== GUI Integration Test ===")

    # Check if test video exists
    test_video_path = "test_video.mp4"
    if not os.path.exists(test_video_path):
        print(f"❌ Test video not found: {test_video_path}")
        print("This test will use camera mode (may not work in WSL)")
        use_video = False
    else:
        print(f"✅ Test video found: {test_video_path}")
        use_video = True

    print("\n1. Creating VideoDiffController...")
    controller = VideoDiffController()
    print("   ✅ Controller created")

    def simulate_user_interaction():
        """Simulate user clicking buttons after GUI starts"""
        time.sleep(2)  # Wait for GUI to fully initialize

        print("\n2. Simulating user interaction...")

        if use_video:
            print("   Switching to video file mode...")
            controller.gui.root.after(0, controller.gui.set_file_mode)
            time.sleep(0.5)

            print("   Setting video file path...")
            controller.gui.video_file_path = test_video_path
            controller.gui.root.after(0, lambda: controller.gui.file_path_label.configure(text=f"Selected: {os.path.basename(test_video_path)}"))
            time.sleep(0.5)

        print("   Clicking Start Detection button...")
        controller.gui.root.after(0, controller.toggle_processing)

        # Let it run for a few seconds
        time.sleep(5)

        print("   Stopping detection...")
        controller.gui.root.after(0, controller.toggle_processing)

        # Give it time to stop
        time.sleep(1)

        print("   Closing application...")
        controller.gui.root.after(0, controller.cleanup)

    # Start user simulation in background
    interaction_thread = threading.Thread(target=simulate_user_interaction, daemon=True)
    interaction_thread.start()

    print("\n3. Starting GUI (will auto-test and close)...")
    try:
        controller.start()
        print("✅ GUI integration test completed!")
    except Exception as e:
        print(f"❌ GUI integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_controller_integration()