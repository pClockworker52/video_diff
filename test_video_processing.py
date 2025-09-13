#!/usr/bin/env python3
"""
Test video processing pipeline without full GUI
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from camera_handler import VideoFileHandler
from diff_engine import DifferenceEngine
import cv2

print("=== Video Processing Test ===")

# Check if test video exists
test_video_path = "test_video.mp4"
if not os.path.exists(test_video_path):
    print(f"❌ Test video not found: {test_video_path}")
    print("Run: python create_test_video.py first")
    sys.exit(1)

try:
    print(f"1. Loading video file: {test_video_path}")
    video_handler = VideoFileHandler(test_video_path)
    if not video_handler.is_opened:
        print("❌ Could not open video file")
        sys.exit(1)
    print("   ✅ Video file opened successfully")

    print("\n2. Testing frame retrieval...")
    frame = video_handler.get_frame()
    if frame is not None:
        print(f"   ✅ Got first frame: {frame.shape}")
    else:
        print("   ❌ Could not get first frame")
        sys.exit(1)

    print("\n3. Testing frame pairs...")
    # Need to get another frame to have a pair
    frame2 = video_handler.get_frame()
    if frame2 is not None:
        print(f"   ✅ Got second frame: {frame2.shape}")
    else:
        print("   ❌ Could not get second frame")
        sys.exit(1)

    prev_frame, curr_frame = video_handler.get_frame_pair()
    if prev_frame is not None and curr_frame is not None:
        print(f"   ✅ Got frame pair: {prev_frame.shape}, {curr_frame.shape}")
    else:
        print("   ❌ Could not get frame pair")
        sys.exit(1)

    print("\n4. Testing difference engine...")
    diff_engine = DifferenceEngine(sensitivity=0.3)
    diff_map, highlights = diff_engine.compute_difference(prev_frame, curr_frame)
    print(f"   ✅ Computed difference: {len(highlights)} highlights")

    viz_frame = diff_engine.visualize_difference(curr_frame, diff_map, highlights)
    print(f"   ✅ Created visualization: {viz_frame.shape}")

    print("\n5. Testing image conversion (like GUI does)...")
    # Test the same conversion process as the GUI
    display_frame = cv2.resize(curr_frame, (640, 480))
    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    print(f"   ✅ Converted frame for display: {rgb_frame.shape}")

    print("\n6. Testing progress tracking...")
    frame_info = video_handler.get_frame_info()
    print(f"   Progress: {frame_info['progress']:.2%} ({frame_info['current_frame']}/{frame_info['total_frames']})")

    video_handler.release()
    print("\n✅ Video processing test completed successfully!")

except Exception as e:
    print(f"❌ Video processing test failed: {e}")
    import traceback
    traceback.print_exc()