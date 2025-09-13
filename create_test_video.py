#!/usr/bin/env python3
"""
Create a simple test video with moving objects for testing
"""
import cv2
import numpy as np
import os

def create_test_video(output_path="test_video.mp4", duration_seconds=10, fps=30):
    """Create a test video with moving objects"""

    # Video properties
    width, height = 640, 480
    total_frames = duration_seconds * fps

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration_seconds}s, FPS: {fps}, Frames: {total_frames}")

    for frame_num in range(total_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray background

        # Add some static elements
        cv2.rectangle(frame, (50, 50), (150, 100), (100, 100, 100), -1)  # Static rectangle
        cv2.circle(frame, (500, 400), 30, (80, 80, 80), -1)  # Static circle

        # Add moving elements
        # Moving circle
        circle_x = int(100 + (frame_num * 2) % (width - 200))
        circle_y = int(200 + 50 * np.sin(frame_num * 0.1))
        cv2.circle(frame, (circle_x, circle_y), 20, (0, 255, 0), -1)  # Green moving circle

        # Moving rectangle
        rect_x = int(50 + (frame_num * 1.5) % (width - 150))
        rect_y = 300
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 50, rect_y + 30), (255, 0, 0), -1)  # Blue rectangle

        # Appearing/disappearing object (every 2 seconds)
        if (frame_num // (fps * 2)) % 2 == 0:
            cv2.rectangle(frame, (300, 100), (350, 150), (0, 0, 255), -1)  # Red rectangle

        # Add frame number text
        cv2.putText(frame, f"Frame: {frame_num:04d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add timestamp
        time_sec = frame_num / fps
        cv2.putText(frame, f"Time: {time_sec:.1f}s", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        out.write(frame)

        # Progress indicator
        if frame_num % (fps * 2) == 0:  # Every 2 seconds
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")

    out.release()
    print(f"✅ Test video created: {output_path}")
    print(f"   Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    return output_path

if __name__ == "__main__":
    # Create test video
    video_path = create_test_video("test_video.mp4", duration_seconds=15, fps=10)

    print(f"\n🎬 Test video ready!")
    print(f"   Location: {os.path.abspath(video_path)}")
    print(f"\nTo test:")
    print(f"1. Run: python src/main.py")
    print(f"2. Select 'Video File' mode")
    print(f"3. Load: {video_path}")
    print(f"4. Click 'Start Detection'")