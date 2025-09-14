#!/usr/bin/env python3
"""
Test regional VLM analysis functionality
"""
import numpy as np
import cv2
import sys
import os

# Add the src directory to sys.path
sys.path.insert(0, 'src')

from vlm_processor import VLMProcessor

def create_test_frames():
    """Create test frames with a moving object"""
    # Create base frame (blue background)
    frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 50  # Dark background
    frame2 = frame1.copy()

    # Add a moving red rectangle
    # Frame 1: rectangle at position (100, 100)
    cv2.rectangle(frame1, (100, 100), (200, 150), (0, 0, 255), -1)

    # Frame 2: rectangle moved to position (300, 200)
    cv2.rectangle(frame2, (300, 200), (400, 250), (0, 0, 255), -1)

    return frame1, frame2

def create_test_highlights():
    """Create test highlight regions"""
    highlights = [
        {
            'bbox': (95, 95, 110, 60),  # Original position region
            'center': (150, 125),
            'area': 6600
        },
        {
            'bbox': (295, 195, 110, 60),  # New position region
            'center': (350, 225),
            'area': 6600
        }
    ]
    return highlights

def test_regional_analysis():
    """Test the regional VLM analysis"""
    print("🧪 Testing Regional VLM Analysis")

    # Initialize VLM processor
    vlm = VLMProcessor()
    print(f"VLM Status: {vlm.get_model_status()}")

    # Create test data
    frame1, frame2 = create_test_frames()
    highlights = create_test_highlights()

    print(f"Created frames: {frame1.shape} and {frame2.shape}")
    print(f"Highlights: {len(highlights)} regions")

    # Save test frames for debugging
    cv2.imwrite("debug_images/test_frame1.jpg", frame1)
    cv2.imwrite("debug_images/test_frame2.jpg", frame2)
    print("Saved test frames to debug_images/")

    # Test regional analysis
    try:
        result = vlm.analyze_difference(frame1, frame2, highlights)
        print(f"✅ VLM Analysis Result: {result}")
        return True
    except Exception as e:
        print(f"❌ VLM Analysis Failed: {e}")
        return False

def test_mock_responses():
    """Test mock response generation"""
    print("\n🧪 Testing Mock Response Generation")

    vlm = VLMProcessor()
    # Force disable to test mock responses
    vlm.is_disabled = True

    highlights = create_test_highlights()

    try:
        result = vlm.analyze_difference(None, None, highlights)
        print(f"✅ Mock Response: {result}")
        return True
    except Exception as e:
        print(f"❌ Mock Response Failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Regional VLM Testing ===")

    # Create debug directory if needed
    os.makedirs("debug_images", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Test regional analysis
    success1 = test_regional_analysis()

    # Test mock responses
    success2 = test_mock_responses()

    if success1 and success2:
        print("\n🚀 All tests passed! Regional VLM analysis is working.")
    else:
        print("\n💥 Some tests failed. Check the output above.")