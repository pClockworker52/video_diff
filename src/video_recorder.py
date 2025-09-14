import cv2
import numpy as np
import time
from datetime import datetime
import os
from pathlib import Path


class VideoRecorder:
    def __init__(self, output_dir="recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.video_writer = None
        self.output_filename = None
        self.is_recording = False
        self.frame_width = None
        self.frame_height = None
        self.fps = 30

        # Subtitle configuration
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_color = (255, 255, 255)  # White text
        self.font_thickness = 2
        self.background_color = (0, 0, 0)  # Black background
        self.padding = 10
        self.max_line_width = 80  # Maximum characters per line

        # Current subtitle state
        self.current_subtitle = ""
        self.subtitle_timestamp = 0
        self.subtitle_duration = 3.0  # Show subtitle for 3 seconds

    def start_recording(self, frame_width, frame_height, fps=30):
        """Start recording video with given dimensions"""
        if self.is_recording:
            print("Already recording!")
            return False

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps

        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = self.output_dir / f"video_diff_{timestamp}.avi"

        # Try different codecs in order of preference
        codecs_to_try = [
            ('XVID', '.avi'),  # Most compatible
            ('MJPG', '.avi'),  # Motion JPEG
            ('mp4v', '.mp4'),  # MP4 fallback
        ]

        self.video_writer = None
        for codec, ext in codecs_to_try:
            self.output_filename = self.output_dir / f"video_diff_{timestamp}{ext}"
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(
                str(self.output_filename),
                fourcc,
                fps,
                (frame_width, frame_height)
            )

            if self.video_writer.isOpened():
                print(f"Successfully initialized video writer with {codec} codec")
                break
            else:
                print(f"Failed to initialize video writer with {codec} codec")
                self.video_writer.release()

        if not self.video_writer or not self.video_writer.isOpened():
            print(f"Failed to open video writer with any codec")
            return False

        self.is_recording = True
        print(f"Started recording to {self.output_filename}")
        return True

    def stop_recording(self):
        """Stop recording and close video file"""
        if not self.is_recording:
            return

        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        print(f"Recording stopped. Video saved to {self.output_filename}")
        return str(self.output_filename)

    def add_subtitle(self, text, duration=3.0):
        """Add a subtitle to be displayed for the specified duration"""
        self.current_subtitle = self._wrap_text(text)
        self.subtitle_timestamp = time.time()
        self.subtitle_duration = duration

    def write_frame(self, frame, subtitle_text=None):
        """Write a frame to the video with optional subtitle overlay"""
        if not self.is_recording or self.video_writer is None:
            return

        # Make a copy to avoid modifying the original frame
        output_frame = frame.copy()

        # Add subtitle if provided or if there's a current subtitle
        if subtitle_text:
            self.add_subtitle(subtitle_text)

        # Check if we should show current subtitle
        if self.current_subtitle and (time.time() - self.subtitle_timestamp) < self.subtitle_duration:
            output_frame = self._overlay_subtitle(output_frame, self.current_subtitle)
        elif (time.time() - self.subtitle_timestamp) >= self.subtitle_duration:
            # Clear expired subtitle
            self.current_subtitle = ""

        # Write frame to video
        self.video_writer.write(output_frame)

    def _wrap_text(self, text):
        """Wrap text to fit within maximum line width"""
        if len(text) <= self.max_line_width:
            return text

        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + word + " ") <= self.max_line_width:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "

        if current_line:
            lines.append(current_line.strip())

        return "\n".join(lines)

    def _overlay_subtitle(self, frame, subtitle_text):
        """Overlay subtitle text onto the frame with background"""
        if not subtitle_text:
            return frame

        lines = subtitle_text.split('\n')
        line_height = int(self.font_scale * 30)  # Approximate line height
        total_height = len(lines) * line_height + 2 * self.padding

        # Calculate text dimensions and position
        max_width = 0
        for line in lines:
            (text_width, text_height), _ = cv2.getTextSize(
                line, self.font, self.font_scale, self.font_thickness
            )
            max_width = max(max_width, text_width)

        # Position subtitle at bottom of frame
        rect_width = max_width + 2 * self.padding
        rect_height = total_height

        # Center horizontally, position at bottom with margin
        rect_x = (frame.shape[1] - rect_width) // 2
        rect_y = frame.shape[0] - rect_height - 20  # 20px margin from bottom

        # Draw background rectangle with some transparency
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (rect_x, rect_y),
            (rect_x + rect_width, rect_y + rect_height),
            self.background_color,
            -1
        )

        # Blend overlay with original frame for semi-transparency
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw text lines
        for i, line in enumerate(lines):
            text_x = rect_x + self.padding
            text_y = rect_y + self.padding + (i + 1) * line_height - 5

            cv2.putText(
                frame,
                line,
                (text_x, text_y),
                self.font,
                self.font_scale,
                self.font_color,
                self.font_thickness
            )

        return frame

    def get_recording_status(self):
        """Get current recording status"""
        return {
            "is_recording": self.is_recording,
            "output_file": str(self.output_filename) if self.output_filename else None,
            "has_subtitle": bool(self.current_subtitle),
            "subtitle_remaining": max(0, self.subtitle_duration - (time.time() - self.subtitle_timestamp)) if self.current_subtitle else 0
        }