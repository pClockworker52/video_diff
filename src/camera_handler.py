import cv2
import numpy as np
from collections import deque
import threading


class CameraHandler:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame_buffer = deque(maxlen=2)
        self.lock = threading.Lock()
        self.is_opened = self.cap.isOpened()

    def get_frame(self):
        """Get current frame from camera"""
        if not self.is_opened:
            return None

        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.frame_buffer.append(frame)
            return frame
        return None

    def get_frame_pair(self):
        """Returns previous and current frame for comparison"""
        with self.lock:
            if len(self.frame_buffer) == 2:
                return self.frame_buffer[0], self.frame_buffer[1]
        return None, None

    def release(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.is_opened = False


class VideoFileHandler:
    def __init__(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        self.frame_buffer = deque(maxlen=2)
        self.lock = threading.Lock()
        self.is_opened = self.cap.isOpened()
        self.current_frame = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_opened else 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) if self.is_opened else 0

    def get_frame(self):
        """Get next frame from video file"""
        if not self.is_opened:
            return None

        ret, frame = self.cap.read()
        if ret:
            with self.lock:
                self.frame_buffer.append(frame)
                self.current_frame += 1
            return frame
        return None

    def get_frame_pair(self):
        """Returns previous and current frame for comparison"""
        with self.lock:
            if len(self.frame_buffer) == 2:
                return self.frame_buffer[0], self.frame_buffer[1]
        return None, None

    def get_progress(self):
        """Get processing progress as percentage"""
        if self.total_frames > 0:
            return self.current_frame / self.total_frames
        return 0.0

    def get_frame_info(self):
        """Get current frame position info"""
        return {
            'current_frame': self.current_frame,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'progress': self.get_progress()
        }

    def seek_frame(self, frame_number):
        """Seek to specific frame"""
        if self.is_opened and 0 <= frame_number < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            # Clear buffer after seek
            with self.lock:
                self.frame_buffer.clear()

    def release(self):
        """Release video file resources"""
        if self.cap:
            self.cap.release()
            self.is_opened = False