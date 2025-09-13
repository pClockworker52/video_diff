import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
from tkinter import filedialog
import os
import numpy as np


class VideoDiffApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Video-Diff: Real-time Change Detection")
        self.root.geometry("1200x800")

        # Main layout
        self.create_widgets()
        self.is_running = False
        self.camera = None
        self.video_file_path = None
        self.processing_mode = "camera"  # "camera" or "file"

    def create_widgets(self):
        # Main video display area
        self.video_display_frame = ctk.CTkFrame(self.root)
        self.video_display_frame.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # Left panel: Camera/Video feed
        self.video_frame = ctk.CTkLabel(
            self.video_display_frame,
            text="Camera Feed",
            width=640,
            height=480
        )
        self.video_frame.pack(side="left", padx=10, pady=10)

        # Right panel: Difference visualization
        self.diff_frame = ctk.CTkLabel(
            self.video_display_frame,
            text="Difference Map",
            width=640,
            height=480
        )
        self.diff_frame.pack(side="left", padx=10, pady=10)

        # Bottom panel: Controls
        self.control_frame = ctk.CTkFrame(self.root)
        self.control_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        # Mode selection frame
        self.mode_frame = ctk.CTkFrame(self.control_frame)
        self.mode_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.mode_label = ctk.CTkLabel(self.mode_frame, text="Mode:")
        self.mode_label.pack(side="left", padx=5)

        # Create shared variable for radio buttons
        self.mode_var = ctk.StringVar(value="camera")
        print(f"GUI: Created mode variable with default: {self.mode_var.get()}")

        self.camera_radio = ctk.CTkRadioButton(
            self.mode_frame,
            text="Live Camera",
            variable=self.mode_var,
            value="camera",
            command=self.set_camera_mode
        )
        self.camera_radio.pack(side="left", padx=5)

        self.file_radio = ctk.CTkRadioButton(
            self.mode_frame,
            text="Video File",
            variable=self.mode_var,
            value="file",
            command=self.set_file_mode
        )
        self.file_radio.pack(side="left", padx=5)

        # File selection frame
        self.file_frame = ctk.CTkFrame(self.control_frame)
        self.file_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.load_file_button = ctk.CTkButton(
            self.file_frame,
            text="Load Video File",
            command=self.load_video_file,
            state="disabled"
        )
        self.load_file_button.pack(side="left", padx=5)

        self.file_path_label = ctk.CTkLabel(
            self.file_frame,
            text="No file selected"
        )
        self.file_path_label.pack(side="left", padx=10)

        # Control buttons frame
        self.button_frame = ctk.CTkFrame(self.control_frame)
        self.button_frame.pack(side="top", fill="x", padx=5, pady=5)

        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Detection",
            command=self.toggle_detection
        )
        self.start_button.pack(side="left", padx=5)

        # Progress bar for file processing
        self.progress_bar = ctk.CTkProgressBar(self.button_frame)
        self.progress_bar.pack(side="left", padx=10, fill="x", expand=True)
        self.progress_bar.set(0)

        # Description text area
        self.description_text = ctk.CTkTextbox(
            self.control_frame,
            height=100,
            width=600
        )
        self.description_text.pack(side="bottom", padx=10, pady=5)

    def set_camera_mode(self):
        """Switch to camera mode"""
        print(f"GUI: set_camera_mode called, mode_var: {self.mode_var.get()}")
        self.processing_mode = "camera"
        self.load_file_button.configure(state="disabled")
        self.file_path_label.configure(text="No file selected")
        self.video_file_path = None
        self.description_text.insert("end", "Switched to live camera mode\n")
        print(f"GUI: Processing mode set to: {self.processing_mode}")

    def set_file_mode(self):
        """Switch to file processing mode"""
        print(f"GUI: set_file_mode called, mode_var: {self.mode_var.get()}")
        self.processing_mode = "file"
        self.load_file_button.configure(state="normal")
        self.description_text.insert("end", "Switched to video file mode\n")
        print(f"GUI: Processing mode set to: {self.processing_mode}")

    def load_video_file(self):
        """Open file dialog to select video file"""
        file_types = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("All files", "*.*")
        ]

        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=file_types
        )

        if file_path:
            self.video_file_path = file_path
            filename = os.path.basename(file_path)
            self.file_path_label.configure(text=f"Selected: {filename}")
            self.description_text.insert("end", f"Loaded video file: {filename}\n")

            # Get video info
            try:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames / fps if fps > 0 else 0

                    self.description_text.insert("end",
                        f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s duration\n")
                    cap.release()
                else:
                    self.description_text.insert("end", "Error: Could not open video file\n")
                    self.video_file_path = None
                    self.file_path_label.configure(text="Error loading file")
            except Exception as e:
                self.description_text.insert("end", f"Error reading video info: {str(e)}\n")

    def toggle_detection(self):
        """Basic toggle detection - will be overridden by controller"""
        print("GUI: toggle_detection called (should be overridden)")
        if not self.is_running:
            if self.processing_mode == "file" and not self.video_file_path:
                self.description_text.insert("end", "Please select a video file first\n")
                return

            self.is_running = True
            self.start_button.configure(text="Stop Detection")
            self.description_text.insert("end", "Processing started\n")
        else:
            self.is_running = False
            self.start_button.configure(text="Start Detection")
            self.progress_bar.set(0)
            self.description_text.insert("end", "Detection stopped\n")

    def process_video_file(self):
        """This method is handled by the controller"""
        pass

    def update_video_display(self, frame, is_diff=False):
        """Update video display with OpenCV frame"""

        if frame is None:
            print("GUI: Warning - Received None frame")
            return

        try:
            # Resize frame to fit display while preserving aspect ratio
            display_frame = self._resize_with_aspect_ratio(frame, 640, 480)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)

            # Convert to PhotoImage for tkinter
            photo = ImageTk.PhotoImage(pil_image)

            # Update appropriate display
            if is_diff:
                self.diff_frame.configure(image=photo, text="")
                self.diff_frame.image = photo  # Keep a reference
            else:
                self.video_frame.configure(image=photo, text="")
                self.video_frame.image = photo  # Keep a reference

        except Exception as e:
            print(f"GUI: Error updating video display: {e}")
            import traceback
            traceback.print_exc()

    def clear_displays(self):
        """Clear video displays"""
        self.video_frame.configure(image=None, text="Camera Feed")
        self.diff_frame.configure(image=None, text="Difference Map")
        # Clear image references
        self.video_frame.image = None
        self.diff_frame.image = None

    def _resize_with_aspect_ratio(self, frame, max_width, max_height):
        """Resize frame while preserving aspect ratio and center it"""
        height, width = frame.shape[:2]

        # Calculate scaling factor to fit within max dimensions
        scale_width = max_width / width
        scale_height = max_height / height
        scale = min(scale_width, scale_height)

        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Create a black canvas of the target size
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)

        # Calculate position to center the resized frame
        x_offset = (max_width - new_width) // 2
        y_offset = (max_height - new_height) // 2

        # Place the resized frame in the center of the canvas
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

        return canvas


if __name__ == "__main__":
    app = VideoDiffApp()
    app.root.mainloop()