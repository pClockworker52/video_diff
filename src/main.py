import asyncio
import threading
import time
import sys
import os

# Add the src directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from gui import VideoDiffApp
from camera_handler import CameraHandler, VideoFileHandler
from diff_engine import DifferenceEngine
from vlm_processor import VLMProcessor
from video_recorder import VideoRecorder


class VideoDiffController:
    def __init__(self):
        print("CONTROLLER: Initializing VideoDiffController")
        self.camera = None
        self.video_handler = None
        print("CONTROLLER: Creating DifferenceEngine")
        self.diff_engine = DifferenceEngine(sensitivity=0.3)
        print("CONTROLLER: Creating VLMProcessor")
        self.vlm = VLMProcessor()
        print("CONTROLLER: Creating VideoRecorder")
        self.recorder = VideoRecorder()
        print("CONTROLLER: Creating GUI")
        self.gui = VideoDiffApp()

        self.is_running = False
        self.baseline_established = False
        self.processing_thread = None
        self.frame_skip_count = 0
        self.frame_skip_interval = 50
        self.vlm_reference_frame = None  # Store frame from N frames ago for VLM comparison

        # Connect GUI callbacks
        print("CONTROLLER: Setting up GUI callbacks")
        self.setup_gui_callbacks()
        print("CONTROLLER: Initialization complete")

    def setup_gui_callbacks(self):
        """Connect GUI events to controller methods"""
        print("CONTROLLER: Setting up callbacks - overriding toggle_detection")

        # Store original method for debugging
        original_method = self.gui.toggle_detection
        print(f"CONTROLLER: Original GUI method: {original_method}")

        # Override the GUI's toggle_detection method
        self.gui.toggle_detection = self.toggle_processing
        print(f"CONTROLLER: New GUI method: {self.gui.toggle_detection}")

        # Also override the button command directly to be sure
        print(f"CONTROLLER: Original button command: {self.gui.start_button.cget('command')}")
        self.gui.start_button.configure(command=self.toggle_processing)
        print(f"CONTROLLER: New button command: {self.gui.start_button.cget('command')}")

        # Set up cleanup on window close
        self.gui.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        print("CONTROLLER: Callbacks setup complete")

    def start(self):
        """Main entry point"""
        self.gui.root.mainloop()

    def toggle_processing(self):
        """Start/stop processing"""
        print("\n=== CONTROLLER: toggle_processing called ===")
        print(f"CONTROLLER: Current is_running state: {self.is_running}")
        print(f"CONTROLLER: Current processing_mode: {self.gui.processing_mode}")
        if not self.is_running:
            # Validation checks
            if self.gui.processing_mode == "file" and not self.gui.video_file_path:
                self.gui.description_text.insert("end", "Please select a video file first\n")
                return

            # Initialize appropriate handler
            if self.gui.processing_mode == "camera":
                self.camera = CameraHandler()
                if not self.camera.is_opened:
                    self.gui.description_text.insert("end", "Error: Could not open camera\n")
                    return
                self.gui.description_text.insert("end", "Camera initialized\n")
            else:
                self.video_handler = VideoFileHandler(self.gui.video_file_path)
                if not self.video_handler.is_opened:
                    self.gui.description_text.insert("end", "Error: Could not open video file\n")
                    return
                self.gui.description_text.insert("end", "Video file loaded\n")

            # Start processing
            self.is_running = True
            self.baseline_established = False
            self.gui.start_button.configure(text="Stop Detection")

            # Auto-start recording when detection starts
            if self.gui.processing_mode == "camera":
                current_frame = self.camera.get_frame()
            else:
                current_frame = self.video_handler.get_frame()

            if current_frame is not None:
                height, width = current_frame.shape[:2]
                success = self.recorder.start_recording(width, height, fps=30)

                if success:
                    output_file = str(self.recorder.output_filename)
                    self.gui.update_recording_status(True, output_file)
                else:
                    self.gui.description_text.insert("end", "Failed to start recording\n")

            print("CONTROLLER: Starting processing thread")
            self.processing_thread = threading.Thread(
                target=self.processing_loop,
                daemon=True
            )
            self.processing_thread.start()
            print(f"CONTROLLER: Thread started: {self.processing_thread.is_alive()}")

        else:
            # Stop processing
            self.is_running = False
            self.gui.start_button.configure(text="Start Detection")
            self.gui.progress_bar.set(0)
            self.gui.clear_displays()

            # Cleanup handlers
            if self.camera:
                self.camera.release()
                self.camera = None
            if self.video_handler:
                self.video_handler.release()
                self.video_handler = None

            # Stop recording if active
            if self.recorder.is_recording:
                output_file = self.recorder.stop_recording()
                self.gui.update_recording_status(False, output_file)

            self.gui.description_text.insert("end", "Detection stopped\n")


    def processing_loop(self):
        """Main processing loop"""
        print("CONTROLLER: Processing loop started")
        while self.is_running:
            try:
                # Get current frame
                if self.gui.processing_mode == "camera":
                    current_frame = self.camera.get_frame()
                    if current_frame is None:
                        time.sleep(0.1)
                        continue
                else:
                    current_frame = self.video_handler.get_frame()
                    if current_frame is None:
                        # End of video file
                        self.gui.description_text.insert("end", "Video processing completed!\n")
                        self.gui.progress_bar.set(1.0)
                        break

                    # Update progress for file processing (thread-safe)
                    progress_info = self.video_handler.get_frame_info()
                    self.gui.root.after(0, lambda p=progress_info['progress']: self.gui.progress_bar.set(p))

                # Update GUI with current frame (thread-safe)
                frame_copy = current_frame.copy()
                self.gui.root.after(0, lambda f=frame_copy: self._update_video_frame(f, False))

                # Establish baseline on first frame
                if not self.baseline_established:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        vlm_status = self.vlm.get_model_status()
                        if vlm_status["available"]:
                            self.gui.description_text.insert("end", f"VLM Status: {vlm_status['mode']} mode with {vlm_status['model_name']}\n")
                            description = self.vlm.analyze_initial_frame(current_frame)

                            # Write baseline to file
                            with open("logs/vlm_output.txt", "a", encoding="utf-8") as f:
                                f.write(f"[{timestamp}] BASELINE: {description}\n")

                            self.gui.description_text.insert("end", f"[BASELINE] {description}\n")
                        else:
                            # Write to file
                            with open("logs/vlm_output.txt", "a", encoding="utf-8") as f:
                                f.write(f"[{timestamp}] VLM not available: {vlm_status}\n")

                            self.gui.description_text.insert("end", f"VLM not available: {vlm_status}\n")
                            self.gui.description_text.insert("end", "[BASELINE] Baseline frame captured - VLM analysis disabled\n")

                        self.gui.description_text.see("end")
                        self.baseline_established = True
                        time.sleep(1)  # Wait before starting difference detection
                        continue

                    except Exception as e:
                        # Write error to file
                        with open("logs/vlm_output.txt", "a", encoding="utf-8") as f:
                            f.write(f"[{timestamp}] VLM baseline error: {str(e)}\n")

                        self.gui.description_text.insert("end", f"VLM baseline error: {str(e)}\n")
                        self.gui.description_text.insert("end", "[BASELINE] Baseline frame captured - continuing with difference detection only\n")
                        self.baseline_established = True  # Continue without VLM

                # Get frame pair for comparison
                if self.gui.processing_mode == "camera":
                    prev_frame, curr_frame = self.camera.get_frame_pair()
                else:
                    prev_frame, curr_frame = self.video_handler.get_frame_pair()

                if prev_frame is None or curr_frame is None:
                    time.sleep(0.1)
                    continue


                # Compute difference
                diff_map, highlights = self.diff_engine.compute_difference(
                    prev_frame,
                    curr_frame
                )

                # Visualize difference (with heatmap for display)
                viz_frame = self.diff_engine.visualize_difference(
                    curr_frame,
                    diff_map,
                    highlights
                )

                # Create clean frame with boxes only for VLM
                boxes_frame = self.diff_engine.get_boxes_only_frame(
                    curr_frame,
                    highlights
                )

                # Update difference visualization (thread-safe)
                viz_copy = viz_frame.copy()
                self.gui.root.after(0, lambda f=viz_copy: self._update_video_frame(f, True))

                # Record frame if recording is active
                if self.recorder.is_recording:
                    # Record the visualization frame (with highlights)
                    self.recorder.write_frame(viz_frame)

                # Analyze changes with VLM (only if significant changes and frame skipping)
                if highlights:
                    change_summary = self.diff_engine.get_change_summary(highlights)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                    # Only process VLM analysis every few frames for more noticeable changes
                    self.frame_skip_count += 1
                    should_analyze_vlm = self.frame_skip_count >= self.frame_skip_interval

                    # Store reference frame at the start of each skip interval
                    if self.frame_skip_count == 1:
                        self.vlm_reference_frame = curr_frame.copy()

                    if should_analyze_vlm:
                        self.frame_skip_count = 0  # Reset counter

                        try:
                            vlm_status = self.vlm.get_model_status()
                            if vlm_status["available"]:
                                # Use the reference frame from N frames ago instead of consecutive frame
                                vlm_prev_frame = self.vlm_reference_frame if self.vlm_reference_frame is not None else prev_frame
                                description = self.vlm.analyze_difference(
                                    vlm_prev_frame,  # Frame from N frames ago
                                    curr_frame,      # Current frame
                                    highlights       # Change coordinates
                                )

                                # Write to file
                                with open("logs/vlm_output.txt", "a", encoding="utf-8") as f:
                                    f.write(f"[{timestamp}] CHANGE: {change_summary} - {description}\n")

                                # Add subtitle to video recording if active
                                if self.recorder.is_recording:
                                    self.recorder.add_subtitle(f"{change_summary} - {description}", duration=5.0)

                                # Also write to GUI
                                self.gui.description_text.insert("end",
                                    f"[CHANGE] {change_summary} - {description}\n")
                            else:
                                # Write to file
                                with open("logs/vlm_output.txt", "a", encoding="utf-8") as f:
                                    f.write(f"[{timestamp}] CHANGE: {change_summary} (VLM offline)\n")

                                self.gui.description_text.insert("end",
                                    f"[CHANGE] {change_summary} (VLM offline)\n")

                            self.gui.description_text.see("end")

                        except RuntimeError as e:
                            # VLM API error - show change detection only
                            with open("logs/vlm_output.txt", "a", encoding="utf-8") as f:
                                f.write(f"[{timestamp}] CHANGE: {change_summary} (VLM error: {str(e)})\n")

                            self.gui.description_text.insert("end",
                                f"[CHANGE] {change_summary} (VLM error: {str(e)})\n")
                            self.gui.description_text.see("end")

                        except Exception as e:
                            # Unexpected error - show change detection only
                            with open("logs/vlm_output.txt", "a", encoding="utf-8") as f:
                                f.write(f"[{timestamp}] CHANGE: {change_summary} (Error: {str(e)})\n")

                            self.gui.description_text.insert("end",
                                f"[CHANGE] {change_summary} (Error: {str(e)})\n")
                            self.gui.description_text.see("end")
                    else:
                        # Show change detection only (no VLM analysis this frame)
                        self.gui.description_text.insert("end",
                            f"[CHANGE] {change_summary}\n")
                        self.gui.description_text.see("end")

                # Control processing rate and ensure GUI updates
                if self.gui.processing_mode == "camera":
                    time.sleep(0.5)  # Adjust based on VLM inference speed
                else:
                    time.sleep(0.1)  # Slower for file processing to see frames

                # Force GUI update
                self.gui.root.update_idletasks()

            except Exception as e:
                self.gui.description_text.insert("end", f"Error in processing loop: {e}\n")
                continue

        # Processing stopped
        self.is_running = False
        self.gui.start_button.configure(text="Start Detection")

    def cleanup(self):
        """Clean shutdown"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)

        if self.camera:
            self.camera.release()
        if self.video_handler:
            self.video_handler.release()

        self.gui.root.destroy()

    def _update_video_frame(self, frame, is_diff):
        """Thread-safe wrapper for GUI video updates"""
        try:
            self.gui.update_video_display(frame, is_diff)
        except Exception as e:
            print(f"Error updating frame: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    app = VideoDiffController()
    app.start()