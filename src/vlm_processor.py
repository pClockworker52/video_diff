import base64
import io
from PIL import Image
import json
import numpy as np
import cv2
import os
from dotenv import load_dotenv
import requests
import time

# Load environment variables
load_dotenv()


class VLMProcessor:
    def __init__(self):
        # Local-only configuration - no API fallback
        self.consecutive_failures = 0
        self.max_failures = 3  # Disable after 3 consecutive failures
        self.is_disabled = False

        self.prompts = self._load_prompts()
        self.local_config = self._load_local_config()

        # Try different endpoints to avoid conversation state issues
        self.chat_endpoint = f"http://{self.local_config['server_host']}:{self.local_config['server_port']}/v1/chat/completions"
        self.completion_endpoint = f"http://{self.local_config['server_host']}:{self.local_config['server_port']}/v1/completions"
        self.model_name = "LFM2-VL-450M-Q8_0"

        self.model = self._initialize_local_model()

    def _load_local_config(self):
        """Load local model configuration"""
        try:
            with open('config/local_model.json', 'r') as f:
                config = json.load(f)
                return config.get('local_model', {})
        except FileNotFoundError:
            return {
                'enabled': False,
                'server_host': 'localhost',
                'server_port': 8000
            }

    def _initialize_local_model(self):
        """Test connection to local llama.cpp server"""
        try:
            test_url = f"http://{self.local_config['server_host']}:{self.local_config['server_port']}/v1/models"
            response = requests.get(test_url, timeout=5)

            if response.status_code == 200:
                print(f"VLM Processor initialized with local model on port {self.local_config['server_port']}")
                return True
            else:
                print(f"Local server connection failed: {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            print("Local llama.cpp server not running. Start it with: python setup_local_model.py")
            return False
        except Exception as e:
            print(f"Failed to connect to local model: {str(e)}")
            return False

    def _load_prompts(self):
        """Load prompts for frame comparison"""
        return {
            "initial": "Describe the objects and content visible in this image.",
            "change": "Compare these two sequential frames and describe what changed in the specified coordinate regions."
        }

    def analyze_initial_frame(self, frame):
        """Analyze the first frame to establish baseline"""
        if self.is_disabled:
            raise RuntimeError("VLM analysis disabled due to consecutive failures")

        # Convert frame to format expected by model
        image_data = self._prepare_image(frame)

        # Call LFM2-VL with initial prompt
        response = self._call_model(
            image_data,
            self.prompts["initial"]
        )

        return response

    def analyze_difference(self, prev_frame, curr_frame, highlights):
        """Analyze differences using full-frame side-by-side comparison"""
        if self.is_disabled:
            raise RuntimeError("VLM analysis disabled due to consecutive failures")

        if not highlights:
            return "No significant changes detected."

        # Use full frames instead of regional extraction
        print(f"VLM: Using full-frame analysis instead of regional extraction")

        # Combine full frames side-by-side for comparison
        combined_frames = self._combine_frames_side_by_side(prev_frame, curr_frame)
        print(f"VLM: Combined frames shape: {combined_frames.shape}")

        # Save the combined image for debugging
        import time
        timestamp = int(time.time())
        debug_path = f"logs/vlm_input_{timestamp}.jpg"
        cv2.imwrite(debug_path, combined_frames)
        print(f"VLM: Saved input image to {debug_path}")

        # Prepare combined frames for model
        combined_frames_data = self._prepare_image(combined_frames)
        print(f"VLM: Prepared combined data length: {len(combined_frames_data)}")

        # Full-frame comparison prompt optimized for office/indoor setting
        comparison_prompt = """You are analyzing two full video frames side-by-side with a white separator.
LEFT = frame from 50 frames ago, RIGHT = current frame.

This is an indoor office/room setting where someone is sitting and may be interacting with objects.

Compare the two frames and describe what changed:
- Did a person's hand or arm move into or out of the frame?
- Was an object (like a mobile phone, book, cup, etc.) picked up, moved, or put down?
- Did someone reach for something or make a gesture?
- Are there any new objects visible that weren't there before?
- Did existing objects change position or orientation?
- Did the person's posture or position change?

Focus on the actual changes you can see between the LEFT and RIGHT frames. This is an indoor office setting, so avoid mentioning cars, outdoor scenes, or multiple people unless you clearly see them.

Be specific about what moved, appeared, or changed between the two frames."""

        # Send combined frames for analysis
        response = self._call_model(combined_frames_data, comparison_prompt)

        return response

    def _extract_region(self, frame, bbox):
        """Extract a specific region from the frame with padding"""
        x, y, w, h = bbox

        # Add 50% padding around the region for more context
        padding_x = int(w * 0.5)
        padding_y = int(h * 0.5)

        # Calculate padded coordinates
        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(frame.shape[1], x + w + padding_x)
        y_end = min(frame.shape[0], y + h + padding_y)

        # Extract the region
        region = frame[y_start:y_end, x_start:x_end]

        # Ensure minimum size (at least 64x64)
        if region.shape[0] < 64 or region.shape[1] < 64:
            region = cv2.resize(region, (max(64, region.shape[1]), max(64, region.shape[0])))

        return region

    def _combine_regions_side_by_side(self, prev_region, curr_region):
        """Combine two regions side-by-side for comparison"""
        import numpy as np

        # Ensure both regions have the same height
        target_height = max(prev_region.shape[0], curr_region.shape[0])

        # Resize regions to same height if needed
        if prev_region.shape[0] != target_height:
            prev_region = cv2.resize(prev_region,
                                   (int(prev_region.shape[1] * target_height / prev_region.shape[0]), target_height))

        if curr_region.shape[0] != target_height:
            curr_region = cv2.resize(curr_region,
                                   (int(curr_region.shape[1] * target_height / curr_region.shape[0]), target_height))

        # Add a white separator line (5 pixels wide)
        separator = np.ones((target_height, 5, 3), dtype=np.uint8) * 255

        # Combine horizontally: prev | separator | curr
        combined = np.hstack([prev_region, separator, curr_region])

        return combined

    def _combine_frames_side_by_side(self, prev_frame, curr_frame):
        """Combine two full frames side-by-side for comparison"""
        import numpy as np

        # Ensure both frames have the same height
        target_height = max(prev_frame.shape[0], curr_frame.shape[0])

        # Resize frames to same height if needed
        if prev_frame.shape[0] != target_height:
            prev_frame = cv2.resize(prev_frame,
                                  (int(prev_frame.shape[1] * target_height / prev_frame.shape[0]), target_height))

        if curr_frame.shape[0] != target_height:
            curr_frame = cv2.resize(curr_frame,
                                  (int(curr_frame.shape[1] * target_height / curr_frame.shape[0]), target_height))

        # Add a white separator line (10 pixels wide for full frames)
        separator = np.ones((target_height, 10, 3), dtype=np.uint8) * 255

        # Combine horizontally: prev | separator | curr
        combined = np.hstack([prev_frame, separator, curr_frame])

        return combined

    def _prepare_image(self, frame):
        """Convert OpenCV frame to model input format optimized for LFM2-VL"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Resize to higher resolution for better detail recognition
        max_size = (512, 512)  # Higher resolution for better detail detection
        pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Use JPEG format for better compatibility with LFM2-VL
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return img_base64

    def _create_coordinate_description(self, highlights):
        """Create coordinate-based description of change regions"""
        if not highlights:
            return "no specific regions"

        descriptions = []
        for i, h in enumerate(highlights, 1):
            x, y, w, h_val = h['bbox']
            center_x, center_y = h['center']

            # Convert to percentage coordinates for better model understanding
            x_pct = int((x / 1280) * 100)  # Assume 1280 width
            y_pct = int((y / 720) * 100)   # Assume 720 height
            w_pct = int((w / 1280) * 100)
            h_pct = int((h_val / 720) * 100)

            descriptions.append(f"Region {i}: x={x_pct}%, y={y_pct}%, width={w_pct}%, height={h_pct}%")

        return "; ".join(descriptions)

    def _get_quadrant(self, x, y, frame_width=1280, frame_height=720):
        """Determine image quadrant for spatial reference"""
        mid_x, mid_y = frame_width // 2, frame_height // 2

        if x < mid_x and y < mid_y:
            return "top-left"
        elif x >= mid_x and y < mid_y:
            return "top-right"
        elif x < mid_x and y >= mid_y:
            return "bottom-left"
        else:
            return "bottom-right"

    def _clear_server_state(self):
        """Force clear server state before each request"""
        try:
            # Send empty completion request to reset state
            clear_payload = {
                "prompt": "",
                "max_tokens": 1,
                "temperature": 0.0,
                "model": self.model_name
            }
            requests.post(
                self.completion_endpoint,
                json=clear_payload,
                headers={"Content-Type": "application/json"},
                timeout=2
            )
        except:
            pass  # Ignore errors in clearing

    def _call_model_with_comparison(self, prev_image_data, curr_image_data, prompt):
        """Call model with two images for comparison"""
        if not self.model:
            raise RuntimeError("VLM model not initialized")

        # Clear server state first
        self._clear_server_state()

        try:
            headers = {"Content-Type": "application/json"}

            # Create conversation with both images
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Previous frame:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{prev_image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Current frame:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{curr_image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": 300,          # Increased tokens for detailed analysis
                "temperature": 0.0,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False,
                "logit_bias": {},
                "n": 1
            }

            response = requests.post(
                self.chat_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    # Reset failure counter on success
                    self.consecutive_failures = 0
                    return result['choices'][0]['message']['content']
                else:
                    raise RuntimeError("Invalid response format from model")
            else:
                self._handle_failure()
                error_msg = f"Model request failed: {response.status_code}"
                if response.text:
                    try:
                        error_detail = response.json()
                        if 'error' in error_detail and 'message' in error_detail['error']:
                            error_msg += f" - {error_detail['error']['message']}"
                        else:
                            error_msg += f" - {response.text}"
                    except:
                        error_msg += f" - {response.text}"
                raise RuntimeError(error_msg)

        except requests.exceptions.Timeout:
            self._handle_failure()
            raise RuntimeError("Local model request timed out")
        except requests.exceptions.ConnectionError:
            self._handle_failure()
            raise RuntimeError("Failed to connect to local llama.cpp server")
        except Exception as e:
            self._handle_failure()
            raise RuntimeError(f"Model call failed: {str(e)}")

    def _call_model(self, image_data, prompt):
        """
        Call local LFM2-VL model via llama.cpp server with stateless approach
        """
        if not self.model:
            raise RuntimeError("VLM model not initialized")

        # Clear server state first
        self._clear_server_state()

        try:
            headers = {"Content-Type": "application/json"}

            # Create completely fresh conversation each time
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 300,          # Increased tokens for detailed descriptions
                "temperature": 0.3,         # Some creativity for varied responses
                "top_p": 0.9,              # Nucleus sampling for variety
                "frequency_penalty": 0.0,   # No frequency penalty
                "presence_penalty": 0.0,    # No presence penalty
                "stream": False,
                "logit_bias": {},          # No token bias
                "n": 1                     # Single completion
            }

            response = requests.post(
                self.chat_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    # Reset failure counter on success
                    self.consecutive_failures = 0
                    return result['choices'][0]['message']['content']
                else:
                    raise RuntimeError("Invalid response format from model")
            else:
                self._handle_failure()
                error_msg = f"Model request failed: {response.status_code}"
                if response.text:
                    try:
                        error_detail = response.json()
                        if 'error' in error_detail and 'message' in error_detail['error']:
                            error_msg += f" - {error_detail['error']['message']}"
                        else:
                            error_msg += f" - {response.text}"
                    except:
                        error_msg += f" - {response.text}"
                raise RuntimeError(error_msg)

        except requests.exceptions.Timeout:
            self._handle_failure()
            raise RuntimeError("Local model request timed out")
        except requests.exceptions.ConnectionError:
            self._handle_failure()
            raise RuntimeError("Failed to connect to local llama.cpp server")
        except Exception as e:
            self._handle_failure()
            raise RuntimeError(f"Model call failed: {str(e)}")

    def _handle_failure(self):
        """Track consecutive failures and disable if threshold reached"""
        self.consecutive_failures += 1

        # Try to clear model cache on first few failures
        if self.consecutive_failures <= 2:
            self._clear_model_cache()

        if self.consecutive_failures >= self.max_failures:
            self.is_disabled = True
            print(f"VLM disabled after {self.consecutive_failures} consecutive failures")

    def _clear_model_cache(self):
        """Attempt to clear the local model's KV cache"""
        try:
            # Send a small request to reset the context
            clear_url = f"http://{self.local_config['server_host']}:{self.local_config['server_port']}/v1/chat/completions"
            clear_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "reset"}],
                "max_tokens": 1,
                "temperature": 0.1
            }

            requests.post(clear_url, json=clear_payload, timeout=5)
            print("Attempted to clear VLM cache")
        except Exception as e:
            print(f"Failed to clear VLM cache: {e}")

    def get_model_status(self):
        """Check if VLM model is available"""
        if self.is_disabled:
            return {
                "available": False,
                "mode": "disabled",
                "reason": f"Auto-disabled after {self.consecutive_failures} consecutive failures"
            }

        return {
            "available": bool(self.model),
            "mode": "local",
            "model_name": self.model_name if self.model else "Not connected",
            "endpoint": self.chat_endpoint if self.model else "None",
            "failures": self.consecutive_failures
        }

    def set_sensitivity(self, sensitivity):
        """Adjust description sensitivity/detail level"""
        # Could be used to modify prompts or model parameters
        pass