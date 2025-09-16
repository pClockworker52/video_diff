import base64
import io
from PIL import Image
import json
import numpy as np
import cv2
import os
import time
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


class VLMProcessor:
    def __init__(self):
        print("Initializing LFM2-VL with Hugging Face transformers...")

        # Model configuration
        self.model_id = "LiquidAI/LFM2-VL-1.6B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Initialize model and processor
        self._load_model()

        print(f"LFM2-VL initialized successfully on {self.device}")

    def _load_model(self):
        """Load the LFM2-VL model and processor"""
        try:
            print("Loading LFM2-VL processor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )

            print("Loading LFM2-VL model...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                device_map="auto" if torch.cuda.is_available() else None,
                dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Move to device if not using device_map
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)

        except Exception as e:
            raise RuntimeError(f"Failed to load LFM2-VL model: {str(e)}")

    def analyze_initial_frame(self, frame):
        """Analyze the first frame to establish baseline"""
        print("VLM: Analyzing initial frame...")

        # Convert frame to PIL Image
        pil_image = self._frame_to_pil(frame)

        # Create ChatML messages for initial analysis
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that can analyze images in detail."
            },
            {
                "role": "user",
                "content": "<image>\n\nDescribe what you see in this image. Focus on objects, their positions, colors, and spatial arrangement."
            }
        ]

        # Get analysis
        response = self._analyze_with_model(pil_image, messages)
        print(f"VLM: Initial frame analysis: {response[:100]}...")

        return response

    def analyze_difference(self, prev_frame, curr_frame, highlights):
        """Analyze differences using regional side-by-side comparison with real VLM"""
        if not highlights:
            return "No significant changes detected."

        print(f"VLM: Analyzing {len(highlights)} change regions with LFM2-VL...")

        # Process each highlight region with real VLM analysis
        region_descriptions = []
        for i, highlight in enumerate(highlights):
            bbox = highlight['bbox']

            # Extract regions from both frames
            prev_region = self._extract_region(prev_frame, bbox)
            curr_region = self._extract_region(curr_frame, bbox)

            print(f"VLM: Processing region {i+1}: prev={prev_region.shape}, curr={curr_region.shape}")

            # Combine regions side-by-side
            combined_region = self._combine_regions_side_by_side(prev_region, curr_region)

            # Save regional comparison for debugging
            timestamp = int(time.time())
            debug_path = f"logs/vlm_region_{i+1}_{timestamp}.jpg"
            cv2.imwrite(debug_path, combined_region)
            print(f"VLM: Saved region {i+1} to {debug_path}")

            # Get spatial context
            center_x, center_y = highlight['center']
            quadrant = self._get_quadrant(center_x, center_y)

            # Analyze this region with real VLM
            try:
                pil_image = self._frame_to_pil(combined_region)

                # Create regional analysis messages
                messages = [
                    {
                        "role": "system",
                        "content": "You are an AI that specializes in detecting and describing changes between images."
                    },
                    {
                        "role": "user",
                        "content": f"""<image>

This image shows a side-by-side comparison of a specific region where change was detected.
- Left side: previous frame
- Right side: current frame
- Location: {quadrant} quadrant of the original frame
- Region size: {bbox[2]}x{bbox[3]} pixels

Compare the left and right sections and describe:
- What specific objects or elements changed
- How they moved, appeared, disappeared, or transformed
- The nature and direction of any motion

Be specific and concise about what you observe."""
                    }
                ]

                response = self._analyze_with_model(pil_image, messages)
                region_descriptions.append(f"Region {i+1} ({quadrant}): {response}")
                print(f"VLM: Region {i+1} analysis: {response[:60]}...")

            except Exception as e:
                print(f"VLM: Failed to analyze region {i+1}: {e}")
                # If individual region fails, still continue with others
                region_descriptions.append(f"Region {i+1} ({quadrant}): Analysis failed - {str(e)}")

        # Combine all regional descriptions
        if region_descriptions:
            full_response = "; ".join(region_descriptions)
            print(f"VLM: Complete analysis: {full_response[:100]}...")
            return full_response
        else:
            return "Change detection completed but VLM analysis unavailable."

    def _analyze_with_model(self, pil_image, messages):
        """Core method to analyze image with LFM2-VL using proper ChatML format"""
        try:
            # Apply chat template
            prompt = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(pil_image, prompt, return_tensors="pt")

            # Move inputs to device
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.device)
            else:
                for key in inputs:
                    if hasattr(inputs[key], 'to'):
                        inputs[key] = inputs[key].to(self.device)

            # Generate response with recommended parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.1,      # Recommended for vision tasks
                    min_p=0.15,          # Key parameter from docs
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Extract and clean response
            full_response = self.processor.decode(outputs[0], skip_special_tokens=True)
            response = self._extract_assistant_response(full_response)

            return response

        except Exception as e:
            raise RuntimeError(f"Model inference failed: {str(e)}")

    def _extract_assistant_response(self, full_response):
        """Extract just the assistant's response from the full ChatML output"""
        assistant_start = full_response.find("<|im_start|>assistant\n")
        if assistant_start != -1:
            response = full_response[assistant_start + len("<|im_start|>assistant\n"):].strip()
            if response.endswith("<|im_end|>"):
                response = response[:-len("<|im_end|>")].strip()
            return response
        else:
            # Fallback if ChatML markers not found
            return full_response.strip()

    def _frame_to_pil(self, frame):
        """Convert OpenCV frame to PIL Image with proper RGB conversion"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)

        # Ensure RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        return pil_image

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

    def get_model_status(self):
        """Check if VLM model is available and ready"""
        return {
            "available": hasattr(self, 'model') and self.model is not None,
            "mode": "transformers",
            "model_name": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype)
        }

    def set_sensitivity(self, sensitivity):
        """Adjust generation parameters based on sensitivity"""
        # Could be used to modify temperature or other parameters
        pass