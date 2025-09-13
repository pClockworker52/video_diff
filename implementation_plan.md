\# Video-Diff Application Implementation Plan

\*\*Windows 11 Laptop Application for Liquid AI Hackathon\*\*



\## 🎯 Project Overview

Build a real-time change detection system that uses LFM2-VL to narrate scene changes captured via webcam.



\## 🏗️ Architecture Overview



```

┌─────────────┐     ┌──────────────┐     ┌─────────────┐

│   Webcam    │────▶│ Frame Buffer │────▶│  Diff Engine│

└─────────────┘     └──────────────┘     └─────────────┘

&nbsp;                                                │

&nbsp;                                                ▼

┌─────────────┐     ┌──────────────┐     ┌─────────────┐

│Video Output │◀────│   LFM2-VL    │◀────│  Highlight  │

│ + Subtitles │     │   Analysis   │     │    Mask     │

└─────────────┘     └──────────────┘     └─────────────┘

&nbsp;                           │

&nbsp;                           ▼

&nbsp;                   ┌──────────────┐

&nbsp;                   │   Log File   │

&nbsp;                   └──────────────┘

```



\## 📦 Tech Stack



\### Core Dependencies

```python

\# requirements.txt

opencv-python==4.8.1          # Webcam capture \& image processing

numpy==1.24.3                 # Array operations for diff calculations

Pillow==10.0.0                # Image manipulation

scikit-image==0.21.0          # Advanced image comparison

imageio-ffmpeg==0.4.9         # Video recording

liquid-sdk                     # Liquid AI's SDK (check actual name)

customtkinter==5.2.0          # Modern GUI

python-dotenv==1.0.0          # Environment variables

```



\## 📋 Implementation Steps



\### Phase 1: Basic Setup \& GUI (Day 1, Morning)



\*\*1. Project Structure\*\*

```

video-diff/

├── src/

│   ├── main.py              # Entry point

│   ├── camera\_handler.py    # Webcam management

│   ├── diff\_engine.py       # Difference detection

│   ├── vlm\_processor.py     # LFM2-VL integration

│   ├── video\_recorder.py    # Recording with subtitles

│   └── gui.py               # User interface

├── config/

│   ├── settings.json        # Configuration

│   └── prompts.json         # VLM prompts

├── logs/                    # Output logs

├── recordings/              # Saved videos

└── README.md

```



\*\*2. Basic GUI (gui.py)\*\*

```python

import customtkinter as ctk

import cv2

from PIL import Image, ImageTk

import threading



class VideoDiffApp:

&nbsp;   def \_\_init\_\_(self):

&nbsp;       self.root = ctk.CTk()

&nbsp;       self.root.title("Video-Diff: Real-time Change Detection")

&nbsp;       self.root.geometry("1200x800")

&nbsp;       

&nbsp;       # Main layout

&nbsp;       self.create\_widgets()

&nbsp;       self.is\_running = False

&nbsp;       self.camera = None

&nbsp;       

&nbsp;   def create\_widgets(self):

&nbsp;       # Left panel: Camera feed

&nbsp;       self.video\_frame = ctk.CTkLabel(self.root, text="Camera Feed")

&nbsp;       self.video\_frame.pack(side="left", padx=10, pady=10)

&nbsp;       

&nbsp;       # Right panel: Difference visualization

&nbsp;       self.diff\_frame = ctk.CTkLabel(self.root, text="Difference Map")

&nbsp;       self.diff\_frame.pack(side="left", padx=10, pady=10)

&nbsp;       

&nbsp;       # Bottom panel: Controls

&nbsp;       self.control\_frame = ctk.CTkFrame(self.root)

&nbsp;       self.control\_frame.pack(side="bottom", fill="x", padx=10, pady=10)

&nbsp;       

&nbsp;       self.start\_button = ctk.CTkButton(

&nbsp;           self.control\_frame, 

&nbsp;           text="Start Detection",

&nbsp;           command=self.toggle\_detection

&nbsp;       )

&nbsp;       self.start\_button.pack(side="left", padx=5)

&nbsp;       

&nbsp;       # Description text area

&nbsp;       self.description\_text = ctk.CTkTextbox(

&nbsp;           self.control\_frame,

&nbsp;           height=100,

&nbsp;           width=600

&nbsp;       )

&nbsp;       self.description\_text.pack(side="left", padx=10)

```



\### Phase 2: Camera \& Difference Detection (Day 1, Afternoon)



\*\*3. Camera Handler (camera\_handler.py)\*\*

```python

import cv2

import numpy as np

from collections import deque

import threading



class CameraHandler:

&nbsp;   def \_\_init\_\_(self, camera\_index=0):

&nbsp;       self.cap = cv2.VideoCapture(camera\_index)

&nbsp;       self.cap.set(cv2.CAP\_PROP\_FRAME\_WIDTH, 1280)

&nbsp;       self.cap.set(cv2.CAP\_PROP\_FRAME\_HEIGHT, 720)

&nbsp;       self.frame\_buffer = deque(maxlen=2)

&nbsp;       self.lock = threading.Lock()

&nbsp;       

&nbsp;   def get\_frame(self):

&nbsp;       ret, frame = self.cap.read()

&nbsp;       if ret:

&nbsp;           with self.lock:

&nbsp;               self.frame\_buffer.append(frame)

&nbsp;           return frame

&nbsp;       return None

&nbsp;   

&nbsp;   def get\_frame\_pair(self):

&nbsp;       """Returns previous and current frame for comparison"""

&nbsp;       with self.lock:

&nbsp;           if len(self.frame\_buffer) == 2:

&nbsp;               return self.frame\_buffer\[0], self.frame\_buffer\[1]

&nbsp;       return None, None

```



\*\*4. Advanced Difference Engine (diff\_engine.py)\*\*

```python

import cv2

import numpy as np

from skimage.metrics import structural\_similarity as ssim

from scipy import ndimage



class DifferenceEngine:

&nbsp;   def \_\_init\_\_(self, sensitivity=0.3):

&nbsp;       self.sensitivity = sensitivity

&nbsp;       self.background\_subtractor = cv2.createBackgroundSubtractorMOG2()

&nbsp;       

&nbsp;   def compute\_difference(self, frame1, frame2):

&nbsp;       """

&nbsp;       Multi-method difference detection for robustness

&nbsp;       """

&nbsp;       # Method 1: Structural Similarity Index

&nbsp;       diff\_map\_ssim = self.\_ssim\_difference(frame1, frame2)

&nbsp;       

&nbsp;       # Method 2: Absolute difference with threshold

&nbsp;       diff\_map\_abs = self.\_absolute\_difference(frame1, frame2)

&nbsp;       

&nbsp;       # Method 3: Background subtraction (for moving objects)

&nbsp;       diff\_map\_bg = self.\_background\_subtraction(frame2)

&nbsp;       

&nbsp;       # Combine methods with weighted average

&nbsp;       combined\_diff = self.\_combine\_methods(

&nbsp;           diff\_map\_ssim, 

&nbsp;           diff\_map\_abs, 

&nbsp;           diff\_map\_bg

&nbsp;       )

&nbsp;       

&nbsp;       # Generate highlight regions

&nbsp;       highlights = self.\_generate\_highlights(combined\_diff)

&nbsp;       

&nbsp;       return combined\_diff, highlights

&nbsp;   

&nbsp;   def \_ssim\_difference(self, frame1, frame2):

&nbsp;       """Structural similarity approach"""

&nbsp;       gray1 = cv2.cvtColor(frame1, cv2.COLOR\_BGR2GRAY)

&nbsp;       gray2 = cv2.cvtColor(frame2, cv2.COLOR\_BGR2GRAY)

&nbsp;       

&nbsp;       # Compute SSIM

&nbsp;       score, diff = ssim(gray1, gray2, full=True)

&nbsp;       diff = (diff \* 255).astype("uint8")

&nbsp;       

&nbsp;       # Invert (SSIM gives similarity, we want difference)

&nbsp;       diff = 255 - diff

&nbsp;       

&nbsp;       return diff

&nbsp;   

&nbsp;   def \_absolute\_difference(self, frame1, frame2):

&nbsp;       """Simple absolute difference with noise reduction"""

&nbsp;       # Convert to grayscale

&nbsp;       gray1 = cv2.cvtColor(frame1, cv2.COLOR\_BGR2GRAY)

&nbsp;       gray2 = cv2.cvtColor(frame2, cv2.COLOR\_BGR2GRAY)

&nbsp;       

&nbsp;       # Apply Gaussian blur to reduce noise

&nbsp;       gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)

&nbsp;       gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)

&nbsp;       

&nbsp;       # Compute absolute difference

&nbsp;       diff = cv2.absdiff(gray1, gray2)

&nbsp;       

&nbsp;       # Threshold to binary

&nbsp;       \_, thresh = cv2.threshold(

&nbsp;           diff, 

&nbsp;           int(255 \* self.sensitivity), 

&nbsp;           255, 

&nbsp;           cv2.THRESH\_BINARY

&nbsp;       )

&nbsp;       

&nbsp;       return thresh

&nbsp;   

&nbsp;   def \_background\_subtraction(self, frame):

&nbsp;       """MOG2 background subtraction for movement"""

&nbsp;       return self.background\_subtractor.apply(frame)

&nbsp;   

&nbsp;   def \_combine\_methods(self, diff1, diff2, diff3):

&nbsp;       """Weighted combination of difference methods"""

&nbsp;       # Normalize all to same shape if needed

&nbsp;       h, w = diff1.shape\[:2]

&nbsp;       diff2 = cv2.resize(diff2, (w, h))

&nbsp;       diff3 = cv2.resize(diff3, (w, h))

&nbsp;       

&nbsp;       # Weighted average (adjust weights based on testing)

&nbsp;       combined = (

&nbsp;           0.4 \* diff1.astype(float) + 

&nbsp;           0.3 \* diff2.astype(float) + 

&nbsp;           0.3 \* diff3.astype(float)

&nbsp;       )

&nbsp;       

&nbsp;       return combined.astype(np.uint8)

&nbsp;   

&nbsp;   def \_generate\_highlights(self, diff\_map):

&nbsp;       """Generate bounding boxes around changed regions"""

&nbsp;       # Apply morphological operations to connect regions

&nbsp;       kernel = np.ones((5,5), np.uint8)

&nbsp;       diff\_map = cv2.morphologyEx(diff\_map, cv2.MORPH\_CLOSE, kernel)

&nbsp;       diff\_map = cv2.morphologyEx(diff\_map, cv2.MORPH\_OPEN, kernel)

&nbsp;       

&nbsp;       # Find contours

&nbsp;       contours, \_ = cv2.findContours(

&nbsp;           diff\_map, 

&nbsp;           cv2.RETR\_EXTERNAL, 

&nbsp;           cv2.CHAIN\_APPROX\_SIMPLE

&nbsp;       )

&nbsp;       

&nbsp;       # Filter small contours and create bounding boxes

&nbsp;       highlights = \[]

&nbsp;       min\_area = 500  # Minimum area threshold

&nbsp;       

&nbsp;       for contour in contours:

&nbsp;           area = cv2.contourArea(contour)

&nbsp;           if area > min\_area:

&nbsp;               x, y, w, h = cv2.boundingRect(contour)

&nbsp;               highlights.append({

&nbsp;                   'bbox': (x, y, w, h),

&nbsp;                   'area': area,

&nbsp;                   'center': (x + w//2, y + h//2)

&nbsp;               })

&nbsp;       

&nbsp;       return highlights

&nbsp;   

&nbsp;   def visualize\_difference(self, original\_frame, diff\_map, highlights):

&nbsp;       """Create visualization with highlights"""

&nbsp;       # Create colored overlay

&nbsp;       overlay = original\_frame.copy()

&nbsp;       

&nbsp;       # Convert diff\_map to color

&nbsp;       diff\_color = cv2.applyColorMap(diff\_map, cv2.COLORMAP\_JET)

&nbsp;       

&nbsp;       # Blend with original

&nbsp;       result = cv2.addWeighted(overlay, 0.7, diff\_color, 0.3, 0)

&nbsp;       

&nbsp;       # Draw bounding boxes

&nbsp;       for highlight in highlights:

&nbsp;           x, y, w, h = highlight\['bbox']

&nbsp;           cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

&nbsp;           

&nbsp;           # Add change indicator

&nbsp;           cv2.putText(

&nbsp;               result, 

&nbsp;               "CHANGE", 

&nbsp;               (x, y-10),

&nbsp;               cv2.FONT\_HERSHEY\_SIMPLEX, 

&nbsp;               0.5, 

&nbsp;               (0, 255, 0), 

&nbsp;               2

&nbsp;           )

&nbsp;       

&nbsp;       return result

```



\### Phase 3: LFM2-VL Integration (Day 1, Evening)



\*\*5. VLM Processor (vlm\_processor.py)\*\*

```python

import base64

import io

from PIL import Image

import json

import numpy as np



class VLMProcessor:

&nbsp;   def \_\_init\_\_(self, api\_key=None):

&nbsp;       # Initialize Liquid AI SDK

&nbsp;       # Note: Actual SDK initialization will depend on Liquid AI's API

&nbsp;       self.model = self.\_initialize\_lfm2\_vl(api\_key)

&nbsp;       self.prompts = self.\_load\_prompts()

&nbsp;       

&nbsp;   def \_initialize\_lfm2\_vl(self, api\_key):

&nbsp;       """

&nbsp;       Initialize LFM2-VL model

&nbsp;       Replace with actual Liquid AI SDK initialization

&nbsp;       """

&nbsp;       # Example (adjust based on actual SDK):

&nbsp;       # from liquid import LFM2VL

&nbsp;       # return LFM2VL(api\_key=api\_key)

&nbsp;       pass

&nbsp;   

&nbsp;   def \_load\_prompts(self):

&nbsp;       """Load optimized prompts for difference description"""

&nbsp;       return {

&nbsp;           "initial": """Describe what you see in this image. 

&nbsp;                        Be concise and focus on the main elements.""",

&nbsp;           

&nbsp;           "difference": """You are comparing two frames. 

&nbsp;                          The highlighted regions show where changes occurred.

&nbsp;                          Describe ONLY what has changed between the frames.

&nbsp;                          Focus on:

&nbsp;                          1. Objects that appeared or disappeared

&nbsp;                          2. Movement of existing objects

&nbsp;                          3. Changes in lighting or shadows

&nbsp;                          4. Any human activity

&nbsp;                          Be specific and concise.""",

&nbsp;           

&nbsp;           "focused": """The green boxes highlight areas of change.

&nbsp;                       Describe ONLY what is happening in these highlighted regions.

&nbsp;                       Ignore everything else in the image."""

&nbsp;       }

&nbsp;   

&nbsp;   def analyze\_initial\_frame(self, frame):

&nbsp;       """Analyze the first frame to establish baseline"""

&nbsp;       # Convert frame to format expected by model

&nbsp;       image\_data = self.\_prepare\_image(frame)

&nbsp;       

&nbsp;       # Call LFM2-VL with initial prompt

&nbsp;       response = self.\_call\_model(

&nbsp;           image\_data, 

&nbsp;           self.prompts\["initial"]

&nbsp;       )

&nbsp;       

&nbsp;       return response

&nbsp;   

&nbsp;   def analyze\_difference(self, frame\_with\_highlights, highlights):

&nbsp;       """Analyze only the differences"""

&nbsp;       if not highlights:

&nbsp;           return "No significant changes detected."

&nbsp;       

&nbsp;       # Prepare image with highlights

&nbsp;       image\_data = self.\_prepare\_image(frame\_with\_highlights)

&nbsp;       

&nbsp;       # Create context about highlight locations

&nbsp;       highlight\_context = self.\_create\_highlight\_context(highlights)

&nbsp;       

&nbsp;       # Enhanced prompt with spatial information

&nbsp;       prompt = f"""{self.prompts\["focused"]}

&nbsp;                   Change regions detected at: {highlight\_context}"""

&nbsp;       

&nbsp;       response = self.\_call\_model(image\_data, prompt)

&nbsp;       

&nbsp;       return response

&nbsp;   

&nbsp;   def \_prepare\_image(self, frame):

&nbsp;       """Convert OpenCV frame to model input format"""

&nbsp;       # Convert BGR to RGB

&nbsp;       rgb\_frame = cv2.cvtColor(frame, cv2.COLOR\_BGR2RGB)

&nbsp;       

&nbsp;       # Convert to PIL Image

&nbsp;       pil\_image = Image.fromarray(rgb\_frame)

&nbsp;       

&nbsp;       # Resize if needed (check model requirements)

&nbsp;       max\_size = (1024, 1024)

&nbsp;       pil\_image.thumbnail(max\_size, Image.Resampling.LANCZOS)

&nbsp;       

&nbsp;       # Convert to base64 (typical for API calls)

&nbsp;       buffered = io.BytesIO()

&nbsp;       pil\_image.save(buffered, format="PNG")

&nbsp;       img\_base64 = base64.b64encode(buffered.getvalue()).decode()

&nbsp;       

&nbsp;       return img\_base64

&nbsp;   

&nbsp;   def \_create\_highlight\_context(self, highlights):

&nbsp;       """Create spatial description of changes"""

&nbsp;       if not highlights:

&nbsp;           return "no specific regions"

&nbsp;       

&nbsp;       # Sort by area (largest first)

&nbsp;       highlights = sorted(highlights, key=lambda x: x\['area'], reverse=True)

&nbsp;       

&nbsp;       # Describe top 3 largest changes

&nbsp;       descriptions = \[]

&nbsp;       for h in highlights\[:3]:

&nbsp;           x, y, w, h\_val = h\['bbox']

&nbsp;           center\_x, center\_y = h\['center']

&nbsp;           

&nbsp;           # Determine quadrant

&nbsp;           quadrant = self.\_get\_quadrant(center\_x, center\_y)

&nbsp;           descriptions.append(f"{quadrant} region")

&nbsp;       

&nbsp;       return ", ".join(descriptions)

&nbsp;   

&nbsp;   def \_get\_quadrant(self, x, y, frame\_width=1280, frame\_height=720):

&nbsp;       """Determine image quadrant for spatial reference"""

&nbsp;       mid\_x, mid\_y = frame\_width // 2, frame\_height // 2

&nbsp;       

&nbsp;       if x < mid\_x and y < mid\_y:

&nbsp;           return "top-left"

&nbsp;       elif x >= mid\_x and y < mid\_y:

&nbsp;           return "top-right"

&nbsp;       elif x < mid\_x and y >= mid\_y:

&nbsp;           return "bottom-left"

&nbsp;       else:

&nbsp;           return "bottom-right"

&nbsp;   

&nbsp;   def \_call\_model(self, image\_data, prompt):

&nbsp;       """

&nbsp;       Call LFM2-VL model

&nbsp;       Replace with actual API call

&nbsp;       """

&nbsp;       # Placeholder - replace with actual Liquid AI API call

&nbsp;       # response = self.model.generate(

&nbsp;       #     image=image\_data,

&nbsp;       #     prompt=prompt,

&nbsp;       #     max\_tokens=100,

&nbsp;       #     temperature=0.7

&nbsp;       # )

&nbsp;       # return response.text

&nbsp;       

&nbsp;       # Temporary mock response

&nbsp;       return "Movement detected in the highlighted region."

```



\### Phase 4: Recording \& Logging (Day 2, Morning)



\*\*6. Video Recorder with Subtitles (video\_recorder.py)\*\*

```python

import cv2

import datetime

import json

from collections import deque

import threading



class VideoRecorder:

&nbsp;   def \_\_init\_\_(self, output\_dir="recordings"):

&nbsp;       self.output\_dir = output\_dir

&nbsp;       self.is\_recording = False

&nbsp;       self.video\_writer = None

&nbsp;       self.subtitle\_buffer = deque(maxlen=3)  # Keep last 3 descriptions

&nbsp;       self.log\_file = None

&nbsp;       

&nbsp;   def start\_recording(self, frame\_width, frame\_height, fps=10):

&nbsp;       """Start recording with timestamp"""

&nbsp;       timestamp = datetime.datetime.now().strftime("%Y%m%d\_%H%M%S")

&nbsp;       

&nbsp;       # Video file

&nbsp;       video\_path = f"{self.output\_dir}/video\_diff\_{timestamp}.mp4"

&nbsp;       fourcc = cv2.VideoWriter\_fourcc(\*'mp4v')

&nbsp;       self.video\_writer = cv2.VideoWriter(

&nbsp;           video\_path, 

&nbsp;           fourcc, 

&nbsp;           fps, 

&nbsp;           (frame\_width, frame\_height)

&nbsp;       )

&nbsp;       

&nbsp;       # Log file

&nbsp;       log\_path = f"logs/session\_{timestamp}.jsonl"

&nbsp;       self.log\_file = open(log\_path, 'w')

&nbsp;       

&nbsp;       self.is\_recording = True

&nbsp;       

&nbsp;   def add\_frame\_with\_subtitle(self, frame, description):

&nbsp;       """Add frame with burned-in subtitle"""

&nbsp;       if not self.is\_recording:

&nbsp;           return

&nbsp;       

&nbsp;       # Add to subtitle buffer

&nbsp;       self.subtitle\_buffer.append({

&nbsp;           'time': datetime.datetime.now().isoformat(),

&nbsp;           'text': description

&nbsp;       })

&nbsp;       

&nbsp;       # Create frame with subtitles

&nbsp;       frame\_with\_subtitle = self.\_add\_subtitle\_overlay(frame)

&nbsp;       

&nbsp;       # Write to video

&nbsp;       self.video\_writer.write(frame\_with\_subtitle)

&nbsp;       

&nbsp;       # Log to file

&nbsp;       self.\_log\_event(description)

&nbsp;   

&nbsp;   def \_add\_subtitle\_overlay(self, frame):

&nbsp;       """Add semi-transparent subtitle bar"""

&nbsp;       frame\_copy = frame.copy()

&nbsp;       height, width = frame.shape\[:2]

&nbsp;       

&nbsp;       # Create subtitle background

&nbsp;       overlay = frame\_copy.copy()

&nbsp;       cv2.rectangle(

&nbsp;           overlay,

&nbsp;           (0, height - 100),

&nbsp;           (width, height),

&nbsp;           (0, 0, 0),

&nbsp;           -1

&nbsp;       )

&nbsp;       

&nbsp;       # Blend with transparency

&nbsp;       frame\_copy = cv2.addWeighted(frame\_copy, 0.7, overlay, 0.3, 0)

&nbsp;       

&nbsp;       # Add text

&nbsp;       if self.subtitle\_buffer:

&nbsp;           latest = self.subtitle\_buffer\[-1]

&nbsp;           text = latest\['text']\[:100]  # Truncate if too long

&nbsp;           

&nbsp;           # Word wrap

&nbsp;           words = text.split()

&nbsp;           lines = \[]

&nbsp;           current\_line = \[]

&nbsp;           

&nbsp;           for word in words:

&nbsp;               current\_line.append(word)

&nbsp;               test\_line = ' '.join(current\_line)

&nbsp;               (text\_width, \_), \_ = cv2.getTextSize(

&nbsp;                   test\_line,

&nbsp;                   cv2.FONT\_HERSHEY\_SIMPLEX,

&nbsp;                   0.6,

&nbsp;                   2

&nbsp;               )

&nbsp;               if text\_width > width - 40:

&nbsp;                   lines.append(' '.join(current\_line\[:-1]))

&nbsp;                   current\_line = \[word]

&nbsp;           

&nbsp;           lines.append(' '.join(current\_line))

&nbsp;           

&nbsp;           # Draw lines

&nbsp;           y\_offset = height - 70

&nbsp;           for line in lines\[:2]:  # Max 2 lines

&nbsp;               cv2.putText(

&nbsp;                   frame\_copy,

&nbsp;                   line,

&nbsp;                   (20, y\_offset),

&nbsp;                   cv2.FONT\_HERSHEY\_SIMPLEX,

&nbsp;                   0.6,

&nbsp;                   (255, 255, 255),

&nbsp;                   2,

&nbsp;                   cv2.LINE\_AA

&nbsp;               )

&nbsp;               y\_offset += 30

&nbsp;       

&nbsp;       return frame\_copy

&nbsp;   

&nbsp;   def \_log\_event(self, description):

&nbsp;       """Log event to JSONL file"""

&nbsp;       if self.log\_file:

&nbsp;           event = {

&nbsp;               'timestamp': datetime.datetime.now().isoformat(),

&nbsp;               'description': description,

&nbsp;               'frame\_number': getattr(self, 'frame\_count', 0)

&nbsp;           }

&nbsp;           self.log\_file.write(json.dumps(event) + '\\n')

&nbsp;           self.log\_file.flush()

```



\### Phase 5: Main Application Loop (Day 2, Afternoon)



\*\*7. Main Application (main.py)\*\*

```python

import asyncio

import threading

import time

from src.gui import VideoDiffApp

from src.camera\_handler import CameraHandler

from src.diff\_engine import DifferenceEngine

from src.vlm\_processor import VLMProcessor

from src.video\_recorder import VideoRecorder



class VideoDiffController:

&nbsp;   def \_\_init\_\_(self):

&nbsp;       self.camera = CameraHandler()

&nbsp;       self.diff\_engine = DifferenceEngine(sensitivity=0.3)

&nbsp;       self.vlm = VLMProcessor()

&nbsp;       self.recorder = VideoRecorder()

&nbsp;       self.gui = VideoDiffApp()

&nbsp;       

&nbsp;       self.is\_running = False

&nbsp;       self.baseline\_established = False

&nbsp;       self.processing\_thread = None

&nbsp;       

&nbsp;   def start(self):

&nbsp;       """Main entry point"""

&nbsp;       self.gui.start\_button.configure(command=self.toggle\_processing)

&nbsp;       self.gui.root.protocol("WM\_DELETE\_WINDOW", self.cleanup)

&nbsp;       self.gui.root.mainloop()

&nbsp;   

&nbsp;   def toggle\_processing(self):

&nbsp;       """Start/stop processing"""

&nbsp;       if not self.is\_running:

&nbsp;           self.is\_running = True

&nbsp;           self.gui.start\_button.configure(text="Stop Detection")

&nbsp;           self.processing\_thread = threading.Thread(

&nbsp;               target=self.processing\_loop,

&nbsp;               daemon=True

&nbsp;           )

&nbsp;           self.processing\_thread.start()

&nbsp;           self.recorder.start\_recording(1280, 720)

&nbsp;       else:

&nbsp;           self.is\_running = False

&nbsp;           self.gui.start\_button.configure(text="Start Detection")

&nbsp;           self.recorder.stop\_recording()

&nbsp;   

&nbsp;   def processing\_loop(self):

&nbsp;       """Main processing loop"""

&nbsp;       while self.is\_running:

&nbsp;           try:

&nbsp;               # Get current frame

&nbsp;               current\_frame = self.camera.get\_frame()

&nbsp;               if current\_frame is None:

&nbsp;                   continue

&nbsp;               

&nbsp;               # Update GUI with current frame

&nbsp;               self.update\_gui\_frame(current\_frame)

&nbsp;               

&nbsp;               # Establish baseline on first frame

&nbsp;               if not self.baseline\_established:

&nbsp;                   description = self.vlm.analyze\_initial\_frame(current\_frame)

&nbsp;                   self.update\_description(f"\[BASELINE] {description}")

&nbsp;                   self.baseline\_established = True

&nbsp;                   time.sleep(2)  # Wait before starting difference detection

&nbsp;                   continue

&nbsp;               

&nbsp;               # Get frame pair for comparison

&nbsp;               prev\_frame, curr\_frame = self.camera.get\_frame\_pair()

&nbsp;               if prev\_frame is None or curr\_frame is None:

&nbsp;                   continue

&nbsp;               

&nbsp;               # Compute difference

&nbsp;               diff\_map, highlights = self.diff\_engine.compute\_difference(

&nbsp;                   prev\_frame, 

&nbsp;                   curr\_frame

&nbsp;               )

&nbsp;               

&nbsp;               # Visualize difference

&nbsp;               viz\_frame = self.diff\_engine.visualize\_difference(

&nbsp;                   curr\_frame,

&nbsp;                   diff\_map,

&nbsp;                   highlights

&nbsp;               )

&nbsp;               

&nbsp;               # Update difference visualization

&nbsp;               self.update\_diff\_frame(viz\_frame)

&nbsp;               

&nbsp;               # Analyze changes with VLM (only if significant changes)

&nbsp;               if highlights:

&nbsp;                   description = self.vlm.analyze\_difference(

&nbsp;                       viz\_frame,

&nbsp;                       highlights

&nbsp;                   )

&nbsp;                   self.update\_description(f"\[CHANGE] {description}")

&nbsp;                   

&nbsp;                   # Record frame with description

&nbsp;                   self.recorder.add\_frame\_with\_subtitle(

&nbsp;                       viz\_frame,

&nbsp;                       description

&nbsp;                   )

&nbsp;               else:

&nbsp;                   # Record frame with no change message

&nbsp;                   self.recorder.add\_frame\_with\_subtitle(

&nbsp;                       curr\_frame,

&nbsp;                       "No significant changes"

&nbsp;                   )

&nbsp;               

&nbsp;               # Control processing rate

&nbsp;               time.sleep(0.5)  # Adjust based on LFM2-VL inference speed

&nbsp;               

&nbsp;           except Exception as e:

&nbsp;               print(f"Error in processing loop: {e}")

&nbsp;               continue

&nbsp;   

&nbsp;   def update\_gui\_frame(self, frame):

&nbsp;       """Update main camera feed in GUI"""

&nbsp;       # Convert and display (implementation depends on GUI framework)

&nbsp;       pass

&nbsp;   

&nbsp;   def update\_diff\_frame(self, frame):

&nbsp;       """Update difference visualization in GUI"""

&nbsp;       pass

&nbsp;   

&nbsp;   def update\_description(self, text):

&nbsp;       """Update description text in GUI"""

&nbsp;       self.gui.description\_text.insert("end", f"{text}\\n")

&nbsp;       self.gui.description\_text.see("end")

&nbsp;   

&nbsp;   def cleanup(self):

&nbsp;       """Clean shutdown"""

&nbsp;       self.is\_running = False

&nbsp;       if self.processing\_thread:

&nbsp;           self.processing\_thread.join(timeout=2)

&nbsp;       self.camera.release()

&nbsp;       self.recorder.stop\_recording()

&nbsp;       self.gui.root.destroy()



if \_\_name\_\_ == "\_\_main\_\_":

&nbsp;   app = VideoDiffController()

&nbsp;   app.start()

```



\## 🎯 Key Optimizations for the Hackathon



\### 1. \*\*Difference Detection Strategy\*\*

\- \*\*Multi-method approach\*\*: Combines SSIM, absolute difference, and background subtraction

\- \*\*Adaptive thresholding\*\*: Adjusts based on scene complexity

\- \*\*Highlight generation\*\*: Creates bounding boxes for VLM focus



\### 2. \*\*VLM Optimization\*\*

\- \*\*Targeted prompts\*\*: Direct the model to describe only changes

\- \*\*Spatial context\*\*: Provide location information for changes

\- \*\*Batch processing\*\*: Group small changes for efficiency



\### 3. \*\*Performance Considerations\*\*

```python

\# config/settings.json

{

&nbsp;   "performance": {

&nbsp;       "frame\_skip": 3,          # Process every Nth frame

&nbsp;       "resize\_factor": 0.5,      # Downscale for processing

&nbsp;       "inference\_timeout": 2.0,   # Max wait for VLM

&nbsp;       "min\_change\_area": 500,    # Ignore tiny changes

&nbsp;       "max\_highlights": 5        # Limit regions sent to VLM

&nbsp;   }

}

```



\## 📊 Testing Scenarios



1\. \*\*Static Background Tests\*\*

&nbsp;  - Person entering/leaving frame

&nbsp;  - Object placement/removal

&nbsp;  - Lighting changes



2\. \*\*Edge Cases\*\*

&nbsp;  - Camera shake (implement stabilization)

&nbsp;  - Gradual changes (shadow movement)

&nbsp;  - Multiple simultaneous changes



\## 🚀 Quick Start for Junior Dev



1\. \*\*Setup Environment\*\*

```bash

\# Create virtual environment

python -m venv venv

venv\\Scripts\\activate



\# Install dependencies

pip install -r requirements.txt



\# Get Liquid AI SDK access

\# Register at their hackathon portal for API keys

```



2\. \*\*Run Basic Test\*\*

```python

\# test\_camera.py

from src.camera\_handler import CameraHandler

import cv2



camera = CameraHandler()

while True:

&nbsp;   frame = camera.get\_frame()

&nbsp;   if frame is not None:

&nbsp;       cv2.imshow('Test', frame)

&nbsp;   if cv2.waitKey(1) \& 0xFF == ord('q'):

&nbsp;       break

```



3\. \*\*Integration Checklist\*\*

\- \[ ] Camera capture working

\- \[ ] Difference detection producing masks

\- \[ ] LFM2-VL API connected

\- \[ ] GUI displaying frames

\- \[ ] Recording with subtitles

\- \[ ] Logging to file



\## 💡 Hackathon Tips



1\. \*\*Start simple\*\*: Get basic flow working before optimizing

2\. \*\*Mock VLM initially\*\*: Use placeholder responses while building

3\. \*\*Focus on demo\*\*: Prepare specific scenarios that showcase the concept

4\. \*\*Highlight uniqueness\*\*: Emphasize real-time edge processing

5\. \*\*Prepare fallbacks\*\*: Have offline mode if API fails



This implementation provides a robust foundation with multiple difference detection methods, proper VLM integration structure, and recording capabilities. The modular design allows easy testing and iteration during the hackathon.


Important resources:

https://leap.liquid.ai/models
https://leap.liquid.ai/docs/laptop-support
https://leap.liquid.ai/docs the Leap docs are here
https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF

