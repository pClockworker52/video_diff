# 📹 Video-Diff: Real-Time Change Detection with Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LFM2-VL](https://img.shields.io/badge/Model-LFM2--VL--1.6B-green.svg)](https://huggingface.co/LiquidAI)

A powerful real-time video analysis application that combines OpenCV change detection with Liquid AI's LFM2-VL vision-language model for intelligent change analysis and description.

## 🚀 Features

- **Real-Time Video Processing**: Live camera feed with instant change detection
- **Advanced VLM Analysis**: LFM2-VL 1.6B model provides detailed descriptions of changes
- **Regional Analysis**: Focuses on specific change regions with contextual padding
- **Temporal Intelligence**: Compares frames 50 frames apart for meaningful change detection
- **GPU Acceleration**: CUDA support for fast inference
- **Comprehensive Logging**: Timestamped VLM outputs for analysis
- **Easy Setup**: One-command model download and server management

## 🎯 What Makes This Special

Unlike simple motion detection, Video-Diff uses a state-of-the-art vision-language model to:
- **Understand context**: "A person picked up their mobile phone"
- **Detect objects**: Cars, people, phones, hands, and more
- **Describe movement**: Detailed analysis of what changed and how
- **Track interactions**: Human-object interactions and activities

## 📋 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/pClockworker52/video_diff.git
cd video_diff
pip install -r requirements.txt
```

### 2. Download and Start LFM2-VL Model
```bash
# Download the recommended 1.6B model (1.25GB + 564MB projector)
python setup_local_model.py Q8_0 1.6B

# Start the optimized server
python restart_server.py
```

### 3. Run the Application
```bash
python src/main.py
```

## 🎮 Usage

1. **Camera Setup**: The app auto-detects your camera
2. **Baseline Capture**: Click "Capture Baseline" to set reference frame
3. **Live Analysis**: Real-time change detection with VLM descriptions
4. **View Logs**: Check `logs/vlm_output.txt` for detailed analysis

### Controls
- **Capture Baseline**: Sets the reference frame for comparison
- **Start/Stop**: Toggle change detection
- **Visual Feedback**: Green boxes highlight detected changes

## 🔧 Configuration

### Model Options

#### LFM2-VL 1.6B (Recommended)
- **Q8_0 (1.25GB)**: Best quality-to-speed ratio ⭐
- **Q4_0 (1.25GB)**: Fastest inference
- **F16 (1.6GB)**: Maximum quality

#### LFM2-VL 450M (Legacy)
- **Q8_0 (379MB)**: Good for lower-end hardware
- **Q4_0 (219MB)**: Basic functionality

### Hardware Requirements

**Recommended for 1.6B Model:**
- **GPU**: RTX 4080/4090 or similar (4GB+ VRAM)
- **RAM**: 8GB+ system memory
- **CPU**: Modern multi-core processor

**Minimum for 450M Model:**
- **RAM**: 4GB+ system memory
- **CPU**: Any modern processor

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Feed   │───▶│  Change Detection │───▶│   VLM Analysis  │
│                 │    │    (OpenCV)      │    │   (LFM2-VL)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         ▼                        ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GUI Display  │    │ Region Extraction│    │ Detailed Logs  │
│   Live Preview  │    │  Side-by-Side   │    │  Timestamped   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

- **`src/main.py`**: Application controller and frame management
- **`src/gui.py`**: Real-time GUI with video preview
- **`src/vlm_processor.py`**: LFM2-VL integration and prompting
- **`src/diff_engine.py`**: OpenCV change detection algorithms
- **`src/camera_handler.py`**: Video capture and frame processing
- **`restart_server.py`**: Automated model server management
- **`setup_local_model.py`**: Model download and configuration

## 📊 Performance

### Inference Times (RTX 4080)
- **1.6B Q8_0**: 2-8 seconds per analysis
- **450M Q8_0**: 1-5 seconds per analysis

### Analysis Quality Examples

**Input**: Person picking up phone
**Output**:
> "A person's hand and arm have moved from their original position, indicating they may be picking up or moving an object. The mobile phone has appeared in the current frame."

**Input**: Car movement
**Output**:
> "A car has moved from its original position in the LEFT region to a new position in the RIGHT region."

## 🔍 Technical Details

### Change Detection Algorithm
1. **Frame Capture**: Continuous video stream processing
2. **Temporal Spacing**: Compares frames 50 frames apart (configurable)
3. **Difference Calculation**: Advanced OpenCV background subtraction
4. **Region Extraction**: Bounding boxes with 50% contextual padding
5. **VLM Processing**: Side-by-side region comparison
6. **Intelligent Analysis**: Detailed change descriptions

### VLM Integration
- **Model**: LFM2-VL (Liquid AI's vision-language model)
- **Inference**: Local llama.cpp server with CUDA acceleration
- **Prompting**: Specialized prompts for change detection tasks
- **Output**: Structured analysis of movements, objects, and interactions

## 🛠️ Advanced Configuration

### Modify Detection Sensitivity
Edit `config/settings.json`:
```json
{
    "detection": {
        "threshold": 25,
        "min_area": 500,
        "frame_skip_interval": 50
    }
}
```

### Custom VLM Prompts
Edit `config/prompts.json`:
```json
{
    "change_analysis": "Analyze these side-by-side regions for changes..."
}
```

### Manual Server Start
```bash
python -m llama_cpp.server \
  --model models/LFM2-VL-1.6B-Q8_0.gguf \
  --clip_model_path models/mmproj-LFM2-VL-1.6B-Q8_0.gguf \
  --port 8000 \
  --host localhost \
  --n_ctx 4096 \
  --n_gpu_layers -1
```

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Kill existing servers and restart
pkill -f "llama_cpp.server"
python restart_server.py
```

### Out of Memory
- Switch to 450M model: `python setup_local_model.py Q8_0 450M`
- Reduce context size in restart_server.py: `--n_ctx 2048`

### Poor Detection Quality
- Increase lighting conditions
- Adjust detection threshold in settings
- Ensure stable camera mounting

### VLM Analysis Issues
- Check server logs for errors
- Verify model files are complete
- Restart server: `python restart_server.py`

## 📁 Project Structure

```
video_diff/
├── src/                      # Core application code
│   ├── main.py              # Application entry point
│   ├── gui.py               # Real-time GUI interface
│   ├── vlm_processor.py     # VLM integration
│   ├── diff_engine.py       # Change detection
│   └── camera_handler.py    # Video capture
├── config/                   # Configuration files
│   ├── settings.json        # App settings
│   ├── prompts.json         # VLM prompts
│   └── local_model.json     # Model configuration
├── models/                   # Model files (gitignored)
├── logs/                     # Analysis logs
├── restart_server.py         # Server management
├── setup_local_model.py      # Model download
└── requirements.txt          # Python dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Liquid AI** for the excellent LFM2-VL vision-language model
- **llama.cpp** team for efficient model inference
- **OpenCV** community for computer vision tools
- **Hackathon participants** for testing and feedback

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/pClockworker52/video_diff/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pClockworker52/video_diff/discussions)

---

*Built with ❤️ for intelligent video analysis*