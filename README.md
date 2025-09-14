# 📹 Video-Diff: Real-Time Change Detection with Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LFM2-VL](https://img.shields.io/badge/Model-LFM2--VL--450M-orange.svg)](https://huggingface.co/LiquidAI)
[![Status](https://img.shields.io/badge/Status-Hackathon%20Ready-brightgreen.svg)]()

A real-time video analysis application that combines advanced OpenCV change detection with automated video recording. Originally designed to integrate Liquid AI's LFM2-VL vision-language model for intelligent scene analysis.

> ⚠️ **Important**: Due to current llama.cpp limitations with LFM2-VL multimodal projector loading, the VLM component provides text-only responses. See [LFM2-VL_MULTIMODAL_LIMITATION.md](LFM2-VL_MULTIMODAL_LIMITATION.md) for technical details.

## 🚀 Features

- **Real-Time Change Detection**: Advanced OpenCV algorithms detect motion and changes
- **Automated Video Recording**: Records change events with timestamped subtitles
- **Full-Frame Analysis**: Processes complete video frames for comprehensive monitoring
- **Temporal Intelligence**: Compares frames 50 frames apart for meaningful change detection
- **Modern GUI Interface**: CustomTkinter-based interface with live preview
- **Comprehensive Logging**: Detailed change detection logs with timestamps
- **Easy Setup**: One-command model download and server management
- **GPU Acceleration**: CUDA support for faster processing (when VLM is functional)

## 🎯 What Makes This Special

This application provides intelligent change detection with:
- **Advanced Algorithms**: Multi-method difference detection (SSIM, background subtraction)
- **Smart Filtering**: Filters out noise and focuses on significant changes
- **Automatic Recording**: Captures and saves change events with descriptive subtitles
- **Professional GUI**: Clean, modern interface for real-time monitoring
- **Extensible Architecture**: Modular design ready for enhanced VLM integration

## 📋 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/pClockworker52/video_diff.git
cd video_diff
pip install -r requirements.txt
```

### 2. Install Dependencies
```bash
pip install opencv-python numpy scikit-image customtkinter Pillow python-dotenv requests
```

### 3. Run the Application
```bash
python src/main.py
```

## 🎮 Usage

1. **Mode Selection**: Choose between Camera or Video File processing
2. **Start Detection**: Click "Start Detection" to begin monitoring
3. **Automatic Recording**: Video recording starts automatically when detection begins
4. **View Output**: Check `recordings/` folder for saved videos with subtitles
5. **View Logs**: Check `logs/vlm_output.txt` for detailed analysis logs

### Controls
- **Camera/File Toggle**: Switch between live camera and video file processing
- **File Selection**: Browse and select video files for analysis
- **Start/Stop Detection**: Toggle change detection and recording
- **Visual Feedback**: Live preview shows detected changes with highlighting

### Optional VLM Setup (Advanced)
For enhanced scene descriptions (currently limited by llama.cpp):
```bash
# Download the 450M model for basic text responses
python setup_local_model.py Q8_0 450M

# Start the server (optional - app works without VLM)
python restart_server.py
```

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

### Processing Speed
- **Change Detection**: ~30 FPS on modern hardware
- **Video Recording**: Real-time recording with subtitle overlay
- **File Processing**: Depends on video resolution and length

### Application Output Examples

**Change Detection Log**:
```
[2024-01-15 14:30:25] BASELINE: Baseline frame captured
[2024-01-15 14:30:28] CHANGE: Motion detected - 2 regions (1240 pixels total)
[2024-01-15 14:30:31] CHANGE: Significant change - 1 region (856 pixels total)
```

**Recorded Video Features**:
- Timestamped change events
- Visual highlighting of detected changes
- Subtitle overlay with change descriptions
- High-quality video output (configurable codec)

## 🔍 Technical Details

### Change Detection Algorithm
1. **Frame Capture**: Continuous video stream processing
2. **Temporal Spacing**: Compares frames 50 frames apart (configurable)
3. **Difference Calculation**: Advanced OpenCV background subtraction
4. **Region Extraction**: Bounding boxes with 50% contextual padding
5. **VLM Processing**: Side-by-side region comparison
6. **Intelligent Analysis**: Detailed change descriptions

### VLM Integration (Limited)
- **Model**: LFM2-VL (Liquid AI's vision-language model)
- **Current Status**: Text-only responses due to llama.cpp multimodal projector limitations
- **Inference**: Local llama.cpp server with CUDA acceleration
- **Architecture**: Ready for future multimodal enhancements

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
- **Known Limitation**: llama.cpp multimodal projector not loading (see [LFM2-VL_MULTIMODAL_LIMITATION.md](LFM2-VL_MULTIMODAL_LIMITATION.md))
- Application works fully without VLM - change detection and recording function normally
- VLM provides text-only responses when enabled

## 📁 Project Structure

```
video_diff/
├── src/                           # Core application code
│   ├── main.py                   # Application controller
│   ├── gui.py                    # CustomTkinter GUI interface
│   ├── vlm_processor.py          # VLM integration (limited)
│   ├── diff_engine.py            # Multi-method change detection
│   ├── camera_handler.py         # Video capture and buffering
│   └── video_recorder.py         # Video recording with subtitles
├── config/                        # Configuration files (optional)
├── models/                        # Model files (gitignored)
├── logs/                          # Analysis and debug logs
├── recordings/                    # Output video recordings
├── restart_server.py              # LFM2-VL server management
├── setup_local_model.py           # Model download utility
├── test_multimodal.py             # Multimodal testing script
├── LFM2-VL_MULTIMODAL_LIMITATION.md  # Technical limitation documentation
└── requirements.txt               # Python dependencies
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