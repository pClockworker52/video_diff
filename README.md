# 📹 Video-Diff: Real-Time Change Detection with Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LFM2-VL](https://img.shields.io/badge/Model-LFM2--VL--1.6B-green.svg)](https://huggingface.co/LiquidAI)
[![Hackathon](https://img.shields.io/badge/Liquid%20AI%20Hackathon-3rd%20Place%20🥉-orange.svg)]()

🥉 **3rd Place Winner - Liquid AI Hackathon**

A real-time video analysis application that combines advanced OpenCV change detection with Liquid AI's LFM2-VL vision-language model for intelligent scene analysis and automated video recording.

## 🚀 Features

- **Real-Time Change Detection**: Advanced OpenCV algorithms detect motion and changes
- **LFM2-VL Vision Analysis**: Real-time intelligent scene analysis with Liquid AI's vision-language model
- **Automated Video Recording**: Records change events with AI-generated descriptive subtitles
- **Regional Analysis**: Focuses on specific change regions with side-by-side comparisons
- **Temporal Intelligence**: Compares frames 50 frames apart for meaningful change detection
- **Modern GUI Interface**: CustomTkinter-based interface with live preview
- **Comprehensive Logging**: Detailed change detection and AI analysis logs
- **GPU Acceleration**: CUDA support for fast LFM2-VL inference

## 🎯 What Makes This Special

This application provides intelligent change detection with:
- **Advanced Algorithms**: Multi-method difference detection (SSIM, background subtraction)
- **Smart Filtering**: Filters out noise and focuses on significant changes
- **Automatic Recording**: Captures and saves change events with descriptive subtitles
- **Professional GUI**: Clean, modern interface for real-time monitoring
- **Extensible Architecture**: Modular design ready for enhanced VLM integration

## 📋 Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/pClockworker52/video_diff.git
cd video_diff
pip install -r requirements.txt
```

### 2. Run the Application
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

The application will automatically download and initialize the LFM2-VL-1.6B model on first run for real-time vision analysis.

## 🔧 Configuration

### Detection Sensitivity
The refactored DifferenceEngine provides clean parameter control:

```python
# In src/diff_engine.py initialization
DifferenceEngine(
    sensitivity=0.2,      # 0.0-1.0, higher = less sensitive
    min_area_ratio=0.001  # 0.001-0.1, minimum region size ratio
)
```

### Hardware Requirements

**Recommended Configuration:**
- **GPU**: RTX 3060 or better (4GB+ VRAM) for optimal LFM2-VL performance
- **RAM**: 8GB+ system memory
- **CPU**: Modern multi-core processor

**Minimum Requirements:**
- **RAM**: 6GB+ system memory (for LFM2-VL-1.6B model)
- **CPU**: Any modern processor (CPU inference supported)

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
- **`src/vlm_processor.py`**: Native transformers LFM2-VL integration with ChatML
- **`src/diff_engine.py`**: Refactored change detection with clean parameter API
- **`src/camera_handler.py`**: Video capture and frame processing
- **`src/video_recorder.py`**: Video recording with AI-generated subtitle overlay

## 📊 Performance

### Processing Speed
- **Change Detection**: ~30 FPS on modern hardware
- **Video Recording**: Real-time recording with subtitle overlay
- **File Processing**: Depends on video resolution and length

### Application Output Examples

**Real LFM2-VL Analysis Examples**:
```
[2025-09-16 11:41:06] BASELINE: The image shows a simple scene with a dark gray background. In the top left corner, there is a light gray rectangle. To the right of the rectangle, there is a red square...

[2025-09-16 11:41:15] CHANGE: 2 change region(s) detected - Region 1 (top-left): The red square has moved to the top-right quadrant of the frame; Region 2 (top-left): The green circle has shifted from the top-left quadrant to the top-right quadrant

[2025-09-16 11:41:24] CHANGE: 1 change region(s) detected - Region 1 (top-left): The blue rectangle has moved horizontally and increased in size from the left side
```

**Recorded Video Features**:
- Timestamped change events with AI-generated descriptions
- Visual highlighting of detected change regions
- Burned-in subtitles with real LFM2-VL analysis
- Regional comparison images saved for inspection
- High-quality video output with XVID codec

## 🔍 Technical Details

### Change Detection Algorithm
1. **Frame Capture**: Continuous video stream processing
2. **Temporal Spacing**: Compares frames 50 frames apart (configurable)
3. **Difference Calculation**: Advanced OpenCV background subtraction
4. **Region Extraction**: Bounding boxes with 50% contextual padding
5. **VLM Processing**: Side-by-side region comparison
6. **Intelligent Analysis**: Detailed change descriptions

### LFM2-VL Integration ✅
- **Model**: LFM2-VL-1.6B (Liquid AI's vision-language model)
- **Implementation**: Native Hugging Face transformers with proper ChatML formatting
- **Inference**: Direct PyTorch inference with CUDA acceleration
- **Capabilities**: Real-time vision analysis with detailed change descriptions
- **Performance**: 2-8 seconds per regional analysis on modern hardware

## 🐛 Troubleshooting

### Installation Issues
```bash
# Ensure you have the latest pip and required build tools
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Poor Detection Quality
- Increase lighting conditions
- Adjust sensitivity in DifferenceEngine initialization
- Ensure stable camera mounting
- Reduce min_area_ratio for smaller objects

### GPU Memory Issues
- LFM2-VL will automatically fall back to CPU if insufficient GPU memory
- Monitor GPU usage with `nvidia-smi` during operation

## 📁 Project Structure

```
video_diff/
├── src/                           # Core application code
│   ├── main.py                   # Application controller
│   ├── gui.py                    # CustomTkinter GUI interface
│   ├── vlm_processor.py          # Native transformers LFM2-VL integration
│   ├── diff_engine.py            # Refactored change detection with clean API
│   ├── camera_handler.py         # Video capture and buffering
│   └── video_recorder.py         # Video recording with AI subtitle overlay
├── config/                        # Configuration files
├── CLAUDE.md                      # Claude Code project instructions
├── LFM2_VL_WORKING_SOLUTION.md    # Technical breakthrough documentation
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

## 🏆 Hackathon Achievement

This project placed **3rd in the Liquid AI Hackathon**, demonstrating successful integration of LFM2-VL for real-time video analysis. The breakthrough came from discovering the correct approach to use native Hugging Face transformers instead of llama.cpp for multimodal inference.

## 🙏 Acknowledgments

- **Liquid AI** for the powerful LFM2-VL vision-language model and hackathon opportunity
- **Hugging Face** for excellent transformers library enabling native LFM2-VL integration
- **OpenCV** community for robust computer vision tools
- **Anthropic** for Claude Code which facilitated rapid development and debugging
- **Hackathon community** for valuable feedback and collaborative spirit

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/pClockworker52/video_diff/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pClockworker52/video_diff/discussions)

---

🥉 **3rd Place Winner - Liquid AI Hackathon**
*Built with ❤️ for intelligent video analysis using LFM2-VL*