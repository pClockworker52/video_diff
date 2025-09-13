# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Video-Diff Application** for real-time change detection using webcam footage and LFM2-VL (Liquid AI's Vision-Language Model) for narrating scene changes. The application is designed for a Windows 11 laptop as part of the Liquid AI Hackathon.

## Architecture

The application follows a modular architecture with these key components:

```
src/
├── main.py              # Entry point and main controller
├── camera_handler.py    # Webcam capture and frame buffering
├── diff_engine.py       # Multi-method difference detection (SSIM, absolute diff, background subtraction)
├── vlm_processor.py     # LFM2-VL integration for scene analysis
├── video_recorder.py    # Video recording with subtitle overlay
└── gui.py               # CustomTkinter-based user interface
```

The processing flow is:
1. **Camera Handler** captures frames and maintains a rolling buffer
2. **Difference Engine** compares frames using multiple detection methods and generates highlight regions
3. **VLM Processor** analyzes highlighted changes using LFM2-VL with targeted prompts
4. **Video Recorder** saves output with burned-in subtitles
5. **GUI** displays real-time feeds and controls

## Key Dependencies

Based on the implementation plan, this project uses:
- `opencv-python` for webcam capture and image processing
- `numpy` for array operations and diff calculations
- `scikit-image` for advanced image comparison (SSIM)
- `customtkinter` for modern GUI
- `liquid-sdk` for LFM2-VL integration (requires Liquid AI API access)
- `Pillow` for image manipulation
- `imageio-ffmpeg` for video recording

## Development Setup

Since no dependency files exist yet, create them based on the implementation plan:

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies (create requirements.txt first)
pip install opencv-python==4.8.1 numpy==1.24.3 scikit-image==0.21.0 customtkinter==5.2.0 Pillow==10.0.0 imageio-ffmpeg==0.4.9 python-dotenv==1.0.0
```

## Running the Application

```bash
python src/main.py
```

## Liquid AI Resources

Key resources for LFM2-VL integration:
- **Models**: https://leap.liquid.ai/models
- **Laptop Support**: https://leap.liquid.ai/docs/laptop-support
- **Documentation**: https://leap.liquid.ai/docs
- **Hugging Face Model**: https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF

## Testing Strategy

For development and testing:

1. **Camera Test**: Verify webcam capture with `test_camera.py`
2. **Difference Detection**: Test with static background scenarios
3. **Edge Cases**: Handle camera shake, gradual changes, multiple simultaneous changes
4. **Integration**: Verify the full pipeline from capture → detection → VLM → recording

## Configuration

The application uses configuration files in `config/`:
- `settings.json` - Performance tuning (frame skip, resize factors, thresholds)
- `prompts.json` - VLM prompt templates for different scenarios

## Documentation

See `implementation_plan.md` for:
- Detailed implementation steps and code examples
- Complete class structures and method implementations
- Performance optimization strategies
- Testing scenarios and edge cases
- Hackathon-specific tips and strategies

## Important Notes

- This is a hackathon project focused on real-time edge processing
- The VLM processor may need mock responses during development if Liquid AI API is unavailable
- Frame processing rate should be adjusted based on LFM2-VL inference speed
- The application generates outputs in `recordings/` (videos) and `logs/` (JSONL event logs)