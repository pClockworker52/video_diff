# Video-Diff Setup Instructions

## 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure Liquid AI API

1. **Get API Key**: Register at [Liquid AI Leap](https://leap.liquid.ai/models) to get your API key

2. **Set up environment variables**:
   ```bash
   # Copy the example file
   cp .env.example .env

   # Edit .env file and add your API key
   LIQUID_AI_API_KEY=your_actual_api_key_here
   ```

3. **Verify API Access**:
   - Check [laptop support documentation](https://leap.liquid.ai/docs/laptop-support)
   - Review [general docs](https://leap.liquid.ai/docs)
   - Model details: [LFM2-VL-450M on Hugging Face](https://huggingface.co/LiquidAI/LFM2-VL-450M-GGUF)

## 3. Run the Application

```bash
python src/main.py
```

## 4. Usage

1. **Live Camera Mode**: Select "Live Camera" and click "Start Detection"
2. **Video File Mode**: Select "Video File", load a video file, then click "Start Detection"

## 5. Troubleshooting

### VLM API Issues
- If API key is missing: Application will show "VLM not available" but continue with difference detection only
- If API connection fails: Error messages will show in the GUI, change detection continues
- Check your internet connection and API key validity

### Camera Issues
- Ensure your camera is not being used by another application
- Try different camera indices (0, 1, 2) if default doesn't work

### Dependencies
- If scikit-image installation fails, try: `pip install --upgrade pip setuptools wheel`
- For GPU acceleration (optional): Install PyTorch with CUDA support

## 6. Output

- **Live Display**: Real-time camera feed and difference visualization
- **Logs**: Saved to `logs/` directory (when implemented)
- **Recordings**: Saved to `recordings/` directory (when implemented)