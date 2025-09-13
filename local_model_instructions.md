# Local LFM2-VL Model Setup

## Quick Setup

1. **Install dependencies and download model:**
   ```bash
   # Download 1.6B model (recommended for best performance)
   python setup_local_model.py Q8_0 1.6B

   # Or download 450M model for faster inference
   python setup_local_model.py Q8_0 450M
   ```

2. **Start the server (recommended):**
   ```bash
   python restart_server.py
   ```

3. **Run the application:**
   ```bash
   python src/main.py
   ```

## Model Options

### LFM2-VL 1.6B (Recommended)
- **Q4_0 (1.25GB)**: Fastest 1.6B inference, excellent quality
- **Q8_0 (1.25GB)**: Best 1.6B quality, moderate speed
- **F16 (1.6GB)**: Maximum 1.6B quality, slower inference

### LFM2-VL 450M (Legacy)
- **Q4_0 (219 MB)**: Fastest inference, basic quality
- **Q8_0 (379 MB)**: Better quality, moderate speed
- **F16 (711 MB)**: Best 450M quality, slower inference

## Advanced Server Setup

### Manual Server Start
```bash
python -m llama_cpp.server \
  --model models/LFM2-VL-1.6B-Q8_0.gguf \
  --clip_model_path models/mmproj-LFM2-VL-1.6B-Q8_0.gguf \
  --port 8000 \
  --host localhost \
  --n_ctx 4096 \
  --n_gpu_layers -1 \
  --seed -1
```

### Configuration
The config is automatically updated when downloading models:
```json
{
    "local_model": {
        "enabled": true,
        "model_path": "models/LFM2-VL-1.6B-Q8_0.gguf",
        "server_port": 8000,
        "server_host": "localhost"
    }
}
```

## Switching Between Local and API

The application automatically detects:
1. If `local_model.enabled = true` → uses local model
2. If local server not running → falls back to API (if key available)
3. If neither available → continues with difference detection only

## Performance Tips

- **GPU**: RTX 4080/4090 recommended for 1.6B models with CUDA acceleration
- **CPU**: Works on any modern CPU, AMD Ryzen/Intel 12th gen+ recommended
- **RAM**: 8GB+ recommended for 1.6B models, 4GB for 450M models
- **Inference Time**:
  - 1.6B model: 2-8 seconds per image (GPU), 5-20 seconds (CPU)
  - 450M model: 1-5 seconds per image depending on hardware
- **VRAM**: 4GB+ recommended for full GPU acceleration

## Key Features

- **Multimodal Vision-Language**: Supports both text and image analysis
- **Regional Change Detection**: Analyzes specific regions with 50% padding for context
- **Side-by-Side Comparison**: Compares frames 50 frames apart for temporal analysis
- **CUDA Acceleration**: Automatic GPU acceleration when available
- **Optimized Prompting**: Specialized prompts for change detection tasks

## Troubleshooting

### Local server won't start
- Check if port 8000 is available
- Verify model file downloaded correctly
- Try different model sizes

### Connection issues
- Ensure server is running before starting application
- Check firewall settings
- Verify port configuration matches in both files

### Performance issues
- Use Q4_0 model for fastest inference
- Close other applications to free up CPU/RAM
- Consider using API mode for faster response times