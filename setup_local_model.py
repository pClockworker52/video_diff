#!/usr/bin/env python3
"""
Script to download and set up LFM2-VL model locally with llama.cpp
"""
import os
import subprocess
import sys
import requests
from pathlib import Path

def download_model(model_size="Q8_0", model_scale="1.6B"):
    """Download LFM2-VL model from Hugging Face"""
    models = {
        "450M": {
            "Q4_0": "LiquidAI/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-Q4_0.gguf",
            "Q8_0": "LiquidAI/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-Q8_0.gguf",
            "F16": "LiquidAI/LFM2-VL-450M-GGUF/resolve/main/LFM2-VL-450M-F16.gguf"
        },
        "1.6B": {
            "Q4_0": "LiquidAI/LFM2-VL-1.6B-GGUF/resolve/main/LFM2-VL-1.6B-Q4_0.gguf",
            "Q8_0": "LiquidAI/LFM2-VL-1.6B-GGUF/resolve/main/LFM2-VL-1.6B-Q8_0.gguf",
            "F16": "LiquidAI/LFM2-VL-1.6B-GGUF/resolve/main/LFM2-VL-1.6B-F16.gguf"
        }
    }

    # Multimodal projector files
    projector_models = {
        "450M": {
            "Q4_0": None,  # No Q4_0 projector available
            "Q8_0": "LiquidAI/LFM2-VL-450M-GGUF/resolve/main/mmproj-LFM2-VL-450M-Q8_0.gguf",
            "F16": "LiquidAI/LFM2-VL-450M-GGUF/resolve/main/mmproj-LFM2-VL-450M-F16.gguf"
        },
        "1.6B": {
            "Q4_0": "LiquidAI/LFM2-VL-1.6B-GGUF/resolve/main/mmproj-LFM2-VL-1.6B-Q4_0.gguf",
            "Q8_0": "LiquidAI/LFM2-VL-1.6B-GGUF/resolve/main/mmproj-LFM2-VL-1.6B-Q8_0.gguf",
            "F16": "LiquidAI/LFM2-VL-1.6B-GGUF/resolve/main/mmproj-LFM2-VL-1.6B-F16.gguf"
        }
    }

    if model_scale not in models or model_size not in models[model_scale]:
        print(f"Invalid model scale or size. Choose scale from: {list(models.keys())}, size from: {list(models['1.6B'].keys())}")
        return None

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_url = f"https://huggingface.co/{models[model_scale][model_size]}"
    model_path = model_dir / f"LFM2-VL-{model_scale}-{model_size}.gguf"

    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return model_path

    print(f"Downloading {model_size} model ({model_url})...")

    def download_file(url, path, description):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r{description}: {progress:.1f}%", end='', flush=True)

            print(f"\n{description} downloaded to {path}")
            return True

        except Exception as e:
            print(f"\nError downloading {description}: {e}")
            return False

    # Download main model
    if not download_file(model_url, model_path, f"{model_scale}-{model_size} model"):
        return None

    # Download projector if available
    projector_url = projector_models[model_scale].get(model_size)
    if projector_url:
        projector_path = model_dir / f"mmproj-LFM2-VL-{model_scale}-{model_size}.gguf"
        if not projector_path.exists():
            projector_url_full = f"https://huggingface.co/{projector_url}"
            download_file(projector_url_full, projector_path, f"{model_scale}-{model_size} projector")

    return model_path

def install_llama_cpp():
    """Install llama-cpp-python for local inference"""
    try:
        print("Installing llama-cpp-python...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python[server]"])
        print("llama-cpp-python installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing llama-cpp-python: {e}")
        return False

def start_local_server(model_path, port=8000):
    """Start llama.cpp server optimized for LFM2-VL"""
    cmd = [
        "python", "-m", "llama_cpp.server",
        "--model", str(model_path),
        "--port", str(port),
        "--host", "localhost",
        "--ctx_size", "4096",        # Larger context window
        "--n_gpu_layers", "-1",      # Use GPU if available
        "--seed", "-1"               # Random seed for variety
    ]

    print(f"Starting llama.cpp server on port {port}...")
    print(f"Command: {' '.join(cmd)}")
    print("Press Ctrl+C to stop the server")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")

def main():
    import sys

    print("LFM2-VL Local Model Setup")
    print("=" * 40)

    # Choose model size (default to Q4_0 if non-interactive)
    if len(sys.argv) > 1:
        model_size = sys.argv[1].upper()
        if model_size not in ["Q4_0", "Q8_0", "F16"]:
            model_size = "Q4_0"
    else:
        # Check if running interactively
        try:
            print("Available model sizes:")
            print("1. Q4_0 (219 MB) - Fastest inference")
            print("2. Q8_0 (379 MB) - Balanced")
            print("3. F16 (711 MB) - Full precision")

            choice = input("Choose model (1-3, default: 1): ").strip() or "1"
            model_sizes = {"1": "Q4_0", "2": "Q8_0", "3": "F16"}
            model_size = model_sizes.get(choice, "Q4_0")
        except EOFError:
            # Non-interactive mode, use default
            print("Running in non-interactive mode, using Q4_0 (fastest)")
            model_size = "Q4_0"

    # Install dependencies
    if not install_llama_cpp():
        return

    # Download model
    model_path = download_model(model_size)
    if not model_path:
        return

    # Create config for local model
    config = {
        "local_model": {
            "enabled": True,
            "model_path": str(model_path),
            "server_port": 8000,
            "server_host": "localhost"
        }
    }

    config_path = Path("config/local_model.json")
    with open(config_path, 'w') as f:
        import json
        json.dump(config, f, indent=2)

    print(f"\nSetup complete!")
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")

    # Ask if user wants to start server (skip in non-interactive mode)
    try:
        start_server = input("\nStart local server now? (y/n): ").strip().lower()
        if start_server in ['y', 'yes']:
            start_local_server(model_path)
    except EOFError:
        print("\nSetup completed in non-interactive mode.")
        print("To start the server manually, run:")
        print(f"python -m llama_cpp.server --model {model_path} --port 8000 --host localhost --ctx_size 4096 --n_gpu_layers -1 --seed -1")

if __name__ == "__main__":
    main()