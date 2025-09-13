#!/usr/bin/env python3
"""
Restart the local LFM2-VL server with optimized parameters
"""
import subprocess
import time
import requests
import sys
import signal
from pathlib import Path

def kill_existing_server():
    """Kill any existing llama.cpp server on port 8000"""
    try:
        # Try to find and kill existing processes
        result = subprocess.run(
            ["pkill", "-f", "llama_cpp.server"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("Killed existing llama.cpp server")
            time.sleep(2)
    except:
        pass

def start_server():
    """Start the LFM2-VL server with optimized settings"""
    model_path = Path("models/LFM2-VL-1.6B-Q8_0.gguf")
    projector_path = Path("models/mmproj-LFM2-VL-1.6B-Q8_0.gguf")

    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Run setup_local_model.py Q8_0 first")
        return False

    if not projector_path.exists():
        print(f"Multimodal projector not found at {projector_path}")
        print("Run setup_local_model.py Q8_0 first")
        return False

    cmd = [
        "python", "-m", "llama_cpp.server",
        "--model", str(model_path),
        "--clip_model_path", str(projector_path),  # Essential for vision capabilities
        "--port", "8000",
        "--host", "localhost",
        "--n_ctx", "4096",           # Larger context window
        "--n_gpu_layers", "-1",      # Use GPU if available
        "--seed", "-1"               # Random seed for variety
    ]

    print("Starting LFM2-VL server with optimized parameters...")
    print(f"Command: {' '.join(cmd)}")

    try:
        # Start server process
        process = subprocess.Popen(cmd)

        # Wait for server to start
        print("Waiting for server to start...")
        for i in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:8000/v1/models", timeout=2)
                if response.status_code == 200:
                    print("✅ Server started successfully!")
                    print("Server is ready for requests")
                    return True
            except:
                pass
            time.sleep(1)
            print(f"  Checking... ({i+1}/30)")

        print("❌ Server failed to start within 30 seconds")
        process.kill()
        return False

    except KeyboardInterrupt:
        print("\n⛔ Server startup interrupted")
        return False

if __name__ == "__main__":
    print("=== LFM2-VL Server Restart ===")

    # Kill existing server
    kill_existing_server()

    # Start new server
    if start_server():
        print("\n🚀 Server is running!")
        print("You can now run: python src/main.py")

        try:
            print("\nPress Ctrl+C to stop the server...")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⛔ Stopping server...")
            kill_existing_server()
    else:
        print("\n❌ Failed to start server")
        sys.exit(1)