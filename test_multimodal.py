#!/usr/bin/env python3
"""
Test script to verify LFM2-VL multimodal functionality
"""
import requests
import json
import base64
from PIL import Image
import io


def create_test_image():
    """Create a simple test image with geometric shapes"""
    img = Image.new('RGB', (200, 200), color='black')
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)

    # Draw simple shapes
    draw.rectangle([50, 50, 100, 100], fill='red')
    draw.ellipse([120, 60, 170, 110], fill='blue')

    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return img_base64


def test_multimodal_api():
    """Test the multimodal API with a simple image"""
    print("🧪 Testing LFM2-VL multimodal API...")

    # Create test image
    image_data = create_test_image()
    print(f"✅ Created test image (base64 length: {len(image_data)})")

    # Test server connectivity
    try:
        models_response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if models_response.status_code == 200:
            models_data = models_response.json()
            print(f"✅ Server connected. Available models: {len(models_data['data'])}")
            for model in models_data['data']:
                print(f"   - {model['id']}")
        else:
            print(f"❌ Server connection failed: {models_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server connection error: {e}")
        return False

    # Test multimodal request
    payload = {
        "model": "models/LFM2-VL-450M-Q8_0.gguf",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful multimodal assistant by Liquid AI."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe the shapes you see in this image."
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1,
        "min_p": 0.15,
        "min_image_tokens": 64,
        "max_image_tokens": 256
    }

    print("🔄 Sending multimodal request...")
    try:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        print(f"📡 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"✅ SUCCESS! VLM Response: {content}")

                # Check if response is meaningful
                if any(word in content.lower() for word in ['red', 'blue', 'rectangle', 'circle', 'shape']):
                    print("🎉 VLM is correctly analyzing the image!")
                    return True
                else:
                    print("⚠️ VLM response doesn't match test image content")
                    return False
            else:
                print(f"❌ Invalid response format: {result}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Response text: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Request error: {e}")
        return False


if __name__ == "__main__":
    success = test_multimodal_api()
    if success:
        print("\n🚀 Multimodal API is working correctly!")
    else:
        print("\n💥 Multimodal API needs debugging")