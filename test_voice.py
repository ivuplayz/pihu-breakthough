import requests
import base64
import sys

file_path = "test.wav"
mime_type = "audio/wav"

print(f"Reading '{file_path}'...")
try:
    with open(file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")
except Exception as e:
    print(f"Error reading audio file: {e}")
    sys.exit(1)

payload = {
    "audio": {
        "data": audio_b64,
        "mime_type": mime_type
    }
}

print("Dispatching audio payload to Pihu on Vercel...")
try:
    url = "https://pihu-breakthough.vercel.app/api/chat"
    response = requests.post(url, json=payload, timeout=45)
    print(f"HTTP Status: {response.status_code}")
    print("Raw Response:", response.text)
except Exception as e:
    print(f"Request failed: {e}")
