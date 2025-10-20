#!/usr/bin/env python3
"""Test camera upload endpoint with sample landmarks data"""

import requests
import json

# Sample landmarks data (simplified MediaPipe format)
test_payload = {
    "user": "test_user",
    "label": "test_gesture", 
    "dialect": "north",
    "session_id": "debug_session_123",
    "frames": [
        {
            "timestamp": 0.0,
            "landmarks": [
                [0.5, 0.5, 0.1],  # x, y, z for first landmark
                [0.6, 0.4, 0.2],  # x, y, z for second landmark
                [0.4, 0.6, 0.15]  # etc...
            ]
        },
        {
            "timestamp": 0.033,
            "landmarks": [
                [0.51, 0.49, 0.11],
                [0.61, 0.39, 0.21], 
                [0.39, 0.61, 0.16]
            ]
        }
    ]
}

try:
    response = requests.post(
        "http://localhost:8000/upload/camera",
        json=test_payload,
        headers={"Content-Type": "application/json"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")