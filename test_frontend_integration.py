#!/usr/bin/env python3
"""
Test script để verify tính tương thích của Backend với Frontend React
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health endpoint"""
    print("🏥 Testing health check endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check: OK")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_cors_headers():
    """Test CORS configuration"""
    print("\n🌐 Testing CORS headers...")
    headers = {
        'Origin': 'http://localhost:5173',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'Content-Type'
    }
    try:
        response = requests.options(f"{BASE_URL}/upload/camera", headers=headers)
        cors_headers = {k: v for k, v in response.headers.items() if 'access-control' in k.lower()}
        if cors_headers:
            print("✅ CORS headers found:")
            for k, v in cors_headers.items():
                print(f"   {k}: {v}")
        else:
            print("❌ No CORS headers found")
    except Exception as e:
        print(f"❌ CORS test error: {e}")

def test_video_upload_format():
    """Test video upload response format"""
    print("\n📹 Testing video upload response format...")
    
    # Create a dummy text file to simulate video upload
    with open("test_video.txt", "w") as f:
        f.write("dummy video content")
    
    try:
        with open("test_video.txt", "rb") as f:
            files = {"file": ("test_video.mp4", f, "video/mp4")}
            data = {
                "user": "test_user",
                "label": "test_label",
                "session_id": "test_session_123"
            }
            
            response = requests.post(f"{BASE_URL}/upload/video", files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Video upload response format:")
            print(f"   success: {result.get('success')}")
            print(f"   task_id: {result.get('task_id', 'Missing')}")
            print(f"   status: {result.get('status', 'Missing')}")
            print(f"   filename: {result.get('filename', 'Missing')}")
            
            # Check if all required fields are present
            required_fields = ['success', 'task_id', 'status', 'filename']
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
            else:
                print("✅ All required fields present")
        else:
            print(f"❌ Video upload failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Video upload test error: {e}")
    finally:
        # Clean up
        import os
        if os.path.exists("test_video.txt"):
            os.remove("test_video.txt")

def test_camera_upload_format():
    """Test camera data upload with MediaPipe landmarks format"""
    print("\n📷 Testing camera upload with MediaPipe format...")
    
    # Sample MediaPipe landmarks format
    sample_payload = {
        "user": "test_user",
        "label": "hello", 
        "session_id": "camera_session_456",
        "frames": [
            {
                "timestamp": 1000,
                "landmarks": {
                    "pose": [
                        {"x": 0.5, "y": 0.5, "z": 0.1, "visibility": 0.9},
                        {"x": 0.52, "y": 0.48, "z": 0.12, "visibility": 0.85}
                    ],
                    "face": [
                        {"x": 0.48, "y": 0.3, "z": 0.05},
                        {"x": 0.52, "y": 0.32, "z": 0.06}
                    ],
                    "left_hand": [
                        {"x": 0.3, "y": 0.6, "z": 0.2},
                        {"x": 0.32, "y": 0.62, "z": 0.22}
                    ],
                    "right_hand": [
                        {"x": 0.7, "y": 0.6, "z": 0.2},
                        {"x": 0.72, "y": 0.62, "z": 0.22}
                    ]
                }
            },
            {
                "timestamp": 1033,
                "landmarks": {
                    "pose": [
                        {"x": 0.51, "y": 0.51, "z": 0.11, "visibility": 0.88},
                        {"x": 0.53, "y": 0.49, "z": 0.13, "visibility": 0.82}
                    ],
                    "face": [
                        {"x": 0.49, "y": 0.31, "z": 0.06},
                        {"x": 0.53, "y": 0.33, "z": 0.07}
                    ],
                    "left_hand": [
                        {"x": 0.31, "y": 0.61, "z": 0.21},
                        {"x": 0.33, "y": 0.63, "z": 0.23}
                    ],
                    "right_hand": [
                        {"x": 0.71, "y": 0.61, "z": 0.21},
                        {"x": 0.73, "y": 0.63, "z": 0.23}
                    ]
                }
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/upload/camera",
            headers={"Content-Type": "application/json"},
            json=sample_payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Camera upload response format:")
            print(f"   success: {result.get('success')}")
            print(f"   task_id: {result.get('task_id', 'Missing')}")
            print(f"   status: {result.get('status', 'Missing')}")
            print(f"   total_frames: {result.get('total_frames', 'Missing')}")
            print(f"   filename: {result.get('filename', 'Missing')}")
            
            # Check if all required fields are present
            required_fields = ['success', 'task_id', 'status', 'total_frames']
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
            else:
                print("✅ All required fields present")
        else:
            print(f"❌ Camera upload failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Camera upload test error: {e}")

def test_job_status():
    """Test job status endpoint"""
    print("\n📊 Testing job status endpoint...")
    try:
        # Use a dummy job ID
        job_id = "test_job_123"
        response = requests.get(f"{BASE_URL}/jobs/{job_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Job status response format:")
            print(f"   job_id: {result.get('job_id')}")
            print(f"   status: {result.get('status')}")
            print(f"   result: {result.get('result')}")
        else:
            print(f"⚠️ Job status: {response.status_code} (expected for non-existent job)")
            
    except Exception as e:
        print(f"❌ Job status test error: {e}")

def main():
    """Run all frontend integration tests"""
    print("🧪 FRONTEND-BACKEND INTEGRATION TESTS")
    print("=" * 50)
    
    test_health_check()
    test_cors_headers() 
    test_video_upload_format()
    test_camera_upload_format()
    test_job_status()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print("- Health endpoint: Required for frontend connectivity check")
    print("- CORS headers: Required for localhost:5173 access")  
    print("- Video upload: Compatible with FormData from frontend")
    print("- Camera upload: Compatible with MediaPipe landmarks")
    print("- Job status: For tracking async video processing")
    
    print("\n💡 NEXT STEPS:")
    print("1. Start backend: docker-compose up --build")
    print("2. Run tests: python test_frontend_integration.py")
    print("3. Verify all endpoints return expected format")

if __name__ == "__main__":
    main()