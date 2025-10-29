#!/usr/bin/env python3
"""
Test script để kiểm tra chi tiết phần camera upload với MediaPipe Hands landmarks
"""

import requests
import json
import numpy as np
import time

BASE_URL = "http://localhost:8000"

def generate_realistic_mediapipe_data(num_frames=30):
    """
    Tạo data MediaPipe giống thật với MediaPipe Hands (chỉ hands, không có pose/face)
    """
    frames = []
    
    for i in range(num_frames):
        timestamp = i * 33  # 30 FPS = 33ms per frame
        
        # BỎ POSE LANDMARKS - không lấy pose nữa
        
        # BỎ FACE LANDMARKS - không lấy face nữa
        
        # Hand landmarks: 21 points × 3 coordinates each hand
        left_hand_landmarks = []
        right_hand_landmarks = []
        for j in range(21):
            # Left hand
            left_hand_landmarks.append({
                "x": 0.3 + np.sin(i * 0.2 + j * 0.2) * 0.1,
                "y": 0.6 + np.cos(i * 0.2 + j * 0.2) * 0.1,
                "z": 0.2 + np.sin(i * 0.1) * 0.05
            })
            
            # Right hand
            right_hand_landmarks.append({
                "x": 0.7 + np.sin(i * 0.2 + j * 0.2) * 0.1,
                "y": 0.6 + np.cos(i * 0.2 + j * 0.2) * 0.1,
                "z": 0.2 + np.sin(i * 0.1) * 0.05
            })
        
        frame_data = {
            "timestamp": timestamp,
            "landmarks": {
                # "pose": REMOVED - không lấy pose landmarks
                # "face": REMOVED - không lấy face landmarks
                "left_hand": left_hand_landmarks,
                "right_hand": right_hand_landmarks
            }
        }
        frames.append(frame_data)
    
    return frames

def test_camera_upload_detailed():
    """Test camera upload với data thực tế"""
    print("📷 TESTING CAMERA UPLOAD WITH REALISTIC HANDS DATA")
    print("=" * 60)
    
    # Generate realistic MediaPipe Hands data
    print("🔄 Generating MediaPipe Hands landmarks data...")
    frames = generate_realistic_mediapipe_data(num_frames=60)  # 2 seconds at 30fps
    
    print(f"✅ Generated {len(frames)} frames")
    print(f"   - Left hand landmarks per frame: {len(frames[0]['landmarks']['left_hand'])}")
    print(f"   - Right hand landmarks per frame: {len(frames[0]['landmarks']['right_hand'])}")
    print(f"   - Pose landmarks: REMOVED (not used)")
    print(f"   - Face landmarks: REMOVED (not used)")
    
    # Calculate expected feature dimensions (hands only)
    hand_dim = 21 * 3 * 2  # x, y, z for both hands
    expected_total_dim = hand_dim  # Only hands, no pose, no face
    
    print(f"   - Expected feature dimensions: {expected_total_dim}")
    print(f"     • Hands: {hand_dim}")
    print(f"     • Pose: 0 (removed)")
    print(f"     • Face: 0 (removed)")
    
    # Prepare payload
    payload = {
        "user": "test_camera_user",
        "label": "hello_gesture",
        "session_id": f"test_session_{int(time.time())}",
        "frames": frames
    }
    
    print(f"\n📤 Uploading to {BASE_URL}/upload/camera...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/upload/camera",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        upload_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Camera upload successful!")
            print(f"   Upload time: {upload_time:.2f} seconds")
            print(f"   Response format:")
            for key, value in result.items():
                print(f"     • {key}: {value}")
            
            # Validate response format
            required_fields = ['success', 'task_id', 'status', 'total_frames', 'filename']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                print(f"❌ Missing fields: {missing_fields}")
            else:
                print("✅ All required response fields present")
            
            return result
            
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    return None

def test_different_scenarios():
    """Test các scenarios khác nhau"""
    print("\n🎯 TESTING DIFFERENT SCENARIOS")
    print("=" * 60)
    
    scenarios = [
        {
            "name": "Minimal data (5 frames)",
            "frames": 5,
            "user": "minimal_user",
            "label": "wave"
        },
        {
            "name": "Medium data (30 frames)",
            "frames": 30,
            "user": "medium_user", 
            "label": "goodbye"
        },
        {
            "name": "Large data (120 frames)",
            "frames": 120,
            "user": "large_user",
            "label": "thank_you"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Testing: {scenario['name']}")
        
        frames = generate_realistic_mediapipe_data(scenario['frames'])
        payload = {
            "user": scenario['user'],
            "label": scenario['label'],
            "session_id": f"scenario_{int(time.time())}",
            "frames": frames
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{BASE_URL}/upload/camera",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60
            )
            upload_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success - {upload_time:.2f}s")
                print(f"      Frames: {result.get('total_frames')}, File: {result.get('filename')}")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_error_handling():
    """Test error handling"""
    print("\n⚠️ TESTING ERROR HANDLING")
    print("=" * 60)
    
    error_tests = [
        {
            "name": "Missing label",
            "payload": {"user": "test", "frames": []}
        },
        {
            "name": "Missing frames",
            "payload": {"user": "test", "label": "test"}
        },
        {
            "name": "Empty frames array",
            "payload": {"user": "test", "label": "test", "frames": []}
        },
        {
            "name": "Invalid landmarks format",
            "payload": {
                "user": "test",
                "label": "test", 
                "frames": [{"timestamp": 0, "landmarks": "invalid"}]
            }
        }
    ]
    
    for test in error_tests:
        print(f"\n🔍 Testing: {test['name']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/upload/camera",
                headers={"Content-Type": "application/json"},
                json=test['payload']
            )
            
            result = response.json()
            if not result.get('success', True):
                print(f"   ✅ Properly handled error: {result.get('detail', 'No detail')}")
            else:
                print(f"   ⚠️ Should have failed but didn't")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")

def check_saved_files():
    """Kiểm tra files đã lưu"""
    print("\n📁 CHECKING SAVED FILES")
    print("=" * 60)
    
    try:
        import os
        dataset_path = "./dataset"
        
        if os.path.exists(dataset_path):
            print(f"📂 Dataset directory found: {dataset_path}")
            
            # List all class directories
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    print(f"   📁 {item}/")
                    
                    # List files in class directory
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            files = os.listdir(subitem_path)
                            print(f"      📁 {subitem}/ ({len(files)} files)")
                            for file in files[:3]:  # Show first 3 files
                                print(f"         📄 {file}")
                            if len(files) > 3:
                                print(f"         ... and {len(files) - 3} more files")
        else:
            print("❌ Dataset directory not found")
            
    except Exception as e:
        print(f"❌ Error checking files: {e}")

def main():
    """Run all camera upload tests"""
    print("🧪 COMPREHENSIVE CAMERA UPLOAD TESTS")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print("❌ Backend not responding correctly")
            return
    except:
        print("❌ Backend not reachable. Start with: docker-compose up -d")
        return
    
    # Run all tests
    test_camera_upload_detailed()
    test_different_scenarios()
    test_error_handling()
    check_saved_files()
    
    print("\n" + "=" * 60)
    print("🎉 CAMERA UPLOAD TESTING COMPLETE")
    print("\n💡 Next steps:")
    print("1. Check dataset/features/ for saved files")
    print("2. Verify feature dimensions match expectations")
    print("3. Test with real MediaPipe data from frontend")

if __name__ == "__main__":
    main()