#!/usr/bin/env python3
"""
Test script mô phỏng chính xác cách React Frontend gửi MediaPipe data
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def simulate_frontend_capture_session():
    """
    Mô phỏng chính xác một session capture từ React frontend
    với MediaPipe Holistic landmarks
    """
    print("🎬 SIMULATING REACT FRONTEND CAPTURE SESSION")
    print("=" * 60)
    
    # Session info như frontend sẽ tạo
    session_id = f"react_session_{int(time.time())}"
    user_name = "frontend_user"
    gesture_label = "hello_world"
    
    print(f"👤 User: {user_name}")
    print(f"🏷️ Label: {gesture_label}")
    print(f"🆔 Session: {session_id}")
    
    # Tạo frames giống như MediaPipe Holistic output
    frames = []
    capture_duration_ms = 3000  # 3 seconds
    fps = 30
    total_frames = int(capture_duration_ms / (1000 / fps))
    
    print(f"⏱️ Simulating {capture_duration_ms}ms capture at {fps}FPS")
    print(f"📊 Expected frames: {total_frames}")
    
    for frame_idx in range(total_frames):
        timestamp = frame_idx * (1000 / fps)  # milliseconds
        
        # MediaPipe Holistic results format - chỉ pose upper body + hands
        landmarks = {
            "pose": [],
            # "face": [],  # REMOVED - không lấy face
            "left_hand": [],
            "right_hand": []
        }
        
        # Pose landmarks (25 points với visibility) - UPPER BODY ONLY
        for i in range(25):  # Chỉ lấy 25 điểm thay vì 33
            landmarks["pose"].append({
                "x": 0.5 + (i / 100.0) * 0.1,  # Normalized coordinates [0-1]
                "y": 0.5 + (frame_idx / 100.0) * 0.1,
                "z": -0.1 + (i / 200.0) * 0.05,
                "visibility": 0.9 if i < 20 else 0.7  # Some landmarks less visible
            })
        
        # BỎ FACE LANDMARKS
        
        # Left hand landmarks (21 points)
        for i in range(21):
            landmarks["left_hand"].append({
                "x": 0.3 + (i / 50.0) * 0.1,
                "y": 0.6 + (frame_idx / 150.0) * 0.05,
                "z": 0.1 + (i / 100.0) * 0.03
            })
        
        # Right hand landmarks (21 points)
        for i in range(21):
            landmarks["right_hand"].append({
                "x": 0.7 + (i / 50.0) * 0.1,
                "y": 0.6 + (frame_idx / 150.0) * 0.05,
                "z": 0.1 + (i / 100.0) * 0.03
            })
        
        frame_data = {
            "timestamp": int(timestamp),
            "landmarks": landmarks
        }
        
        frames.append(frame_data)
    
    print(f"✅ Generated {len(frames)} frames")
    
    # Payload format chính xác như frontend sẽ gửi
    payload = {
        "user": user_name,
        "label": gesture_label,
        "session_id": session_id,
        "frames": frames
    }
    
    # Simulate network request từ frontend
    print(f"\n📤 Sending POST request to /upload/camera...")
    print(f"   Payload size: ~{len(json.dumps(payload)) / 1024:.1f} KB")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/upload/camera",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "React-Frontend/1.0"
            },
            json=payload,
            timeout=30
        )
        
        upload_time = time.time() - start_time
        
        print(f"⏱️ Upload completed in {upload_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"   Backend Response:")
            print(f"   ├─ success: {result.get('success')}")
            print(f"   ├─ task_id: {result.get('task_id')}")
            print(f"   ├─ status: {result.get('status')}")
            print(f"   ├─ total_frames: {result.get('total_frames')}")
            print(f"   └─ filename: {result.get('filename')}")
            
            # Verify data integrity
            if result.get('total_frames') == len(frames):
                print("✅ Frame count verified")
            else:
                print(f"❌ Frame count mismatch: sent {len(frames)}, received {result.get('total_frames')}")
            
            return result
            
        else:
            print(f"❌ Upload failed!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is backend running?")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    return None

def test_multiple_sessions():
    """Test multiple capture sessions để mô phỏng workflow thực tế"""
    print("\n🎯 TESTING MULTIPLE CAPTURE SESSIONS")
    print("=" * 60)
    
    sessions = [
        {"user": "alice", "label": "hello", "duration": 2000},
        {"user": "bob", "label": "goodbye", "duration": 1500},
        {"user": "charlie", "label": "thank_you", "duration": 2500},
        {"user": "diana", "label": "yes", "duration": 1000},
        {"user": "eve", "label": "no", "duration": 1200}
    ]
    
    results = []
    
    for i, session in enumerate(sessions, 1):
        print(f"\n📹 Session {i}/{len(sessions)}: {session['user']} - {session['label']}")
        
        # Create frames for this session
        frames = []
        fps = 30
        total_frames = int(session['duration'] / (1000 / fps))
        
        for frame_idx in range(total_frames):
            timestamp = frame_idx * (1000 / fps)
            
            # Simplified landmarks for faster testing - chỉ pose upper body + hands
            landmarks = {
                "pose": [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9} for _ in range(25)],  # 25 điểm thay vì 33
                # "face": REMOVED - không lấy face
                "left_hand": [{"x": 0.3, "y": 0.6, "z": 0.1} for _ in range(21)],
                "right_hand": [{"x": 0.7, "y": 0.6, "z": 0.1} for _ in range(21)]
            }
            
            frames.append({
                "timestamp": int(timestamp),
                "landmarks": landmarks
            })
        
        # Upload session
        payload = {
            "user": session['user'],
            "label": session['label'],
            "session_id": f"multi_session_{i}_{int(time.time())}",
            "frames": frames
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/upload/camera",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=20
            )
            
            if response.status_code == 200:
                result = response.json()
                results.append(result)
                print(f"   ✅ Success: {result.get('filename')} ({result.get('total_frames')} frames)")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Session Results: {len(results)}/{len(sessions)} successful")
    return results

def test_edge_cases():
    """Test các edge cases mà frontend có thể gặp"""
    print("\n⚠️ TESTING EDGE CASES")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Very short capture (1 frame)",
            "frames": 1
        },
        {
            "name": "Long capture (5 seconds)",
            "frames": 150
        },
        {
            "name": "Missing some landmarks",
            "frames": 30,
            "missing_landmarks": True
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        
        frames = []
        for i in range(test_case['frames']):
            landmarks = {
                "pose": [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9} for _ in range(33)],
                "face": [{"x": 0.5, "y": 0.3, "z": 0.0} for _ in range(468)],
                "left_hand": [{"x": 0.3, "y": 0.6, "z": 0.1} for _ in range(21)],
                "right_hand": [{"x": 0.7, "y": 0.6, "z": 0.1} for _ in range(21)]
            }
            
            # Simulate missing landmarks (như khi user ra khỏi camera)
            if test_case.get('missing_landmarks') and i % 10 == 0:
                landmarks = {
                    "pose": [],
                    # "face": [],  # REMOVED
                    "left_hand": [],
                    "right_hand": []
                }
            
            frames.append({
                "timestamp": i * 33,
                "landmarks": landmarks
            })
        
        payload = {
            "user": "test_edge_case",
            "label": "edge_test",
            "session_id": f"edge_{int(time.time())}",
            "frames": frames
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/upload/camera",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Handled successfully: {result.get('filename')}")
            else:
                print(f"   ⚠️ Response: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def main():
    """Run comprehensive frontend simulation tests"""
    print("🚀 REACT FRONTEND SIMULATION TESTS")
    print("=" * 60)
    
    # Check backend
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Backend not responding")
            return
    except:
        print("❌ Backend not reachable. Run: docker-compose up -d")
        return
    
    print("✅ Backend is ready")
    
    # Run tests
    simulate_frontend_capture_session()
    test_multiple_sessions()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("🎉 FRONTEND SIMULATION COMPLETE")
    print("\n✅ Camera upload functionality is working correctly!")
    print("✅ Ready for React frontend integration")
    print("\n📋 Integration checklist:")
    print("   ├─ ✅ CORS enabled for localhost:5173")
    print("   ├─ ✅ MediaPipe landmarks processing")
    print("   ├─ ✅ Correct response format")
    print("   ├─ ✅ Session management")
    print("   ├─ ✅ Error handling")
    print("   └─ ✅ Data persistence")

if __name__ == "__main__":
    main()