#!/usr/bin/env python3
"""
Test script để kiểm tra augmentation cho camera upload
"""

import requests
import json
import time
import uuid

API_BASE = "http://localhost:8000"

def test_camera_upload_augmentation():
    """Test xem camera upload có tạo multiple samples không"""
    
    # Tạo test data
    test_frames = []
    for i in range(30):  # 30 frames
        frame = {
            "timestamp": i * 33,  # ~30fps
            "landmarks": {
                "left_hand": [{"x": 0.3 + 0.1 * (i/30), "y": 0.6, "z": 0.1} for _ in range(21)],
                "right_hand": [{"x": 0.7 - 0.1 * (i/30), "y": 0.6, "z": 0.1} for _ in range(21)]
            }
        }
        test_frames.append(frame)
    
    # Payload for upload
    payload = {
        "user": "test_user_aug",
        "label": "test_augmentation",
        "session_id": uuid.uuid4().hex,
        "dialect": "vietnamese",
        "frames": test_frames
    }
    
    print("🧪 Testing camera upload augmentation...")
    print(f"📊 Input: {len(test_frames)} frames")
    
    try:
        response = requests.post(
            f"{API_BASE}/upload/camera",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"📁 Response: {json.dumps(result, indent=2)}")
            
            # Check if multiple samples were created
            total_samples = result.get("total_samples", 1)
            paths = result.get("paths", [result.get("path")])
            
            print(f"\n📈 Augmentation Results:")
            print(f"   Total samples created: {total_samples}")
            print(f"   Number of paths: {len(paths) if paths else 0}")
            
            if total_samples > 1:
                print("✅ Augmentation is working - multiple samples created!")
            else:
                print("❌ Augmentation not working - only 1 sample created")
                
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_augmentation_functions():
    """Test augmentation functions directly"""
    print("\n🔧 Testing augmentation functions directly...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
        
        from backend.app.processing.augmenter import generate_augmented_sequences
        import numpy as np
        
        # Create test sequence
        test_seq = np.random.rand(60, 126).astype(np.float32)  # 60 frames, 126 features
        
        print(f"📊 Input sequence shape: {test_seq.shape}")
        
        # Generate augmented sequences
        augmented_list = generate_augmented_sequences(test_seq)
        
        print(f"📈 Generated {len(augmented_list)} augmented sequences:")
        for i, aug_seq in enumerate(augmented_list):
            print(f"   {i+1}. Shape: {aug_seq.shape}, Type: {type(aug_seq)}")
        
        if len(augmented_list) > 1:
            print("✅ Augmentation functions working correctly!")
        else:
            print("❌ Augmentation functions not generating multiple sequences")
            
    except Exception as e:
        print(f"❌ Error testing augmentation functions: {e}")

if __name__ == "__main__":
    print("🚀 Starting augmentation tests...\n")
    
    # Test 1: Direct function test
    test_augmentation_functions()
    
    # Test 2: API test (requires server to be running)
    print(f"\n{'='*50}")
    test_camera_upload_augmentation()
    
    print(f"\n{'='*50}")
    print("🏁 Test completed!")