#!/usr/bin/env python3
"""
Ví dụ thu thập nhiều sample trong một session
"""

import requests
import json
import time
import uuid

API_BASE = "http://localhost:8000"
session_id = uuid.uuid4().hex  # Session chung cho tất cả samples

def generate_sample_frames(gesture_name, frame_count=30):
    """Tạo frames mô phỏng cho một gesture"""
    frames = []
    for i in range(frame_count):
        # Mô phỏng landmarks khác nhau cho từng gesture
        if gesture_name == "hello":
            # Wave motion
            hand_x = 0.5 + 0.2 * math.sin(i * 0.3)
            hand_y = 0.5 + 0.1 * math.cos(i * 0.3)
        elif gesture_name == "goodbye":
            # Different wave pattern
            hand_x = 0.6 + 0.15 * math.cos(i * 0.4)
            hand_y = 0.4 + 0.2 * math.sin(i * 0.4)
        else:
            # Default motion
            hand_x = 0.5
            hand_y = 0.5
            
        frame = {
            "timestamp": i * 33,  # ~30fps
            "landmarks": {
                "pose": generate_pose_landmarks(),
                "left_hand": generate_hand_landmarks(hand_x - 0.1, hand_y),
                "right_hand": generate_hand_landmarks(hand_x + 0.1, hand_y)
            }
        }
        frames.append(frame)
    return frames

def generate_pose_landmarks():
    """Generate basic pose landmarks"""
    pose = []
    for i in range(25):  # Upper body only
        pose.append({
            "x": 0.5 + (i-12) * 0.02,
            "y": 0.3 + i * 0.02,
            "z": 0.1,
            "visibility": 0.9
        })
    return pose

def generate_hand_landmarks(center_x, center_y):
    """Generate hand landmarks around center point"""
    hand = []
    for i in range(21):
        hand.append({
            "x": center_x + (i-10) * 0.01,
            "y": center_y + (i-10) * 0.01,
            "z": 0.2
        })
    return hand

def upload_sample(user, label, frames, session_id):
    """Upload một sample với nhiều frames"""
    payload = {
        "user": user,
        "label": label,
        "session_id": session_id,
        "dialect": "vietnamese",
        "frames": frames
    }
    
    response = requests.post(f"{API_BASE}/upload/camera", json=payload)
    return response.json()

def main():
    user_name = "test_user"
    
    # Danh sách gestures cần thu thập trong session này
    gestures = [
        ("hello", 45),      # 45 frames cho gesture hello
        ("goodbye", 40),    # 40 frames cho gesture goodbye  
        ("thank_you", 35),  # 35 frames cho gesture thank_you
        ("yes", 30),        # 30 frames cho gesture yes
        ("no", 35)          # 35 frames cho gesture no
    ]
    
    print(f"🎬 Bắt đầu thu thập session: {session_id}")
    print(f"📋 Sẽ thu {len(gestures)} gestures khác nhau")
    
    import math  # Import needed for math functions
    
    successful_uploads = 0
    
    for i, (gesture, frame_count) in enumerate(gestures, 1):
        print(f"\n📹 [{i}/{len(gestures)}] Thu thập gesture: {gesture} ({frame_count} frames)")
        
        try:
            # Tạo frames cho gesture này
            frames = generate_sample_frames(gesture, frame_count)
            
            # Upload sample
            result = upload_sample(user_name, gesture, frames, session_id)
            
            if result.get("success"):
                print(f"✅ Thành công: {result.get('message')}")
                print(f"   📁 File path: {result.get('path', 'N/A')}")
                successful_uploads += 1
            else:
                print(f"❌ Thất bại: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Lỗi khi upload {gesture}: {e}")
        
        # Nghỉ giữa các uploads
        if i < len(gestures):
            time.sleep(1)
    
    print(f"\n🎯 Hoàn thành session!")
    print(f"✅ Thành công: {successful_uploads}/{len(gestures)} samples")
    print(f"🆔 Session ID: {session_id}")

if __name__ == "__main__":
    main()