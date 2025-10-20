#!/usr/bin/env python3
"""
Test script để demo API endpoints của Sign Dataset Backend
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    """Demo các API endpoints chính"""
    
    print("🔍 TESTING SIGN DATASET BACKEND APIs")
    print("=" * 50)
    
    # 1. Test health check
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print(f"✅ Swagger UI: {response.status_code}")
    except:
        print("❌ Server chưa khởi động")
        return
    
    # 2. Test create label
    print("\n📝 Creating new label...")
    label_data = {
        "label": "hello",
        "notes": "Greeting gesture",
        "version": "v1"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/dataset/labels", data=label_data)
        print(f"Label creation: {response.status_code}")
        if response.status_code == 200:
            label_info = response.json()
            print(f"   Class ID: {label_info['class_idx']}")
            print(f"   Folder: {label_info['folder_name']}")
    except Exception as e:
        print(f"❌ Label creation failed: {e}")
    
    # 3. Test list labels
    print("\n📋 Listing all labels...")
    try:
        response = requests.get(f"{BASE_URL}/dataset/labels")
        labels = response.json()
        print(f"Found {len(labels)} labels:")
        for label in labels:
            print(f"   - {label['label_original']} (class_{label['class_idx']})")
    except Exception as e:
        print(f"❌ List labels failed: {e}")
    
    # 4. Test upload endpoint info
    print("\n📤 Upload Video Endpoint:")
    print(f"   POST {BASE_URL}/upload/video")
    print("   Form fields:")
    print("   - file: video file")
    print("   - user: username") 
    print("   - label: gesture label")
    print("   - session_id: optional session ID")
    
    # 5. Test job status endpoint info
    print("\n📊 Job Status Endpoint:")
    print(f"   GET {BASE_URL}/jobs/{{job_id}}")
    print("   Returns: job status, result, errors")

if __name__ == "__main__":
    test_api_endpoints()