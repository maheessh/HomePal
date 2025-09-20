#!/usr/bin/env python3
"""
Test script to verify the connection error fixes
"""

import requests
import time
import json

def test_stream_endpoint():
    """Test the improved stream endpoint"""
    print("🧪 Testing improved stream endpoint...")
    
    try:
        # Test stream endpoint when camera server is not running
        response = requests.get('http://localhost:5000/stream', timeout=5)
        print(f"✅ Stream endpoint response: {response.status_code}")
        
        if response.status_code == 503:
            try:
                data = response.json()
                print(f"   Error message: {data.get('error', 'Unknown error')}")
            except:
                print("   Response is not JSON")
        elif response.status_code == 200:
            print("   Stream is working!")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Stream test failed: {e}")

def test_camera_api():
    """Test camera API endpoints"""
    print("\n🧪 Testing camera API endpoints...")
    
    # Test status endpoint
    try:
        response = requests.get('http://localhost:5000/api/camera/status', timeout=5)
        print(f"✅ Camera status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Camera status test failed: {e}")
    
    # Test start endpoint
    try:
        response = requests.post('http://localhost:5000/api/camera/start', timeout=10)
        print(f"✅ Camera start: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Response: {data.get('message', 'No message')}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Camera start test failed: {e}")

if __name__ == "__main__":
    print("🚀 Testing connection error fixes...")
    print("⚠️  Make sure the main application is running on http://localhost:5000")
    
    try:
        test_stream_endpoint()
        test_camera_api()
        print("\n✅ Connection fix tests completed!")
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
