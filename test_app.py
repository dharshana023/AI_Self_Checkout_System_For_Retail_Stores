#!/usr/bin/env python3
"""
Test script for AI Self-Checkout System
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing dependencies...")
    
    try:
        import streamlit
        print(f"✅ Streamlit: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not found")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO available")
    except ImportError:
        print("❌ Ultralytics not found")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not found")
        return False
    
    return True

def test_model_files():
    """Test if model files exist"""
    print("\n📁 Testing model files...")
    
    yolo_model = "yolov8n.pt"
    custom_model = "custom_model.h5"
    
    if os.path.exists(yolo_model):
        size = os.path.getsize(yolo_model) / (1024 * 1024)  # MB
        print(f"✅ YOLO model found: {yolo_model} ({size:.1f} MB)")
    else:
        print(f"❌ YOLO model not found: {yolo_model}")
        return False
    
    if os.path.exists(custom_model):
        size = os.path.getsize(custom_model) / (1024 * 1024)  # MB
        print(f"✅ Custom model found: {custom_model} ({size:.1f} MB)")
    else:
        print(f"⚠️ Custom model not found: {custom_model}")
        print("   Product classification will not work, but YOLO detection will still function.")
    
    return True

def test_sample_images():
    """Test if sample images exist"""
    print("\n🖼️ Testing sample images...")
    
    sample_dir = "sample"
    if os.path.exists(sample_dir):
        images = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif'))]
        print(f"✅ Sample directory found with {len(images)} images")
        for img in images[:5]:  # Show first 5
            print(f"   - {img}")
        if len(images) > 5:
            print(f"   ... and {len(images) - 5} more")
        return True
    else:
        print(f"⚠️ Sample directory not found: {sample_dir}")
        return False

def create_test_image():
    """Create a simple test image"""
    print("\n🎨 Creating test image...")
    
    try:
        # Create a simple test image with colored rectangles
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
        
        # Add some colored rectangles to simulate products
        cv2.rectangle(img, (50, 50), (200, 150), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(img, (250, 50), (400, 150), (255, 0, 0), -1)  # Blue rectangle
        cv2.rectangle(img, (450, 50), (550, 150), (0, 0, 255), -1)  # Red rectangle
        
        # Add text labels
        cv2.putText(img, "Test Product 1", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Test Product 2", (260, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Test Product 3", (460, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save test image
        test_image_path = "test_image.jpg"
        cv2.imwrite(test_image_path, img)
        print(f"✅ Test image created: {test_image_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating test image: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 AI Self-Checkout System - Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Test dependencies
    if not test_dependencies():
        all_passed = False
    
    # Test model files
    if not test_model_files():
        all_passed = False
    
    # Test sample images
    test_sample_images()
    
    # Create test image
    create_test_image()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All critical tests passed! You can run the Streamlit app.")
        print("\n🚀 To start the application, run:")
        print("   streamlit run streamlit_app.py")
        print("\n   Or use the helper script:")
        print("   python run_app.py")
    else:
        print("❌ Some tests failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    
    print("\n📝 Note: If the custom model is missing, only YOLO detection will work.")
    print("   The app will still function but won't classify specific products.")

if __name__ == "__main__":
    main()