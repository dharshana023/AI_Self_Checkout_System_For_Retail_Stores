import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import time
from datetime import datetime
import pandas as pd
import threading
import queue
import os

# Configure Streamlit page
st.set_page_config(
    page_title="AI Self-Checkout - Live Camera",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Product classes and prices
classes = {
    'dove shampoo': 185,
    'lays': 10,
    'marble cake': 30,
    'maaza': 42,
    'munch': 5,
    'thums up': 50,
    'timepass biscuit': 25
}

# YOLO class names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli","carrot", "hot dog", 
    "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed","diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "mobile phone","microwave","oven","toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors","teddy bear", "hair drier", "toothbrush"
]

@st.cache_resource
def load_models():
    """Load YOLO and custom product classification models"""
    try:
        # Load YOLO model
        yolo_model = YOLO('yolov8n.pt')
        
        # Load custom product classification model
        products_model = None
        if os.path.exists('custom_model.h5'):
            try:
                products_model = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(224, 224, 3)),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                    
                    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                    
                    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'),
                    
                    tf.keras.layers.Flatten(),
                    
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(7, activation='softmax')
                ])
                products_model.load_weights('custom_model.h5')
                st.success("✅ Custom product classification model loaded successfully!")
            except Exception as e:
                st.warning(f"⚠️ Error loading custom model: {str(e)}")
                products_model = None
        else:
            st.warning("⚠️ Custom model weights not found. Only YOLO detection will work.")
        
        return yolo_model, products_model
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

def make_square_with_padding(image):
    """Add padding to make image square"""
    try:
        h, w, _ = image.shape
        max_side = max(h, w)
        
        square_img = np.ones((max_side, max_side, 3), dtype=np.uint8) * 255
        
        x_center = (max_side - w) // 2
        y_center = (max_side - h) // 2
        
        square_img[y_center:y_center+h, x_center:x_center+w] = image
        
        return square_img
    except Exception as e:
        return image

def process_image_for_classification(image):
    """Process image for product classification"""
    try:
        target_size = (224, 224)
        
        # Ensure image is in correct format
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # BGR
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:  # Already RGB
                image_rgb = image
        else:
            # Grayscale to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        img = cv2.resize(image_rgb, target_size)
        img = img.astype(np.float32) / 255.0  
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        return None

def detect_products_in_frame(frame, yolo_model, products_model, detection_threshold=0.5):
    """Detect and classify products in a single frame"""
    try:
        results = yolo_model(frame, stream=False, verbose=False)
        detected_items = []
        annotated_frame = frame.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    try:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        conf = float(box.conf[0].cpu().numpy())
                        id = int(box.cls[0].cpu().numpy())
                        
                        if id < len(classNames):
                            class_name = classNames[id]
                        else:
                            class_name = "unknown"
                        
                        # Draw bounding box
                        color = (0, 255, 0) if conf > detection_threshold else (255, 255, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Process objects that are not persons and have reasonable confidence
                        if class_name != "person" and conf > 0.3:
                            # Validate bounding box
                            if y2 > y1 and x2 > x1 and y1 >= 0 and x1 >= 0:
                                # Crop the detected object
                                cropped_img = frame[y1:y2, x1:x2]
                                
                                if cropped_img.size > 0 and cropped_img.shape[0] > 10 and cropped_img.shape[1] > 10:
                                    try:
                                        if products_model is not None:
                                            # Process for product classification
                                            padded_img = make_square_with_padding(cropped_img)
                                            processed_img = process_image_for_classification(padded_img)
                                            
                                            if processed_img is not None:
                                                prediction = products_model.predict(processed_img, verbose=0)
                                                max_confidence = np.max(prediction)
                                                
                                                if max_confidence > 0.4:  # Lower threshold for better detection
                                                    predicted_class_index = np.argmax(prediction)
                                                    predicted_class = list(classes.keys())[predicted_class_index]
                                                    
                                                    detected_items.append({
                                                        'name': predicted_class,
                                                        'price': classes[predicted_class],
                                                        'confidence': max_confidence,
                                                        'bbox': (x1, y1, x2, y2),
                                                        'yolo_class': class_name,
                                                        'yolo_conf': conf
                                                    })
                                                    
                                                    # Add label to frame
                                                    label = f'{predicted_class} - Rs.{classes[predicted_class]}'
                                                    cv2.putText(annotated_frame, label, (x1, max(40, y1-10)), 
                                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                                else:
                                                    # Show YOLO detection with low product confidence
                                                    label = f'{class_name} (unidentified)'
                                                    cv2.putText(annotated_frame, label, (x1, max(40, y1-10)), 
                                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                        else:
                                            # No product model, just show YOLO detection
                                            label = f'{class_name} ({conf:.2f})'
                                            cv2.putText(annotated_frame, label, (x1, max(40, y1-10)), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    except Exception as e:
                                        continue
                        else:
                            # Label persons and low-confidence objects
                            label = f'{class_name} ({conf:.2f})'
                            cv2.putText(annotated_frame, label, (x1, max(40, y1-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    except Exception as e:
                        continue
        
        return detected_items, annotated_frame
    
    except Exception as e:
        return [], frame

def generate_invoice(cart_items, total_price):
    """Generate a digital invoice/receipt"""
    try:
        # Group items by name and count
        item_counts = {}
        for item in cart_items:
            name = item['name']
            if name in item_counts:
                item_counts[name]['count'] += 1
            else:
                item_counts[name] = {'item': item, 'count': 1}
        
        # Create invoice text
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        transaction_id = f"TXN{int(time.time())}"
        
        invoice_text = f"""
╔═══════════════════════════════════════╗
║            AI SELF-CHECKOUT           ║
╠═══════════════════════════════════════╣
║ Date: {current_time:<25} ║
║ Transaction ID: {transaction_id:<19} ║
╠═══════════════════════════════════════╣
║ Items Purchased:                      ║
╠═════���═════════════════════════════════╣"""
        
        for name, data in item_counts.items():
            item = data['item']
            count = data['count']
            subtotal = item['price'] * count
            if count > 1:
                item_line = f"║ {name.title():<20} {count}x₹{item['price']} = ₹{subtotal:>6} ║"
            else:
                item_line = f"║ {name.title():<25} ₹{item['price']:>6} ║"
            invoice_text += f"\n{item_line}"
        
        invoice_text += f"""
╠═══════════════════════════════════════╣
║ Total Amount:              ₹{total_price:>8} ║
╚═══════════════════════════════════════╝
        Thank you for shopping with us!
        """
        
        return invoice_text, transaction_id
    
    except Exception as e:
        return None, None

class LiveCameraDetection:
    def __init__(self, yolo_model, products_model):
        self.yolo_model = yolo_model
        self.products_model = products_model
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.detection_queue = queue.Queue(maxsize=5)
        
    def start_camera(self, camera_index=0):
        """Start camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                return False
            
            self.running = True
            return True
        except Exception as e:
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def get_frame(self):
        """Get current frame from camera"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None
    
    def detect_in_frame(self, frame):
        """Detect products in current frame"""
        return detect_products_in_frame(frame, self.yolo_model, self.products_model)

def main():
    st.title("🛒 AI Self-Checkout - Live Camera Detection")
    st.markdown("---")
    
    # Initialize session state
    if 'cart_items' not in st.session_state:
        st.session_state.cart_items = []
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        yolo_model, products_model = load_models()
    
    if yolo_model is None:
        st.error("❌ Failed to load YOLO model. Please check if yolov8n.pt exists.")
        return
    
    # Sidebar - Cart Summary
    st.sidebar.title("🛒 Shopping Cart")
    if st.session_state.cart_items:
        cart_total = sum(item['price'] for item in st.session_state.cart_items)
        
        # Group items for display
        item_counts = {}
        for item in st.session_state.cart_items:
            name = item['name']
            if name in item_counts:
                item_counts[name]['count'] += 1
            else:
                item_counts[name] = {'item': item, 'count': 1}
        
        st.sidebar.markdown(f"**Items in Cart: {len(st.session_state.cart_items)}**")
        st.sidebar.markdown(f"**Total: ₹{cart_total}**")
        
        # Display cart items
        for name, data in item_counts.items():
            item = data['item']
            count = data['count']
            subtotal = item['price'] * count
            st.sidebar.write(f"• {name.title()} × {count} = ₹{subtotal}")
        
        st.sidebar.markdown("---")
        
        # Cart actions
        if st.sidebar.button("🛒 Checkout", type="primary"):
            invoice_text, transaction_id = generate_invoice(st.session_state.cart_items, cart_total)
            if invoice_text:
                st.success(f"✅ Payment of ₹{cart_total} processed successfully!")
                st.balloons()
                
                # Display invoice
                st.subheader("🧾 Invoice/Receipt")
                st.text(invoice_text)
                
                # Download invoice
                st.download_button(
                    label="📥 Download Receipt",
                    data=invoice_text,
                    file_name=f"receipt_{transaction_id}.txt",
                    mime="text/plain"
                )
                
                # Clear cart
                st.session_state.cart_items = []
                st.rerun()
        
        if st.sidebar.button("🗑️ Clear Cart"):
            st.session_state.cart_items = []
            st.success("Cart cleared!")
            st.rerun()
    else:
        st.sidebar.info("Cart is empty")
    
    # Main content
    st.header("📹 Live Camera Detection")
    
    # Camera controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📹 Start Camera", type="primary"):
            st.session_state.camera_active = True
    
    with col2:
        if st.button("⏹️ Stop Camera"):
            st.session_state.camera_active = False
    
    with col3:
        detection_sensitivity = st.slider("Detection Sensitivity", 0.3, 0.9, 0.5, 0.1)
    
    # Camera feed and detection
    if st.session_state.camera_active:
        st.info("🔴 Camera is active - Real-time detection in progress...")
        
        # Create camera detection instance
        camera_detector = LiveCameraDetection(yolo_model, products_model)
        
        if camera_detector.start_camera():
            # Create placeholders for dynamic content
            frame_placeholder = st.empty()
            detection_placeholder = st.empty()
            
            # Detection loop
            frame_count = 0
            last_detection_time = time.time()
            
            try:
                while st.session_state.camera_active:
                    frame = camera_detector.get_frame()
                    
                    if frame is not None:
                        frame_count += 1
                        
                        # Detect products every few frames to reduce processing load
                        if frame_count % 10 == 0:  # Process every 10th frame
                            detected_items, annotated_frame = camera_detector.detect_in_frame(frame)
                            
                            # Display annotated frame
                            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                            
                            # Update detection results
                            if detected_items:
                                with detection_placeholder.container():
                                    st.subheader("🎯 Live Detections")
                                    
                                    cols = st.columns(min(len(detected_items), 3))
                                    for i, item in enumerate(detected_items):
                                        with cols[i % 3]:
                                            st.markdown(f"**{item['name'].title()}**")
                                            st.markdown(f"₹{item['price']} ({item['confidence']:.2f})")
                                            
                                            if st.button(f"➕ Add", key=f"live_add_{frame_count}_{i}"):
                                                st.session_state.cart_items.append(item)
                                                st.success(f"Added {item['name']} to cart!")
                                                time.sleep(0.5)  # Brief pause to show success message
                                                st.rerun()
                            
                            last_detection_time = time.time()
                        else:
                            # Just display the current frame without detection
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.1)
                    
                    # Check if we should stop (this is a simplified check)
                    if time.time() - last_detection_time > 30:  # Stop after 30 seconds of inactivity
                        break
                        
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
            finally:
                camera_detector.stop_camera()
        else:
            st.error("❌ Failed to start camera. Please check camera permissions and availability.")
            st.session_state.camera_active = False
    
    else:
        st.markdown("""
        ### 📸 How to use Live Camera Detection:
        
        1. **Click "Start Camera"** to begin live detection
        2. **Position products** in front of the camera
        3. **Products will be detected automatically** in real-time
        4. **Click "Add"** to add detected items to your cart
        5. **Use the sidebar** to manage your cart and checkout
        
        ### 💡 Tips for best results:
        - Ensure good lighting conditions
        - Hold products steady in camera view
        - Avoid rapid movements
        - Keep products clearly visible and unobstructed
        - Adjust detection sensitivity if needed
        
        ### ⚙️ System Requirements:
        - Camera/webcam access required
        - Good internet connection for model processing
        - Modern browser with camera permissions enabled
        """)
    
    # Detection history
    if st.session_state.detection_history:
        with st.expander("📊 Detection History"):
            for i, detection in enumerate(st.session_state.detection_history[-10:]):  # Show last 10
                st.write(f"{i+1}. {detection['name']} - ₹{detection['price']} ({detection['time']})")

if __name__ == "__main__":
    main()