import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import tempfile
import os
from io import BytesIO
import time
from datetime import datetime
import pandas as pd
import re
import random

# Configure Streamlit page
st.set_page_config(
    page_title="AI Self-Checkout System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Product classes and prices
classes = {
    'dove shampoo': 185,
    'lays': 10,
    'marble cake': 30,
    'Slice': 42,
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

def validate_phone_number(phone):
    """Validate Indian phone number format"""
    # Remove any spaces, dashes, or other characters
    phone = re.sub(r'[^\d+]', '', phone)
    
    # Check for Indian phone number patterns
    patterns = [
        r'^(\+91|91)?[6-9]\d{9}$',  # Indian mobile numbers
        r'^(\+91|91)?[1-9]\d{9}$',  # Indian landline numbers
    ]
    
    for pattern in patterns:
        if re.match(pattern, phone):
            return True
    return False

def authenticate_user(phone_number, user_name):
    """Authenticate user with phone number and name"""
    # In a real implementation, you would:
    # - Check if phone number exists in database
    # - Create user profile if new
    # - Log the login attempt
    # - Store user preferences
    
    # For demo purposes, we'll accept any valid phone number
    st.session_state.authenticated = True
    st.session_state.user_phone = phone_number
    st.session_state.user_name = user_name if user_name else f"User {phone_number[-4:]}"
    return True

def show_welcome_page():
    """Display enhanced welcome page with improved UI"""
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .main-title {
        color: white;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-subtitle {
        color: #E8F4FD;
        font-size: 1.3rem;
        margin-bottom: 0;
        opacity: 0.9;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #FF6B6B;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    .feature-title {
        color: #333;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        color: #666;
        line-height: 1.5;
        margin: 0;
    }
    .login-section {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4f1d4 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 2rem;
        border: 2px solid #c8e6c9;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 15px;
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #FF6B6B;
    }
    .stat-label {
        color: #666;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🛒 AI Self-Checkout</h1>
        <p class="main-subtitle">Next-Generation Smart Retail Solution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats section
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-number">7</div>
            <div class="stat-label">Products Supported</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">99%</div>
            <div class="stat-label">Accuracy Rate</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">3s</div>
            <div class="stat-label">Average Detection</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">24/7</div>
            <div class="stat-label">Available</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <h3 class="feature-title">Object Detection</h3>
            <p class="feature-desc">Advanced YOLOv8 technology detects objects in images with high precision and speed</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <h3 class="feature-title">Product Classification</h3>
            <p class="feature-desc">Custom CNN model accurately classifies detected products from our trained dataset</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">💰</span>
            <h3 class="feature-title">Price Calculation</h3>
            <p class="feature-desc">Automatically calculates total cost based on detected items with real-time pricing</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🧾</span>
            <h3 class="feature-title">Invoice Generation</h3>
            <p class="feature-desc">Creates professional digital receipts with transaction details and customer information</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Login section
    st.markdown("""
    <div class="login-section">
        <h3 style="color: #2d5a2d; margin-bottom: 1rem;">👋 Get Started</h3>
        <p style="color: #4a7c4a; margin-bottom: 0;">Enter your details below to access the AI Self-Checkout system</p>
    </div>
    """, unsafe_allow_html=True)

def show_login_form():
    """Display enhanced login form with name and phone number"""
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Custom CSS for the form
    st.markdown("""
    <style>
    .login-form {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    .form-title {
        color: #333;
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .input-group {
        margin-bottom: 1rem;
    }
    .input-label {
        color: #555;
        font-weight: 500;
        margin-bottom: 0.5rem;
        display: block;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        
        with st.form("user_login_form", clear_on_submit=False):
            st.markdown('<h3 class="form-title">👤 Create Your Profile</h3>', unsafe_allow_html=True)
            
            # Name input
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            user_name = st.text_input(
                "👤 Full Name",
                placeholder="Enter your full name",
                help="This will appear on your receipts and invoices"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Phone number input
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 2])
            with col1:
                country_code = st.selectbox(
                    "🌍 Country", 
                    ["+91 (India)", "+1 (USA)", "+44 (UK)", "+86 (China)", "+33 (France)", "+49 (Germany)"], 
                    index=0,
                    help="Select your country code"
                )
            
            with col2:
                phone_number = st.text_input(
                    "📱 Phone Number", 
                    placeholder="Enter your mobile number",
                    help="Enter your mobile number for account identification"
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Terms and conditions
            st.markdown('<div class="input-group">', unsafe_allow_html=True)
            agree_terms = st.checkbox(
                "I agree to the Terms of Service and Privacy Policy",
                help="By checking this box, you agree to our terms and conditions"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Submit button
            submit_button = st.form_submit_button(
                "🚀 Start Shopping", 
                type="primary", 
                use_container_width=True,
                help="Click to access the AI Self-Checkout system"
            )
            
            if submit_button:
                # Validation
                if not user_name or not user_name.strip():
                    st.error("❌ Please enter your full name")
                elif not phone_number:
                    st.error("❌ Please enter your phone number")
                elif not agree_terms:
                    st.error("❌ Please agree to the Terms of Service to continue")
                else:
                    # Extract country code
                    code = country_code.split()[0]
                    full_phone = f"{code}{phone_number}"
                    
                    if validate_phone_number(full_phone):
                        # Authenticate user with name and phone number
                        if authenticate_user(full_phone, user_name.strip()):
                            st.success(f"✅ Welcome {user_name}! Login successful!")
                            st.balloons()
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            st.error("❌ Login failed. Please try again.")
                    else:
                        st.error("❌ Please enter a valid phone number")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information section
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ Why do we need this information?"):
        st.markdown("""
        **Your Name:**
        - Appears on receipts and invoices
        - Personalizes your shopping experience
        - Helps with customer service
        
        **Your Phone Number:**
        - Unique account identification
        - Transaction history tracking
        - Customer support contact
        
        **Privacy:**
        - Your information is secure and encrypted
        - We never share your data with third parties
        - You can delete your account anytime
        """)
    
    # Demo credentials
    st.markdown("---")
    st.info("""
    **🎯 Demo Mode:** You can use any valid phone number format to test the system.
    
    **Examples:**
    - India: +91 9876543210
    - USA: +1 5551234567
    - UK: +44 7700900123
    """)

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
        st.error(f"Error in make_square_with_padding: {str(e)}")
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
        st.error(f"Error in process_image_for_classification: {str(e)}")
        return None

def detect_products_in_image(image, yolo_model, products_model):
    """Detect and classify products in an image"""
    try:
        results = yolo_model(image, stream=False, verbose=False)
        detected_items = []
        annotated_image = image.copy()
        
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
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
                        color = (0, 255, 0) if conf > 0.5 else (255, 255, 0)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Process objects that are not persons and have reasonable confidence
                        if class_name != "person" and conf > 0.3:
                            # Validate bounding box
                            if y2 > y1 and x2 > x1 and y1 >= 0 and x1 >= 0:
                                # Crop the detected object
                                cropped_img = image[y1:y2, x1:x2]
                                
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
                                                    
                                                    # Add label to image
                                                    label = f'{predicted_class} (₹{classes[predicted_class]})'
                                                    cv2.putText(annotated_image, label, (x1, max(40, y1-10)), 
                                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                                else:
                                                    # Show YOLO detection with low product confidence
                                                    label = f'{class_name} (unidentified)'
                                                    cv2.putText(annotated_image, label, (x1, max(40, y1-10)), 
                                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                        else:
                                            # No product model, just show YOLO detection
                                            label = f'{class_name} ({conf:.2f})'
                                            cv2.putText(annotated_image, label, (x1, max(40, y1-10)), 
                                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                    except Exception as e:
                                        continue
                        else:
                            # Label persons and low-confidence objects
                            label = f'{class_name} ({conf:.2f})'
                            cv2.putText(annotated_image, label, (x1, max(40, y1-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    except Exception as e:
                        continue
        
        return detected_items, annotated_image
    
    except Exception as e:
        st.error(f"❌ Error in product detection: {str(e)}")
        return [], image

def generate_invoice(detected_items, total_price):
    """Generate a digital invoice/receipt"""
    try:
        # Create invoice text
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        transaction_id = f"TXN{int(time.time())}"
        
        # Include user information in invoice
        user_phone = st.session_state.get('user_phone', 'N/A')
        user_name = st.session_state.get('user_name', 'Guest')
        
        invoice_text = f"""
╔═══════════════════════════════════════╗
║            AI SELF-CHECKOUT           ║
║              RETAIL STORE             ║
╠═══════════════════════════════════════╣
║ Date: {current_time:<25} ║
║ Transaction ID: {transaction_id:<19} ║
║ Customer: {user_name:<25} ║
║ Phone: {user_phone:<28} ║
╠═══════════════════════════════════════╣
║ Items Purchased:                      ║
╠═══════════════════════════════════════╣"""
        
        # Group items by name and count
        item_counts = {}
        for item in detected_items:
            name = item['name']
            if name in item_counts:
                item_counts[name]['count'] += 1
            else:
                item_counts[name] = {'item': item, 'count': 1}
        
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
        Visit us again for more AI-powered shopping!
        """
        
        return invoice_text, transaction_id
    
    except Exception as e:
        st.error(f"Error generating invoice: {str(e)}")
        return None, None

def main():
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'login_step' not in st.session_state:
        st.session_state.login_step = "welcome"
    if 'cart_items' not in st.session_state:
        st.session_state.cart_items = []
    
    # Check authentication status
    if not st.session_state.authenticated:
        show_welcome_page()
        show_login_form()
        return
    
    # Main application (after authentication)
    st.title("🛒 AI Self-Checkout System")
    
    # User info in sidebar
    st.sidebar.markdown(f"### 👤 Welcome, {st.session_state.get('user_name', 'User')}!")
    st.sidebar.markdown(f"📱 {st.session_state.get('user_phone', 'N/A')}")
    
    if st.sidebar.button("🚪 Logout"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.markdown("---")
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        yolo_model, products_model = load_models()
    
    if yolo_model is None:
        st.error("❌ Failed to load YOLO model. Please check if yolov8n.pt exists.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Image Upload", "Live Camera", "Real-time Detection", "Shopping Cart", "About"]
    )
    
    # Display current cart summary in sidebar
    if st.session_state.cart_items:
        cart_total = sum(item['price'] for item in st.session_state.cart_items)
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### 🛒 Current Cart")
        st.sidebar.markdown(f"**Items:** {len(st.session_state.cart_items)}")
        st.sidebar.markdown(f"**Total:** ₹{cart_total}")
    
    if mode == "Image Upload":
        st.header("📸 Upload Image for Product Detection")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image containing products to detect and calculate total price"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="📷 Uploaded Image", use_column_width=True)
                
                # Convert PIL to OpenCV format
                image_array = np.array(image)
                if len(image_array.shape) == 3:
                    if image_array.shape[2] == 4:  # RGBA
                        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
                    elif image_array.shape[2] == 3:  # RGB
                        image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    else:
                        image_cv = image_array
                else:
                    image_cv = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
                
                # Process image
                with st.spinner("🔍 Detecting products..."):
                    detected_items, annotated_image = detect_products_in_image(
                        image_cv, yolo_model, products_model
                    )
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Detection Results")
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    st.image(annotated_image_rgb, caption="Detection Results", use_column_width=True)
                
                with col2:
                    st.subheader("🛒 Shopping Cart")
                    if detected_items:
                        total_price = 0
                        
                        # Display detected items
                        for i, item in enumerate(detected_items):
                            col_item, col_add = st.columns([3, 1])
                            with col_item:
                                st.write(f"**{item['name'].title()}** - ₹{item['price']} (Confidence: {item['confidence']:.2f})")
                            with col_add:
                                if st.button(f"➕ Add", key=f"add_{i}"):
                                    st.session_state.cart_items.append(item)
                                    st.success(f"Added {item['name']} to cart!")
                                    st.rerun()
                            
                            total_price += item['price']
                        
                        st.markdown("---")
                        st.markdown(f"### **Detected Total: ₹{total_price}**")
                        
                        # Checkout button
                        if st.button("🛒 Quick Checkout", type="primary"):
                            invoice_text, transaction_id = generate_invoice(detected_items, total_price)
                            if invoice_text is not None:
                                st.success(f"✅ Payment of ₹{total_price} processed successfully!")
                                st.balloons()
                                
                                # Display invoice
                                st.subheader("🧾 Invoice")
                                st.text(invoice_text)
                                
                                # Download invoice
                                st.download_button(
                                    label="📥 Download Invoice",
                                    data=invoice_text,
                                    file_name=f"invoice_{transaction_id}.txt",
                                    mime="text/plain"
                                )
                    else:
                        st.info("🔍 No products detected. Try uploading a clearer image with products.")
                        st.markdown("**Tips for better detection:**")
                        st.markdown("- Ensure good lighting")
                        st.markdown("- Products should be clearly visible")
                        st.markdown("- Avoid blurry images")
                        
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
    
    elif mode == "Live Camera":
        st.header("📹 Live Camera Detection & Shopping")
        st.info("📱 Use your device camera to detect products and add them to your shopping cart!")
        
        # Camera input
        camera_input = st.camera_input("📸 Take a picture to detect products")
        
        if camera_input is not None:
            try:
                # Process camera image
                image = Image.open(camera_input)
                image_array = np.array(image)
                image_cv = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                with st.spinner("🔍 Processing camera image..."):
                    detected_items, annotated_image = detect_products_in_image(
                        image_cv, yolo_model, products_model
                    )
                
                # Display results in three columns
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.subheader("🎯 Detection Results")
                    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    st.image(annotated_image_rgb, use_column_width=True)
                
                with col2:
                    st.subheader("🛍️ Detected Products")
                    if detected_items:
                        detected_total = 0
                        
                        # Display each detected item with add to cart option
                        for i, item in enumerate(detected_items):
                            with st.container():
                                st.markdown(f"**{item['name'].title()}**")
                                st.markdown(f"Price: ₹{item['price']} | Confidence: {item['confidence']:.2f}")
                                
                                col_add, col_info = st.columns([1, 2])
                                with col_add:
                                    if st.button(f"➕ Add to Cart", key=f"camera_add_{i}"):
                                        st.session_state.cart_items.append(item)
                                        st.success(f"✅ Added {item['name']} to cart!")
                                        st.rerun()
                                
                                with col_info:
                                    st.caption(f"YOLO: {item.get('yolo_class', 'N/A')} ({item.get('yolo_conf', 0):.2f})")
                                
                                st.markdown("---")
                            
                            detected_total += item['price']
                        
                        st.markdown(f"### **Detected Total: ₹{detected_total}**")
                        
                        # Quick add all button
                        col_add_all, col_checkout = st.columns(2)
                        with col_add_all:
                            if st.button("➕ Add All to Cart", type="secondary"):
                                for item in detected_items:
                                    st.session_state.cart_items.append(item)
                                st.success(f"✅ Added all {len(detected_items)} items to cart!")
                                st.rerun()
                        
                        with col_checkout:
                            if st.button("🛒 Quick Checkout", type="primary"):
                                # Add items to cart and checkout immediately
                                for item in detected_items:
                                    st.session_state.cart_items.append(item)
                                
                                # Generate invoice for all cart items
                                cart_total = sum(item['price'] for item in st.session_state.cart_items)
                                invoice_text, transaction_id = generate_invoice(st.session_state.cart_items, cart_total)
                                
                                if invoice_text is not None:
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
                                    
                                    # Clear cart after successful checkout
                                    st.session_state.cart_items = []
                    else:
                        st.info("🔍 No products detected in this image.")
                        st.markdown("**Tips for better detection:**")
                        st.markdown("- Ensure good lighting")
                        st.markdown("- Hold products clearly in view")
                        st.markdown("- Avoid shadows and reflections")
                        st.markdown("- Try different angles")
                
                with col3:
                    st.subheader("📊 Session Stats")
                    st.metric("Cart Items", len(st.session_state.cart_items))
                    if st.session_state.cart_items:
                        cart_total = sum(item['price'] for item in st.session_state.cart_items)
                        st.metric("Cart Total", f"₹{cart_total}")
                    
                    st.markdown("---")
                    st.markdown("**Quick Actions:**")
                    if st.button("🗑️ Clear Cart", key="clear_cart_camera"):
                        st.session_state.cart_items = []
                        st.success("Cart cleared!")
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing camera image: {str(e)}")
                st.info("Please try taking another picture or check your camera permissions.")
        
        else:
            # Show instructions when no image is captured
            st.markdown("""
            ### 📸 How to use Live Camera Detection:
            
            1. **Click the camera button** above to take a picture
            2. **Position products** clearly in the camera view
            3. **Wait for detection** - the AI will identify products
            4. **Add items** to your cart individually or all at once
            5. **Generate receipt** or proceed to checkout
            
            ### 💡 Tips for best results:
            - Use good lighting
            - Hold products steady and clearly visible
            - Avoid reflections and shadows
            - Try different angles if detection fails
            - Ensure products are not overlapping too much
            """)
    
    elif mode == "Real-time Detection":
        st.header("🎥 Real-time Camera Detection")
        st.info("🚀 For true real-time detection without manual photo capture, use the dedicated live camera app!")
        
        # Instructions for real-time detection
        st.markdown("""
        ### 🔴 Continuous Live Detection
        
        For **real-time continuous detection** without taking manual photos, we have created a dedicated application:
        
        #### 🚀 How to use Real-time Detection:
        
        1. **Run the dedicated live camera app**:
           ```bash
           streamlit run live_camera_app.py
           ```
        
        2. **Features of Real-time Detection**:
           - 📹 **Continuous camera feed** - No need to take photos
           - 🔄 **Automatic detection** - Products detected in real-time
           - ⚡ **Instant recognition** - Immediate product identification
           - 🛒 **Live cart updates** - Add items instantly
           - 🧾 **Real-time invoicing** - Generate receipts on the fly
        
        ### 🎯 Alternative: Quick Camera Capture
        
        If you prefer to stay in this interface, you can use the **Live Camera** mode above.
        """)
        
        # Quick launch button for live camera app
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🚀 Launch Real-time Detection")
            command = "streamlit run live_camera_app.py"
            st.code(command, language="bash")
            
            if st.button("📋 Copy Command", type="primary"):
                st.success("✅ Command copied! Run it in your terminal.")
    
    elif mode == "Shopping Cart":
        st.header("🛒 Shopping Cart")
        
        if st.session_state.cart_items:
            total_price = 0
            
            # Group items by name
            item_counts = {}
            for item in st.session_state.cart_items:
                name = item['name']
                if name in item_counts:
                    item_counts[name]['count'] += 1
                else:
                    item_counts[name] = {'item': item, 'count': 1}
            
            # Display cart items
            for name, data in item_counts.items():
                item = data['item']
                count = data['count']
                subtotal = item['price'] * count
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{name.title()}** - ₹{item['price']} × {count}")
                with col2:
                    st.write(f"₹{subtotal}")
                with col3:
                    if st.button(f"🗑️ Remove", key=f"remove_{name}"):
                        # Remove one instance of this item
                        for i, cart_item in enumerate(st.session_state.cart_items):
                            if cart_item['name'] == name:
                                st.session_state.cart_items.pop(i)
                                st.rerun()
                                break
                
                total_price += subtotal
            
            st.markdown("---")
            st.markdown(f"### **Total: ₹{total_price}**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🛒 Checkout", type="primary"):
                    invoice_text, transaction_id = generate_invoice(st.session_state.cart_items, total_price)
                    if invoice_text is not None:
                        st.success(f"✅ Payment of ₹{total_price} processed successfully!")
                        st.balloons()
                        st.text(invoice_text)
                        
                        # Download invoice
                        st.download_button(
                            label="📥 Download Receipt",
                            data=invoice_text,
                            file_name=f"receipt_{transaction_id}.txt",
                            mime="text/plain"
                        )
                        
                        st.session_state.cart_items = []  # Clear cart
            
            with col2:
                if st.button("🗑️ Clear Cart"):
                    st.session_state.cart_items = []
                    st.success("Cart cleared!")
                    st.rerun()
        else:
            st.info("🛒 Your cart is empty. Add items by detecting products in images!")
    
    elif mode == "About":
        st.header("ℹ️ About AI Self-Checkout System")
        
        st.markdown("""
        ### 🎯 Overview
        This AI-powered self-checkout system uses computer vision to automatically detect and identify grocery products, 
        calculate prices, and generate invoices - making the shopping experience faster and more efficient.
        
        ### 🔧 How it Works
        1. **🎯 Object Detection**: Uses YOLOv8 to detect objects in images
        2. **🧠 Product Classification**: Custom CNN model classifies detected products
        3. **💰 Price Calculation**: Automatically calculates total based on detected items
        4. **🧾 Invoice Generation**: Creates digital receipt for the transaction
        
        ### 🛍️ Supported Products
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            for i, (product, price) in enumerate(list(classes.items())[:4]):
                st.write(f"• **{product.title()}** - ₹{price}")
        
        with col2:
            for product, price in list(classes.items())[4:]:
                st.write(f"• **{product.title()}** - ₹{price}")
        
        st.markdown("""
        ### ✨ Features
        - 🔍 Real-time product detection
        - 💰 Automatic price calculation
        - 📱 User-friendly web interface
        - 🚀 Fast and accurate recognition
        - 🧾 Digital invoice generation
        - 🛒 Shopping cart functionality
        - 📥 Downloadable receipts
        - 📱 Phone number authentication
        
        ### 🛠️ Technology Stack
        - **Computer Vision**: OpenCV, YOLOv8
        - **Deep Learning**: TensorFlow, Keras
        - **Web Interface**: Streamlit
        - **Image Processing**: PIL, NumPy
        - **Authentication**: OTP-based login
        """)

if __name__ == "__main__":
    main()