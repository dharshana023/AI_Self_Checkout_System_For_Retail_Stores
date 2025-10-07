# AI-Self-Checkout

Automated AI self checkout system using CNN and Yolo. This project was created in collaboration with &nbsp;[Rishi Bhattasali](https://github.com/Rishi2403) for Tech AI hackathon organized by IIC, TMSL.

## Sample Video

https://github.com/user-attachments/assets/4e32061c-bb0e-4733-a0e6-3e0ed40aa990

## About

This projects aims to enhance the efficiency of grocery retail outlets with our AI-driven self-checkout system, leveraging computer vision technology to enable quick and seamless transactions, ensuring a streamlined shopping experience for customers. Our solution aims to replace traditional cashiers, thereby improving profit margins and enabling automated AI-driven sales tracking and inventory replenishment.

## Data

The datataset used in this project is created by our team and is available in the `data` folder which includes images for 7 Indian daily used grocery items. 

This is a sample dataset, our future goal is to create a large dataset for various items to increase the efficiency as well coverage of our model.

## Usage

### Option 1: Streamlit Web Application (Recommended)

1. After downloading the dataset, unzip it and place in the root directory.
   
2. Clone the repository
```
git clone https://github.com/priyanshudutta04/AI-Self-Checkout.git
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Run the Streamlit Web Application
```
streamlit run streamlit_app.py
```

Or use the helper script:
```
python run_app.py
```

The application will open in your web browser at `http://localhost:8501`

### Option 2: Original OpenCV Application

Run the original camera-based application:
```
python yolo_products.py
```

### Features

#### Streamlit Web App
- 🖼️ **Image Upload**: Upload images for product detection
- 📷 **Camera Integration**: Use your device camera for real-time detection
- 🛒 **Shopping Cart**: Automatic price calculation and checkout
- 📱 **User-Friendly Interface**: Clean, intuitive web interface
- 📊 **Digital Receipts**: Automatic invoice generation

#### Original App
- 🎥 **Real-time Detection**: Live camera feed with product recognition
- ⌨️ **Keyboard Controls**: Press 'Q' to quit, 'C' to clear cart

<br/>

*Note: If GPU is available install `cuda toolkit` and `cuDNN` for faster execution during Model-Training*

## Contributing

Contributions are welcome! If you have ideas for improving the model or adding new features, please feel free to fork the repository and submit a pull request or open a issue.

## Support

If you like this project, do give it a ⭐and share it with your friends.






