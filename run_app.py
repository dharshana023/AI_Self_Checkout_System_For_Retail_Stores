#!/usr/bin/env python3
"""
Script to run the AI Self-Checkout Streamlit application
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application"""
    try:
        # Check if streamlit is installed
        subprocess.run([sys.executable, "-c", "import streamlit"], check=True, capture_output=True)
        
        # Run the Streamlit app
        print("Starting AI Self-Checkout System...")
        print("The application will open in your default web browser.")
        print("Press Ctrl+C to stop the application.")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
        
    except subprocess.CalledProcessError:
        print("Error: Streamlit is not installed.")
        print("Please install the required dependencies:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error running the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()