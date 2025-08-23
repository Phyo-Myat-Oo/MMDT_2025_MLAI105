#!/usr/bin/env python3
"""
Helper script to run the Myanmar Article Classification Streamlit app
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['streamlit', 'tensorflow', 'numpy', 'pandas']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit app"""
    print("🚀 Starting Myanmar Article Classification App...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\nTo stop the app, press Ctrl+C\n")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

def main():
    """Main function"""
    print("🇲🇲 Myanmar Article Classification - Streamlit App")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('streamlit_app.py'):
        print("❌ Error: streamlit_app.py not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Check if model exists
    model_path = os.path.join("2_training", "output_model", "bilstm_model.h5")
    if not os.path.exists(model_path):
        print("❌ Error: Trained model not found!")
        print(f"Expected: {model_path}")
        print("Please train the model first using the bilstm_pipeline.py")
        return
    
    print("✅ Model found!")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        install_choice = input("Install missing packages automatically? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_dependencies():
                print("❌ Failed to install dependencies. Please install manually:")
                print("pip install -r requirements.txt")
                return
        else:
            print("Please install missing packages manually:")
            print("pip install -r requirements.txt")
            return
    
    print("✅ All dependencies available!")
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main()