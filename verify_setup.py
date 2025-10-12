"""Verify Lock-In setup and dependencies"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
        return False


def check_import(module_name, display_name=None):
    """Check if a module can be imported"""
    if display_name is None:
        display_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {display_name}")
        return True
    except ImportError:
        print(f"✗ {display_name} (not installed)")
        return False


def check_camera():
    """Check if camera is accessible"""
    try:
        import cv2
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            print("✓ Webcam accessible")
            cam.release()
            return True
        else:
            print("✗ Webcam not accessible")
            return False
    except Exception as e:
        print(f"✗ Webcam check failed: {e}")
        return False


def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU available: {gpu_name}")
            return True
        else:
            print("⚠ GPU not available (CPU mode will be used)")
            return False
    except Exception as e:
        print(f"⚠ GPU check failed: {e}")
        return False


def check_model():
    """Check if model exists"""
    model_path = Path("models/distraction_classifier.pt")
    if model_path.exists():
        print(f"✓ Model found: {model_path}")
        return True
    else:
        print(f"⚠ Model not found: {model_path}")
        print("  → Train a model first: python -m src.train --data_dir <path>")
        return False


def check_config():
    """Check if config exists"""
    config_path = Path("config.yaml")
    if config_path.exists():
        print(f"✓ Config found: {config_path}")
        return True
    else:
        print(f"✗ Config not found: {config_path}")
        return False


def main():
    """Run all checks"""
    print("="*60)
    print("Lock-In Monitor - Setup Verification")
    print("="*60)
    
    results = []
    
    print("\n[Python Version]")
    results.append(check_python_version())
    
    print("\n[Required Dependencies]")
    results.append(check_import("torch", "PyTorch"))
    results.append(check_import("torchvision", "torchvision"))
    results.append(check_import("cv2", "OpenCV (cv2)"))
    results.append(check_import("numpy", "NumPy"))
    results.append(check_import("PIL", "Pillow"))
    results.append(check_import("yaml", "PyYAML"))
    
    print("\n[Optional Dependencies]")
    check_import("win10toast", "win10toast (Windows notifications)")
    check_import("plyer", "plyer (cross-platform notifications)")
    check_import("tqdm", "tqdm (progress bars)")
    check_import("pandas", "pandas (data analysis)")
    
    print("\n[Hardware]")
    results.append(check_camera())
    check_gpu()  # GPU is optional
    
    print("\n[Project Files]")
    results.append(check_config())
    check_model()  # Model is optional for initial setup
    
    print("\n" + "="*60)
    
    if all(results):
        print("✓ All critical checks passed! You're ready to go.")
        print("\nNext steps:")
        print("  1. Train a model: python -m src.train --data_dir <path>")
        print("  2. Start monitoring: python -m src.app")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nInstall missing dependencies:")
        print("  pip install -r requirements.txt")
    
    print("="*60)


if __name__ == "__main__":
    main()

