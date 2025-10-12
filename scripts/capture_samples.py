"""Utility script to capture personal calibration samples"""

import cv2
import argparse
from pathlib import Path
import time
from datetime import datetime


def capture_samples(output_dir: str, class_name: str, duration: int = 60, interval: int = 2):
    """
    Capture calibration samples for a specific class
    
    Args:
        output_dir: Directory to save samples
        class_name: Name of the class being captured (e.g., 'focused')
        duration: Total capture duration in seconds
        interval: Seconds between captures
    """
    # Create output directory
    class_dir = Path(output_dir) / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Failed to open camera")
    
    print(f"\nCapturing samples for class: {class_name}")
    print(f"Duration: {duration}s, Interval: {interval}s")
    print(f"Saving to: {class_dir}")
    print(f"\nPlease position yourself as: {class_name}")
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    start_time = time.time()
    count = 0
    
    try:
        while time.time() - start_time < duration:
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Save frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{class_name}_{timestamp}_{count:04d}.jpg"
            filepath = class_dir / filename
            cv2.imwrite(str(filepath), frame)
            
            count += 1
            remaining = duration - (time.time() - start_time)
            print(f"Captured {count} samples | {remaining:.1f}s remaining", end='\r')
            
            # Show preview
            cv2.imshow('Capture Preview (Press Q to quit)', frame)
            if cv2.waitKey(interval * 1000) & 0xFF == ord('q'):
                break
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
    
    print(f"\n\nCapture complete! Saved {count} samples to {class_dir}")


def main():
    parser = argparse.ArgumentParser(description="Capture personal calibration samples")
    parser.add_argument('--output', type=str, default='data/personal', 
                       help='Output directory for samples')
    parser.add_argument('--class_name', type=str, required=True,
                       choices=['focused', 'looking_away', 'using_phone', 'yawning', 'sleepy'],
                       help='Class to capture')
    parser.add_argument('--duration', type=int, default=60,
                       help='Capture duration in seconds')
    parser.add_argument('--interval', type=int, default=2,
                       help='Interval between captures in seconds')
    
    args = parser.parse_args()
    
    capture_samples(args.output, args.class_name, args.duration, args.interval)


if __name__ == "__main__":
    main()

