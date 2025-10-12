"""Extract frames from YawDD video dataset"""

import cv2
import argparse
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, fps: int = 1, max_frames: int = None):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract (1 = one frame per second)
        max_frames: Maximum frames to extract (None = all)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            output_file = output_dir / f"{video_path.stem}_frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(output_file), frame)
            saved_count += 1
            
            if max_frames and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    return saved_count


def process_yawdd_dataset(input_dir: str, output_dir: str, fps: int = 1):
    """
    Process YawDD dataset videos
    
    Expects structure:
      input_dir/
        ├── Yawn/
        │   ├── video1.avi
        │   └── video2.avi
        └── Normal/
            ├── video1.avi
            └── video2.avi
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Process Yawn videos
    yawn_dir = input_path / "Yawn"
    if yawn_dir.exists():
        yawn_output = output_path / "yawning"
        yawn_output.mkdir(parents=True, exist_ok=True)
        
        print("Processing yawn videos...")
        total = 0
        for video_file in yawn_dir.glob("*.avi"):
            print(f"  Extracting {video_file.name}...")
            count = extract_frames(video_file, yawn_output, fps=fps)
            total += count
        print(f"Extracted {total} yawn frames")
    
    # Process Normal videos (can be used for focused class)
    normal_dir = input_path / "Normal"
    if normal_dir.exists():
        normal_output = output_path / "focused"
        normal_output.mkdir(parents=True, exist_ok=True)
        
        print("\nProcessing normal videos...")
        total = 0
        for video_file in normal_dir.glob("*.avi"):
            print(f"  Extracting {video_file.name}...")
            count = extract_frames(video_file, normal_output, fps=fps)
            total += count
        print(f"Extracted {total} normal frames")
    
    print(f"\nProcessing complete! Output: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from YawDD video dataset"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to YawDD videos directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for extracted frames'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=1,
        help='Frames per second to extract (default: 1)'
    )
    
    args = parser.parse_args()
    
    process_yawdd_dataset(args.input, args.output, args.fps)


if __name__ == "__main__":
    main()

