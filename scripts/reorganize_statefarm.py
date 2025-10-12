"""Reorganize State Farm dataset into Lock-In classes"""

import shutil
from pathlib import Path
import argparse


# Mapping from State Farm classes to Lock-In classes
STATEFARM_TO_LOCKIN = {
    'c0': 'focused',        # Safe driving
    'c1': 'using_phone',    # Texting - right
    'c2': 'using_phone',    # Phone - right
    'c3': 'using_phone',    # Texting - left
    'c4': 'using_phone',    # Phone - left
    'c5': 'looking_away',   # Operating radio
    'c6': 'looking_away',   # Drinking
    'c7': 'looking_away',   # Reaching behind
    'c8': 'looking_away',   # Hair and makeup
    'c9': 'looking_away',   # Talking to passenger
}


def reorganize_statefarm(input_dir: str, output_dir: str, copy: bool = True):
    """
    Reorganize State Farm dataset structure
    
    Args:
        input_dir: Path to State Farm imgs/train directory
        output_dir: Output directory for Lock-In structure
        copy: If True, copy files; if False, create symlinks
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output class directories
    for lockin_class in set(STATEFARM_TO_LOCKIN.values()):
        (output_path / lockin_class).mkdir(parents=True, exist_ok=True)
    
    # Process each State Farm class
    total_files = 0
    for statefarm_class, lockin_class in STATEFARM_TO_LOCKIN.items():
        class_dir = input_path / statefarm_class
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping")
            continue
        
        # Copy/link all images
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        for img_path in image_files:
            output_file = output_path / lockin_class / f"{statefarm_class}_{img_path.name}"
            
            if copy:
                shutil.copy2(img_path, output_file)
            else:
                if output_file.exists():
                    output_file.unlink()
                output_file.symlink_to(img_path.absolute())
            
            total_files += 1
        
        print(f"Processed {statefarm_class} ({len(image_files)} images) -> {lockin_class}")
    
    print(f"\nTotal files processed: {total_files}")
    print(f"Output directory: {output_path.absolute()}")
    
    # Print class distribution
    print("\nClass distribution:")
    for lockin_class in sorted(set(STATEFARM_TO_LOCKIN.values())):
        count = len(list((output_path / lockin_class).glob("*")))
        print(f"  {lockin_class}: {count} images")


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize State Farm dataset for Lock-In training"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to State Farm imgs/train directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for reorganized data'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='Create symlinks instead of copying files (saves space)'
    )
    
    args = parser.parse_args()
    
    reorganize_statefarm(args.input, args.output, copy=not args.symlink)


if __name__ == "__main__":
    main()

