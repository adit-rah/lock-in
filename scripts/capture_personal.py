"""Capture personal webcam footage for fine-tuning.

Records webcam frames into data/personal/{train,val}/{focused,distracted}/
matching the layout `src/train.py` expects. Two passes per class — one for
train, one for val — so you get a real driver-disjoint-ish split without
re-recording.

Typical use:

    # 1 minute focused (looking at the screen, working normally)
    python -m scripts.capture_personal --class_name focused --train_seconds 60

    # 1 minute distracted (looking at phone, looking away, etc.)
    python -m scripts.capture_personal --class_name distracted --train_seconds 60

After both classes are captured, fine-tune from your existing model:

    python -m src.train \\
        --data_dir data/personal \\
        --resume checkpoints/best_model_epoch_3.pth \\
        --epochs 5 --lr 0.0001 --freeze_backbone

You can re-run capture for either class to add more samples; existing files
aren't deleted.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


VALID_CLASSES = ("focused", "distracted")


def _drain_buffer(cap: cv2.VideoCapture, n: int = 5) -> None:
    for _ in range(n):
        cap.read()


def capture_split(cap: cv2.VideoCapture, class_name: str, out_dir: Path,
                  duration_s: float, fps: float, split_label: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / fps if fps > 0 else 0.0
    print(f"\n[{split_label}] Recording '{class_name}' for {duration_s:.0f}s at {fps:g} fps")
    print("  Get into position. Starting in 3...")
    for i in range(3, 0, -1):
        time.sleep(1)
        print(f"  {i}...")
    print("  Recording. Behave naturally; vary your pose/angle a little.")

    start = time.time()
    count = 0
    next_capture = start
    while time.time() - start < duration_s:
        # Drop stale frames so we get the freshest one
        cap.grab(); cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"{class_name}_{split_label}_{stamp}.jpg"
        cv2.imwrite(str(out_path), frame)
        count += 1
        next_capture += interval
        sleep_for = max(0.0, next_capture - time.time())
        if sleep_for > 0:
            time.sleep(sleep_for)
        remaining = duration_s - (time.time() - start)
        print(f"\r  captured {count:4d}  |  {remaining:5.1f}s left", end="", flush=True)
    print()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--class_name", required=True, choices=VALID_CLASSES,
                        help="Which class this recording is of.")
    parser.add_argument("--output", default="data/personal",
                        help="Output root (default: data/personal).")
    parser.add_argument("--train_seconds", type=float, default=60.0,
                        help="Seconds to record into train/ (default: 60).")
    parser.add_argument("--val_seconds", type=float, default=20.0,
                        help="Seconds to record into val/ (default: 20). "
                             "Set to 0 to skip the val pass.")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Capture rate (default: 2 fps — gives you ~120 frames per minute). "
                             "Higher = more samples but more near-duplicates.")
    parser.add_argument("--camera", type=int, default=0, help="cv2 camera index (default: 0).")
    args = parser.parse_args()

    out_root = Path(args.output)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera}")

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass
    _drain_buffer(cap)

    try:
        if args.train_seconds > 0:
            n_train = capture_split(
                cap, args.class_name,
                out_root / "train" / args.class_name,
                args.train_seconds, args.fps, "train",
            )
        else:
            n_train = 0

        if args.val_seconds > 0:
            print("\nNow the validation pass. Move slightly — change angle, lighting if you can.")
            time.sleep(1)
            n_val = capture_split(
                cap, args.class_name,
                out_root / "val" / args.class_name,
                args.val_seconds, args.fps, "val",
            )
        else:
            n_val = 0
    finally:
        cap.release()

    print(f"\nDone. {args.class_name}: train={n_train}, val={n_val}")
    print(f"Output: {out_root.resolve()}")
    print("\nWhen you've captured both classes, fine-tune:")
    print("  python -m src.train --data_dir data/personal \\")
    print("      --resume checkpoints/best_model_epoch_3.pth \\")
    print("      --epochs 5 --lr 0.0001 --freeze_backbone")


if __name__ == "__main__":
    main()
