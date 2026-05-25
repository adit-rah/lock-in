"""Prepare State Farm distracted-driver dataset for binary Lock-In training.

State Farm provides 10 classes (c0 = safe driving, c1-c9 = various distractions)
across ~22.4K training images. This script:

1. Maps c0 -> focused, c1..c9 -> distracted.
2. Reads driver_imgs_list.csv (ships with the Kaggle release) and splits
   subjects between train/ and val/ so the same driver never appears in
   both splits. This is the realistic split — random per-image splits
   produce inflated metrics because consecutive frames of the same driver
   are nearly identical.
3. Symlinks the images into out_dir/{train,val}/{focused,distracted}/ by
   default (no 4GB copy). Use --copy if your filesystem can't symlink.

Usage:
    python scripts/prepare_state_farm.py \
        --kaggle_dir /path/to/state-farm-distracted-driver-detection \
        --out_dir data/state_farm_binary

The kaggle_dir should be the extracted archive root containing imgs/train/
and driver_imgs_list.csv.
"""

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


STATE_FARM_TO_BINARY: Dict[str, str] = {
    'c0': 'focused',
    'c1': 'distracted',
    'c2': 'distracted',
    'c3': 'distracted',
    'c4': 'distracted',
    'c5': 'distracted',
    'c6': 'distracted',
    'c7': 'distracted',
    'c8': 'distracted',
    'c9': 'distracted',
}


def load_driver_imgs_list(csv_path: Path) -> List[Tuple[str, str, str]]:
    """Returns list of (subject, classname, img) tuples from driver_imgs_list.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"driver_imgs_list.csv not found at {csv_path}. "
            "This file ships with the Kaggle State Farm release and is required "
            "for driver-disjoint splits."
        )
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row['subject'], row['classname'], row['img']))
    return rows


def split_subjects(subjects: List[str], val_fraction: float, seed: int) -> Tuple[set, set]:
    """Driver-disjoint split: each subject goes entirely to train or val."""
    rng = random.Random(seed)
    unique = sorted(set(subjects))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    val_set = set(unique[:n_val])
    train_set = set(unique[n_val:])
    return train_set, val_set


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def prepare(kaggle_dir: Path, out_dir: Path, val_fraction: float, seed: int, copy: bool) -> Dict:
    imgs_train = kaggle_dir / "imgs" / "train"
    if not imgs_train.is_dir():
        raise FileNotFoundError(
            f"Expected State Farm images at {imgs_train}. "
            "Did you point --kaggle_dir at the extracted archive root?"
        )

    rows = load_driver_imgs_list(kaggle_dir / "driver_imgs_list.csv")
    subjects = [r[0] for r in rows]
    train_subjects, val_subjects = split_subjects(subjects, val_fraction, seed)

    counts: Dict[str, Counter] = defaultdict(Counter)
    per_subject: Dict[str, int] = Counter()

    for subject, classname, img in rows:
        binary_class = STATE_FARM_TO_BINARY.get(classname)
        if binary_class is None:
            continue
        src = imgs_train / classname / img
        if not src.exists():
            continue
        split = "train" if subject in train_subjects else "val"
        dst = out_dir / split / binary_class / f"{subject}_{classname}_{img}"
        link_or_copy(src, dst, copy=copy)
        counts[split][binary_class] += 1
        per_subject[subject] += 1

    return {
        'train_subjects': sorted(train_subjects),
        'val_subjects': sorted(val_subjects),
        'counts': {split: dict(c) for split, c in counts.items()},
        'total_images': sum(per_subject.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--kaggle_dir', required=True,
                        help='Extracted State Farm archive root (contains imgs/train/ and driver_imgs_list.csv)')
    parser.add_argument('--out_dir', required=True,
                        help='Output root; will create train/{focused,distracted} and val/{focused,distracted}')
    parser.add_argument('--val_fraction', type=float, default=0.2,
                        help='Fraction of unique drivers held out for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed for driver split (default: 42)')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files instead of symlinking (slower, uses ~4GB disk)')
    args = parser.parse_args()

    kaggle_dir = Path(args.kaggle_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = prepare(kaggle_dir, out_dir, args.val_fraction, args.seed, copy=args.copy)

    print(f"\nWrote {report['total_images']} images to {out_dir}")
    print(f"Drivers — train: {len(report['train_subjects'])}, val: {len(report['val_subjects'])}")
    for split in ('train', 'val'):
        c = report['counts'].get(split, {})
        focused = c.get('focused', 0)
        distracted = c.get('distracted', 0)
        total = focused + distracted
        ratio = focused / total if total else 0
        print(f"  {split}: focused={focused}, distracted={distracted} ({ratio*100:.1f}% focused)")


if __name__ == "__main__":
    main()
