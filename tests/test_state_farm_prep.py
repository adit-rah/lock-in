"""Tests for scripts/prepare_state_farm.py — binary mapping and driver-disjoint splits."""

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_state_farm import (
    STATE_FARM_TO_BINARY,
    prepare,
    split_subjects,
)


def _build_fake_kaggle_dir(root: Path, subjects_per_class: int = 3, imgs_per_subject: int = 2) -> Path:
    """Create a tiny State Farm-shaped tree: imgs/train/c0..c9/ + driver_imgs_list.csv."""
    rows = []
    img_idx = 0
    for cls_idx in range(10):
        cls = f"c{cls_idx}"
        cls_dir = root / "imgs" / "train" / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for s in range(subjects_per_class):
            subject = f"p{cls_idx:02d}{s}"
            for _ in range(imgs_per_subject):
                img_name = f"img_{img_idx}.jpg"
                Image.new("RGB", (4, 4), color=(cls_idx * 25, 0, 0)).save(cls_dir / img_name)
                rows.append({'subject': subject, 'classname': cls, 'img': img_name})
                img_idx += 1

    csv_path = root / "driver_imgs_list.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'classname', 'img'])
        writer.writeheader()
        writer.writerows(rows)
    return root


def test_class_mapping_is_binary():
    assert STATE_FARM_TO_BINARY['c0'] == 'focused'
    for i in range(1, 10):
        assert STATE_FARM_TO_BINARY[f'c{i}'] == 'distracted'
    assert set(STATE_FARM_TO_BINARY.values()) == {'focused', 'distracted'}


def test_split_subjects_is_disjoint():
    subjects = [f"p{i}" for i in range(20)]
    train, val = split_subjects(subjects, val_fraction=0.25, seed=7)
    assert train.isdisjoint(val)
    assert train | val == set(subjects)
    assert len(val) == 5


def test_prepare_writes_binary_split(tmp_path):
    kaggle_dir = _build_fake_kaggle_dir(tmp_path / "kaggle")
    out_dir = tmp_path / "out"
    report = prepare(kaggle_dir, out_dir, val_fraction=0.34, seed=1, copy=True)

    # Driver-disjoint
    assert set(report['train_subjects']).isdisjoint(report['val_subjects'])

    # Both splits exist, both class folders exist in each
    for split in ('train', 'val'):
        for cls in ('focused', 'distracted'):
            d = out_dir / split / cls
            assert d.is_dir(), f"missing {d}"

    # c0 -> focused, c1..c9 -> distracted
    # focused image count == imgs_per_subject(2) * focused subjects
    total_focused = report['counts']['train'].get('focused', 0) + report['counts']['val'].get('focused', 0)
    total_distracted = report['counts']['train'].get('distracted', 0) + report['counts']['val'].get('distracted', 0)
    assert total_focused == 3 * 2  # 3 subjects in c0 * 2 imgs each
    assert total_distracted == 9 * 3 * 2  # 9 distracted classes * 3 subjects * 2 imgs

    # Total matches
    assert report['total_images'] == total_focused + total_distracted


def test_prepare_symlinks_by_default(tmp_path):
    kaggle_dir = _build_fake_kaggle_dir(tmp_path / "kaggle", subjects_per_class=1, imgs_per_subject=1)
    out_dir = tmp_path / "out"
    prepare(kaggle_dir, out_dir, val_fraction=0.5, seed=0, copy=False)

    # At least one symlink should exist in the output tree
    found_symlink = False
    for p in out_dir.rglob("*"):
        if p.is_symlink():
            found_symlink = True
            break
    assert found_symlink, "expected --copy=False to produce symlinks"
