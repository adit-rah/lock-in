"""Training script for distraction classifier"""

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from tqdm import tqdm

from .config import Config, load_config
from .model import create_model, pick_device, save_model_torchscript


class DistractionDataset(Dataset):
    """Image-folder dataset for distraction detection.

    Expects subdirectories under root_dir, one per class. Class index is
    determined by sorted directory order so that two datasets pointing at
    train/ and val/ siblings agree on label encoding.
    """

    def __init__(self, root_dir: str, transform=None, class_to_idx: Optional[Dict[str, int]] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        if class_to_idx is None:
            self.classes = [d.name for d in class_dirs]
            self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        else:
            self.class_to_idx = dict(class_to_idx)
            self.classes = [c for c, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

        for class_dir in class_dirs:
            label = self.class_to_idx.get(class_dir.name)
            if label is None:
                continue
            for img_path in class_dir.rglob("*"):
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    self.samples.append((str(img_path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(input_size: int = 224, augment: bool = True):
    """Train + val transforms with ImageNet normalization."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform


def build_datasets(data_dir: str, train_transform, val_transform, validation_split: float,
                   class_to_idx: Optional[Dict[str, int]] = None):
    """Resolve data_dir into (train_dataset, val_dataset, classes).

    If data_dir contains train/ and val/ subdirectories, treat them as explicit
    splits (recommended for State Farm with driver-grouped splitting). Otherwise,
    fall back to a random split of a single image-folder root.

    `class_to_idx` lets callers pin a specific class order (e.g. from
    config.classes) so the model's output index → class-name mapping stays
    stable across retrains and matches what `config.yaml` claims at inference.
    """
    root = Path(data_dir)
    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.is_dir() and val_dir.is_dir():
        train_dataset = DistractionDataset(str(train_dir), transform=train_transform,
                                           class_to_idx=class_to_idx)
        val_dataset = DistractionDataset(
            str(val_dir), transform=val_transform, class_to_idx=train_dataset.class_to_idx
        )
        classes = train_dataset.classes
        print(f"Detected explicit train/ and val/ splits under {data_dir}")
        return train_dataset, val_dataset, classes

    print(f"No train/val subdirs found; doing random split with validation_split={validation_split}")
    full_dataset = DistractionDataset(data_dir, transform=train_transform, class_to_idx=class_to_idx)
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    # The val subset shares the underlying transform; wrap it so it uses val_transform.
    class _ValView(Dataset):
        def __init__(self, base_dataset, indices, transform):
            self.base = base_dataset
            self.indices = list(indices)
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            img_path, label = self.base.samples[self.indices[i]]
            image = Image.open(img_path).convert('RGB')
            return self.transform(image), label

    val_dataset = _ValView(full_dataset, val_subset.indices, val_transform)
    return train_subset, val_dataset, full_dataset.classes


def _make_sampler(train_dataset) -> Optional[WeightedRandomSampler]:
    """Class-balanced sampler. Compensates for skewed datasets like State Farm binary."""
    labels = []
    if hasattr(train_dataset, 'samples'):
        labels = [lbl for _, lbl in train_dataset.samples]
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset, 'indices'):
        base_samples = train_dataset.dataset.samples
        labels = [base_samples[i][1] for i in train_dataset.indices]
    if not labels:
        return None

    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[lbl] for lbl in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return running_loss / total, 100.0 * correct / total


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device, class_names) -> Dict:
    model.eval()
    running_loss = 0.0
    total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(total, 1)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision, recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'macro_f1': float(macro_f1),
        'per_class_f1': per_class_f1.tolist(),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'confusion_matrix': cm.tolist(),
        'y_true': y_true,
        'y_pred': y_pred,
    }


def train_model(data_dir: str, config: Optional[Config] = None, config_path: str = "config.yaml",
                resume: Optional[str] = None, epochs: Optional[int] = None,
                lr: Optional[float] = None, freeze_backbone: bool = False):
    """Main training function.

    Args:
        data_dir: image-folder root (or root with train/val subdirs).
        config: pre-loaded Config; loaded from config_path if None.
        config_path: path to config.yaml (used only if config is None).
        resume: optional .pth checkpoint to initialize weights from. Useful
            for fine-tuning on personal data starting from State Farm weights.
        epochs: override config.training.epochs.
        lr: override config.training.learning_rate.
        freeze_backbone: if True, freeze all params except the final
            classifier/fc layer. Recommended when fine-tuning on small
            personal datasets to avoid catastrophic forgetting.
    """

    if config is None:
        config = load_config(config_path)
    if epochs is not None:
        config.training.epochs = epochs
    if lr is not None:
        config.training.learning_rate = lr

    device = pick_device(config.model.use_gpu)
    print(f"Using device: {device}")

    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model.model_path).parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    train_transform, val_transform = get_transforms(config.model.input_size, augment=True)

    # Pin class order to config.yaml so inference and training agree on label
    # indices. Falls back to sorted directory order if config.classes is empty.
    class_to_idx = {name: i for i, name in enumerate(config.classes)} if config.classes else None

    train_dataset, val_dataset, classes = build_datasets(
        data_dir, train_transform, val_transform, config.training.validation_split,
        class_to_idx=class_to_idx,
    )
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Classes (label-index order): {classes}")

    sampler = _make_sampler(train_dataset) if config.training.use_class_balanced_sampler else None
    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=2, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.training.batch_size,
        shuffle=False, num_workers=2, pin_memory=pin_memory,
    )

    print(f"Creating model: {config.model.architecture}")
    model = create_model(
        num_classes=config.model.num_classes,
        architecture=config.model.architecture,
        pretrained=(resume is None),  # if resuming, skip the ImageNet download
    )

    if resume is not None:
        print(f"Resuming weights from {resume}")
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  load_state_dict — missing: {len(missing)}, unexpected: {len(unexpected)}")
        if 'val_macro_f1' in ckpt:
            print(f"  Source checkpoint val_macro_f1: {ckpt['val_macro_f1']:.4f}")

    if freeze_backbone:
        print("Freezing backbone; only the classifier head will be trained.")
        for p in model.parameters():
            p.requires_grad = False
        # Re-enable the final classifier layer(s). DistractionClassifier wraps
        # the backbone, so we target backbone.fc (ResNet) or backbone.classifier
        # (MobileNet) — whichever exists.
        head = getattr(model.backbone, 'fc', None) or getattr(model.backbone, 'classifier', None)
        if head is None:
            raise RuntimeError("Could not locate classifier head on backbone")
        for p in head.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_f1 = -1.0
    patience_counter = 0
    checkpoint_path = None
    metrics_history = []

    print(f"\nStarting training for {config.training.epochs} epochs...")
    print("Press Ctrl-C to stop early; metrics + TorchScript will still be saved from the best epoch.")
    interrupted = False
    for epoch in range(config.training.epochs):
        try:
            print(f"\nEpoch {epoch+1}/{config.training.epochs}")
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = validate(model, val_loader, criterion, device, classes)
            epoch_time = time.time() - t0
        except KeyboardInterrupt:
            print("\n\nCtrl-C received. Finalizing with whatever we have so far...")
            interrupted = True
            break

        scheduler.step(val_metrics['macro_f1'])

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
              f"Macro F1: {val_metrics['macro_f1']:.4f} ({epoch_time:.1f}s)")
        for i, name in enumerate(classes):
            print(f"  {name:>12s}: precision={val_metrics['per_class_precision'][i]:.3f} "
                  f"recall={val_metrics['per_class_recall'][i]:.3f} "
                  f"f1={val_metrics['per_class_f1'][i]:.3f}")

        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_macro_f1': val_metrics['macro_f1'],
            'per_class_f1': val_metrics['per_class_f1'],
            'confusion_matrix': val_metrics['confusion_matrix'],
        })

        if val_metrics['macro_f1'] > best_f1:
            best_f1 = val_metrics['macro_f1']
            patience_counter = 0
            checkpoint_path = Path(config.training.checkpoint_dir) / f"best_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_f1,
                'val_metrics': {k: v for k, v in val_metrics.items() if k not in ('y_true', 'y_pred')},
                'classes': classes,
                'architecture': config.model.architecture,
            }, checkpoint_path)
            print(f"Saved best model (F1={best_f1:.4f}) to {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    status = "interrupted" if interrupted else "completed"
    print(f"\nTraining {status}! Best validation macro F1: {best_f1:.4f}")

    # Write metrics JSON for release notes / README.
    metrics_path = Path(config.training.checkpoint_dir) / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'best_val_macro_f1': best_f1,
            'classes': classes,
            'architecture': config.model.architecture,
            'history': metrics_history,
        }, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")

    if checkpoint_path is not None:
        print("Saving final model as TorchScript...")
        best_checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        save_model_torchscript(
            model, config.model.model_path,
            input_size=(1, 3, config.model.input_size, config.model.input_size),
        )
        print(f"TorchScript model saved to {config.model.model_path}")

    return model, best_f1


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train distraction classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to training data directory (single image-folder root, "
                             "or a directory containing train/ and val/ subdirs).")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pth checkpoint to initialize weights from "
                             "(useful for fine-tuning on personal data).")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config.training.epochs (e.g. --epochs 5 when fine-tuning).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config.training.learning_rate (e.g. --lr 0.0001 when fine-tuning).")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Train only the classifier head; freeze the backbone. "
                             "Strongly recommended when fine-tuning on small personal datasets.")
    args = parser.parse_args()

    train_model(
        args.data_dir, config_path=args.config,
        resume=args.resume, epochs=args.epochs, lr=args.lr,
        freeze_backbone=args.freeze_backbone,
    )


if __name__ == "__main__":
    main()
