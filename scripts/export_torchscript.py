"""Re-export a saved .pth checkpoint as a TorchScript .pt model.

Use this if a training run finished but the final TorchScript export failed
(e.g. an MPS device-mismatch crash). Picks up the best checkpoint by val_macro_f1.

    python scripts/export_torchscript.py \
        --checkpoint checkpoints/best_model_epoch_3.pth \
        --out models/distraction_classifier.pt
"""

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.model import create_model, save_model_torchscript


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--out", default=None,
                        help="Destination .pt path (default: from config.model.model_path)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    arch = ckpt.get("architecture", config.model.architecture)
    classes = ckpt.get("classes", config.classes)
    num_classes = len(classes) if classes else config.model.num_classes

    model = create_model(num_classes=num_classes, architecture=arch, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])

    out_path = args.out or config.model.model_path
    save_model_torchscript(
        model, out_path,
        input_size=(1, 3, config.model.input_size, config.model.input_size),
    )
    f1 = ckpt.get("val_macro_f1")
    if f1 is not None:
        print(f"Source checkpoint macro F1: {f1:.4f}")
    print(f"Classes (label index order): {classes}")
    print(f"Wrote TorchScript model to {Path(out_path).resolve()}")


if __name__ == "__main__":
    main()
