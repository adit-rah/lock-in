"""Model architecture and training utilities"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
from pathlib import Path


def pick_device(use_gpu: bool = True) -> torch.device:
    """Return the best available torch device.

    Preference order: CUDA -> Apple Silicon MPS -> CPU. Honors the `use_gpu`
    config flag — set it to False to force CPU regardless of hardware.
    """
    if not use_gpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


class DistractionClassifier(nn.Module):
    """CNN classifier for distraction detection"""
    
    def __init__(self, num_classes: int = 5, architecture: str = "mobilenet_v3_small", pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Load backbone
        if architecture == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.Hardswish(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, num_classes)
            )
        elif architecture == "mobilenet_v3_large":
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            in_features = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.Hardswish(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, num_classes)
            )
        elif architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif architecture == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(num_classes: int = 5, architecture: str = "mobilenet_v3_small", pretrained: bool = True) -> DistractionClassifier:
    """Factory function to create model"""
    return DistractionClassifier(num_classes=num_classes, architecture=architecture, pretrained=pretrained)


def save_model_torchscript(model: nn.Module, save_path: str, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)):
    """Save model as TorchScript for optimized inference.

    Always traces on CPU so the resulting .pt is portable across devices —
    `load_model_torchscript` uses `map_location` to move it to CUDA/MPS/CPU on
    load. Tracing on the source device (e.g. MPS) silently breaks when the
    example tensor and weights end up on different devices.
    """
    model = model.to("cpu")
    model.eval()
    example = torch.rand(*input_size)

    traced_script_module = torch.jit.trace(model, example)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    traced_script_module.save(save_path)
    print(f"Model saved as TorchScript to {save_path}")


def load_model_torchscript(model_path: str, device: Optional[torch.device] = None) -> torch.jit.ScriptModule:
    """Load TorchScript model for inference"""
    if device is None:
        device = pick_device()

    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def load_model_checkpoint(model_path: str, num_classes: int = 5, architecture: str = "mobilenet_v3_small", device: Optional[torch.device] = None) -> DistractionClassifier:
    """Load model from standard PyTorch checkpoint"""
    if device is None:
        device = pick_device()

    model = create_model(num_classes=num_classes, architecture=architecture, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

