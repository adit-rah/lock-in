"""Evaluate model performance on test dataset"""

import torch
import argparse
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.train import DistractionDataset
from src.model import load_model_torchscript, load_model_checkpoint
from src.config import load_config


def evaluate_model(model_path: str, test_data_dir: str, config_path: str = "config.yaml"):
    """Evaluate model on test dataset"""
    
    # Load config
    config = load_config(config_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config.model.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    if model_path.endswith('.pt') and 'torchscript' in model_path.lower():
        model = load_model_torchscript(model_path, device)
    else:
        model = load_model_checkpoint(model_path, config.model.num_classes, config.model.architecture, device)
    
    model.eval()
    
    # Load test dataset
    print(f"Loading test data from {test_data_dir}...")
    test_transform = transforms.Compose([
        transforms.Resize((config.model.input_size, config.model.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = DistractionDataset(test_data_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")
    
    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).mean()
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Per-class confidence
    print("\nMean Confidence per Class:")
    for i, class_name in enumerate(test_dataset.classes):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            mean_conf = all_probs[class_mask, i].mean()
            print(f"  {class_name}: {mean_conf*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Evaluate distraction classifier")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model file")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_data, args.config)


if __name__ == "__main__":
    main()

