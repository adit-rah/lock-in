"""Training script for distraction classifier"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from pathlib import Path
from typing import Tuple, Optional, Dict
import time
from PIL import Image
import numpy as np
from tqdm import tqdm

from .model import create_model, save_model_torchscript
from .config import Config, load_config


class DistractionDataset(Dataset):
    """Custom dataset for distraction detection"""
    
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: Directory with subdirectories for each class
                      (e.g., root_dir/focused/, root_dir/looking_away/, etc.)
            transform: Optional transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = []
        
        # Load all image paths and labels
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.classes:
                    self.classes.append(class_name)
                class_idx = self.classes.index(class_name)
                
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.samples.append((str(img_path), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(input_size: int = 224, augment: bool = True):
    """Get data transforms for training and validation"""
    
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train for one epoch"""
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
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> Tuple[float, float]:
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(data_dir: str, config: Optional[Config] = None, config_path: str = "config.yaml"):
    """Main training function"""
    
    if config is None:
        config = load_config(config_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config.model.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model.model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    train_transform, val_transform = get_transforms(config.model.input_size, augment=True)
    
    full_dataset = DistractionDataset(data_dir, transform=train_transform)
    
    # Split dataset
    val_size = int(len(full_dataset) * config.training.validation_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Classes: {full_dataset.classes}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    print(f"Creating model: {config.model.architecture}")
    model = create_model(
        num_classes=config.model.num_classes,
        architecture=config.model.architecture,
        pretrained=True
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate, 
                          weight_decay=config.training.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                      patience=5, verbose=True)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {config.training.epochs} epochs...")
    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch+1}/{config.training.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint if best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint_path = Path(config.training.checkpoint_dir) / f"best_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': full_dataset.classes
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model and save as TorchScript
    print("Saving final model as TorchScript...")
    best_checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    save_model_torchscript(model, config.model.model_path, 
                          input_size=(1, 3, config.model.input_size, config.model.input_size))
    
    print(f"Model training complete! TorchScript model saved to {config.model.model_path}")
    return model, best_val_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train distraction classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    train_model(args.data_dir, config_path=args.config)

