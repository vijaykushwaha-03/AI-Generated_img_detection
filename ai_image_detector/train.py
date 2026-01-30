import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm
from model.architecture import DualStreamDetector

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class AugmentedDataset(Dataset):
    """
    Custom Dataset to apply Albumentations to generic data.
    Wraps a list of file paths and labels.
    """
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load image via PIL, convert to RGB
        try:
            image = np.array(Image.open(path).convert("RGB"))
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a blank black image to prevent crash
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
            
        return image, label

def get_transforms(img_size=224):
    """
    Define training and validation transforms.
    Includes JPEG compression simulation for AI artifact robustness.
    """
    # Albumentations uses RGB logic (H, W, C)
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=img_size, width=img_size, p=0.5), # Only if larger, else resize covers it
        # Crucial for AI Grid/Artifact detection: simulate compression
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, val_transform

def load_data(data_dir):
    """
    Auto-detects data structure.
    Mode 1: "Split" - data_dir/train and data_dir/val exist.
    Mode 2: "Flat" - data_dir/real and data_dir/ai (or fake) exist directly.
    """
    classes = {'real': 0, 'ai': 1, 'fake': 1} # Map 'fake' folder to 'ai' class (1)
    
    # Check for split structure
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f"Found standard train/val structure in {data_dir}")
        datasets = {}
        for split in ['train', 'val']:
            paths = []
            labels = []
            split_path = os.path.join(data_dir, split)
            for cls_name, cls_idx in classes.items():
                cls_folder = os.path.join(split_path, cls_name)
                if not os.path.exists(cls_folder):
                    continue
                for fname in os.listdir(cls_folder):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        paths.append(os.path.join(cls_folder, fname))
                        labels.append(cls_idx)
            datasets[split] = (paths, labels)
        return datasets
        
    else:
        print(f"Standard train/val not found. Looking for class folders directly in {data_dir}...")
        all_paths = []
        all_labels = []
        
        for cls_name, cls_idx in classes.items():
            cls_folder = os.path.join(data_dir, cls_name)
            if not os.path.exists(cls_folder):
                continue
            
            print(f"Found class folder: {cls_name} -> Class {cls_idx}")
            files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            for fname in files:
                all_paths.append(os.path.join(cls_folder, fname))
                all_labels.append(cls_idx)
                
        if not all_paths:
            print("No valid images found.")
            return {'train': ([], []), 'val': ([], [])}
            
        # Shuffle and Split 80/20
        combined = list(zip(all_paths, all_labels))
        random.shuffle(combined)
        all_paths, all_labels = zip(*combined)
        
        split_idx = int(0.8 * len(all_paths))
        train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
        train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]
        
        print(f"Auto-split: {len(train_paths)} Train, {len(val_paths)} Val")
        
        return {
            'train': (list(train_paths), list(train_labels)),
            'val': (list(val_paths), list(val_labels))
        }

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / total, 'acc': correct / total})
        
    return running_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser(description='Train AI Image Detector')
    parser.add_argument('--data_dir', type=str, default='ai_image_detector/data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    set_seed()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    print("Loading data...")
    data_info = load_data(args.data_dir)
    train_paths, train_labels = data_info['train']
    val_paths, val_labels = data_info['val']
    
    if len(train_paths) == 0:
        print("No training data found! Please populate ai_image_detector/data/train with 'real' and 'ai' folders.")
        print("Creating dummy data for verification purposes...")
        # Create dummy data if valid
        # For now, just exit or user can verify with dummy inputs in verify script.
        # Let's create dummy in memory for this run if empty?
        # Better to warn.
        return 

    train_transform, val_transform = get_transforms()
    
    train_dataset = AugmentedDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = AugmentedDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print(f"Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")
    
    # 2. Model Setup
    print("Initializing Dual-Stream Model...")
    model = DualStreamDetector(pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. Training Loop
    best_acc = 0.0
    save_path = "ai_image_detector/models/ai_image_detector.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with accuracy: {best_acc:.4f}")
            
    print("Training Complete.")

if __name__ == "__main__":
    main()
