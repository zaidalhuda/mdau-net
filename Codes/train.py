"""
Training script for MDAU-Net crack segmentation.

Usage:
    python Codes/train.py --dataset deepcrack --epochs 100 --batch_size 8
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import mdau_net
from losses import HybridLoss
from data_loader import get_dataset
from utils import save_checkpoint, calculate_metrics, set_seed


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, masks in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    # Calculate metrics
    import numpy as np
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_metrics(all_predictions, all_targets)
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, metrics


def plot_training_curves(train_losses, val_losses, val_metrics, save_path):
    """Plot training curves."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # mIoU curve
    miou_values = [m['miou'] for m in val_metrics]
    plt.subplot(1, 3, 2)
    plt.plot(epochs, miou_values, 'g-', label='mIoU')
    plt.title('Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)
    
    # F1-Score curve
    f1_values = [m['f1'] for m in val_metrics]
    plt.subplot(1, 3, 3)
    plt.plot(epochs, f1_values, 'purple', label='F1-Score')
    plt.title('Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train MDAU-Net for crack segmentation')
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['deepcrack', 'crack500'], help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='Datasets',
                       help='Root directory of datasets')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, default=512, help='Input image size')
    
    # Model parameters
    parser.add_argument('--base_channels', type=int, default=64, help='Base channels')
    parser.add_argument('--scales', nargs='+', type=float, default=[0.5, 1.0, 1.5],
                       help='Multi-scale factors')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='Checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = get_dataset(args.dataset, 'train', args.data_root, args.image_size, augmentation=True)
    val_dataset = get_dataset(args.dataset, 'val', args.data_root, args.image_size, augmentation=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Create model
    model = mdau_net(
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
        scales=args.scales
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = HybridLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    val_metrics = []
    best_miou = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss, metrics = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_metrics.append(metrics)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val mIoU: {metrics['miou']:.4f}")
        print(f"Val F1: {metrics['f1']:.4f}")
        print(f"Val Precision: {metrics['precision']:.4f}")
        print(f"Val Recall: {metrics['recall']:.4f}")
        
        # Save best model
        if metrics['miou'] > best_miou:
            best_miou = metrics['miou']
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'metrics': metrics
            }, os.path.join(args.save_dir, f'mdau_net_{args.dataset}.pth'))
            print(f"New best model saved! mIoU: {best_miou:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'metrics': metrics
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, val_metrics, 
        f'Results/training_curves_{args.dataset}.png'
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"Model saved to: {args.save_dir}/mdau_net_{args.dataset}.pth")


if __name__ == '__main__':
    main()