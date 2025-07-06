"""
Utility functions for MDAU-Net crack segmentation.

This module provides essential utilities for:
- Reproducibility (seed setting)
- Model checkpointing (save/load)
- Metrics calculation
- Visualization helpers
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Optional, Union
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def save_checkpoint(state: Dict, filepath: str, is_best: bool = False):
    """
    Save model checkpoint to file.
    
    Args:
        state (Dict): Dictionary containing model state and metadata
        filepath (str): Path to save the checkpoint
        is_best (bool): Whether this is the best model so far
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save checkpoint
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")
    
    # Save best model copy if needed
    if is_best:
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(filepath: str, model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """
    Load model checkpoint from file.
    
    Args:
        filepath (str): Path to checkpoint file
        model (torch.nn.Module): Model to load state into
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
        
    Returns:
        Dict: Loaded checkpoint information
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, 
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate comprehensive segmentation metrics.
    
    Args:
        predictions (np.ndarray): Predicted masks [B, H, W] or [B, 1, H, W]
        targets (np.ndarray): Ground truth masks [B, H, W] or [B, 1, H, W]
        threshold (float): Threshold for converting predictions to binary
        
    Returns:
        Dict[str, float]: Dictionary containing computed metrics
    """
    # Ensure correct shape - remove channel dimension if present
    if predictions.ndim == 4:
        predictions = predictions[:, 0]
    if targets.ndim == 4:
        targets = targets[:, 0]
    
    # Convert predictions to binary
    pred_binary = (predictions > threshold).astype(np.uint8)
    targets_binary = targets.astype(np.uint8)
    
    # Flatten arrays for calculation
    pred_flat = pred_binary.flatten()
    target_flat = targets_binary.flatten()
    
    # Calculate confusion matrix components
    true_positive = np.sum((pred_flat == 1) & (target_flat == 1))
    true_negative = np.sum((pred_flat == 0) & (target_flat == 0))
    false_positive = np.sum((pred_flat == 1) & (target_flat == 0))
    false_negative = np.sum((pred_flat == 0) & (target_flat == 1))
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-7
    
    # Calculate metrics
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # IoU calculation
    intersection = true_positive
    union = true_positive + false_positive + false_negative
    iou = intersection / (union + epsilon)
    
    # Mean IoU (considering both foreground and background)
    iou_background = true_negative / (true_negative + false_positive + false_negative + epsilon)
    miou = (iou + iou_background) / 2
    
    # Accuracy
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative + epsilon)
    
    # Specificity
    specificity = true_negative / (true_negative + false_positive + epsilon)
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'miou': float(miou),
        'accuracy': float(accuracy),
        'specificity': float(specificity)
    }


def visualize_training_progress(train_losses: list, val_losses: list, 
                              val_metrics: list, save_path: str = None):
    """
    Visualize training progress with loss curves and metrics.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        val_metrics (list): Validation metrics per epoch
        save_path (str, optional): Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # mIoU curve
    miou_values = [metrics['miou'] for metrics in val_metrics]
    axes[0, 1].plot(epochs, miou_values, 'g-', label='mIoU', linewidth=2)
    axes[0, 1].set_title('Validation mIoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision, Recall, F1
    precision_values = [metrics['precision'] for metrics in val_metrics]
    recall_values = [metrics['recall'] for metrics in val_metrics]
    f1_values = [metrics['f1'] for metrics in val_metrics]
    
    axes[1, 0].plot(epochs, precision_values, 'b-', label='Precision', linewidth=2)
    axes[1, 0].plot(epochs, recall_values, 'r-', label='Recall', linewidth=2)
    axes[1, 0].plot(epochs, f1_values, 'g-', label='F1-Score', linewidth=2)
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy and IoU
    accuracy_values = [metrics['accuracy'] for metrics in val_metrics]
    iou_values = [metrics['iou'] for metrics in val_metrics]
    
    axes[1, 1].plot(epochs, accuracy_values, 'purple', label='Accuracy', linewidth=2)
    axes[1, 1].plot(epochs, iou_values, 'orange', label='IoU', linewidth=2)
    axes[1, 1].set_title('Additional Metrics')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(images: torch.Tensor, predictions: torch.Tensor, 
                         targets: torch.Tensor, save_path: str = None, 
                         num_samples: int = 4):
    """
    Visualize model predictions alongside ground truth.
    
    Args:
        images (torch.Tensor): Input images [B, 3, H, W]
        predictions (torch.Tensor): Predicted masks [B, 1, H, W]
        targets (torch.Tensor): Ground truth masks [B, 1, H, W]
        save_path (str, optional): Path to save visualization
        num_samples (int): Number of samples to visualize
    """
    batch_size = min(images.shape[0], num_samples)
    
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Convert tensors to numpy and handle normalization
        if isinstance(images, torch.Tensor):
            image = images[i].cpu().permute(1, 2, 0).numpy()
            # Denormalize image
            image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            image = np.clip(image, 0, 1)
        else:
            image = images[i]
        
        if isinstance(predictions, torch.Tensor):
            prediction = predictions[i, 0].cpu().numpy()
        else:
            prediction = predictions[i, 0]
        
        if isinstance(targets, torch.Tensor):
            target = targets[i, 0].cpu().numpy()
        else:
            target = targets[i, 0]
        
        # Plot original image
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(target, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(prediction, cmap='gray')
        axes[i, 2].set_title(f'Prediction {i+1}')
        axes[i, 2].axis('off')
        
        # Plot overlay
        overlay = image.copy()
        binary_pred = (prediction > 0.5).astype(np.uint8)
        overlay[binary_pred > 0] = [1, 0, 0]  # Red for predicted cracks
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title(f'Overlay {i+1}')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count the number of parameters in a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        Dict[str, int]: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def print_model_info(model: torch.nn.Module):
    """
    Print comprehensive model information.
    
    Args:
        model (torch.nn.Module): PyTorch model
    """
    param_info = count_parameters(model)
    model_size_mb = param_info['total_parameters'] * 4 / (1024 * 1024)  # Assuming float32
    
    print("="*50)
    print("MODEL INFORMATION")
    print("="*50)
    print(f"Total Parameters:     {param_info['total_parameters']:,}")
    print(f"Trainable Parameters: {param_info['trainable_parameters']:,}")
    print(f"Model Size:           {model_size_mb:.2f} MB")
    print("="*50)


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test seed setting
    set_seed(42)
    
    # Test metrics calculation
    pred = np.random.rand(5, 256, 256)
    target = np.random.randint(0, 2, (5, 256, 256))
    
    metrics = calculate_metrics(pred, target)
    print("Sample metrics:", metrics)
    
    print("All utility functions working correctly!")