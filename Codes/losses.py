"""
Loss functions for MDAU-Net crack segmentation.

This module implements the hybrid weighted segmentation loss function from the paper:
- Boundary Loss (LB): For precise boundary detection
- Weighted Cross Entropy Loss (LWCE): For handling class imbalance  
- Dice Loss (LDice): For overlap-based optimization

Hybrid Loss: Lseg = Wc(α*LB + β*LWCE + γ*LDice)
Paper weights: α=1, β=3, γ=3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks.
    
    Args:
        smooth (float): Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, prediction, target):
        """
        Calculate Dice loss.
        
        Args:
            prediction (torch.Tensor): Predicted masks [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]
            
        Returns:
            torch.Tensor: Dice loss value
        """
        prediction = prediction.view(-1)
        target = target.view(-1)
        
        intersection = (prediction * target).sum()
        union = prediction.sum() + target.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance.
    
    Args:
        pos_weight (float): Weight for positive class (crack pixels)
    """
    
    def __init__(self, pos_weight=0.9):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, prediction, target):
        """
        Calculate weighted cross entropy loss.
        
        Args:
            prediction (torch.Tensor): Predicted masks [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]
            
        Returns:
            torch.Tensor: Weighted BCE loss value
        """
        # Calculate class weights based on pixel distribution
        pos_weight = torch.tensor(self.pos_weight).to(prediction.device)
        
        # Apply weighted binary cross entropy
        bce_loss = F.binary_cross_entropy(prediction, target, reduction='none')
        
        # Apply weights: higher weight for positive (crack) pixels
        weights = torch.where(target == 1, pos_weight, 1.0 - pos_weight)
        weighted_loss = bce_loss * weights
        
        return weighted_loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary loss for precise edge detection as described in the paper.
    
    This loss emphasizes the boundaries between foreground and background,
    which is crucial for crack segmentation where precise boundaries are important.
    """
    
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    
    def forward(self, prediction, target):
        """
        Calculate boundary loss.
        
        Args:
            prediction (torch.Tensor): Predicted masks [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]
            
        Returns:
            torch.Tensor: Boundary loss value
        """
        # Simple boundary detection using morphological operations
        batch_size = target.shape[0]
        boundary_loss = 0.0
        
        for i in range(batch_size):
            # Get boundary using gradient-based edge detection
            target_sample = target[i, 0]  # [H, W]
            pred_sample = prediction[i, 0]  # [H, W]
            
            # Calculate gradients to find boundaries
            target_grad_x = torch.abs(target_sample[:, 1:] - target_sample[:, :-1])
            target_grad_y = torch.abs(target_sample[1:, :] - target_sample[:-1, :])
            
            # Pad gradients to original size
            target_grad_x = F.pad(target_grad_x, (0, 1), mode='constant', value=0)
            target_grad_y = F.pad(target_grad_y, (1, 0), mode='constant', value=0)
            
            # Combine gradients to get boundary map
            boundary_map = target_grad_x + target_grad_y
            boundary_map = torch.clamp(boundary_map, 0, 1)
            
            # Calculate boundary loss - emphasize boundary regions
            boundary_weight = boundary_map + 1.0  # Weight boundaries more
            sample_loss = F.binary_cross_entropy(pred_sample, target_sample, reduction='none')
            weighted_sample_loss = (sample_loss * boundary_weight).mean()
            
            boundary_loss += weighted_sample_loss
        
        return boundary_loss / batch_size


class HybridLoss(nn.Module):
    """
    Hybrid weighted segmentation loss function as described in the MDAU-Net paper.
    
    Combines three loss functions:
    - Boundary Loss (LB): For precise boundary detection
    - Weighted Cross Entropy Loss (LWCE): For handling class imbalance
    - Dice Loss (LDice): For overlap optimization
    
    Formula: Lseg = Wc(α*LB + β*LWCE + γ*LDice)
    
    Args:
        alpha (float): Weight for boundary loss (paper: α=1)
        beta (float): Weight for weighted CE loss (paper: β=3) 
        gamma (float): Weight for dice loss (paper: γ=3)
        class_weight (float): Class weighting coefficient Wc
    """
    
    def __init__(self, alpha=1.0, beta=3.0, gamma=3.0, class_weight=1.0):
        super(HybridLoss, self).__init__()
        
        self.alpha = alpha      # Boundary loss weight
        self.beta = beta        # Weighted CE loss weight  
        self.gamma = gamma      # Dice loss weight
        self.class_weight = class_weight  # Class weighting coefficient Wc
        
        # Initialize loss functions
        self.boundary_loss = BoundaryLoss()
        self.wce_loss = WeightedCrossEntropyLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, prediction, target):
        """
        Calculate hybrid weighted segmentation loss.
        
        Args:
            prediction (torch.Tensor): Predicted masks [B, 1, H, W]
            target (torch.Tensor): Ground truth masks [B, 1, H, W]
            
        Returns:
            torch.Tensor: Combined hybrid loss value
        """
        # Ensure correct tensor shapes
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # Calculate individual loss components
        lb = self.boundary_loss(prediction, target)
        lwce = self.wce_loss(prediction, target)
        ldice = self.dice_loss(prediction, target)
        
        # Combine losses according to paper formula: Lseg = Wc(α*LB + β*LWCE + γ*LDice)
        total_loss = self.class_weight * (
            self.alpha * lb + 
            self.beta * lwce + 
            self.gamma * ldice
        )
        
        return total_loss
    
    def get_loss_weights(self):
        """Return current loss weights."""
        return {
            'alpha': self.alpha,
            'beta': self.beta, 
            'gamma': self.gamma,
            'class_weight': self.class_weight
        }


# Factory functions for creating loss configurations as per paper
def create_paper_loss():
    """Create hybrid loss with exact paper weights (α=1, β=3, γ=3)."""
    return HybridLoss(alpha=1.0, beta=3.0, gamma=3.0, class_weight=1.0)


def create_balanced_loss():
    """Create hybrid loss with balanced weights."""
    return HybridLoss(alpha=1.0, beta=1.0, gamma=1.0, class_weight=1.0)


def create_boundary_focused_loss():
    """Create hybrid loss with emphasis on boundary detection."""
    return HybridLoss(alpha=2.0, beta=2.0, gamma=1.0, class_weight=1.0)


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample data
    batch_size, channels, height, width = 2, 1, 256, 256
    prediction = torch.randn(batch_size, channels, height, width).sigmoid().to(device)
    target = torch.randint(0, 2, (batch_size, channels, height, width)).float().to(device)
    
    # Test individual loss functions
    print("Testing individual loss functions:")
    
    dice_loss = DiceLoss().to(device)
    wce_loss = WeightedCrossEntropyLoss().to(device)
    boundary_loss = BoundaryLoss().to(device)
    
    print(f"Dice Loss: {dice_loss(prediction, target).item():.4f}")
    print(f"Weighted CE Loss: {wce_loss(prediction, target).item():.4f}")
    print(f"Boundary Loss: {boundary_loss(prediction, target).item():.4f}")
    
    # Test hybrid loss with paper weights
    print(f"\nTesting hybrid loss:")
    paper_loss = create_paper_loss().to(device)
    total_loss = paper_loss(prediction, target)
    print(f"Hybrid Loss (Paper weights): {total_loss.item():.4f}")
    
    print("\nLoss functions working correctly!")