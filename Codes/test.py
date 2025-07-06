"""
Testing script for MDAU-Net crack segmentation.

Usage:
    python Codes/test.py --dataset deepcrack --model Checkpoints/mdau_net_deepcrack.pth
"""

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import mdau_net
from data_loader import get_dataset
from utils import load_checkpoint, calculate_metrics


def test_model(model, test_loader, device):
    """Test model and return metrics."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return metrics, all_predictions, all_targets


def visualize_results(images, predictions, targets, save_path, num_samples=4):
    """Visualize test results."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Original image (denormalize)
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        img = np.clip(img, 0, 1)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        axes[i, 1].imshow(targets[i, 0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[i, 0], cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test MDAU-Net')
    parser.add_argument('--dataset', type=str, required=True, choices=['deepcrack', 'crack500'])
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data_root', type=str, default='Datasets')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    test_dataset = get_dataset(args.dataset, 'test', args.data_root, args.image_size, augmentation=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Load model
    model = mdau_net().to(device)
    load_checkpoint(args.model, model)
    print(f"Model loaded from {args.model}")
    
    # Test model
    metrics, predictions, targets = test_model(model, test_loader, device)
    
    # Print results
    print("\nTest Results:")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"F1-Score:   {metrics['f1']:.4f}")
    print(f"IoU:        {metrics['iou']:.4f}")
    print(f"mIoU:       {metrics['miou']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    
    # Save results
    import os
    os.makedirs('Results', exist_ok=True)
    
    # Visualize some results
    sample_batch = next(iter(test_loader))
    with torch.no_grad():
        sample_pred = model(sample_batch[0].to(device))
    
    visualize_results(
        sample_batch[0], sample_pred.cpu(), sample_batch[1],
        f'Results/test_results_{args.dataset}.png'
    )
    print(f"Results saved to Results/test_results_{args.dataset}.png")


if __name__ == '__main__':
    main()