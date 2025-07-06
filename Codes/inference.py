"""
Inference script for MDAU-Net.

Usage:
    python Codes/inference.py --image path/to/image.jpg --model Checkpoints/mdau_net_deepcrack.pth
"""

import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model import mdau_net
from utils import load_checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2


def preprocess_image(image_path, image_size=512):
    """Preprocess image for inference."""
    # Load image
    image = cv2.imread(image_path)
    original_size = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transform
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)
    
    return image_tensor, original_size, image


def postprocess_prediction(prediction, original_size):
    """Postprocess prediction to original size."""
    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (original_size[1], original_size[0]))
    return prediction


def visualize_result(original_image, prediction, save_path=None, threshold=0.5):
    """Visualize inference result."""
    binary_mask = (prediction > threshold).astype(np.uint8) * 255
    
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    overlay = original_image.copy()
    overlay[binary_mask > 0] = [255, 0, 0]  # Red for cracks
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='MDAU-Net Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, help='Path to save result')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary mask')
    parser.add_argument('--image_size', type=int, default=512, help='Input image size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = mdau_net().to(device)
    load_checkpoint(args.model, model)
    model.eval()
    print(f"Model loaded from {args.model}")
    
    # Preprocess image
    image_tensor, original_size, original_image = preprocess_image(args.image, args.image_size)
    image_tensor = image_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Postprocess
    prediction = postprocess_prediction(prediction, original_size)
    
    # Generate output path if not provided
    if not args.output:
        input_path = Path(args.image)
        args.output = f"Results/{input_path.stem}_result.png"
        Path("Results").mkdir(exist_ok=True)
    
    # Visualize and save result
    visualize_result(original_image, prediction, args.output, args.threshold)
    
    print(f"Inference completed for {args.image}")


if __name__ == '__main__':
    main()