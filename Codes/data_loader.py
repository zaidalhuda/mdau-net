"""
Data loading utilities for MDAU-Net crack segmentation.

This module provides:
- Dataset classes for crack segmentation
- Data augmentation pipelines
- Support for DeepCrack and Crack500 datasets
- Flexible data loading with transforms
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CrackDataset(Dataset):
    """
    Generic dataset class for crack segmentation.
    
    Args:
        images_dir (str): Directory containing images
        masks_dir (str): Directory containing masks
        image_size (int): Target image size for resizing
        transform (callable, optional): Data augmentation transforms
        image_ext (str): Image file extension
        mask_ext (str): Mask file extension
    """
    
    def __init__(
        self, 
        images_dir: str,
        masks_dir: str,
        image_size: int = 512,
        transform: Optional[Callable] = None,
        image_ext: str = '.jpg',
        mask_ext: str = '.png'
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.transform = transform
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        
        # Get all image files
        self.image_files = self._get_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
    
    def _get_image_files(self) -> list:
        """Get list of valid image files that have corresponding masks."""
        image_files = []
        
        # Look for images with specified extension
        for img_file in self.images_dir.glob(f'*{self.image_ext}'):
            # Check if corresponding mask exists
            mask_file = self.masks_dir / f"{img_file.stem}{self.mask_ext}"
            if mask_file.exists():
                image_files.append(img_file)
            else:
                print(f"Warning: No mask found for {img_file.name}")
        
        # If no .jpg files found, try .png
        if not image_files and self.image_ext == '.jpg':
            for img_file in self.images_dir.glob('*.png'):
                mask_file = self.masks_dir / f"{img_file.stem}{self.mask_ext}"
                if mask_file.exists():
                    image_files.append(img_file)
        
        return sorted(image_files)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image, mask) tensors
        """
        # Get file paths
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / f"{img_path.stem}{self.mask_ext}"
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Normalize mask to [0, 1] range
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure mask has correct shape [1, H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Get training data augmentation transforms.
    
    Args:
        image_size (int): Target image size
        
    Returns:
        A.Compose: Albumentations transform pipeline
    """
    return A.Compose([
        # Resize to target size
        A.Resize(image_size, image_size),
        
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        
        # Elastic deformation
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.3
        ),
        
        # Photometric transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.7
        ),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        
        # Color transforms
        A.OneOf([
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=1.0
            ),
            A.CLAHE(clip_limit=2, p=1.0),
        ], p=0.3),
        
        # Normalization and tensor conversion
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """
    Get validation/test data transforms (no augmentation).
    
    Args:
        image_size (int): Target image size
        
    Returns:
        A.Compose: Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_dataset(dataset_name: str, split: str, data_root: str, 
               image_size: int = 512, augmentation: bool = True) -> Dataset:
    """
    Factory function to get dataset based on name and split.
    
    Args:
        dataset_name (str): Name of dataset ('deepcrack', 'crack500')
        split (str): Data split ('train', 'val', 'test')
        data_root (str): Root directory of datasets
        image_size (int): Target image size
        augmentation (bool): Whether to apply data augmentation
        
    Returns:
        Dataset: Configured dataset instance
    """
    # Construct dataset path
    dataset_path = os.path.join(data_root, dataset_name)
    
    # Handle special cases
    if split == 'val' and dataset_name == 'crack500':
        # Crack500 typically doesn't have validation split
        print("Warning: Crack500 doesn't have validation split, using train split")
        split = 'train'
    
    # Construct image and mask directories
    images_dir = os.path.join(dataset_path, split, 'images')
    masks_dir = os.path.join(dataset_path, split, 'masks')
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(masks_dir):
        raise ValueError(f"Masks directory not found: {masks_dir}")
    
    # Get transforms based on split and augmentation setting
    if split == 'train' and augmentation:
        transform = get_train_transforms(image_size)
    else:
        transform = get_val_transforms(image_size)
    
    # Determine file extensions based on dataset
    if dataset_name.lower() == 'deepcrack':
        image_ext = '.jpg'
        mask_ext = '.png'
    elif dataset_name.lower() == 'crack500':
        image_ext = '.jpg'
        mask_ext = '.png'
    else:
        # Default extensions
        image_ext = '.jpg'
        mask_ext = '.png'
    
    # Create dataset
    dataset = CrackDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=image_size,
        transform=transform,
        image_ext=image_ext,
        mask_ext=mask_ext
    )
    
    return dataset


class DeepCrackDataset(CrackDataset):
    """
    Specialized dataset class for DeepCrack dataset.
    
    DeepCrack dataset structure:
    - 537 images total (300 train, 100 val, 137 test)
    - Original resolution: 544×384 pixels
    - Contains various crack types and scenarios
    """
    
    def __init__(self, data_root: str, split: str = 'train', **kwargs):
        dataset_path = os.path.join(data_root, 'deepcrack')
        images_dir = os.path.join(dataset_path, split, 'images')
        masks_dir = os.path.join(dataset_path, split, 'masks')
        
        super().__init__(
            images_dir=images_dir,
            masks_dir=masks_dir,
            image_ext='.jpg',
            mask_ext='.png',
            **kwargs
        )


class Crack500Dataset(CrackDataset):
    """
    Specialized dataset class for Crack500 dataset.
    
    Crack500 dataset structure:
    - 3,020 images total (1,896 train, 1,124 test)
    - Original resolution: 360×640 pixels
    - More challenging with complex backgrounds
    """
    
    def __init__(self, data_root: str, split: str = 'train', **kwargs):
        dataset_path = os.path.join(data_root, 'crack500')
        
        # Crack500 typically doesn't have val split
        if split == 'val':
            split = 'train'
            print("Crack500 doesn't have validation split, using train split")
        
        images_dir = os.path.join(dataset_path, split, 'images')
        masks_dir = os.path.join(dataset_path, split, 'masks')
        
        super().__init__(
            images_dir=images_dir,
            masks_dir=masks_dir,
            image_ext='.jpg',
            mask_ext='.png',
            **kwargs
        )


def create_data_loaders(dataset_name: str, data_root: str, 
                       batch_size: int = 8, num_workers: int = 4,
                       image_size: int = 512) -> dict:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        dataset_name (str): Name of dataset
        data_root (str): Root directory of datasets
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        image_size (int): Target image size
        
    Returns:
        dict: Dictionary containing data loaders for each split
    """
    from torch.utils.data import DataLoader
    
    data_loaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = get_dataset(
                dataset_name=dataset_name,
                split=split,
                data_root=data_root,
                image_size=image_size,
                augmentation=(split == 'train')
            )
            
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == 'train')
            )
            
            data_loaders[split] = data_loader
            
        except Exception as e:
            print(f"Warning: Could not create {split} loader for {dataset_name}: {e}")
    
    return data_loaders


def visualize_dataset_samples(dataset: Dataset, num_samples: int = 4,
                            save_path: Optional[str] = None):
    """
    Visualize random samples from the dataset.
    
    Args:
        dataset (Dataset): Dataset to visualize
        num_samples (int): Number of samples to show
        save_path (str, optional): Path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Get random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, idx in enumerate(indices):
        image, mask = dataset[idx]
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            # Denormalize image
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
            image = image.permute(1, 2, 0).numpy()
        
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().numpy()
        
        # Plot image and mask
        axes[0, i].imshow(image)
        axes[0, i].set_title(f'Image {idx}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title(f'Mask {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dataset samples saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test data loading functionality
    print("Testing data loading functionality...")
    
    try:
        # Test with sample data structure
        data_root = "Datasets"
        
        for dataset_name in ['deepcrack', 'crack500']:
            print(f"\nTesting {dataset_name} dataset:")
            
            try:
                # Test dataset creation
                train_dataset = get_dataset(
                    dataset_name=dataset_name,
                    split='train',
                    data_root=data_root,
                    image_size=256,  # Smaller size for testing
                    augmentation=True
                )
                
                print(f"✓ {dataset_name} train dataset created: {len(train_dataset)} samples")
                
                # Test data loading
                if len(train_dataset) > 0:
                    sample_image, sample_mask = train_dataset[0]
                    print(f"✓ Sample loaded - Image: {sample_image.shape}, Mask: {sample_mask.shape}")
                
            except Exception as e:
                print(f"✗ Error with {dataset_name}: {e}")
        
        print("\nData loading test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure datasets are properly organized in Datasets/ directory")