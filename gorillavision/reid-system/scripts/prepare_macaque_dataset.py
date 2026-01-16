#!/usr/bin/env python3
import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        source_dir: Path to the source directory with individual folders
        target_dir: Path to create the split datasets
        train_ratio: Ratio of images to use for training
        val_ratio: Ratio of images to use for validation
        test_ratio: Ratio of images to use for testing
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    
    # Clean and recreate target directory
    if os.path.exists(target_dir):
        print(f"Removing existing target directory: {target_dir}")
        shutil.rmtree(target_dir)
    
    print(f"Creating fresh target directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    
    # Create target directories
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    test_dir = os.path.join(target_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Process each individual's folder
    for individual in os.listdir(source_dir):
        individual_dir = os.path.join(source_dir, individual)
        
        # Skip if not a directory
        if not os.path.isdir(individual_dir):
            continue
        
        # Get all image files
        image_files = []
        for filename in os.listdir(individual_dir):
            if filename.startswith('._'):  # Skip macOS hidden files
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                image_files.append(filename)
        
        # Skip if not enough images
        if len(image_files) < 3:
            print(f"Warning: {individual} has fewer than 3 images, skipping.")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split sizes
        n_train = max(2, int(len(image_files) * train_ratio))
        n_val = max(1, int(len(image_files) * val_ratio))
        n_test = max(1, len(image_files) - n_train - n_val)
        
        # Split images
        train_images = image_files[:n_train]
        val_images = image_files[n_train:n_train+n_val]
        test_images = image_files[n_train+n_val:n_train+n_val+n_test]
        
        # Create individual folders in each split
        os.makedirs(os.path.join(train_dir, individual), exist_ok=True)
        os.makedirs(os.path.join(val_dir, individual), exist_ok=True)
        os.makedirs(os.path.join(test_dir, individual), exist_ok=True)
        
        # Copy images to respective splits
        for img in train_images:
            shutil.copy2(
                os.path.join(individual_dir, img),
                os.path.join(train_dir, individual, img)
            )
        
        for img in val_images:
            shutil.copy2(
                os.path.join(individual_dir, img),
                os.path.join(val_dir, individual, img)
            )
        
        for img in test_images:
            shutil.copy2(
                os.path.join(individual_dir, img),
                os.path.join(test_dir, individual, img)
            )
        
        print(f"Processed {individual}: {n_train} train, {n_val} val, {n_test} test images")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train, validation, and test sets')
    parser.add_argument('--source', type=str, required=True, help='Source directory with individual folders')
    parser.add_argument('--target', type=str, required=True, help='Target directory for the split datasets')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of images for training')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Ratio of images for validation')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Ratio of images for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Ensure ratios sum to 1
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
        print(f"Warning: Ratios adjusted to sum to 1: {args.train_ratio:.2f}, {args.val_ratio:.2f}, {args.test_ratio:.2f}")
    
    split_dataset(
        args.source,
        args.target,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

if __name__ == '__main__':
    main() 