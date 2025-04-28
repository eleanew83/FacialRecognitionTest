#!/usr/bin/env python3
import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(source_dir, target_dir, train_ratio=0.7, db_ratio=0.15, eval_ratio=0.15, random_seed=42):
    """
    Split the dataset into train, database and evaluation sets.
    
    Args:
        source_dir: Path to the source directory with individual folders
        target_dir: Path to create the split datasets
        train_ratio: Ratio of images to use for training
        db_ratio: Ratio of images to use for database creation
        eval_ratio: Ratio of images to use for evaluation
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    
    # Create target directories
    train_dir = os.path.join(target_dir, 'train')
    db_dir = os.path.join(target_dir, 'database_set')
    eval_dir = os.path.join(target_dir, 'eval')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
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
        n_db = max(1, int(len(image_files) * db_ratio))
        n_eval = max(1, len(image_files) - n_train - n_db)
        
        # Split images
        train_images = image_files[:n_train]
        db_images = image_files[n_train:n_train+n_db]
        eval_images = image_files[n_train+n_db:n_train+n_db+n_eval]
        
        # Create individual folders in each split
        os.makedirs(os.path.join(train_dir, individual), exist_ok=True)
        os.makedirs(os.path.join(db_dir, individual), exist_ok=True)
        os.makedirs(os.path.join(eval_dir, individual), exist_ok=True)
        
        # Copy images to respective splits
        for img in train_images:
            shutil.copy2(
                os.path.join(individual_dir, img),
                os.path.join(train_dir, individual, img)
            )
        
        for img in db_images:
            shutil.copy2(
                os.path.join(individual_dir, img),
                os.path.join(db_dir, individual, img)
            )
        
        for img in eval_images:
            shutil.copy2(
                os.path.join(individual_dir, img),
                os.path.join(eval_dir, individual, img)
            )
        
        print(f"Processed {individual}: {n_train} train, {n_db} database, {n_eval} eval images")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train, database and evaluation sets')
    parser.add_argument('--source', type=str, required=True, help='Source directory with individual folders')
    parser.add_argument('--target', type=str, required=True, help='Target directory for the split datasets')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Ratio of images for training')
    parser.add_argument('--db-ratio', type=float, default=0.15, help='Ratio of images for database')
    parser.add_argument('--eval-ratio', type=float, default=0.15, help='Ratio of images for evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Ensure ratios sum to 1
    total_ratio = args.train_ratio + args.db_ratio + args.eval_ratio
    if abs(total_ratio - 1.0) > 0.001:
        args.train_ratio /= total_ratio
        args.db_ratio /= total_ratio
        args.eval_ratio /= total_ratio
        print(f"Warning: Ratios adjusted to sum to 1: {args.train_ratio:.2f}, {args.db_ratio:.2f}, {args.eval_ratio:.2f}")
    
    split_dataset(
        args.source,
        args.target,
        args.train_ratio,
        args.db_ratio,
        args.eval_ratio,
        args.seed
    )

if __name__ == '__main__':
    main() 