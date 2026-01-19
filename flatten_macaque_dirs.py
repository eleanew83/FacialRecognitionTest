#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def flatten_directory_structure(source_dir, target_dir):
    """
    Flatten the nested directory structure to individual macaque directories.
    
    Args:
        source_dir: Path to the source directory with nested structure
        target_dir: Path to create the flattened directory structure
    """
    # Remove target directory if it exists, then recreate it
    if os.path.exists(target_dir):
        print(f"Removing existing target directory: {target_dir}")
        shutil.rmtree(target_dir)
    
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)
    print(f"Created fresh target directory: {target_dir}")
    
    # Walk through the directory structure
    for location in os.listdir(source_dir):
        location_path = os.path.join(source_dir, location)
        
        # Skip if not a directory or hidden files
        if not os.path.isdir(location_path) or location.startswith('.'):
            continue
        
        # Process gender directories (males/females)
        for gender in os.listdir(location_path):
            gender_path = os.path.join(location_path, gender)
            
            # Skip if not a directory or hidden files
            if not os.path.isdir(gender_path) or gender.startswith('.'):
                continue
            
            # Process individual macaque directories
            for individual in os.listdir(gender_path):
                individual_path = os.path.join(gender_path, individual)
                
                # Skip if not a directory or hidden files
                if not os.path.isdir(individual_path) or individual.startswith('.'):
                    continue
                
                # Use just the individual name instead of the full path structure
                unique_name = individual
                target_individual_path = os.path.join(target_dir, unique_name)
                
                # Create target directory
                os.makedirs(target_individual_path, exist_ok=True)
                
                # Copy all image files
                for filename in os.listdir(individual_path):
                    if filename.startswith('.'):  # Skip hidden files
                        continue
                    
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png']:
                        source_file = os.path.join(individual_path, filename)
                        target_file = os.path.join(target_individual_path, filename)
                        shutil.copy2(source_file, target_file)
                
                print(f"Processed {unique_name}")

if __name__ == '__main__':
    # Choose the corresponding local or remote home directories
    local_home_dir = '/Users/eleanew83/Documents/OneDrive - University of Cambridge/Cambridge'
    # remote_home_dir = '/home/ylj20'
    source_dir = local_home_dir + '/Gibraltar_Macaques_Photos_Cleaned'
    target_dir = local_home_dir + '/macaque_flattened'
    
    flatten_directory_structure(source_dir, target_dir)
    print("Flattening complete. Now you can run prepare_macaque_dataset.py on the flattened directory.")
