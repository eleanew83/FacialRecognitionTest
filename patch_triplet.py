#!/usr/bin/env python3
"""
Script to patch TripletLoss class for PyTorch Lightning v2.0 compatibility
"""
import sys
import os
import re

def patch_triplet_file(triplet_path):
    print(f"Patching file: {triplet_path}")
    
    with open(triplet_path, 'r') as f:
        content = f.read()
    
    # Replace training_epoch_end with on_train_epoch_end
    if 'def training_epoch_end' in content:
        content = content.replace('def training_epoch_end', 'def on_train_epoch_end')
        print("Replaced training_epoch_end with on_train_epoch_end")
    
    # Replace validation_epoch_end with on_validation_epoch_end if it exists
    if 'def validation_epoch_end' in content:
        content = content.replace('def validation_epoch_end', 'def on_validation_epoch_end')
        print("Replaced validation_epoch_end with on_validation_epoch_end")
    
    # Save the patched file
    with open(triplet_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully patched {triplet_path}")

if __name__ == "__main__":
    # Check if a path was provided
    if len(sys.argv) > 1:
        triplet_path = sys.argv[1]
    else:
        # Default path based on project structure
        triplet_path = "gorillavision/reid-system/gorillavision/model/triplet.py"
    
    # Make sure the file exists
    if not os.path.exists(triplet_path):
        print(f"Error: File {triplet_path} not found")
        sys.exit(1)
    
    patch_triplet_file(triplet_path) 