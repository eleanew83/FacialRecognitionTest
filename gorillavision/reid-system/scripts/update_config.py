#!/usr/bin/env python3
import json
import os
import argparse
import glob

def find_latest_model(models_dir):
    """Find the latest model file in the models directory"""
    model_files = glob.glob(os.path.join(models_dir, "Model_*.ckpt"))
    if not model_files:
        return None
    
    # Sort by modification time (most recent first)
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def update_config(config_path, model_path):
    """Update the config file with the model path"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update model paths in create_db and eval sections
    if model_path:
        config['create_db']['model_path'] = model_path
        config['eval']['model_path'] = model_path
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated config at {config_path} with model path: {model_path}")

def main():
    parser = argparse.ArgumentParser(description='Update config file with latest model path')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--models-dir', type=str, required=True, help='Directory containing model files')
    
    args = parser.parse_args()
    
    latest_model = find_latest_model(args.models_dir)
    if not latest_model:
        print("No model files found. Config not updated.")
        return
    
    update_config(args.config, latest_model)

if __name__ == '__main__':
    main() 