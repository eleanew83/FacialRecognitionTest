#!/usr/bin/env python3
"""
Simplified training script for GorillaVision that works without wandb and runs offline
"""
import sys
import os
import json
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Add the reid-system directory to the path
sys.path.insert(0, '/gorilla-reidentification/reid-system')

# Disable wandb completely by making a mock version
class MockWandb:
    def __init__(self):
        self.run = MockRun()
        self.config = {}
        
    def init(self, **kwargs):
        print("[INFO] Wandb disabled. Running in offline mode.")
        return self.run
        
    def log(self, data, **kwargs):
        pass
        
    def watch(self, model, **kwargs):
        pass
        
    def finish(self):
        pass
        
    def Table(self, *args, **kwargs):
        return {}
        
    def Image(self, *args, **kwargs):
        return {}
        
    def login(self, *args, **kwargs):
        pass

class MockRun:
    def __init__(self):
        self.name = "offline_run"
        self.id = "offline_id"
        
    def log(self, data, **kwargs):
        pass
        
    def finish(self):
        pass
        
    @property
    def dir(self):
        return os.getcwd()

# Create mock module
sys.modules['wandb'] = MockWandb()

# Now import GorillaVision modules
from gorillavision.model.triplet import TripletLoss
from gorillavision.utils.dataset_utils import load_data
from gorillavision.utils.logger import logger

# Also patch any imports within the module itself
for module_name in list(sys.modules.keys()):
    if module_name.startswith('gorillavision'):
        module = sys.modules[module_name]
        if 'wandb' in module.__dict__:
            module.__dict__['wandb'] = sys.modules['wandb']

def train(df, lr, batch_size, input_width, input_height, embedding_size, nb_epochs, sampler, use_augmentation,
          augment_config, model_save_path, train_val_split_overlapping, class_sampler_config, cutoff_classes, 
          l2_factor, img_preprocess, backbone, experiment_desc="-"):
    
    logger.info("Initializing Model")
    img_size = (input_width, input_height)

    if not os.path.exists(model_save_path):
        logger.warning(f"Path {model_save_path} does not exist. Creating it...")
        os.makedirs(model_save_path, exist_ok=True)

    model = TripletLoss(
        df=df,
        embedding_size=embedding_size,
        lr=lr,
        batch_size=batch_size,
        sampler=sampler,
        use_augmentation=use_augmentation,
        augment_config=augment_config,
        train_val_split_overlapping=train_val_split_overlapping,
        class_sampler_config=class_sampler_config,
        cutoff_classes=cutoff_classes,
        l2_factor=l2_factor,
        img_size=img_size,
        img_preprocess=img_preprocess,
        backbone=backbone
    )

    logger.info("Initializing Trainer")
    checkpointCallback = ModelCheckpoint(
        dirpath=model_save_path,
        filename=f"Model_macaque_{{epoch}}-loss-{{val_loss:.2f}}",
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    # CPU training
    trainer = pl.Trainer(
        max_epochs=nb_epochs,
        callbacks=[checkpointCallback],
        accelerator='cpu',  # Use CPU explicitly
        devices=1  # Use 1 CPU device
    )

    logger.info("Starting Training")
    trainer.fit(model)
    logger.info("Model trained.")
    
    # Find best model
    best_loss = float('inf')
    best_model = ""
    if os.path.exists(model_save_path):
        for model_name in list(filter(lambda file_name: file_name.startswith("Model"), os.listdir(model_save_path))):
            parts = model_name.split("-loss-")
            if len(parts) > 1:
                try:
                    loss_part = parts[1].split('.')[0]  # Get the part before the extension
                    loss = float(loss_part)
                    if loss < best_loss:
                        best_loss = loss
                        best_model = model_name
                except ValueError:
                    continue

    if best_model:
        best_model_path = os.path.join(model_save_path, best_model)
        logger.info(f"Best model: {best_model_path}")
        return best_model_path
    else:
        logger.warning("No models found.")
        return None

def main():
    logger.info("Loading config...")
    argparser = argparse.ArgumentParser(description='Train and validate a model on any dataset')
    argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')
    args = argparser.parse_args()
    conf_name = args.conf
    
    config_path = os.path.join("/gorilla-reidentification/reid-system/gorillavision/configs", conf_name)
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    
    # Check if data path exists
    data_path = config["train"]["dataset"]["path"]
    if not os.path.exists(data_path):
        logger.error(f"Data path not found: {data_path}")
        print(f"Available directories in /data: {os.listdir('/data')}")
        sys.exit(1)
    
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    logger.info(f"Loaded dataset with {len(df)} images from {df['labels'].nunique()} individuals")
    
    # Extract configuration
    lr = config["train"]["learning_rate"]
    batch_size = config["train"]["batch_size"]
    input_width = config['model']['input_width']
    input_height = config['model']['input_height']
    embedding_size = config["model"]["embedding_size"]
    nb_epochs = config["train"]["nb_epochs"]
    sampler = config["train"]["sampler"]
    use_augmentation = config["train"]["use_augmentation"]
    augment_config = config["train"]["augment_config"]
    model_save_path = config["train"]["model_save_path"]
    train_val_split_overlapping = config["train"]["train_val_split_overlapping"]
    class_sampler_config = config["train"]["class_sampler_config"]
    cutoff_classes = config["model"]["cutoff_classes"]
    l2_factor = config["train"]["l2_factor"]
    img_preprocess = config["model"]["img_preprocess"]
    backbone = config["model"]["backbone"]
    
    # Run training
    model_path = train(
        df, lr, batch_size, input_width, input_height, embedding_size, nb_epochs, 
        sampler, use_augmentation, augment_config, model_save_path, 
        train_val_split_overlapping, class_sampler_config, cutoff_classes, 
        l2_factor, img_preprocess, backbone
    )
    
    logger.info(f"Training completed. Best model: {model_path}")
    
    # Update config with model path
    if model_path:
        config["create_db"]["model_path"] = model_path
        config["eval"]["model_path"] = model_path
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Updated config file with best model path: {model_path}")

if __name__ == '__main__':
    main() 