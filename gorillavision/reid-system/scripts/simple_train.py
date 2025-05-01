#!/usr/bin/env python3
"""
Simplified training script for GorillaVision that works without wandb
"""
import sys
import os
import json
import argparse

# Make sure backports.tarfile is available
try:
    import backports.tarfile
except ImportError:
    # Create a simple mock module
    class MockTarfile:
        @staticmethod
        def open(*args, **kwargs):
            pass
    sys.modules['backports.tarfile'] = MockTarfile()
    
# Make sure setuptools is properly set up
os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'stdlib'

# Now import pytorch lightning
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Add the reid-system directory to the path
sys.path.insert(0, '/gorilla-reidentification/reid-system')

# Set up mock wandb
class MockWandb:
    def __init__(self):
        self.run = MockRun()
        
    def init(self, **kwargs):
        print("Mock wandb.init() called with:", kwargs)
        return self.run
        
    def log(self, data, **kwargs):
        print("Mock wandb.log() called")
        
    def watch(self, model, **kwargs):
        print("Mock wandb.watch() called")
        
    def finish(self):
        print("Mock wandb.finish() called")

class MockRun:
    def __init__(self):
        self.name = "mock_run"
        
    def log(self, data, **kwargs):
        pass
        
    def finish(self):
        pass

# Only create a mock if needed
try:
    import wandb
    print("Using real wandb")
except ImportError:
    # Create mock module
    sys.modules['wandb'] = MockWandb()
    wandb = sys.modules['wandb']

from gorillavision.model.triplet import TripletLoss
from gorillavision.utils.dataset_utils import load_data
from gorillavision.utils.logger import logger

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
        filename=f"Model_macaque_{{epoch}}-loss-{{val_loss:.50f}}",
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    
    # CPU training
    trainer = pl.Trainer(
        max_epochs=nb_epochs,
        callbacks=[checkpointCallback],
        enable_progress_bar=False,
        enable_model_summary=True,
        log_every_n_steps=1,
        detect_anomaly=True,  # Helps catch exploding gradients or NaNs
        num_sanity_val_steps=0
    )

    logger.info("⚠️ Manually calling prepare_data for debug")
    model.prepare_data()
    logger.info("⚠️ Manually calling train_dataloader for debug")
    try:
        dl = model.train_dataloader()
        logger.info("✅ train_dataloader returned successfully")
        # Optional: fetch one batch to force the iteration
        first_batch = next(iter(dl))
        logger.info(f"✅ Retrieved first batch with keys: {list(first_batch.keys())}")
    except Exception as e:
        logger.error(f"❌ Error in train_dataloader: {e}")

    print("🟡 About to call trainer.fit(model)")
    logger.info("Starting Training")
    trainer.fit(model)
    print("✅ trainer.fit(model) returned")
    logger.info("Model trained.")
    
    # Find best model
    best_loss = float('inf')
    best_model = ""
    for model_name in list(filter(lambda file_name: file_name.startswith("Model"), os.listdir(model_save_path))):
        def get_loss(model_name):
            return float(model_name.split("=")[-1][:-5])
        
        try:
            loss = get_loss(model_name)
            if loss < best_loss:
                best_loss = loss
                best_model = model_name
        except:
            continue

    return os.path.join(model_save_path, best_model) if best_model else None

def main():
    logger.info("Loading config...")
    argparser = argparse.ArgumentParser(description='Train and validate a model on any dataset')
    argparser.add_argument('-c','--conf', help='name of the configuration file in config folder', default='config.json')
    args = argparser.parse_args()
    conf_name = args.conf
    
    config_path = os.path.join("/gorilla-reidentification/reid-system/gorillavision/configs", conf_name)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    
    df = load_data(config["train"]["dataset"]["path"])
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
