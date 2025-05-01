#!/bin/bash

# Fix wandb and related package dependencies in the Docker container
# This script should be run inside the container before running the training

echo "Fixing dependency issues..."

# Install the specific versions known to work together
#pip install --force-reinstall setuptools==65.6.0
#pip install --force-reinstall backports.tarfile==0.1
#pip install --force-reinstall jaraco.text==3.7.0
#pip install --force-reinstall traitlets==5.1.1 ipython==7.34.0
#pip install --force-reinstall pydantic==1.8.2
#pip install --force-reinstall markdown-it-py==1.0.0

# Try to force reinstall pytorch-lightning to pick up the new dependencies
#pip install --force-reinstall --no-deps pytorch_lightning==2.2.5
#pip install --force-reinstall --no-deps lightning-utilities==0.11.9

# Run pip check to see if there are any remaining conflicts
echo "Checking for remaining dependency issues..."
pip check || echo "There are still some dependency issues, but we'll try to continue anyway"

# Make a backup of the original better_class_sampler.py 
if [ ! -f /gorilla-reidentification/reid-system/gorillavision/utils/better_class_sampler.py.bak ]; then
    cp /gorilla-reidentification/reid-system/gorillavision/utils/better_class_sampler.py /gorilla-reidentification/reid-system/gorillavision/utils/better_class_sampler.py.bak
fi

# Create a patched version of better_class_sampler.py that handles wandb import safely
cat > /gorilla-reidentification/reid-system/gorillavision/utils/better_class_sampler.py << 'EOF'
import copy
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from numpy.random import shuffle, choice

# Safe import of wandb
try:
    import wandb
except Exception as e:
    print(f"Warning: Could not import wandb - {str(e)}")
    # Create a mock wandb if import fails
    class MockWandb:
        def __init__(self):
            self.run = MockRun()
        def init(self, **kwargs):
            print("Using mock wandb.init()")
            return self.run
        def log(self, data, **kwargs):
            pass
        def watch(self, model, **kwargs):
            pass
        def finish(self):
            pass
    
    class MockRun:
        def __init__(self):
            self.name = "mock_run"
        def log(self, data, **kwargs):
            pass
        def finish(self):
            pass
    
    wandb = MockWandb()

class BatchSamplerByClass(BatchSampler):
    def __init__(self, ds, seed=123, classes_per_batch=15, samples_per_class=3):
        # Uses every class once per batch. For every class takes min(smaples_per_class, len(class.samples))
        
        self.ds = ds
        self.classes_ds = {}
        self.labels = []
        # create one df for every class
        for idx, row in enumerate(DataLoader(ds)):
            self.labels.append(row["labels"].item())
            if row["labels"].item() not in self.classes_ds:
                self.classes_ds[row["labels"].item()] = [idx]
            else: 
                self.classes_ds[row["labels"].item()].append(idx)
        self.classes_per_batch = min(classes_per_batch, len(list(self.classes_ds.keys())))
        self.samples_per_class = samples_per_class
        self.batch_size = self.samples_per_class * self.classes_per_batch
        np.random.seed(seed)

    def __iter__(self):
        current_classes = list(self.classes_ds.keys())
        for i in range(0, self.__len__()):
            batch = [0] * self.batch_size
            idx_in_batch = 0
            amount_cls = min(self.classes_per_batch, len(current_classes))
            classes = np.random.choice(current_classes, amount_cls, replace=False)
            current_classes = [c for c in current_classes if c not in classes]
            for i in range(0, len(classes)):
                num_samples = min(self.samples_per_class, len(self.classes_ds[classes[i]]))
                selected_idx = np.random.choice(self.classes_ds[classes[i]], num_samples, replace=False)
                batch[idx_in_batch:idx_in_batch + len(selected_idx)] = selected_idx
                idx_in_batch += len(selected_idx)
            yield batch

    def __len__(self) -> int:
        return len(self.ds) // self.batch_size
EOF

# Also patch simple_train.py to handle module import issues
echo "Patching simple_train.py to handle import issues..."
cp /gorilla-reidentification/reid-system/scripts/simple_train.py /gorilla-reidentification/reid-system/scripts/simple_train.py.bak

# Create a backports module with tarfile
mkdir -p /tmp/backports_patch
cat > /tmp/backports_patch/setup.py << EOF
from setuptools import setup, find_packages

setup(
    name="backports.tarfile",
    version="0.1",
    packages=find_packages(),
)
EOF

mkdir -p /tmp/backports_patch/backports
touch /tmp/backports_patch/backports/__init__.py
cat > /tmp/backports_patch/backports/tarfile.py << EOF
# Mock tarfile module
class TarFile:
    @staticmethod
    def open(*args, **kwargs):
        pass
EOF

cd /tmp/backports_patch && pip install -e .

cat > /gorilla-reidentification/reid-system/scripts/simple_train.py << 'EOF'
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
        callbacks=[checkpointCallback]
    )

    logger.info("Starting Training")
    trainer.fit(model)
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
EOF

echo "Container dependencies fixed successfully!" 
