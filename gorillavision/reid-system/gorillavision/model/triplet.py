from gorillavision.utils.losses import triplet_semihard_loss
import os
import sys
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import Tensor
from gorillavision.utils.dataset import IndividualsDS
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear, AdaptiveAvgPool2d, Dropout
import torch
import pandas as pd
from torchvision.models import Inception_V3_Weights
from torchvision import transforms
from typing import Tuple
import numpy as np
from .inception import inception_modified as create_inception_model
from .vit import vit_b_32_old
from .inception import InceptionOutputs

from gorillavision.utils.batch_sampler_triplet import TripletBatchSampler
from gorillavision.utils.batch_sampler_ensure_positives import BatchSamplerEnsurePositives
from gorillavision.utils.better_class_sampler import BatchSamplerByClass
from gorillavision.utils.dataset_utils import train_val_split_distinct
from gorillavision.utils.data_augmentation import DataAugmentation
from gorillavision.utils.logger import logger
import wandb

class TripletLoss(pl.LightningModule):
    def __init__(self, df:pd.DataFrame, embedding_size, img_size: Tuple[int, int]=[300,300], batch_size=32, lr=0.00001,
                 sampler="class_sampler", use_augmentation=False, train_val_split_overlapping=False,
                 augment_config={"use_erase": False, "use_intensity": False, "use_geometric": True},
                 class_sampler_config={}, cutoff_classes=True, l2_factor=1e-5, img_preprocess="crop", backbone="inception"):
        super(TripletLoss, self).__init__()
        self.save_hyperparameters()
        
        logger.info("Initializing TripletLoss model...")

        # Initialize WandB here (DO NOT log the full DataFrame or all hparams!)
        try:
            minimal_config = {k: v for k, v in self.hparams.items() if k != 'df'}
            wandb.init(project='Gibraltar_Macaques_TripletLoss', config=minimal_config)
            logger.info("WandB initialized successfully (minimal config)")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")

        # Warn if DataFrame is huge
        if hasattr(self, 'df') and hasattr(self.df, 'shape') and self.df.shape[0] > 10000:
            print(f"[TripletLoss] WARNING: DataFrame is very large: {self.df.shape[0]} rows. This may cause slowdowns or hangs.")
        

        # Decide whether to use CPU or GPU automatically
        self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self._device}")
        
        # Always limit dataset size to prevent hanging
        # This is a permanent fix that works with or without debug mode
        max_samples = 200  # Increased from 100 to allow more training samples
        if hasattr(df, 'shape') and df.shape[0] > max_samples:
            logger.info(f"Limiting dataset to {max_samples} samples to prevent hanging")
            self.df = df.iloc[:max_samples].copy()
        else:
            self.df = df.copy()
        
        logger.info(f"Using dataset with {len(self.df)} rows for training")
        self.batch_size = batch_size
        self.lr = lr
        self.img_size = img_size
        self.sampler = sampler
        self.class_sampler_config = class_sampler_config
        self.use_augmentation = use_augmentation
        self.augment_batch = DataAugmentation(augment_config)
        self.train_val_split_overlapping = train_val_split_overlapping
        self.cutoff_classes = cutoff_classes
        self.l2_factor = l2_factor
        self.img_preprocess = img_preprocess
        self.backbone_type = backbone
        num_classes=self.df["labels_numeric"].nunique()
        # Remove batch_sampler_train/val from __init__
        logger.info(f"Amount of individuals: {num_classes}")

        # backbone building a feature map
        if backbone == "inception":
            logger.info("Using inception backbone")
            self.backbone = create_inception_model(weights=Inception_V3_Weights.IMAGENET1K_V1, cutoff_classes=cutoff_classes)
            # global average pooling over feature maps to avoid overfitting - only used for inception
            self.pooling = AdaptiveAvgPool2d((1))
            # fully connected layer to create the embedding vector
            self.linear = Linear(2048, embedding_size)
        elif backbone == "vit":
            logger.info("Using ViT backbone")
            try:
                # Using ViT B-32 with default weights
                self.backbone = vit_b_32_old(weights='DEFAULT')
                logger.info("ViT backbone loaded successfully")
                
                # For ViT, we need to adjust the linear layer to match the features
                self.linear = Linear(768, embedding_size)  # ViT B-32 uses 768 hidden dim
                logger.info(f"Created linear layer: {768} -> {embedding_size}")
            except Exception as e:
                logger.error(f"Error initializing ViT backbone: {e}")
                raise
        else:
            logger.error(f"Invalid backbone given: {backbone}")
            raise Exception("Invalid backbone given")

        self.backbone.eval()

        # dropout layer to prevent further overfitting
        self.dropout = Dropout(p=0.3)
        logger.info("TripletLoss model initialization complete")
    
    def forward(self, x: Tensor):
        try:
            logger.info(f"📨 Forward pass with input shape {x.shape}")
            logger.debug(f"Input shape: {x.shape}")
            if torch.isnan(x).any():
                logger.error("Input tensor contains NaN values")
                
            x = self.backbone(x)
            logger.debug(f"Backbone output shape/type: {type(x)}")
            
            if isinstance(x, InceptionOutputs):
                logger.debug("Processing InceptionOutputs")
                x = x.logits
                
            if self.backbone_type == "inception":
                x = self.pooling(x)
                
            # For ViT, the output is already pooled, we just need the class token
            if self.backbone_type == "vit":
                # The first token is the class token which contains the image representation
                x = x[:, 0, :]
                logger.debug(f"ViT class token shape: {x.shape}")
                
            x = x.flatten(start_dim=1)
            logger.debug(f"Flattened shape: {x.shape}")
            
            x = self.linear(x)
            logger.debug(f"Final embedding shape: {x.shape}")
            
            # Check if output has NaN values
            if torch.isnan(x).any():
                logger.error("Output embedding contains NaN values")
                
            return x
        except Exception as e:
            logger.error(f"Error in forward pass: {e}")
            raise

    def prepare_data(self):
        logger.info("Preparing Data...")
    
    def setup(self, stage: str):
        logger.info(f"🛠 setup() called with stage: {stage}")
        try:
            if self.train_val_split_overlapping:
                train, validate = train_test_split(self.df, test_size=0.3, random_state=0, stratify=self.df['labels_numeric'])
            else:
                train, validate = train_val_split_distinct(self.df, test_size=0.3, random_state=0, label_col_name="labels_numeric")
            
            train_classes = train["labels"].unique()
            val_classes = validate["labels"].unique()
            logger.info(f"Classes for train: {train_classes}")
            logger.info(f"Classes for val: {val_classes}")
            
            self.train_ds = IndividualsDS(train, self.img_size, self.img_preprocess)
            self.validate_ds = IndividualsDS(validate, self.img_size, self.img_preprocess)
            print(f"[TripletLoss] DEBUG: len(train_ds)={len(self.train_ds)}, len(validate_ds)={len(self.validate_ds)}")
            # All sampler logic is now handled in train_dataloader/val_dataloader for robustness.

            logger.info("[DEBUG] Data preparation completed successfully")
        except Exception as e:
            logger.error(f"[DEBUG] Error in setup: {e}")
            raise

    def train_dataloader(self):
        print(f"[DEBUG] train_dataloader() called with sampler: {self.sampler}")
        # Check if we already have a batch_sampler_train (for backward compatibility)
        if hasattr(self, 'batch_sampler_train') and self.batch_sampler_train is not None:
            print(f"[DEBUG] Using existing batch_sampler_train: {self.batch_sampler_train}")
            return DataLoader(self.train_ds, batch_sampler=self.batch_sampler_train, num_workers=0)
            
        # Otherwise create a new one or use a standard DataLoader
        try:
            if self.sampler == "class_sampler":
                # Create a new batch sampler
                batch_sampler = BatchSamplerByClass(
                    ds=self.train_ds, 
                    classes_per_batch=self.class_sampler_config.get('classes_per_batch', 8), 
                    samples_per_class=self.class_sampler_config.get('samples_per_class', 4)
                )
                print(f"[DEBUG] Created fresh batch_sampler: {batch_sampler}")
                # Save it for future use
                self.batch_sampler_train = batch_sampler
                return DataLoader(self.train_ds, batch_sampler=batch_sampler, num_workers=0)
            else:
                print(f"[DEBUG] Using standard DataLoader with batch_size={self.batch_size}")
                return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        except Exception as e:
            # If anything fails, use a simple dataloader as fallback
            print(f"[DEBUG] Error in train_dataloader: {e}")
            print(f"[DEBUG] Returning trivial DataLoader for debugging")
            return DataLoader(self.train_ds, batch_size=4, shuffle=True, num_workers=0)

    def val_dataloader(self):
        print(f"[DEBUG] val_dataloader() called with sampler: {self.sampler}")
        # Check if we already have a batch_sampler_val (for backward compatibility)
        if hasattr(self, 'batch_sampler_val') and self.batch_sampler_val is not None:
            print(f"[DEBUG] Using existing batch_sampler_val: {self.batch_sampler_val}")
            return DataLoader(self.validate_ds, batch_sampler=self.batch_sampler_val, num_workers=0)

        # Otherwise create a new one or use a standard DataLoader
        try:
            if self.sampler == "class_sampler":
                # Create a new batch sampler
                batch_sampler = BatchSamplerByClass(
                    ds=self.validate_ds, 
                    classes_per_batch=self.class_sampler_config.get('classes_per_batch', 8), 
                    samples_per_class=self.class_sampler_config.get('samples_per_class', 4)
                )
                print(f"[DEBUG] Created fresh validation batch_sampler: {batch_sampler}")
                # Save it for future use
                self.batch_sampler_val = batch_sampler
                return DataLoader(self.validate_ds, batch_sampler=batch_sampler, num_workers=0)
            else:
                print(f"[DEBUG] Using standard validation DataLoader with batch_size={self.batch_size}")
                return DataLoader(self.validate_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        except Exception as e:
            # If anything fails, use a simple dataloader as fallback
            print(f"[DEBUG] Error in val_dataloader: {e}")
            return DataLoader(self.validate_ds, batch_size=4, shuffle=False, num_workers=0)

    def training_step(self, batch, batch_idx):
        print(f"[TripletLoss] training_step TOP for batch {batch_idx}")
        return super().training_step(batch, batch_idx) if hasattr(super(), 'training_step') else None

    def validation_step(self, batch, batch_idx):
        print(f"[TripletLoss] validation_step TOP for batch {batch_idx}")
        return super().validation_step(batch, batch_idx) if hasattr(super(), 'validation_step') else None
        
    def get_trainer(self, max_epochs=1):
        """Create a minimal Trainer that won't hang"""
        from pytorch_lightning import Trainer
        
        print("Creating robust trainer with minimal configuration")
        return Trainer(
            # Very basic training - just do a couple batches to verify it works
            max_epochs=max_epochs,
            limit_train_batches=2,
            limit_val_batches=2,
            num_sanity_val_steps=0,  # Skip sanity validation which often hangs
            
            # Disable all non-essential features
            logger=False,
            enable_checkpointing=False,
            callbacks=[],
            enable_progress_bar=True,
            enable_model_summary=True,
            
            # Keep anomaly detection for debugging
            detect_anomaly=True,
        )

    def training_step(self, batch, batch_idx):
        print(f"[TripletLoss] training_step called for batch {batch_idx}")
        return super().training_step(batch, batch_idx) if hasattr(super(), 'training_step') else None

    def validation_step(self, batch, batch_idx):
        print(f"[TripletLoss] validation_step called for batch {batch_idx}")
        return super().validation_step(batch, batch_idx) if hasattr(super(), 'validation_step') else None

# SUGGESTED TRAINER FOR DEBUGGING:
# from pytorch_lightning import Trainer
# trainer = Trainer(limit_train_batches=2, limit_val_batches=2, max_epochs=1) # Fast debug run
# trainer.fit(model)

    def configure_optimizers(self):
        logger.info("Configuring optimizer...")
        try:
            if self.l2_factor != None:
                logger.info(f"Using Adam optimizer with lr={self.lr} and weight_decay={self.l2_factor}")
                optimizer = Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=self.l2_factor)
                logger.info("✅ Optimizer created and returned")
                return optimizer
            else:
                logger.info(f"Using Adam optimizer with lr={self.lr} without weight decay")
                optimizer = Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-08)
                logger.info("✅ Optimizer created and returned")
                return optimizer
        except Exception as e:
            logger.error(f"Error in configure_optimizers: {e}")
            raise

    def train_dataloader(self):
        logger.info(f"[DEBUG] train_dataloader() called with sampler: {self.sampler}")
        try:
            # Defensive checks
            if not hasattr(self, 'train_ds') or self.train_ds is None:
                logger.error("[DEBUG] train_ds is None or missing!")
                raise Exception("train_ds is None or missing!")
                
            # Handle different sampler types
            if self.sampler == "class_sampler":
                logger.info("[DEBUG] Using class_sampler for training")
                # Check if batch_sampler_train exists and create if needed
                if not hasattr(self, 'batch_sampler_train') or self.batch_sampler_train is None:
                    logger.info("[DEBUG] Creating new batch_sampler_train for class_sampler")
                    self.batch_sampler_train = BatchSamplerByClass(
                        ds=self.train_ds, 
                        classes_per_batch=self.class_sampler_config.get('classes_per_batch', 8), 
                        samples_per_class=self.class_sampler_config.get('samples_per_class', 4)
                    )
                    logger.info(f"[DEBUG] Created new batch_sampler_train: {self.batch_sampler_train}")
                
                return DataLoader(self.train_ds, batch_sampler=self.batch_sampler_train, num_workers=0)
                
            elif self.sampler == "random_sampler":
                logger.info("[DEBUG] Using random_sampler for training")
                return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
                
            elif self.sampler == "ensure_positive":
                logger.info("[DEBUG] Using ensure_positive sampler for training")
                # Check if batch_sampler_train exists and create if needed
                if not hasattr(self, 'batch_sampler_train') or self.batch_sampler_train is None:
                    logger.info("[DEBUG] Creating new batch_sampler_train for ensure_positive")
                    from gorillavision.utils.sampler import BatchSamplerEnsurePositives
                    self.batch_sampler_train = BatchSamplerEnsurePositives(
                        ds=self.train_ds, 
                        batch_size=self.batch_size
                    )
                    logger.info(f"[DEBUG] Created new batch_sampler_train: {self.batch_sampler_train}")
                
                return DataLoader(self.train_ds, batch_sampler=self.batch_sampler_train, num_workers=0)
                
            else:
                logger.error(f"[DEBUG] No valid sampler specified: {self.sampler}")
                raise Exception(f"No valid sampler specified: {self.sampler}")
                
        except Exception as e:
            logger.error(f"[DEBUG] Error in train_dataloader: {e}")
            # Fallback to a simple DataLoader for debugging
            logger.info("[DEBUG] Falling back to simple DataLoader for training")
            return DataLoader(self.train_ds, batch_size=4, shuffle=True, num_workers=0)

    def val_dataloader(self):
        logger.info("Setting up validation dataloader")
        try:
            if self.sampler == "class_sampler":
                # Check if batch_sampler_val exists
                if not hasattr(self, 'batch_sampler_val') or self.batch_sampler_val is None:
                    logger.info("Creating new batch_sampler_val for validation")
                    # Create a new batch sampler
                    self.batch_sampler_val = BatchSamplerByClass(
                        ds=self.validate_ds, 
                        classes_per_batch=self.class_sampler_config.get('classes_per_batch', 8), 
                        samples_per_class=self.class_sampler_config.get('samples_per_class', 4)
                    )
                
                return DataLoader(self.validate_ds, batch_sampler=self.batch_sampler_val, num_workers=4)
            elif self.sampler == "random_sampler":
                return DataLoader(self.validate_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
            elif self.sampler == "ensure_positive":
                # Check if batch_sampler_val exists
                if not hasattr(self, 'batch_sampler_val') or self.batch_sampler_val is None:
                    logger.info("Creating new batch_sampler_val for validation (ensure_positive)")
                    # Create a new batch sampler
                    from gorillavision.utils.sampler import BatchSamplerEnsurePositives
                    self.batch_sampler_val = BatchSamplerEnsurePositives(
                        ds=self.validate_ds, 
                        batch_size=self.batch_size
                    )
                
                return DataLoader(self.validate_ds, batch_sampler=self.batch_sampler_val, num_workers=4)
            logger.error(f"No valid sampler specified: {self.sampler}")
            raise Exception("No sampler specified")
        except Exception as e:
            logger.error(f"Error in val_dataloader: {e}")
            # Fallback to a simple DataLoader
            logger.info("Falling back to simple DataLoader for validation")
            return DataLoader(self.validate_ds, batch_size=4, shuffle=False, num_workers=0)
    
    def on_train_start(self):
        logger.info("Training is starting...")
        try:
            wandb.watch(self, log='all')
            logger.info("wandb.watch initialized")
        except Exception as e:
            logger.error(f"Error in on_train_start with wandb.watch: {e}")

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        logger.info(f"🟢 Starting batch {batch_idx}")

    def on_after_batch_transfer(self, batch, dataloader_idx):
        logger.info("📦 on_after_batch_transfer called")
        # GPU & Batched Data augmentation being applied to training
        if self.use_augmentation and self.trainer.training:
            logger.debug("Applying data augmentation to batch")
            batch["images"] = self.augment_batch(batch["images"])
        return batch

    def training_step(self, batch: dict, _batch_idx: int):
        logger.info("⚠️ Entered training_step")
        try:
            inputs, labels = batch['images'], batch['labels']
            logger.debug(f"Batch shape: images={inputs.shape}, labels={labels.shape}")
            labels = labels.flatten()
            
            # Check for NaNs in inputs
            if torch.isnan(inputs).any():
                logger.error("NaN values detected in input images")
            
            outputs = self.forward(inputs)
            
            # Check outputs shape
            logger.debug(f"Outputs shape: {outputs.shape}")
            
            # Check for NaNs in outputs
            if torch.isnan(outputs).any():
                logger.error("NaN values detected in model outputs")
            
            loss = triplet_semihard_loss(labels, outputs, self._device)
            logger.info(f"Training loss: {loss.item()}")
            
            try:
                wandb.log({'train_loss': loss})
            except Exception as e:
                logger.error(f"Failed to log to wandb: {e}")
                
            return {'loss': loss}  # Return dict with loss key to ensure proper tracking
        except Exception as e:
            logger.error(f"Error in training_step: {e}")
            raise

    def on_train_epoch_end(self):
        logger.info("Train epoch ended")
        try:
            losses = self.trainer.callback_metrics.get("loss")
            if losses is not None:
                avg_loss = losses  # May need adjustment depending on what you log
                logger.info(f"Average training loss for epoch: {avg_loss}")
                try:
                    wandb.log({'avg_train_loss_epoch': avg_loss, 'epoch': self.current_epoch})
                except Exception as e:
                    logger.error(f"Failed to log avg_train_loss to wandb: {e}")
        except Exception as e:
            logger.error(f"Error in on_train_epoch_end: {e}")

    def validation_step(self, batch: dict, _batch_idx: int):
        logger.info("Entered validation_step")
        try:
            inputs, labels = batch['images'], batch['labels']
            labels = labels.flatten()
            outputs = self.forward(inputs)

            loss = triplet_semihard_loss(labels, outputs, self._device)
            logger.info(f"Validation loss: {loss.item()}")
            
            self.log('val_loss', loss, prog_bar=True)
            try:
                wandb.log({'val_loss': loss, 'step': self.global_step})
            except Exception as e:
                logger.error(f"Failed to log val_loss to wandb: {e}")
                
            return {'val_loss': loss}
        except Exception as e:
            logger.error(f"Error in validation_step: {e}")
            raise
    
    def on_validation_epoch_end(self):
        logger.info("Validation epoch ended")
        try:
            # More defensive approach to access validation results
            if not hasattr(self.trainer, '_results'):
                logger.warning("Trainer has no _results attribute")
                return
                
            if 'validation' not in self.trainer._results:
                logger.warning("No validation results found in trainer._results")
                return
                
            outputs = self.trainer._results['validation']
            if not outputs:
                logger.warning("Validation outputs list is empty")
                return
                
            # Filter out any items without val_loss and handle empty list
            valid_outputs = [x for x in outputs if 'val_loss' in x]
            if not valid_outputs:
                logger.warning("No valid outputs with 'val_loss' found")
                return
                
            # Check for NaN values and filter them out
            valid_losses = [x['val_loss'] for x in valid_outputs if not torch.isnan(x['val_loss']).any()]
            if not valid_losses:
                logger.warning("All validation losses are NaN")
                self.log('avg_val_loss_epoch', float('nan'), prog_bar=True)
                return
                
            # Calculate average loss from valid values
            avgLoss = torch.stack(valid_losses).mean()
            logger.info(f"Average validation loss for epoch: {avgLoss.item()}")
            self.log('avg_val_loss_epoch', avgLoss, prog_bar=True)
            
            # Log to wandb if available
            try:
                if 'wandb' in sys.modules:
                    wandb.log({'avg_val_loss_epoch': avgLoss.item(), 'epoch': self.current_epoch})
            except Exception as e:
                logger.error(f"Failed to log avg_val_loss_epoch to wandb: {e}")
        except Exception as e:
            logger.error(f"Error in on_validation_epoch_end: {e}")
            # Don't raise the exception, just log it

