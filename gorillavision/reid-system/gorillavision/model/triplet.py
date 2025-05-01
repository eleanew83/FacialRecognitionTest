from gorillavision.utils.losses import triplet_semihard_loss
import pytorch_lightning as pl
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
# from utils.batch_sampler_by_class import BatchSamplerByClass
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

        # Initialize WandB here
        try:
            wandb.init(project='Gibraltar_Macaques_TripletLoss', config=self.hparams)  # Pass hyperparameters to WandB
            logger.info("WandB initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WandB: {e}")

        # Decide whether to use CPU or GPU automatically
        self._device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self._device}")

        self.df = df
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
            
            if self.sampler == "class_sampler":
                classes_per_batch = self.class_sampler_config["classes_per_batch"]
                samples_per_class = self.class_sampler_config["samples_per_class"]
                logger.info(f"Using BatchSamplerByClass with {classes_per_batch} classes per batch, {samples_per_class} samples per class")
                self.batch_sampler_train = BatchSamplerByClass(ds=self.train_ds, classes_per_batch=classes_per_batch, samples_per_class=samples_per_class)
                self.batch_sampler_val = BatchSamplerByClass(ds=self.validate_ds, classes_per_batch=classes_per_batch, samples_per_class=samples_per_class)
            elif self.sampler == "ensure_positive":
                logger.info(f"Using BatchSamplerEnsurePositives with batch size {self.batch_size}")
                self.batch_sampler_train = BatchSamplerEnsurePositives(ds=self.train_ds, batch_size=self.batch_size)
                self.batch_sampler_val = BatchSamplerEnsurePositives(ds=self.validate_ds, batch_size=self.batch_size)
            else:
                logger.warning(f"Using default sampler: {self.sampler}")
            
            logger.info("Data preparation completed successfully")
        except Exception as e:
            logger.error(f"Error in prepare_data: {e}")
            raise

    def configure_optimizers(self):
        logger.info("Configuring optimizer...")
        try:
            if self.l2_factor != None:
                logger.info(f"Using Adam optimizer with lr={self.lr} and weight_decay={self.l2_factor}")
                return Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=self.l2_factor)
            else:
                logger.info(f"Using Adam optimizer with lr={self.lr} without weight decay")
                return Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-08)
        except Exception as e:
            logger.error(f"Error in configure_optimizers: {e}")
            raise

    def train_dataloader(self):
        logger.info(f"Setting up train dataloader with sampler: {self.sampler}")
        try:
            if self.sampler == "class_sampler":
                logger.info("Using class_sampler for training")
                return DataLoader(self.train_ds, batch_sampler=self.batch_sampler_train, num_workers=0)
            elif self.sampler == "random_sampler":
                logger.info("Using random_sampler for training")
                return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
            elif self.sampler == "ensure_positive":
                logger.info("Using ensure_positive sampler for training")
                return DataLoader(self.train_ds, batch_sampler=self.batch_sampler_train, num_workers=0)
            logger.error(f"No valid sampler specified: {self.sampler}")
            raise Exception("No sampler specified")
        except Exception as e:
            logger.error(f"Error in train_dataloader: {e}")
            raise

    def val_dataloader(self):
        logger.info("Setting up validation dataloader")
        try:
            if self.sampler == "class_sampler":
                return DataLoader(self.validate_ds, batch_sampler=self.batch_sampler_val, num_workers=0)
            elif self.sampler == "random_sampler":
                return DataLoader(self.validate_ds, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)
            elif self.sampler == "ensure_positive":
                return DataLoader(self.validate_ds, batch_sampler=self.batch_sampler_val, num_workers=0)
            logger.error(f"No valid sampler specified: {self.sampler}")
            raise Exception("No sampler specified")
        except Exception as e:
            logger.error(f"Error in val_dataloader: {e}")
            raise
    
    def on_train_start(self):
        logger.info("Training is starting...")
        try:
            wandb.watch(self, log='all')
            logger.info("wandb.watch initialized")
        except Exception as e:
            logger.error(f"Error in on_train_start with wandb.watch: {e}")

    def on_after_batch_transfer(self, batch, dataloader_idx):
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

    def on_train_epoch_end(self, training_step_outputs):
        logger.info("Train epoch ended")
        try:
            # Compute average loss
            if training_step_outputs:
                avg_loss = torch.stack([x['loss'] for x in training_step_outputs if 'loss' in x]).mean()
                logger.info(f"Average training loss for epoch: {avg_loss.item()}")
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
    
    def on_validation_epoch_end(self, validationStepOutputs):
        logger.info("Validation epoch ended")
        try:
            if validationStepOutputs:
                avgLoss = torch.stack([x['val_loss'] for x in validationStepOutputs]).mean()
                logger.info(f"Average validation loss for epoch: {avgLoss.item()}")
                self.log('avg_val_loss_epoch', avgLoss, prog_bar=True)
                try:
                    wandb.log({'avg_val_loss_epoch': avgLoss, 'epoch': self.current_epoch})
                except Exception as e:
                    logger.error(f"Failed to log avg_val_loss_epoch to wandb: {e}")
        except Exception as e:
            logger.error(f"Error in on_validation_epoch_end: {e}")

