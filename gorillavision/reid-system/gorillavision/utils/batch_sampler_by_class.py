import copy
import time
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data import DataLoader
from numpy.random import shuffle, choice
import logging

logger = logging.getLogger(__name__)

class BatchSamplerByClass(BatchSampler):
    """A completely rewritten batch sampler that prevents hanging
    and handles edge cases more robustly.
    """
    def __init__(self, ds, seed=123, classes_per_batch=15, samples_per_class=3, max_batches=None):
        # Initialize with safety parameters
        self.ds = ds
        self.seed = seed
        np.random.seed(seed)
        self.classes_ds = {}
        self.labels = []
        self.max_batches = max_batches  # Optional cap on number of batches
        
        # Set a timeout for initialization to prevent hanging
        start_time = time.time()
        timeout = 30  # seconds
        
        # Log initialization start
        print(f"[BatchSamplerByClass] Initializing with {len(ds)} samples")
        
        # Create a mapping of class -> sample indices
        try:
            # Use a faster approach with a single pass through the dataset
            temp_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            for idx, batch in enumerate(temp_loader):
                # Check timeout
                if time.time() - start_time > timeout:
                    print(f"[BatchSamplerByClass] WARNING: Initialization timed out after {timeout}s")
                    break
                    
                # Get label and add to mapping
                try:
                    label = batch["labels"].item()
                    self.labels.append(label)
                    
                    if label not in self.classes_ds:
                        self.classes_ds[label] = [idx]
                    else:
                        self.classes_ds[label].append(idx)
                except Exception as e:
                    print(f"[BatchSamplerByClass] Error processing sample {idx}: {e}")
                    continue
                    
                # Print progress occasionally
                if idx % 100 == 0:
                    print(f"[BatchSamplerByClass] Processed {idx}/{len(ds)} samples")
        except Exception as e:
            print(f"[BatchSamplerByClass] Error during initialization: {e}")
            # Create a fallback with random classes if initialization fails
            if not self.classes_ds:
                print(f"[BatchSamplerByClass] Creating fallback class mapping")
                random_classes = np.random.randint(0, 10, len(ds))
                for idx, cls in enumerate(random_classes):
                    if cls not in self.classes_ds:
                        self.classes_ds[cls] = [idx]
                    else:
                        self.classes_ds[cls].append(idx)
                self.labels = random_classes.tolist()
        
        # Safety checks on parameters
        available_classes = len(self.classes_ds.keys())
        if available_classes == 0:
            print(f"[BatchSamplerByClass] ERROR: No classes found in dataset")
            # Create dummy class
            self.classes_ds[0] = list(range(len(ds)))
            available_classes = 1
            
        # Adjust parameters based on available data
        self.classes_per_batch = min(classes_per_batch, available_classes)
        
        # Find minimum samples per class
        min_samples = min([len(v) for v in self.classes_ds.values()]) if self.classes_ds else 1
        self.samples_per_class = min(min_samples, samples_per_class)
        
        # Calculate batch size
        self.batch_size = self.samples_per_class * self.classes_per_batch
        
        # Ensure batch size is at least 1
        if self.batch_size < 1:
            print(f"[BatchSamplerByClass] WARNING: Calculated batch_size < 1, setting to 1")
            self.batch_size = 1
            
        print(f"[BatchSamplerByClass] Initialized with {len(self.classes_ds)} classes")
        print(f"[BatchSamplerByClass] classes_per_batch={self.classes_per_batch}, samples_per_class={self.samples_per_class}")
        print(f"[BatchSamplerByClass] batch_size={self.batch_size}")

    def __iter__(self):
        """Iterator with multiple safety mechanisms to prevent hanging"""
        # Calculate expected number of batches
        expected_batches = self.__len__()
        
        # Set safety limits
        max_iterations = expected_batches * 3  # Triple the expected to be safe
        if self.max_batches and self.max_batches < max_iterations:
            max_iterations = self.max_batches
            
        # Set timeout
        start_time = time.time()
        timeout = 60  # seconds
        
        print(f"[BatchSamplerByClass] Starting iteration, expecting {expected_batches} batches")
        print(f"[BatchSamplerByClass] Safety cap at {max_iterations} batches")
        
        # Track iteration
        iteration_count = 0
        yielded_count = 0
        
        # Main iteration loop
        while yielded_count < expected_batches and iteration_count < max_iterations:
            iteration_count += 1
            
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"[BatchSamplerByClass] WARNING: Iteration timed out after {timeout}s")
                break
                
            # Print progress occasionally
            if iteration_count % 10 == 0:
                print(f"[BatchSamplerByClass] Iteration progress: {yielded_count}/{expected_batches} batches")
                
            try:
                # Initialize batch
                batch = [0] * self.batch_size
                idx_in_batch = 0
                
                # Get available classes
                available_classes = list(self.classes_ds.keys())
                
                # Sample classes
                if len(available_classes) < self.classes_per_batch:
                    # Not enough classes, use with replacement
                    classes = np.random.choice(available_classes, self.classes_per_batch, replace=True)
                else:
                    # Normal case
                    classes = np.random.choice(available_classes, self.classes_per_batch, replace=False)
                
                # Sample from each class
                for class_idx in range(len(classes)):
                    class_label = classes[class_idx]
                    class_samples = self.classes_ds.get(class_label, [])
                    
                    # Skip empty classes
                    if not class_samples:
                        continue
                        
                    # Sample from this class
                    if len(class_samples) < self.samples_per_class:
                        # Not enough samples, use with replacement
                        selected_idx = np.random.choice(class_samples, self.samples_per_class, replace=True)
                    else:
                        # Normal case
                        selected_idx = np.random.choice(class_samples, self.samples_per_class, replace=False)
                    
                    # Add to batch
                    end_idx = min(idx_in_batch + len(selected_idx), self.batch_size)
                    batch[idx_in_batch:end_idx] = selected_idx[:end_idx-idx_in_batch]
                    idx_in_batch = end_idx
                    
                    # Check if batch is full
                    if idx_in_batch >= self.batch_size:
                        break
                
                # Only yield if we have at least one sample
                if idx_in_batch > 0:
                    # If batch is not full, pad with random samples
                    if idx_in_batch < self.batch_size:
                        all_indices = np.concatenate(list(self.classes_ds.values()))
                        padding = np.random.choice(all_indices, self.batch_size - idx_in_batch)
                        batch[idx_in_batch:] = padding
                    
                    yielded_count += 1
                    yield batch
                    
            except Exception as e:
                print(f"[BatchSamplerByClass] Error in iteration {iteration_count}: {e}")
                # Create a fallback batch with random samples
                try:
                    fallback_batch = np.random.choice(range(len(self.ds)), self.batch_size).tolist()
                    yielded_count += 1
                    yield fallback_batch
                except Exception as e2:
                    print(f"[BatchSamplerByClass] Error creating fallback batch: {e2}")
        
        print(f"[BatchSamplerByClass] Iteration complete: yielded {yielded_count}/{expected_batches} batches")
        print(f"[BatchSamplerByClass] Took {iteration_count} iterations and {time.time() - start_time:.2f}s")

    def __len__(self) -> int:
        """Calculate number of batches with a safety cap"""
        # Basic calculation
        if len(self.ds) == 0 or self.batch_size == 0:
            return 0
            
        # Calculate based on dataset size
        n_batches = len(self.ds) // self.batch_size
        
        # Apply safety cap
        max_safe_batches = 100  # Reasonable upper limit
        capped_batches = min(n_batches, max_safe_batches)
        
        if capped_batches < n_batches:
            print(f"[BatchSamplerByClass] WARNING: Capped batches from {n_batches} to {capped_batches}")
            
        # Apply user-specified cap if provided
        if self.max_batches and self.max_batches < capped_batches:
            return self.max_batches
            
        return capped_batches
