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
        print("[BatchSamplerByClass] __init__ called")
        # Uses every class once per batch. For every class takes min(smaples_per_class, len(class.samples))
        
        self.ds = ds
        self.classes_ds = {}
        self.labels = []
        # create one df for every class (do NOT use DataLoader here!)
        for idx in range(len(ds)):
            row = ds[idx]
            label = row["labels"].item() if hasattr(row["labels"], 'item') else row["labels"]
            self.labels.append(label)
            if label not in self.classes_ds:
                self.classes_ds[label] = [idx]
            else:
                self.classes_ds[label].append(idx)
        self.classes_per_batch = min(classes_per_batch, len(list(self.classes_ds.keys())))
        self.samples_per_class = samples_per_class
        self.batch_size = self.samples_per_class * self.classes_per_batch
        np.random.seed(seed)
        print(f"[BatchSamplerByClass] Constructed: {len(self.classes_ds)} classes, {self.batch_size} batch size")
        print("[BatchSamplerByClass] __init__ finished")


    def __iter__(self):
        print("[BatchSamplerByClass] __iter__ called")
        current_classes = list(self.classes_ds.keys())
        max_batches = min(self.__len__(), 10)  # Hard safety cap for debug
        for i in range(0, max_batches):
            print(f"[BatchSamplerByClass] Yielding batch {i+1} of {max_batches}")
            batch = []
            amount_cls = min(self.classes_per_batch, len(current_classes))
            if amount_cls < self.classes_per_batch:
                print(f"[BatchSamplerByClass] Not enough classes left to fill batch, resetting class pool.")
                current_classes = list(self.classes_ds.keys())
                amount_cls = min(self.classes_per_batch, len(current_classes))
            if amount_cls == 0:
                raise RuntimeError("[BatchSamplerByClass] No classes available to form a batch!")
            classes = np.random.choice(current_classes, amount_cls, replace=False)
            current_classes = [c for c in current_classes if c not in classes]
            for c in classes:
                num_samples = min(self.samples_per_class, len(self.classes_ds[c]))
                if num_samples == 0:
                    print(f"[BatchSamplerByClass] ERROR: No samples for class {c}")
                    continue
                if num_samples < self.samples_per_class:
                    print(f"[BatchSamplerByClass] Not enough samples for class {c}, using {num_samples}.")
                selected_idx = np.random.choice(self.classes_ds[c], num_samples, replace=False)
                batch.extend(selected_idx.tolist())
            if len(batch) == 0:
                print(f"[BatchSamplerByClass] ERROR: Empty batch at batch {i+1}, skipping!")
                continue
            print(f"[BatchSamplerByClass] Batch {i+1} indices: {batch}")
            yield batch
        print("[BatchSamplerByClass] __iter__ finished")

    def __len__(self) -> int:
        print("[BatchSamplerByClass] __len__ called")
        size = len(self.ds) // self.batch_size
        # Safety cap to prevent infinite epochs
        capped_size = min(size, 100)
        print(f"[BatchSamplerByClass] __len__ = {size}, capped to {capped_size}")
        print("[BatchSamplerByClass] __len__ finished")
        return capped_size
