from gorillavision.utils.transformations import EnsureSize, FillSizePad
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Normalize
from typing import Tuple
from torchvision.utils import save_image

class IndividualsDS(Dataset):

    def __init__(self, dataFrame, img_size: Tuple[int, int], img_preprocess):
        self.dataFrame = dataFrame
        
        # ! apply changes to processing of images here, also to image.py
        transformations_crop = Compose([
            ToPILImage(),
            Resize(img_size),
            ToTensor()
        ])

        transformations_pad = Compose([
            ToPILImage(),
            EnsureSize(img_size),
            FillSizePad(img_size),
            ToTensor()
        ])
        self.transformations = transformations_crop if img_preprocess == "crop" else transformations_pad
    
    def __getitem__(self, key):
        print(f"[IndividualsDS] __getitem__ called for index: {key}")
        row = self.dataFrame.iloc[key]
        try:
            img = self.transformations(row['images'])
            print(f"[IndividualsDS] Image transformed for index: {key} (shape: {getattr(img, 'shape', 'unknown')})")
        except Exception as e:
            print(f"[IndividualsDS] ERROR transforming image at index {key}: {e}")
            raise
        try:
            label = tensor([row['labels_numeric']], dtype=long)
        except Exception as e:
            print(f"[IndividualsDS] ERROR creating label at index {key}: {e}")
            raise
        return {
            'images': img,
            'labels': label,
        }
    
    def __len__(self):
        l = len(self.dataFrame.index)
        print(f"[IndividualsDS] __len__ called: {l}")
        return l