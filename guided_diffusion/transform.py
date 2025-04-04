import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class TransformDataset(Dataset):

    def __init__(self, dataset: Dataset, image_size: int):
        self.dataset = dataset
        self.transform = A.Compose(transforms=[A.Resize(image_size, image_size),
                                            A.HorizontalFlip(p=0.5),
                                            ToTensorV2()])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        mask, image = self.dataset[idx]
        if isinstance(mask, list):
            transformed = self.transform(image=image, masks=mask)
            mask = [m.unsqueeze(0).to(torch.float32) for m in transformed["masks"]]
        else:
            transformed = self.transform(image=image, mask=mask)
            mask = transformed["mask"].unsqueeze(0).to(torch.float32)
        return mask, transformed["image"].to(torch.float32)