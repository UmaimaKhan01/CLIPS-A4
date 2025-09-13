# utils/har_data.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from .config import DATA_ROOT, IMAGE_SIZE

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])  # CLIP norm
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711])
        ])

def get_loaders(micro_batch, num_workers=0):
    train_ds = ImageFolder((DATA_ROOT / "train").as_posix(), transform=get_transforms(True))
    test_ds  = ImageFolder((DATA_ROOT / "test").as_posix(),  transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=micro_batch, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, train_ds.classes
