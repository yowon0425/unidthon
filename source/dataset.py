import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import CenterCrop, Resize

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class TrainScanImageDataset(Dataset):
    def __init__(self, noisy_image_dir_path, clean_image_dir_path, patch_size=128, transform=None):
        self.clean_image_file_paths = [os.path.join(noisy_image_dir_path, x) for x in os.listdir(noisy_image_dir_path)]
        self.noisy_image_file_paths = [os.path.join(clean_image_dir_path, x) for x in os.listdir(clean_image_dir_path)]
        self.transform = transform
        self.center_crop = CenterCrop(1024)
        self.resize = Resize((224, 224))
        self.patch_size = patch_size
        self.noisy_clean_pairs = self._create_noisy_clean_pairs()

    def _create_noisy_clean_pairs(self):
        clean_to_noisy = {}
        for clean_path in self.clean_image_file_paths:
            clean_id = '_'.join(os.path.basename(clean_path).split('_')[:-1])
            clean_to_noisy[clean_id] = clean_path
        
        noisy_clean_pairs = []
        for noisy_path in self.noisy_image_file_paths:
            noisy_id = '_'.join(os.path.basename(noisy_path).split('_')[:-1])
            if noisy_id in clean_to_noisy:
                clean_path = clean_to_noisy[noisy_id]
                noisy_clean_pairs.append((noisy_path, clean_path))
            else:
                pass
        
        return noisy_clean_pairs

    def __len__(self):
        return len(self.noisy_clean_pairs)

    def __getitem__(self, index):
        noisy_image_path, clean_image_path = self.noisy_clean_pairs[index]
        
        noisy_image = Image.open(noisy_image_path).convert("RGB")
        clean_image = Image.open(clean_image_path).convert("RGB")
        
        # Central Crop and Resize
        noisy_image = self.center_crop(noisy_image)
        clean_image = self.center_crop(clean_image)
        noisy_image = self.resize(noisy_image)
        clean_image = self.resize(clean_image)
        
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

class ValidScanImageDataset(Dataset):

    def __init__(self, noisy_image_dir_path, clean_image_dir_path, patch_size=128, transform=None):
        self.clean_image_file_paths = [os.path.join(noisy_image_dir_path, x) for x in os.listdir(noisy_image_dir_path)]
        self.noisy_image_file_paths = [os.path.join(clean_image_dir_path, x) for x in os.listdir(clean_image_dir_path)]
        self.transform = transform
        self.center_crop = CenterCrop(1024)
        self.resize = Resize((224, 224))
        self.patch_size = patch_size
        self.noisy_clean_pairs = self._create_noisy_clean_pairs()

    def _create_noisy_clean_pairs(self):
        clean_to_noisy = {}
        for clean_path in self.clean_image_paths:
            clean_id = '_'.join(os.path.basename(clean_path).split('_')[:-1])
            clean_to_noisy[clean_id] = clean_path
        
        noisy_clean_pairs = []
        for noisy_path in self.noisy_image_paths:
            noisy_id = '_'.join(os.path.basename(noisy_path).split('_')[:-1])
            if noisy_id in clean_to_noisy:
                clean_path = clean_to_noisy[noisy_id]
                noisy_clean_pairs.append((noisy_path, clean_path))
            else:
                pass
        
        return noisy_clean_pairs

    def __len__(self):
        return len(self.noisy_clean_pairs)

    def __getitem__(self, index):
        noisy_image_path, clean_image_path = self.noisy_clean_pairs[index]

        noisy_image = Image.open(noisy_image_path).convert("RGB")
        clean_image = Image.open(clean_image_path).convert("RGB")
        
        # Central Crop and Resize
        noisy_image = self.center_crop(noisy_image)
        clean_image = self.center_crop(clean_image)
        noisy_image = self.resize(noisy_image)
        clean_image = self.resize(clean_image)
        
        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        
        return noisy_image, clean_image

class ScanImageTestDataset(Dataset):
    def __init__(self, noisy_image_paths, transform=None):
        self.noisy_image_paths = [os.path.join(noisy_image_paths, x) for x in os.listdir(noisy_image_paths)]
        self.transform = transform

    def __len__(self):
        return len(self.noisy_image_paths)

    def __getitem__(self, index):
        noisy_image_path = self.noisy_image_paths[index]
        noisy_image = load_img(self.noisy_image_paths[index])

        # Convert numpy array to PIL image
        if isinstance(noisy_image, np.ndarray):
            noisy_image = Image.fromarray(noisy_image)

        if self.transform:
            noisy_image = self.transform(noisy_image)

        return noisy_image, noisy_image_path
