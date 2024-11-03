from torchvision.transforms import Compose, Normalize, ToTensor, ColorJitter
from torchvision.transforms.v2 import AugMix, GaussianBlur, RandomErasing
import random
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

#torchvision.transforms.RandomHorizontalFlip(p=0.5)
#torchvision.transforms.RandomVerticalFlip(p=0.5)

class DirectionalBlur:
    def __init__(self, kernel_size=15, directions=['horizontal', 'vertical'], probability=0.5):
        self.kernel_size = kernel_size
        self.directions = directions
        self.probability = probability

    def __call__(self, image):
        if random.random() < self.probability:
            direction = random.choice(self.directions)
            
            # Convert PIL Image to NumPy array
            if isinstance(image, Image.Image):
                image = np.array(image)
            elif not isinstance(image, np.ndarray):
                raise ValueError(f"Expected PIL Image or NumPy array, got {type(image)}")

            # Ensure the image is 3-channel (RGB)
            if image.ndim == 3 and image.shape[2] == 3:
                if direction == 'horizontal':
                    kernel = np.zeros((1, self.kernel_size))
                    kernel[0, :] = 1 / self.kernel_size
                elif direction == 'vertical':
                    kernel = np.zeros((self.kernel_size, 1))
                    kernel[:, 0] = 1 / self.kernel_size

                # Apply the kernel to the image
                blurred_image = cv2.filter2D(image, -1, kernel)
                # Convert the NumPy array back to PIL Image
                return Image.fromarray(blurred_image)
            else:
                raise ValueError("Input image must be a 3-channel (RGB) image.")
        
        # If probability condition is not met, return the original image
        return image

'''
def get_transform():
    return Compose([
        ToTensor(),
        GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        ColorJitter(brightness =.3, contrast = .5, saturation = .3),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        RandomErasing()
    ])
'''
def get_transform():
    transform_1 = transforms.Compose([
        DirectionalBlur(kernel_size=10, directions=['horizontal', 'vertical'], probability=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            transforms.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.3),
        ])
    ])
    transform_3 = transforms.Compose([
        transforms.ToTensor(),
        AugMix(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Apply either transform_1 or transform_2 with a 50% chance for each
    return transforms.RandomChoice([transform_1, transform_3])


def get_test_transform():
    return Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
