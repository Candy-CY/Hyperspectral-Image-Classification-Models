#Test augmentation
import numpy as np
from src import augmentation
import torch

def test_train_augmentation():
    image = torch.randn(20, 369, 11, 11)    
    transformer = augmentation.train_augmentation(image_size=11)
    transformed_image = transformer(image)
    assert transformed_image.shape == image.shape
    assert not np.array_equal(image, transformed_image)
