import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter,ImageEnhance

import torchvision.transforms.functional as F

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}

class RandomCrop(object):
    def __init__(self, crop_size, fill=0):
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)

        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class RandomScaleCrop(object):
    def __init__(self, crop_size, mi, ma, fill=0):
        self.crop_size = crop_size
        self.fill = fill
        self.mi = mi
        self.ma = ma

    def __call__(self, sample):

        img = sample['image']
        mask = sample['label']

        # random scale (short edge)

        w, h = img.size

        base_size = min(h,w)

        short_size = random.randint(int(base_size * self.mi), int(base_size * self.ma))

        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size

        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)

        img = img.resize((ow, oh), Image.BILINEAR)
        if short_size < self.crop_size:
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        mask = mask.resize((ow, oh), Image.NEAREST)
        if short_size < self.crop_size:
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class CenterCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        img = F.center_crop(img, self.crop_size)
        mask = F.center_crop(mask, self.crop_size)

        return {'image': img,
                'label': mask}