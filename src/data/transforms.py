from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ClipTransform:
    """Applies the same spatial transform to every frame in a clip."""

    def __init__(self, image_size: int, train: bool):
        self.image_size = image_size
        self.train = train
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __call__(self, frames: list[np.ndarray]) -> torch.Tensor:
        pil_frames = [Image.fromarray(f) for f in frames]
        if self.train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                pil_frames[0], scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            flip = random.random() < 0.5
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            out = []
            for img in pil_frames:
                img = transforms.functional.resized_crop(
                    img, i, j, h, w, (self.image_size, self.image_size)
                )
                if flip:
                    img = transforms.functional.hflip(img)
                img = transforms.functional.adjust_brightness(img, brightness)
                img = transforms.functional.adjust_contrast(img, contrast)
                tensor = transforms.functional.to_tensor(img)
                out.append(self.normalize(tensor))
            return torch.stack(out, dim=0)

        out = []
        resize = transforms.Resize(self.image_size + 32)
        crop = transforms.CenterCrop(self.image_size)
        for img in pil_frames:
            tensor = transforms.functional.to_tensor(crop(resize(img)))
            out.append(self.normalize(tensor))
        return torch.stack(out, dim=0)


def eval_transform(image_size: int) -> ClipTransform:
    return ClipTransform(image_size=image_size, train=False)
