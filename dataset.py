import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from torch.utils.data import Dataset


IMAGE_SUFFIXES = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def generate_boundary_label(mask, width=2, mode="fixed", adaptive_alpha=0.25, adaptive_max_width=2):
    """Generate an in-mask boundary band, matching the paper implementation."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().float().numpy()
    foreground = np.asarray(mask, dtype=np.float32) > 0.5
    if not foreground.any():
        return np.zeros(foreground.shape, dtype=np.float32)

    mode = str(mode).lower()
    if mode == "fixed":
        distance = distance_transform_edt(foreground)
        return ((distance <= int(width)) & foreground).astype(np.float32)
    if mode != "adaptive":
        raise ValueError(f"unsupported boundary mode: {mode}")

    components, count = label(foreground)
    boundary = np.zeros(foreground.shape, dtype=np.float32)
    for component_id in range(1, count + 1):
        instance = components == component_id
        area = int(instance.sum())
        instance_width = min(
            int(adaptive_max_width),
            max(1, int(round(float(adaptive_alpha) * math.sqrt(area)))),
        )
        distance = distance_transform_edt(instance)
        boundary[(distance <= instance_width) & instance] = 1.0
    return boundary


def pad_to_multiple(array, multiple=32):
    height, width = array.shape
    target_height = math.ceil(height / multiple) * multiple
    target_width = math.ceil(width / multiple) * multiple
    return np.pad(array, ((0, target_height - height), (0, target_width - width)), mode="constant")


def _resolve_file(directory, sample_id):
    candidate = directory / sample_id
    if candidate.is_file():
        return candidate
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{sample_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"cannot find sample '{sample_id}' under {directory}")


def _random_crop(image, mask, patch_size, positive_probability):
    height, width = image.shape
    if min(height, width) < patch_size:
        target_height = max(height, patch_size)
        target_width = max(width, patch_size)
        image = np.pad(image, ((0, target_height - height), (0, target_width - width)), mode="constant")
        mask = np.pad(mask, ((0, target_height - height), (0, target_width - width)), mode="constant")
        height, width = image.shape

    use_positive = mask.max() > 0 and random.random() <= positive_probability
    if use_positive:
        rows, columns = np.where(mask > 0)
        index = random.randrange(len(rows))
        row_start = random.randint(max(0, rows[index] - patch_size), min(rows[index], height - patch_size))
        column_start = random.randint(max(0, columns[index] - patch_size), min(columns[index], width - patch_size))
    else:
        row_start = random.randint(0, height - patch_size)
        column_start = random.randint(0, width - patch_size)
    return (
        image[row_start:row_start + patch_size, column_start:column_start + patch_size],
        mask[row_start:row_start + patch_size, column_start:column_start + patch_size],
    )


def _augment(image, mask):
    if random.random() < 0.5:
        image, mask = image[::-1, :], mask[::-1, :]
    if random.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if random.random() < 0.5:
        image, mask = image.T, mask.T
    return image, mask


class IRSTDDataset(Dataset):
    """Dataset for the BasicIRSTD images/masks/img_idx directory convention."""

    def __init__(
        self,
        root,
        index_file,
        mean,
        std,
        training=False,
        patch_size=256,
        positive_probability=0.5,
        boundary_width=2,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        with open(index_file, "r", encoding="utf-8") as stream:
            self.sample_ids = [line.strip() for line in stream if line.strip()]
        self.mean = float(mean)
        self.std = float(std)
        self.training = bool(training)
        self.patch_size = int(patch_size)
        self.positive_probability = float(positive_probability)
        self.boundary_width = int(boundary_width)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        image = np.asarray(Image.open(_resolve_file(self.image_dir, sample_id)).convert("I"), dtype=np.float32)
        mask = np.asarray(Image.open(_resolve_file(self.mask_dir, sample_id)), dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        # Preserve the reference loader's normalized mask values. Some
        # datasets contain antialiased boundary pixels between 0 and 1.
        mask = (mask / 255.0).astype(np.float32)
        image = (image - self.mean) / self.std
        original_size = image.shape

        if self.training:
            image, mask = _random_crop(image, mask, self.patch_size, self.positive_probability)
            image, mask = _augment(image, mask)
            boundary = generate_boundary_label(mask, width=self.boundary_width)
        else:
            image = pad_to_multiple(image)
            mask = pad_to_multiple(mask)
            boundary = np.zeros_like(mask, dtype=np.float32)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(image[None])).float(),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])).float(),
            "boundary": torch.from_numpy(np.ascontiguousarray(boundary[None])).float(),
            "size": torch.tensor(original_size, dtype=torch.int64),
            "id": sample_id,
        }
