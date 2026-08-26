import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PRIZM_TARGET_SIZE = (300, 300)
DEEPLAB_COMPAT_SIZE = (304, 304)

try:
    PIL_BICUBIC = Image.Resampling.BICUBIC
    PIL_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover
    PIL_BICUBIC = Image.BICUBIC
    PIL_NEAREST = Image.NEAREST


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VALID_IMAGE_EXTS


def _strip_simple_segmentation_token(stem: str) -> str:
    token = "_Simple Segmentation"
    if token in stem:
        return stem.replace(token, "")
    return stem


def _image_to_rgb_uint8(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_rgb_u8_from_path(image_path: str) -> np.ndarray:
    with Image.open(image_path) as img:
        return _image_to_rgb_uint8(img)


def _load_mask_u8_from_path(mask_path: str) -> np.ndarray:
    with Image.open(mask_path) as mask_img:
        return np.asarray(mask_img, dtype=np.uint8)


def _jitter_color_hsv(
    rgb_u8: np.ndarray,
    contrast: float = 0.5,
    saturation: float = 0.5,
    brightness: float = 0.5,
) -> np.ndarray:
    """
    Apply the PRIZM HSV augmentation:
    - random saturation offset in [-saturation, saturation]
    - random brightness offset in [-brightness, brightness]
    - random contrast scale in [1-contrast, 1+contrast]
    applied on the HSV value/saturation channels.
    """
    rgb = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    sat_offset = float(np.random.uniform(-saturation, saturation))
    bright_offset = float(np.random.uniform(-brightness, brightness))
    contrast_scale = float(np.random.uniform(1.0 - contrast, 1.0 + contrast))

    hsv[..., 1] = np.clip(hsv[..., 1] + sat_offset, 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * contrast_scale + bright_offset, 0.0, 1.0)

    rgb_jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.clip(np.round(rgb_jittered * 255.0), 0.0, 255.0).astype(np.uint8)


def _rgb_to_gray_uint8(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_u8, dtype=np.float32)
    gray = (0.2989 * rgb[..., 0]) + (0.5870 * rgb[..., 1]) + (0.1140 * rgb[..., 2])
    return np.clip(np.round(gray), 0.0, 255.0).astype(np.uint8)


def _contrast_limits_uint8(gray_u8: np.ndarray, tol=(0.01, 0.99)) -> Tuple[float, float]:
    gray = np.asarray(gray_u8, dtype=np.uint8)
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.int64)
    cdf = np.cumsum(hist)
    total = int(cdf[-1]) if cdf.size else 0
    if total <= 0:
        return 0.0, 1.0

    low_target = float(tol[0]) * float(total)
    high_target = float(tol[1]) * float(total)
    low_idx = int(np.searchsorted(cdf, low_target, side="left"))
    high_idx = int(np.searchsorted(cdf, high_target, side="left"))
    low_idx = int(np.clip(low_idx, 0, 255))
    high_idx = int(np.clip(high_idx, 0, 255))
    if high_idx <= low_idx:
        return 0.0, 1.0
    return low_idx / 255.0, high_idx / 255.0


def _rescale_intensity_uint8(gray_u8: np.ndarray, in_range: Tuple[float, float]) -> np.ndarray:
    low, high = float(in_range[0]), float(in_range[1])
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high):
        high = 1.0
    if high <= low:
        low, high = 0.0, 1.0

    gray = np.asarray(gray_u8, dtype=np.float32) / 255.0
    out = np.clip(gray, low, high)
    out = (out - low) / max(high - low, 1e-6)
    return np.clip(np.round(out * 255.0), 0.0, 255.0).astype(np.uint8)


def _resize_gray_uint8(gray_u8: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(gray_u8, dtype=np.uint8)).resize(size, resample=PIL_BICUBIC),
        dtype=np.uint8,
    )


def _pad_image_to_deeplab_compat(image_chw: torch.Tensor) -> torch.Tensor:
    _, height, width = image_chw.shape
    if (height, width) == DEEPLAB_COMPAT_SIZE:
        return image_chw

    pad_h = max(DEEPLAB_COMPAT_SIZE[0] - height, 0)
    pad_w = max(DEEPLAB_COMPAT_SIZE[1] - width, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return F.pad(image_chw.unsqueeze(0), (left, right, top, bottom), mode="replicate").squeeze(0)


def _pad_one_hot_mask_to_deeplab_compat(mask_chw: torch.Tensor) -> torch.Tensor:
    _, height, width = mask_chw.shape
    if (height, width) == DEEPLAB_COMPAT_SIZE:
        return mask_chw

    pad_h = max(DEEPLAB_COMPAT_SIZE[0] - height, 0)
    pad_w = max(DEEPLAB_COMPAT_SIZE[1] - width, 0)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = torch.zeros(
        (mask_chw.shape[0], height + pad_h, width + pad_w),
        dtype=mask_chw.dtype,
    )
    padded[0].fill_(1.0)
    padded[:, top : top + height, left : left + width] = mask_chw
    return padded


def _preprocess_gray_uint8_from_rgb(
    rgb_u8: np.ndarray,
    apply_random_transform: bool = True,
) -> np.ndarray:
    """
    Convert RGB to grayscale, apply percentile-based contrast adjustment,
    resize to 300x300, and replicate grayscale to three channels.

    DeepLab compatibility padding to 304x304 happens afterward so the
    augmentation semantics remain 300x300.
    """
    if apply_random_transform:
        rgb_u8 = _jitter_color_hsv(
            rgb_u8,
            contrast=0.5,
            saturation=0.5,
            brightness=0.5,
        )

    gray_u8 = _rgb_to_gray_uint8(rgb_u8)
    low, high = _contrast_limits_uint8(gray_u8)
    if apply_random_transform:
        low *= 0.5 + 0.5 * float(np.random.rand())
        high *= 0.7 + 0.3 * float(np.random.rand())
    gray_u8 = _rescale_intensity_uint8(gray_u8, (low, high))
    gray_u8 = _resize_gray_uint8(gray_u8, PRIZM_TARGET_SIZE)

    return gray_u8


def _preprocess_gray_uint8(
    image: Image.Image,
    apply_random_transform: bool = True,
) -> np.ndarray:
    rgb_u8 = _image_to_rgb_uint8(image)
    return _preprocess_gray_uint8_from_rgb(
        rgb_u8,
        apply_random_transform=apply_random_transform,
    )


def _gray_u8_to_padded_image_tensor(gray_u8: np.ndarray) -> torch.Tensor:
    gray = torch.from_numpy(np.asarray(gray_u8, dtype=np.float32) / 255.0).unsqueeze(0)
    image_chw = gray.repeat(3, 1, 1)
    return _pad_image_to_deeplab_compat(image_chw)


def _preprocess_training_image(
    image: Image.Image,
    apply_random_transform: bool = True,
) -> torch.Tensor:
    gray_u8 = _preprocess_gray_uint8(
        image,
        apply_random_transform=apply_random_transform,
    )
    return _gray_u8_to_padded_image_tensor(gray_u8)


def _mask_to_one_hot(mask_u8: np.ndarray) -> torch.Tensor:
    """Convert raw label IDs (0,2,4) into one-hot [3,H,W] float."""
    mask_u8 = np.asarray(mask_u8, dtype=np.uint8)
    one_hot = np.zeros((3, *mask_u8.shape), dtype=np.float32)
    one_hot[0] = (mask_u8 == 0).astype(np.float32)
    one_hot[1] = (mask_u8 == 2).astype(np.float32)
    one_hot[2] = (mask_u8 == 4).astype(np.float32)
    return torch.from_numpy(one_hot)


def _resize_mask_u8(mask_u8: np.ndarray) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(mask_u8, dtype=np.uint8)).resize(
            PRIZM_TARGET_SIZE, resample=PIL_NEAREST
        ),
        dtype=np.uint8,
    )


def _safe_train_test_split(indices: np.ndarray, labels: Sequence[str], test_size: float, seed: int):
    labels_arr = np.asarray(labels, dtype=object)
    unique, counts = np.unique(labels_arr, return_counts=True)
    stratify_ok = unique.size > 1 and np.all(counts >= 2)
    stratify = labels_arr if stratify_ok else None
    return train_test_split(indices, test_size=test_size, random_state=seed, stratify=stratify)


def discover_segmentation_pairs(dataset_dir: str) -> List[Tuple[str, str, str]]:
    """
    Discover paired images and segmentation masks for PRIZM training.

    Returns list of (image_path, mask_path, folder_label).
    """
    root = Path(dataset_dir)
    search_dir = root / "train" if (root / "train").is_dir() else root

    files = [p for p in search_dir.rglob("*") if p.is_file() and _is_image_file(str(p))]
    images = [p for p in files if "_Simple Segmentation" not in p.name]
    masks = [p for p in files if "_Simple Segmentation" in p.name]

    mask_map: Dict[str, Path] = {}
    for mp in masks:
        k = _strip_simple_segmentation_token(mp.stem)
        if k not in mask_map:
            mask_map[k] = mp

    pairs: List[Tuple[str, str, str]] = []
    for ip in sorted(images):
        k = ip.stem
        if k in mask_map:
            folder_label = ip.parent.name
            pairs.append((str(ip), str(mask_map[k]), folder_label))

    return pairs


def split_segmentation_pairs(
    pairs: Sequence[Tuple[str, str, str]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[str, str, str]]]:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total:.6f}")
    if len(pairs) < 3:
        raise ValueError("Need at least 3 paired samples for train/val/test split.")

    indices = np.arange(len(pairs))
    labels = [pairs[i][2] for i in indices]

    train_idx, hold_idx = _safe_train_test_split(
        indices,
        labels,
        test_size=(1.0 - train_ratio),
        seed=seed,
    )

    hold_labels = [pairs[i][2] for i in hold_idx]
    hold_test_frac = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = _safe_train_test_split(
        hold_idx,
        hold_labels,
        test_size=hold_test_frac,
        seed=seed,
    )

    def _subset(idxs):
        return [pairs[int(i)] for i in sorted(idxs)]

    return {
        "train": _subset(train_idx),
        "val": _subset(val_idx),
        "test": _subset(test_idx),
    }


class PRIZMSegmentationDataset(Dataset):
    """PRIZM training dataset with joint image and mask preprocessing."""

    def __init__(
        self,
        pairs: Sequence[Tuple[str, str, str]],
        apply_random_transform: bool = True,
        random_rot90: bool = False,
        preload_to_memory: bool = True,
    ):
        self.pairs = list(pairs)
        self.apply_random_transform = bool(apply_random_transform)
        self.random_rot90 = bool(random_rot90)
        self.preload_to_memory = bool(preload_to_memory)
        self.cached_sources: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        self.cached_tensors: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        if self.preload_to_memory:
            self._preload()

    def __len__(self) -> int:
        return len(self.pairs)

    def _preload(self) -> None:
        if not self.pairs:
            self.cached_sources = []
            self.cached_tensors = []
            return

        if self.apply_random_transform:
            cached_sources: List[Tuple[np.ndarray, np.ndarray]] = []
            for image_path, mask_path, _ in tqdm(
                self.pairs,
                desc="Preloading train pairs",
                dynamic_ncols=True,
                leave=False,
            ):
                rgb_u8 = _load_rgb_u8_from_path(image_path)
                mask_u8 = _load_mask_u8_from_path(mask_path)
                cached_sources.append((rgb_u8, mask_u8))
            self.cached_sources = cached_sources
        else:
            cached_tensors: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for image_path, mask_path, _ in tqdm(
                self.pairs,
                desc="Preloading eval pairs",
                dynamic_ncols=True,
                leave=False,
            ):
                rgb_u8 = _load_rgb_u8_from_path(image_path)
                mask_u8 = _load_mask_u8_from_path(mask_path)
                gray_u8 = _preprocess_gray_uint8_from_rgb(
                    rgb_u8,
                    apply_random_transform=False,
                )
                mask_np = _resize_mask_u8(mask_u8)
                image = _gray_u8_to_padded_image_tensor(gray_u8)
                mask = _pad_one_hot_mask_to_deeplab_compat(_mask_to_one_hot(mask_np))
                cached_tensors.append((image.float(), mask.float()))
            self.cached_tensors = cached_tensors

    def __getitem__(self, idx: int):
        if self.cached_tensors is not None:
            return self.cached_tensors[idx]

        if self.cached_sources is not None:
            rgb_u8, mask_u8 = self.cached_sources[idx]
        else:
            image_path, mask_path, _ = self.pairs[idx]
            rgb_u8 = _load_rgb_u8_from_path(image_path)
            mask_u8 = _load_mask_u8_from_path(mask_path)

        gray_u8 = _preprocess_gray_uint8_from_rgb(
            rgb_u8,
            apply_random_transform=self.apply_random_transform,
        )
        mask_np = _resize_mask_u8(mask_u8)

        if self.apply_random_transform and self.random_rot90:
            k = int(np.random.randint(0, 4))
            if k:
                gray_u8 = np.ascontiguousarray(np.rot90(gray_u8, k=k))
                mask_np = np.ascontiguousarray(np.rot90(mask_np, k=k))

        image = _gray_u8_to_padded_image_tensor(gray_u8)
        mask = _pad_one_hot_mask_to_deeplab_compat(_mask_to_one_hot(mask_np))

        return image.float(), mask.float()


# Define the dataset class for image and mask pairing
class PRIZM_Dataset(Dataset):
    def __init__(self, args, mode):
        self.base_dir = args.dataset_dir
        self.mode = mode
        print(f"Loading {mode} data from {self.base_dir}")
        # Get all files in the directory
        if mode == "train" or mode == "val":
            all_files = glob.glob(os.path.join(self.base_dir, "train", "*"))
        else:
            all_files = glob.glob(os.path.join(self.base_dir, mode, "*"))

        # Separate the segmentation labels from the images
        images = [
            file
            for file in all_files
            if "_Simple Segmentation" not in file and _is_image_file(file)
        ]
        print(f"Found {len(images)} images")
        images.sort()
        
        if mode == "train" or mode == "val":
            labels = [
                file
                for file in all_files
                if "_Simple Segmentation" in file and _is_image_file(file)
            ]
            print(f"Found {len(labels)} masks")
            # Check that the number of images and masks match
            assert len(images) == len(labels), "Number of images and masks must match"
            labels.sort()
        
        if args.augmentation == False:
            transform = transforms.Compose([
                transforms.Resize((304, 304)),  # Resize image to 300x300
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()  # Convert image to a tensor
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((304, 304)),  # Resize image to 300x300
                
                transforms.Grayscale(num_output_channels=1),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Apply color jitter
                
                transforms.ToTensor()  # Convert image to a tensor
            ])

        # Load the images and masks
        self.data = []
        for i in tqdm(range(len(images)), desc=f"Loading {mode} data"):
            image = Image.open(images[i])
            image = transform(image)

            if mode == "train":
                label = Image.open(labels[i])
                label = torch.tensor(np.array(label), dtype=torch.long)

                if args.model == 'deeplabv3plus':
                    # resize the mask to 304x304
                    label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=(304, 304), mode='nearest').long().squeeze(0).squeeze(0)
                label = self.convert_label(label)

                self.data.append((image, label))
            elif mode == "val":
                label = Image.open(labels[i])
                label = torch.tensor(np.array(label), dtype=torch.long)

                if args.model == 'deeplabv3plus':
                    # resize the mask to 304x304
                    label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=(304, 304), mode='nearest').long().squeeze(0).squeeze(0)
                label = self.convert_label(label)

                self.data.append((image, label, images[i]))
            else:
                self.data.append((image, []))

    def convert_label(self, label):
        unique_labels = torch.tensor([0, 2, 4])
        num_classes = unique_labels.numel()  # Count of unique labels
        one_hot = torch.zeros((num_classes, *label.shape), dtype=torch.long)
    
        for idx, class_label in enumerate(unique_labels):
            one_hot[idx] = (label == class_label).float()
        
        return one_hot
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        if self.mode == "train" or self.mode == "test":
            image, mask = self.data[idx]
            return image, mask
        else:
            image, mask, image_path = self.data[idx]
            return image, mask, image_path
