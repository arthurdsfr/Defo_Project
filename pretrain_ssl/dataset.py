from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from data.DeforestationDataset import AMAZON_RO, CustomDatasetClassification


class AmazonSSLDataset(Dataset):
    """
    SSL dataset over AMAZON_RO using existing patch extraction code.
    Returns multi-crop tensors for DINO-style training.
    """

    def __init__(
        self,
        data_path,
        transforms=None,
        input_patch_size=224,
        overlap_porcent=0.5,
        set_id=1,
        task="ssl_pretrain",
        porcent_pos_current_ref=0,
        debug=True,
        max_debug_samples=3,
    ):
        super().__init__()
        self.transforms = transforms
        self.debug = debug
        self.max_debug_samples = max_debug_samples
        self._debug_counter = 0

        args = SimpleNamespace(
            data_path=data_path,
            phase="train",
            input_patch_size=input_patch_size,
            overlap_porcent=overlap_porcent,
            task=task,
            porcent_pos_current_ref=porcent_pos_current_ref,
        )

        self.amazon = AMAZON_RO(args)
        classes = max(1, len(np.unique(self.amazon.reference)) - 1)
        self.patch_dataset = CustomDatasetClassification(self.amazon, classes, set_id, args)

        self.coordinates = self.patch_dataset.coordinates
        self.images_padded = self.patch_dataset.images_padded
        self.num_channels = self.patch_dataset.num_channels

        if self.debug:
            print("\n[SSL Dataset] AMAZON_RO loaded")
            print("[SSL Dataset] image_t1 shape:", self.amazon.image_t1.shape)
            print("[SSL Dataset] image_t2 shape:", self.amazon.image_t2.shape)
            print("[SSL Dataset] stacked padded shape:", self.images_padded.shape)
            print("[SSL Dataset] num channels:", self.num_channels)
            print("[SSL Dataset] num patches:", len(self.coordinates))
            print("[SSL Dataset] input_patch_size:", input_patch_size)
            if len(self.coordinates) > 0:
                print("[SSL Dataset] first patch coord [y0, x0, y1, x1]:", self.coordinates[0].astype(int).tolist())

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coords = self.coordinates[idx]
        y0, x0, y1, x1 = [int(v) for v in coords]
        patch_hwc = self.images_padded[y0:y1, x0:x1, :]
        patch_chw = torch.from_numpy(np.transpose(patch_hwc, (2, 0, 1))).to(torch.float32)

        crops = self.transforms(patch_chw) if self.transforms is not None else [patch_chw]

        if self.debug and self._debug_counter < self.max_debug_samples:
            print(f"\n[SSL Dataset][sample {self._debug_counter}] idx={idx}")
            print("[SSL Dataset] coord:", [y0, x0, y1, x1])
            print("[SSL Dataset] raw patch shape (CHW):", tuple(patch_chw.shape))
            print("[SSL Dataset] raw patch min/max:", float(patch_chw.min()), float(patch_chw.max()))
            print("[SSL Dataset] number of crops:", len(crops))
            print("[SSL Dataset] crop shapes:", [tuple(c.shape) for c in crops])
            self._debug_counter += 1

        return {
            "crops": crops,
            "index": idx,
            "coords": torch.tensor([y0, x0, y1, x1], dtype=torch.int32),
        }


def multicrop_collate_fn(batch):
    """
    Collate function for list-of-crops outputs.
    Produces a list where each item is a stacked batch for that crop index.
    """
    n_crops = len(batch[0]["crops"])
    crops = [torch.stack([item["crops"][i] for item in batch], dim=0) for i in range(n_crops)]
    indices = torch.as_tensor([item["index"] for item in batch], dtype=torch.long)
    coords = torch.stack([item["coords"] for item in batch], dim=0)

    return {
        "crops": crops,
        "index": indices,
        "coords": coords,
    }
