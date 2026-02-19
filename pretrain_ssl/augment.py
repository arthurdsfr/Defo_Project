import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class MultiCropAugmentation:
    """
    DINO-style multi-crop augmentation for multispectral tensors.
    Input expected as torch.Tensor in CHW format.
    """

    def __init__(
        self,
        global_crops_size=224,
        local_crops_size=96,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.1, 0.4),
        local_crops_number=6,
        horizontal_flip_p=0.5,
        vertical_flip_p=0.5,
        noise_std=0.01,
        debug=False,
    ):
        self.local_crops_number = local_crops_number
        self.noise_std = noise_std
        self.debug = debug
        self._printed_once = False

        self.global_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=global_crops_size,
                    scale=global_crops_scale,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=horizontal_flip_p),
                T.RandomVerticalFlip(p=vertical_flip_p),
            ]
        )

        self.local_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=local_crops_size,
                    scale=local_crops_scale,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=horizontal_flip_p),
                T.RandomVerticalFlip(p=vertical_flip_p),
            ]
        )

    def _to_chw_float(self, image):
        if isinstance(image, torch.Tensor):
            tensor = image
        else:
            tensor = torch.as_tensor(image)

        if tensor.ndim != 3:
            raise ValueError(f"Expected a 3D tensor/image, got shape {tuple(tensor.shape)}")

        # HWC -> CHW
        if tensor.shape[0] > tensor.shape[-1]:
            tensor = tensor.permute(2, 0, 1)

        return tensor.to(torch.float32)

    def _add_noise(self, tensor):
        if self.noise_std <= 0:
            return tensor
        return tensor + torch.randn_like(tensor) * self.noise_std

    def __call__(self, image):
        image = self._to_chw_float(image)

        crops = []
        global_1 = self._add_noise(self.global_transform(image))
        global_2 = self._add_noise(self.global_transform(image))
        crops.extend([global_1, global_2])

        for _ in range(self.local_crops_number):
            crops.append(self._add_noise(self.local_transform(image)))

        if self.debug and not self._printed_once:
            print("[SSL Augment] Input shape:", tuple(image.shape))
            print("[SSL Augment] Number of crops:", len(crops))
            print("[SSL Augment] Crop shapes:", [tuple(c.shape) for c in crops])
            self._printed_once = True

        return crops
