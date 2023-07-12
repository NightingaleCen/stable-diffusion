import os
import numpy as np
from pathlib import Path
import PIL
from PIL import Image
from torch.utils.data import Dataset


class ArtworkDataset(Dataset):
    def __init__(
        self,
        data_root,
        watermark_path,
        file_exts=["jpg", "jpeg", "png"],
        size=None,
        interpolation="bicubic",
    ):  # TODO: now the watermark has the same size to the artwork,
        # maybe do it more flexible or combine them together
        if not isinstance(file_exts, list):
            file_exts = [file_exts]

        self.images = []
        data_path = Path(data_root)
        for ext in file_exts:
            assert isinstance(ext, str), f"{ext} is not a valid file extension."
            self.images.extend(data_path.glob(f"*.{ext}"))

        self._length = len(self.images)
        self.size = size
        self.interpolation = {
            "linear": PIL.Image.LINEAR,
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.watermark = self._read_watermark(watermark_path)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self._preprocess(image)

        file_name, extension = str(os.path.basename(image_path)).split(".")

        return {
            "image": image,
            "watermark": self.watermark,
            "file_name": file_name,
            "extension": extension,
        }

    def _read_watermark(self, watermark_path):
        watermark = Image.open(watermark_path)
        if not watermark.mode == "RGB":
            watermark = watermark.convert("RGB")
        return self._preprocess(watermark)

    def _preprocess(self, img):
        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image
