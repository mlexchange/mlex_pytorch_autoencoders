from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import tifffile
from PIL import Image
from torchvision import transforms

from datasets.base_dataset import BaseDataset, DetectorSource


class CustomDirectoryDataset(BaseDataset):
    def __init__(
        self,
        data: list,
        target_size: tuple,
        augmentation: list,
        augm_invariant: Optional[bool] = False,
        log: Optional[bool] = False,
        detector_uri: Optional[str] = None,
        detector_source: Optional[str] = DetectorSource.PYFAI.value,
        detector_tiled_api_key: Optional[str] = None,
    ):
        super().__init__(detector_uri, detector_source, detector_tiled_api_key)
        self.data = data
        augmentation.insert(0, transforms.Resize(target_size))
        self.data_augmentation = transforms.Compose(augmentation)
        self.augm_invariant = augm_invariant
        self.simple_transform = transforms.Compose(
            [transforms.Resize(target_size), transforms.ToTensor()]
        )
        self.log = log

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data[idx]
        is_tiff = file_path.endswith(".tif") or file_path.endswith(".tiff")

        # Handle TIFF files that are not 8-bit
        if is_tiff and self._get_dtype(file_path) != np.uint8:
            image = tifffile.imread(file_path)

            if image.ndim == 3:
                # If channels-first [3, H, W], transpose to [H, W, 3]
                if image.shape[0] == 3:
                    image = np.moveaxis(image, 0, -1)
                # If there's a singleton dimension either at [H, W, 1] or [1, H, W], squeeze it out
                elif image.shape[-1] == 1 or image.shape[0] == 1:
                    image = np.squeeze(image)

            # Apply log transform or percentile normalization
            image = (
                self._apply_log_transform(image)
                if self.log
                else self._normalize_percentiles(image)
            )
            image = Image.fromarray(image)

        else:
            # Non-TIFF or 8-bit TIFF => read with PIL
            # Convert to single-channel (L)
            image = Image.open(file_path).convert("L")
            if self.log:
                arr = np.array(image)  # convert to np for transformations
                arr = self._apply_log_transform(arr)
                image = Image.fromarray(arr)

        tensor_image = self.data_augmentation(image)

        # Return either one or two items in the tuple
        if self.augm_invariant:
            return (tensor_image, self.simple_transform(image))
        else:
            return (tensor_image,)

    def _get_dtype(self, file_path):
        """Get the data type of the first page of a TIFF file."""
        with tifffile.TiffFile(file_path) as tif:
            dtype = tif.pages[0].dtype
        return dtype

    def __getitems__(self, index_list):
        with ThreadPoolExecutor() as executor:
            data = list(executor.map(self.__getitem__, index_list))
        return data
