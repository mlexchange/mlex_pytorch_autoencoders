from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
import numpy as np
from PIL import Image
from tiled.client import from_uri
from torchvision import transforms

from datasets.base_dataset import BaseDataset, DetectorSource


class CustomTiledDataset(BaseDataset):
    def __init__(
        self,
        root_uri: str,
        sub_uris: list,
        target_size: tuple,
        augmentation: list,
        augm_invariant: Optional[bool] = False,
        log: Optional[bool] = False,
        data_tiled_api_key: Optional[str] = None,
        detector_uri: Optional[str] = None,
        detector_source: Optional[str] = DetectorSource.PYFAI.value,
        detector_tiled_api_key: Optional[str] = None,
    ):
        super().__init__(detector_uri, detector_source, detector_tiled_api_key)
        self.tiled_client = from_uri(
            root_uri, api_key=data_tiled_api_key, timeout=httpx.Timeout(120)
        )
        self.sub_uris = sub_uris
        self.data_augmentation = transforms.Compose(augmentation)
        self.augm_invariant = augm_invariant
        self.simple_transform = transforms.Compose(
            [transforms.Resize(target_size), transforms.ToTensor()]
        )
        self.log = log
        self.cum_sizes = []
        self._get_cumulative_sizes(sub_uris)

    def __len__(self):
        return self.cum_sizes[-1]

    def _get_cumulative_sizes(self, sub_uris):
        # Calculate cumulative sizes per sub_uri
        cum_size = 0
        for sub_uri in sub_uris:
            if len(self.tiled_client[sub_uri].shape) > 2:
                cum_size += len(self.tiled_client[sub_uri])
            else:
                cum_size += 1
            self.cum_sizes.append(cum_size)
        pass

    def _get_tiled_index(self, index):
        """Get the sub_uri and relative index within the list of sub_uris."""
        for count, cum_size in enumerate(self.cum_sizes):
            if index < cum_size:
                if count == 0:
                    return self.sub_uris[count], index
                else:
                    return self.sub_uris[count], index - self.cum_sizes[count - 1]
        return None

    def __getitem__(self, idx):
        sub_uri, index = self._get_tiled_index(idx)
        sub_uri_client = self.tiled_client[sub_uri]

        frame = sub_uri_client[index] if sub_uri_client.ndim > 2 else sub_uri_client
        if frame.ndim == 3:
            if frame.shape[0] == 3:
                # Move channel dimension to the last axis
                frame = np.moveaxis(frame, 0, -1)
            elif frame.shape[-1] == 1 or frame.shape[0] == 1:
                frame = np.squeeze(frame)

        # Apply log transform and/or percentile normalization
        frame = (
            self._apply_log_transform(frame)
            if self.log
            else self._normalize_percentiles(frame)
        )

        # Convert to PIL image and apply data augmentation
        image = Image.fromarray(frame).convert("L")
        tensor_image = self.data_augmentation(image)

        # Return according to augm_invariant flag
        if self.augm_invariant:
            return (tensor_image, self.simple_transform(image))
        else:
            return (tensor_image,)

    def __getitems__(self, index_list):
        with ThreadPoolExecutor() as executor:
            data = list(executor.map(self.__getitem__, index_list))
        return data
