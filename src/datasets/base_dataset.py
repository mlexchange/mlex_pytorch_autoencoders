from enum import Enum
from typing import Optional

import httpx
import numpy as np
import pyFAI
from tiled.client import from_uri
from torch.utils.data import Dataset


class DetectorSource(Enum):
    PYFAI = "pyFAI"
    TILED = "tiled"
    FILE = "file"


class BaseDataset(Dataset):
    def __init__(
        self,
        detector_uri: Optional[str] = None,
        detector_source: str = DetectorSource.PYFAI.value,
        detector_tiled_api_key: Optional[str] = None,
    ):
        """
        Base class for datasets that require detector masks.

        Args:
            detector_uri (str, optional): URI to the detector mask file.
            detector_source (str): Source of the detector mask.
                                   Options are 'pyFAI', 'tiled', and 'file'.
                                   Default is 'pyFAI'.
            detector_tiled_api_key (str, optional): API key for the detector mask, required if the source is 'tiled'.
        Raises:
            ValueError: If an invalid detector_source is provided.
        """

        if detector_source not in DetectorSource._value2member_map_:
            raise ValueError(
                f"Invalid detector_source: {detector_source}. "
                f"Valid options are {[s.value for s in DetectorSource]}"
            )

        self.detector_uri = detector_uri
        self.detector_source = detector_source
        self.detector_tiled_api_key = detector_tiled_api_key
        self.detector_tiled_client = None
        self.mask_det = None

        if detector_uri is not None:
            self.mask_det = self._load_detector_mask()

    def _load_detector_mask(self) -> Optional[np.ndarray]:
        """Loads the detector mask based on the source."""

        if self.detector_source == DetectorSource.PYFAI.value:
            detector = pyFAI.detector_factory(self.detector_uri)
            return detector.mask

        elif self.detector_source == DetectorSource.TILED.value:
            self.detector_tiled_client = from_uri(
                self.detector_uri,
                api_key=self.detector_tiled_api_key,
                timeout=httpx.Timeout(120),
            )
            return self.detector_tiled_client[:]

        elif self.detector_source == DetectorSource.FILE.value:
            return np.load(self.detector_uri)

        else:
            raise RuntimeError(f"Unexpected detector_source: {self.detector_source}")

    def _apply_log_transform(self, image, threshold=0.000000000001) -> np.ndarray:
        # Mask negative, NaN values, and detector mask
        nan_img = np.isnan(image)
        img_neg = image < 0.0
        mask_neg = np.array(img_neg)
        mask_nan = np.array(nan_img)
        if self.mask_det is not None:
            mask = mask_nan + mask_neg + self.mask_det
        else:
            mask = mask_nan + mask_neg
        x = np.ma.array(image, mask=mask)

        # Apply log transform
        image = np.log(x + threshold)
        x = np.ma.array(image, mask=mask)

        # Normalize to [0, 1] and scale to [0, 255]
        x = self._normalize_percentiles(x)
        return x

    @staticmethod
    def _normalize_percentiles(x, low_perc=0.01, high_perc=99) -> np.ndarray:
        """Normalize the input array to [0, 1] and scale to [0, 255]."""
        low = np.percentile(x.ravel(), low_perc)
        high = np.percentile(x.ravel(), high_perc)
        x = (np.clip((x - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        return x
