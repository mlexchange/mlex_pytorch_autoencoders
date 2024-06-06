import numpy as np
import pyFAI
import torch.nn.functional as F
from PIL import Image
from tiled.client import from_uri
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDirectoryDataset(Dataset):
    def __init__(self, data, target_size, augmentation, augm_invariant, log):
        augmentation.insert(0, transforms.Resize(target_size))
        self.data_augmentation = transforms.Compose(augmentation)
        self.data = data
        self.augm_invariant = augm_invariant
        self.simple_transform = transforms.Compose(
            [transforms.Resize(target_size), transforms.ToTensor()]
        )
        self.log = log

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert("L")
        if self.log:
            image = np.log1p(np.array(image))
            image = (
                ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255
            ).astype(np.uint8)
            image = Image.fromarray(image)
        tensor_image = self.data_augmentation(image)
        if not self.augm_invariant:
            return (tensor_image, tensor_image)
        else:
            return (tensor_image, self.simple_transform(image))


class InterpolateTransform:
    def __init__(self, output_size, mode="bilinear"):
        """
        output_size: Desired output size (height, width)
        mode: Interpolation mode to use ('nearest', 'linear', 'bilinear', etc.)
        """
        self.output_size = output_size
        self.mode = mode

    def __call__(self, data):
        data = data.unsqueeze(0)
        resized_data = F.interpolate(
            data,
            size=self.output_size,
            mode=self.mode,
            align_corners=None if self.mode in ["nearest", "area"] else False,
        )
        resized_data = resized_data.squeeze(0)
        return resized_data


class CustomTiledDataset(Dataset):
    def __init__(
        self,
        sub_uri_list,
        root_uri,
        target_size,
        log=False,
        api_key=None,
        detector_name=None,
    ):
        # Connect to Tiled
        self.tiled_client = from_uri(root_uri, api_key=api_key)

        # Get size of each sub_tiled_uri and prep tiled_data
        # tiled_data: List of tuples (sub_tiled_uri, cum_data_size)
        tiled_data = []
        cum_data_size = 0
        for uri in sub_uri_list:
            dataset = self.tiled_client[uri]
            if len(dataset.shape) == 4 or len(dataset.shape) == 3:
                cum_data_size += len(dataset)
            else:
                cum_data_size += 1
            tiled_data.append((uri, cum_data_size))

        self.cum_data_size = cum_data_size  # Total size of dataset
        self.tiled_data = tiled_data  # List of tuples (uri, cum_data_size)
        self.simple_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                InterpolateTransform(target_size),
            ]
        )
        self.log = log  # Apply log transformation

        # If detector name is provided, get detector mask
        if detector_name is not None:
            detector = pyFAI.detector_factory(detector_name)
            self.mask_det = detector.mask
        else:
            self.mask_det = None
        pass

    def __len__(self):
        return self.cum_data_size

    def _mask_image(self, contents):
        # Mask negative values and NaNs
        nan_img = np.isnan(contents)
        img_neg = contents < 0.0
        mask_neg = np.array(img_neg)
        mask_nan = np.array(nan_img)

        # Add detector mask if it was previously defined
        if self.mask_det is not None:
            mask = mask_nan + mask_neg + self.mask_det
        else:
            mask = mask_nan + mask_neg

        # Define masked array
        x = np.ma.array(contents, mask=mask)

        # Normalize according to percentiles 1-99
        low = np.percentile(x.ravel(), 1)
        high = np.percentile(x.ravel(), 99)
        x = np.clip((x - low) / (high - low), 0, 1)

        # Apply log transformation
        if self.log:
            x = np.log(x + 0.000000000001)
            # Normalize to 0-1
            low = np.min(x)
            high = np.max(x)
            x = np.clip((x - low) / (high - low), 0, 1)

        # Apply mask to data and return as float32
        mask = 1 - (mask > 0)
        res = x.data * mask
        return res.astype(np.float32)

    def __getitem__(self, idx):
        # Find sub_tiled_uri and relative index
        prev_cum_data_size = 0
        cum_data_size = 0
        for tiled_uri, cum_data_size in self.tiled_data:
            if idx < cum_data_size:
                # TODO: Replace with logging
                print(
                    f"Loading {tiled_uri} with {idx} from {prev_cum_data_size} to {cum_data_size}"
                )
                tiled_data = self.tiled_client[tiled_uri]
                if len(tiled_data.shape) == 4 or len(tiled_data.shape) == 3:
                    contents = tiled_data[prev_cum_data_size - idx - 1]
                else:
                    contents = tiled_data
                    contents = np.expand_dims(contents, axis=0)
                break
            prev_cum_data_size = cum_data_size

        # If contents is not uint8, apply mask (and log transformation if log=True)
        if contents.dtype != np.uint8:
            contents = self._mask_image(np.squeeze(contents))

        # Otherwise, apply percentile normalization (and log transformation if log=True)
        else:
            # Normalize according to percentiles 1-99
            low = np.percentile(contents.ravel(), 1)
            high = np.percentile(contents.ravel(), 99)
            contents = np.clip((contents - low) / (high - low), 0, 1)
            contents = (contents * 255).astype(np.uint8)  # Convert to uint8, 0-255

            # Apply log transformation
            if self.log:
                contents = np.log1p(np.array(contents))  # Apply log(1+x)

                # Normalize to 0-255
                contents = (
                    (
                        (contents - np.min(contents))
                        / (np.max(contents) - np.min(contents))
                    )
                    * 255
                ).astype(np.uint8)

            # Convert to grayscale
            contents = Image.fromarray(contents).convert("L")

        # Apply simple transform and return
        tensor_image = self.simple_transform(contents)
        return (tensor_image, tensor_image)
