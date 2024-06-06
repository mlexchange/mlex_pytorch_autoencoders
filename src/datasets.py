from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import tifffile
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
        file_path = self.data[idx]
        is_tiff = file_path.endswith(".tif") or file_path.endswith(".tiff")

        if is_tiff and self._get_dtype(file_path) != np.uint8:
            image = tifffile.imread(file_path)
            image = np.array(image)
            if len(image.shape) == 3:
                if image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                elif image.shape[2] == 1 or image.shape[0] == 1:
                    image = np.squeeze(image)
            if self.log:
                image = self._apply_log_transform(image)
            else:
                image = self._normalize_percentiles(image)
            image = Image.fromarray(image)
        else:
            image = Image.open(file_path).convert("L")
            if self.log:
                image = self._apply_log_transform(np.array(image))
                image = Image.fromarray(image)

        tensor_image = self.data_augmentation(image)
        if not self.augm_invariant:
            return (tensor_image,)
        else:
            return (tensor_image, self.simple_transform(image))

    def _get_dtype(self, file_path):
        with tifffile.TiffFile(file_path) as tif:
            dtype = tif.pages[0].dtype
        return dtype

    def _apply_log_transform(self, image, threshold=0.000000000001):
        # Mask negative and NaN values
        nan_img = np.isnan(image)
        img_neg = image < 0.0
        mask_neg = np.array(img_neg)
        mask_nan = np.array(nan_img)
        mask = mask_nan + mask_neg
        x = np.ma.array(image, mask=mask)

        image = np.log(x + threshold)
        x = np.ma.array(image, mask=mask)
        x = self._normalize_percentiles(x)
        return x

    def _normalize_percentiles(self, x, low_perc=0.01, high_perc=99):
        low = np.percentile(x.ravel(), low_perc)
        high = np.percentile(x.ravel(), high_perc)
        x = (np.clip((x - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        return x

    def __getitems__(self, index_list):
        with ThreadPoolExecutor() as executor:
            data = list(executor.map(self.__getitem__, index_list))
        return data


class CustomTiledDataset(Dataset):
    def __init__(
        self,
        root_uri,
        sub_uris,
        target_size,
        log,
        api_key=None,
    ):
        self.sub_uris = sub_uris
        self.simple_transform = transforms.Compose(
            [transforms.Resize(target_size), transforms.ToTensor()]
        )
        self.log = log
        self.tiled_client = from_uri(
            root_uri, api_key=api_key, timeout=httpx.Timeout(120)
        )
        self.cum_sizes = []
        cum_size = 0
        for sub_uri in sub_uris:
            if len(self.tiled_client[sub_uri].shape) > 2:
                cum_size += len(self.tiled_client[sub_uri])
            else:
                cum_size += 1
            self.cum_sizes.append(cum_size)

    def __len__(self):
        return self.cum_sizes[-1]

    def _get_tiled_index(self, index):
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
        if len(sub_uri_client.shape) > 2:
            frame = sub_uri_client[index,]
        else:
            frame = sub_uri_client[:]
        if len(frame.shape) == 3:
            if frame.shape[0] == 3:
                frame = np.transpose(frame, (1, 2, 0))
            elif frame.shape[2] == 1 or frame.shape[0] == 1:
                frame = np.squeeze(frame)
        if self.log:
            frame = self._apply_log_transform(frame)
        else:
            frame = self._normalize_percentiles(frame)
        image = Image.fromarray(frame)
        image = image.convert("L")
        tensor_image = self.simple_transform(image)
        return (tensor_image,)

    def _apply_log_transform(self, image, threshold=0.000000000001):
        # Mask negative and NaN values
        nan_img = np.isnan(image)
        img_neg = image < 0.0
        mask_neg = np.array(img_neg)
        mask_nan = np.array(nan_img)
        mask = mask_nan + mask_neg
        x = np.ma.array(image, mask=mask)

        image = np.log(x + threshold)
        x = np.ma.array(image, mask=mask)
        x = self._normalize_percentiles(x)
        return x

    def _normalize_percentiles(self, x, low_perc=0.01, high_perc=99):
        low = np.percentile(x.ravel(), low_perc)
        high = np.percentile(x.ravel(), high_perc)
        x = (np.clip((x - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        return x

    def __getitems__(self, index_list):
        with ThreadPoolExecutor() as executor:
            data = list(executor.map(self.__getitem__, index_list))
        return data
