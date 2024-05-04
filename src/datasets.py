import numpy as np
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


class CustomTiledDataset(Dataset):
    def __init__(self, root_uri, sub_uris, target_size, log, api_key=None):
        self.sub_uris = sub_uris
        self.simple_transform = transforms.Compose(
            [transforms.Resize(target_size), transforms.ToTensor()]
        )
        self.log = log
        self.tiled_client = from_uri(root_uri, api_key=api_key)
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
                return self.sub_uris[count], index - cum_size
        return None

    def __getitem__(self, idx):
        sub_uri, index = self._get_tiled_index(idx)
        if index > 0:
            image = self.tiled_client[sub_uri][index,]
        else:
            image = self.tiled_client[sub_uri][:]
        if image.dtype != np.uint8:
            # Normalize according to percentiles 1-99
            low = np.percentile(image.ravel(), 1)
            high = np.percentile(image.ravel(), 99)
            image = np.clip((image - low) / (high - low), 0, 1)
            image = (image * 255).astype(np.uint8)  # Convert to uint8, 0-255
        if len(image.shape) == 3:
            if image.shape[2] == 3 or image.shape[3] == 1:
                image = np.transpose(image, (2, 0, 1))
            elif image.shape[0] == 1:
                image = np.squeeze(image)
            else:
                raise ValueError("Not a valid image shape")
        if self.log:
            image = np.log1p(np.array(image))
            image = (
                ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255
            ).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.convert("L")
        tensor_image = self.simple_transform(image)
        return (tensor_image, tensor_image)
