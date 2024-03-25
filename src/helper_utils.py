import glob
from functools import reduce

import torch
from torchvision import transforms

from model import CustomDirectoryDataset, CustomTiledDataset

# List of allowed and not allowed formats
FORMATS = [
    "**/*.[pP][nN][gG]",
    "**/*.[jJ][pP][gG]",
    "**/*.[jJ][pP][eE][gG]",
    "**/*.[tT][iI][fF]",
    "**/*.[tT][iI][fF][fF]",
]
NOT_ALLOWED_FORMATS = [
    "**/__pycache__/**",
    "**/.*",
    "cache/",
    "cache/**/",
    "cache/**",
    "tiled_local_copy/",
    "**/tiled_local_copy/**",
    "**/tiled_local_copy/**/",
    "mlexchange_store/**/",
    "mlexchange_store/**",
    "labelmaker_outputs/**/",
    "labelmaker_outputs/**",
]


def walk_directory(dataset_path, formats=FORMATS):
    """
    This function walks through the directory and returns the list of files
    Args:
        directory:      Directory to walk through
    Returns:
        List of files in the directory
    """
    all_paths = list(
        reduce(
            lambda list1, list2: list1 + list2,
            (
                [
                    path
                    for path in glob.glob(str(dataset_path) + "/" + t, recursive=False)
                ]
                for t in formats
            ),
        )
    )
    # Find paths that match the not allowed file/directory formats
    not_allowed_paths = list(
        reduce(
            lambda list1, list2: list1 + list2,
            (
                [
                    path
                    for path in glob.glob(str(dataset_path) + "/" + t, recursive=False)
                ]
                for t in NOT_ALLOWED_FORMATS
            ),
        )
    )
    # Remove not allowed filepaths from filepaths of interest
    paths = list(set(all_paths) - set(not_allowed_paths))
    paths.sort()
    return paths


def split_dataset(dataset, val_pct):
    """
    This function splits the input dataset according to the splitting ratios
    Args:
        dataset:            Full dataset to be split
        val_pct:           Percentage for validation [0-100]
    Returns:
        train_set:          Training torch subset
        val_set:            Testing torch subset
    """
    data_size = len(dataset)
    val_size = int(val_pct * data_size / 100)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [data_size - val_size, val_size]
    )
    return train_set, val_set


def get_dataloaders(
    data_uris,
    root_uri,
    data_type,
    batch_size,
    num_workers,
    shuffle=False,
    target_size=None,
    horz_flip_prob=0,
    vert_flip_prob=0,
    brightness=0,
    contrast=0,
    saturation=0,
    hue=0,
    val_pct=0,
    augm_invariant=False,
    log=False,
    train=True,
    api_key=None,
):
    """
    This function creates the dataloaders in PyTorch from directory or npy files
    Args:
        data_uris:      List[str] List of data URIs
        data_type:      [str] Type of data
        root_uri:       [str] Root URI
        batch_size:     [int] Batch size
        num_workers:    [int] Number of workers
        shuffle:        [bool] Shuffle data
        target_size:    [tuple] Target size
        horz_flip_prob: [float] Probability of horizontal flip
        vert_flip_prob: [float] Probability of vertical flip
        ############################################################################################
        Following descriptions were taken from https://pytorch.org/vision/main/generated/torchvision
        .transforms.ColorJitter.html
        brightness:     [float, non negative] How much to jitter brightness, brightness_factor is
                        chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
        contrast:       [float, non negative] How much to jitter contrast, contrast_factor is chosen
                        uniformly from [max(0, 1 - contrast), 1 + contrast]
        saturation:     [float, non negative] How much to jitter saturation. saturation_factor is
                        chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue:            [float] How much to jitter hue, hue_factor is chosen uniformly from
                        [-hue, hue]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
                        To jitter hue, the pixel values of the input image has to be non-negative
                        for conversion to HSV space; thus it does not work if you normalize your
                        image to an interval with negative values, or use an interpolation that
                        generates negative values before using this function.
        ############################################################################################
        val_pct:        [int] Percentage for validation [0-100]
        augm_invariant: [bool] Ground truth changes (or not) according to selected transformations
        log:            [bool] Log information
        api_key:        [str] API key for tiled
    Returns:
        PyTorch DataLoaders
        Image size, e.g. (input_channels, width, height)
        List of data URIs
    """
    # Load data information
    data_transform = []

    if train:
        # TODO: Fix train
        data_info = None
        # Definition of data transforms
        if brightness > 0 or contrast > 0 or saturation > 0 or hue > 0:
            data_transform.append(
                transforms.ColorJitter(brightness, contrast, saturation, hue)
            )
        if horz_flip_prob > 0:
            data_transform.append(transforms.RandomHorizontalFlip(p=horz_flip_prob))
        if vert_flip_prob > 0:
            data_transform.append(transforms.RandomVerticalFlip(p=vert_flip_prob))
        data_transform.append(transforms.ToTensor())

        # Get local file location of tiled data if needed
        if "local_uri" in data_info:
            local_uri = data_info["local_uri"]
        else:
            local_uri = data_info["uri"]

        # Create dataset and dataloaders
        dataset = CustomDirectoryDataset(
            local_uri, target_size, data_transform, augm_invariant, log
        )
        (input_channels, width, height) = dataset[0][0].shape

        # Split dataset into train and validation
        train_set, val_set = split_dataset(dataset, val_pct)
        train_loader = torch.utils.data.DataLoader(
            train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
        )

        if val_pct > 0:
            val_loader = torch.utils.data.DataLoader(
                val_set, shuffle=False, batch_size=batch_size, num_workers=num_workers
            )
            data_loader = [train_loader, val_loader]
        else:
            data_loader = [train_loader, None]
    else:
        if data_type == "tiled":
            dataset = CustomTiledDataset(
                data_uris, root_uri, target_size, log, api_key=api_key
            )
        else:
            data_transform.append(transforms.ToTensor())
            filepaths = []
            for uri in data_uris:
                filepaths += walk_directory(root_uri + uri)
            dataset = CustomDirectoryDataset(
                filepaths, target_size, data_transform, augm_invariant, log
            )

        (input_channels, width, height) = dataset[0][0].shape
        data_loader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers
        )

    return data_loader, (input_channels, width, height)


def embed_imgs(model, data_loader):
    """
    This function finds the latent space representation of the input data
    Args:
        model:          Trained model
        data_loader:    PyTorch DataLoaders
    Returns:
        Latent space representation of the data
    """
    embed_list = []
    reconstruct_list = []
    for imgs in data_loader:
        with torch.no_grad():
            z = model.encoder(imgs[0].to(model.device))
            x_hat = model.decoder(z)
        embed_list.append(z)
        reconstruct_list.append(x_hat)
    return torch.cat(embed_list, dim=0), torch.cat(reconstruct_list, dim=0)
