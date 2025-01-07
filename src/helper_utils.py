import glob
import os
from functools import reduce

import numpy as np
import torch
from tiled.client import from_uri
from tiled.structures.array import ArrayStructure, BuiltinDtype  # noqa: F401
from tiled.structures.core import StructureFamily  # noqa: F401
from tiled.structures.data_source import Asset, DataSource, Management  # noqa: F401
from tiled.structures.table import TableStructure
from torchvision import transforms

from datasets import CustomDirectoryDataset, CustomTiledDataset

FORMATS = [
    "*.[pP][nN][gG]",
    "*.[jJ][pP][gG]",
    "*.[jJ][pP][eE][gG]",
    "*.[tT][iI][fF]",
    "*.[tT][iI][fF][fF]",
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


def filepaths_from_directory(
    root_uri, selected_sub_uris=None, formats=FORMATS, sort=True
):
    """
    This function returns the list of file paths from the directory
    Args:
        root_uri:           [str] Root URI
        selected_sub_uris:  [list] List of selected sub URIs
    Returns:
        List of file paths
    """
    filenames = []
    for dataset in selected_sub_uris:
        dataset_path = os.path.join(root_uri, dataset)
        if os.path.isdir(dataset_path):
            # Find paths that match the format of interest
            all_paths = list(
                reduce(
                    lambda list1, list2: list1 + list2,
                    (
                        [
                            path
                            for path in glob.glob(
                                str(dataset_path) + "/" + t, recursive=False
                            )
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
                            for path in glob.glob(
                                str(dataset_path) + "/" + t, recursive=False
                            )
                        ]
                        for t in NOT_ALLOWED_FORMATS
                    ),
                )
            )
            # Remove not allowed filepaths from filepaths of interest
            paths = list(set(all_paths) - set(not_allowed_paths))
            if sort:
                paths.sort()
            filenames += paths
    return filenames


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
    sub_uris,
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
    detector_name=None,
):
    """
    This function creates the dataloaders in PyTorch from directory or npy files
    Args:
        sub_uris:       List[str] List of data URIs
        data_type:      [str] Type of data
        root_uri:       [str] Root URI
        batch_size:     [int] Batch size
        num_workers:    [int] Number of workers
        shuffle:        [bool] Shuffle data
        target_size:    [tuple] Target size
        horz_flip_prob: [float] Probability of horizontal flip
        vert_flip_prob: [float] Probability of vertical flip
        ############################################################################################
        Following descriptions were taken from
        https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html
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
        detector_name:  [str] Detector name
    Returns:
        PyTorch DataLoaders
        Image size, e.g. (input_channels, width, height)
        List of data URIs
    """
    # Load data information
    data_transform = []
    if num_workers > 0:
        persistent_workers = True
    else:
        persistent_workers = False

    data_uris = filepaths_from_directory(root_uri, selected_sub_uris=sub_uris)

    if train:
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

        # Create dataset and dataloaders
        dataset = CustomDirectoryDataset(
            data_uris, target_size, data_transform, augm_invariant, log
        )
        (input_channels, width, height) = dataset[0][0].shape

        # Split dataset into train and validation
        train_set, val_set = split_dataset(dataset, val_pct)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
        )

        if val_pct > 0:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                pin_memory=True,
            )
            data_loader = [train_loader, val_loader]
        else:
            data_loader = [train_loader, None]
    else:
        if data_type == "tiled":
            dataset = CustomTiledDataset(
                root_uri,
                data_uris,
                target_size,
                log,
                api_key,
            )
        else:
            data_transform.append(transforms.ToTensor())
            dataset = CustomDirectoryDataset(
                data_uris, target_size, data_transform, augm_invariant, log
            )

        (input_channels, width, height) = dataset[0][0].shape

        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=True,
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


def write_results(
    feature_vectors,
    io_parameters,
    feature_vectors_path,
    reconstructions,
    recons_size,
    metadata=None,
):
    # Prepare Tiled parent node
    uid_save = io_parameters.uid_save
    write_client = from_uri(
        io_parameters.results_tiled_uri, api_key=io_parameters.results_tiled_api_key
    )
    write_client = write_client.create_container(key=uid_save)

    # Save latent vectors to Tiled
    structure = TableStructure.from_pandas(feature_vectors)

    # Remove API keys from metadata
    if metadata:
        metadata["io_parameters"].pop("data_tiled_api_key", None)
        metadata["io_parameters"].pop("results_tiled_api_key", None)

    frame = write_client.new(
        structure_family="table",
        data_sources=[
            DataSource(
                structure_family="table",
                structure=structure,
                mimetype="application/x-parquet",
                assets=[
                    Asset(
                        data_uri=f"file://{feature_vectors_path}",
                        is_directory=False,
                        parameter="data_uris",
                        num=1,
                    )
                ],
            )
        ],
        metadata=metadata,
        key="feature_vectors",
    )

    frame.write(feature_vectors)

    # Save reconstructions to Tiled
    # structure = ArrayStructure(
    #     data_type=BuiltinDtype.from_numpy_dtype(np.dtype('int32')),
    #     shape=recons_size,
    #     chunks=((1,)*recons_size[0], (recons_size[1],), (recons_size[2],))
    # )

    write_client.write_array(
        reconstructions.astype(np.float32),
        metadata=metadata,
        key="reconstructions",
    )

    # write_client.new(
    #     structure_family=StructureFamily.array,
    #     data_sources=[
    #         DataSource(
    #             management=Management.external,
    #             mimetype="multipart/related;type=image/tiff",
    #             structure_family=StructureFamily.array,
    #         structure=structure,
    #             assets=[
    #                 Asset(
    #                     data_uri=f"file://{filepath}",
    #                     is_directory=False,
    #                     parameter="data_uri",
    #                     num=count
    #                 )
    #                 for count, filepath
    #                 in enumerate(reconstructions)
    #             ],
    #         ),
    #     ],
    #     metadata=metadata,
    #     key="reconstructions",
    # )

    pass
