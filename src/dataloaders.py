from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from datasets.base_dataset import DetectorSource
from datasets.directory_dataset import CustomDirectoryDataset
from datasets.tiled_dataset import CustomTiledDataset
from helper_utils import (
    filepaths_from_directory,
    setup_data_transformations,
    split_dataset,
)


def setup_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    persistent_workers: bool,
) -> DataLoader:
    """
    Helper function to set up a DataLoader.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )


def get_train_dataloaders(
    sub_uris: List[str],
    root_uri: str,
    data_type: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    target_size: Optional[tuple] = None,
    horz_flip_prob: float = 0.0,
    vert_flip_prob: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    val_pct: float = 0.0,
    augm_invariant: bool = False,
    log: bool = False,
    data_tiled_api_key: Optional[str] = None,
    detector_uri: Optional[str] = None,
    detector_source: Optional[str] = DetectorSource.PYFAI.value,
    detector_tiled_api_key: Optional[str] = None,
) -> Tuple[List[DataLoader], tuple]:
    """
    Creates train and validation dataloaders in PyTorch from directory or tiled data.
    """
    persistent_workers = num_workers > 0

    # Set up data transformations
    data_transform = setup_data_transformations(
        brightness, contrast, saturation, hue, horz_flip_prob, vert_flip_prob
    )

    # Define dataset
    if data_type == "tiled":
        dataset = CustomTiledDataset(
            root_uri,
            sub_uris,
            target_size,
            data_transform,
            augm_invariant=augm_invariant,
            log=log,
            data_tiled_api_key=data_tiled_api_key,
            detector_uri=detector_uri,
            detector_source=detector_source,
            detector_tiled_api_key=detector_tiled_api_key,
        )
    else:
        data_uris = filepaths_from_directory(root_uri, selected_sub_uris=sub_uris)
        dataset = CustomDirectoryDataset(
            data_uris,
            target_size,
            data_transform,
            augm_invariant=augm_invariant,
            log=log,
            detector_uri=detector_uri,
            detector_source=detector_source,
            detector_tiled_api_key=detector_tiled_api_key,
        )

    # Get input shape
    input_channels, width, height = dataset[0][0].shape

    # Split dataset into train and validation
    train_set, val_set = split_dataset(dataset, val_pct)

    # Create train and validation dataloaders
    train_loader = setup_dataloader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        persistent_workers=persistent_workers,
    )

    val_loader = None
    if val_pct > 0:
        val_loader = setup_dataloader(
            val_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            persistent_workers=persistent_workers,
        )

    return [train_loader, val_loader], (input_channels, width, height)


def get_inference_dataloaders(
    sub_uris: List[str],
    root_uri: str,
    data_type: str,
    batch_size: int,
    num_workers: int,
    target_size: Optional[tuple] = None,
    log: bool = False,
    data_tiled_api_key: Optional[str] = None,
    detector_uri: Optional[str] = None,
    detector_source: Optional[str] = DetectorSource.PYFAI.value,
    detector_tiled_api_key: Optional[str] = None,
) -> Tuple[DataLoader, tuple]:
    """
    Creates inference dataloaders in PyTorch from directory or tiled data.
    """
    persistent_workers = num_workers > 0
    data_transform = setup_data_transformations()

    # Define dataset
    if data_type == "tiled":
        dataset = CustomTiledDataset(
            root_uri,
            sub_uris,
            target_size,
            data_transform,
            augm_invariant=False,
            log=log,
            data_tiled_api_key=data_tiled_api_key,
            detector_uri=detector_uri,
            detector_source=detector_source,
            detector_tiled_api_key=detector_tiled_api_key,
        )
    else:
        data_uris = filepaths_from_directory(root_uri, selected_sub_uris=sub_uris)
        dataset = CustomDirectoryDataset(
            data_uris,
            target_size,
            data_transform,
            augm_invariant=False,
            log=log,
            detector_uri=detector_uri,
            detector_source=detector_source,
            detector_tiled_api_key=detector_tiled_api_key,
        )

    input_channels, width, height = dataset[0][0].shape

    # Create dataloader for inference
    data_loader = setup_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=persistent_workers,
    )

    return data_loader, (input_channels, width, height)
