import os

import einops
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms

from model import CustomTensorDataset, CustomDirectoryDataset #, CustomSplashDataset


def split_dataset(dataset, val_pct):
    '''
    This function splits the input dataset according to the splitting ratios
    Args:
        dataset:            Full dataset to be split
        val_pct:           Percentage for validation [0-100]
    Returns:
        train_set:          Training torch subset
        val_set:            Testing torch subset
    '''
    data_size = len(dataset)
    val_size = int(val_pct*data_size/100)
    train_set, val_set = torch.utils.data.random_split(dataset, [data_size - val_size, val_size])
    return train_set, val_set


def get_dataloaders(data, #splash_uri,
                    batch_size, num_workers, shuffle=False, target_size=None,
                    horz_flip_prob=0, vert_flip_prob=0, brightness=0, contrast=0, saturation=0, hue=0,
                    val_pct=None):
    '''
    This function creates the dataloaders in PyTorch from directory or npy files
    Args:
        splash_uri:     [str] Splash URI to Tagging Event UID for tracking purposes
        batch_size:     [int] Batch size
        num_workers:    [int] Number of workers
        shuffle:        [bool] Shuffle data
        target_size:    [tuple] Target size
        horz_flip_prob: [float] Probability of horizontal flip
        vert_flip_prob: [float] Probability of vertical flip
        brightness:     [float]
        contrast:
        saturation:
        hue:
        val_pct:        [int] Percentage for validation [0-100]
    Returns:
        PyTorch DataLoaders
    '''

    # Definition of data transforms
    data_transform = []
    if brightness>0 or contrast>0 or saturation>0 or hue>0:
        data_transform.append(transforms.ColorJitter(brightness, contrast, saturation, hue))
    if horz_flip_prob>0:
        data_transform.append(transforms.RandomHorizontalFlip(p=horz_flip_prob))
    if vert_flip_prob>0:
        data_transform.append(transforms.RandomVerticalFlip(p=vert_flip_prob))

    if target_size:
        data_transform.append(transforms.Resize(target_size))
    data_transform.append(transforms.ToTensor())
    data_info = pd.read_parquet(data, engine='pyarrow')
    if 'local_uri' in data_info:
        dataset = CustomDirectoryDataset(data_info['local_uri'], transforms.Compose(data_transform))
    else:
        dataset = CustomDirectoryDataset(data_info['uri'], transforms.Compose(data_transform))
    (input_channels, width, height) = dataset[0][0].shape
    datasets_uris = data_info['uri']

    if val_pct:
        train_set, val_set = split_dataset(dataset, val_pct)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers)
        if val_pct > 0:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers)
            data_loader = [train_loader, val_loader]
        else:
            data_loader = [train_loader, None]
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers)
        data_loader = [data_loader, None]
    return data_loader, (input_channels, width, height), datasets_uris


def embed_imgs(model, data_loader):
    '''
    This function finds the latent space representation of the input data
    Args:
        model:          Trained model
        data_loader:    PyTorch DataLoaders
    Returns:
        Latent space representation of the data
    '''
    embed_list = []
    for counter, imgs in enumerate(data_loader):
        with torch.no_grad():
            z = model.encoder(imgs[0].to(model.device))
        embed_list.append(z)
    return torch.cat(embed_list, dim=0)
