<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

# How To Guide

This repository provides an machine learning (ML)-framework to train and test convolutional autoencoders for latent space exploration.

## Data Format
Currently, this ML algorithm supports data access through [Tiled](https://blueskyproject.io/tiled/). or filesystem. The supported image formats are: TIFF, TIF, JPG, JPEG, and PNG.

## Installation

### Using Docker
You can use the docker image provided in this repository. In a new terminal window, you can download the docker image with the command:
   ```
   docker pull ghcr.io/mlexchange/mlex_pytorch_autoencoders:main
   ```

## Training
To train a model from a Dash application, please follow the following steps:

1. Choose your dataset. Click on "Open File Manager", and choose your dataset.
2. Modify the [training parameters](./concepts.md) as needed.
3. Click Train/Run.
4. The training job has been successfully submitted! You can check the progress of this job in dropdown `Trained Jobs`, where you can select the corresponding job to display the training stats and/or logs.

## Inference
To run inference, please follow the following steps:

1. Choose your dataset. Click on "Open File Manager", and choose your dataset.
2. Modify the [inference parameters](./concepts.md) as needed.
4. Choose a trained model from the dropdown `Trained Jobs`.
4. Click Inference/Run.
5. The inference job has been successfully submitted! You can check the progress of this job in dropdown `Inference Jobs`, where you can select the corresponding job to display the results and/or logs.
