import argparse
import logging
import time
import warnings
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from helper_utils import embed_imgs, get_dataloaders
from parameters import InferenceParameters, IOParameters
from src.model import Autoencoder, Decoder, Encoder  # noqa: F401

warnings.filterwarnings("ignore")
logger = logging.getLogger("pytorch_lightning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    io_parameters = IOParameters.parse_obj(parameters["io_parameters"])
    inference_parameters = InferenceParameters.parse_obj(parameters)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("Device:" + str(device))

    if inference_parameters.target_width * inference_parameters.target_height > 0:
        target_size = (
            inference_parameters.target_width,
            inference_parameters.target_height,
        )
    else:
        target_size = None

    # Define dataloaders
    inference_loader, (temp_channels, temp_w, temp_h) = get_dataloaders(
        io_parameters.data_uris,
        io_parameters.root_uri,
        io_parameters.data_type,
        inference_parameters.batch_size,
        inference_parameters.num_workers,
        shuffle=False,
        target_size=target_size,
        log=inference_parameters.log,
        train=False,
        api_key=io_parameters.data_tiled_api_key,
        detector_name=inference_parameters.detector_name,
    )

    model = Autoencoder.load_from_checkpoint(io_parameters.model_dir)

    # Get latent space representation of inference images and reconstructed images
    start = time.time()
    inference_img_embeds, inference_result = embed_imgs(model, inference_loader)
    logger.info(f"Time taken to embed images: {time.time() - start:.2f} seconds")

    # Create output directory if it does not exist
    output_dir = Path(f"{io_parameters.output_dir}/{io_parameters.uid_save}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save latent space representation of inference images
    df = pd.DataFrame(inference_img_embeds.cpu().detach().numpy())
    df.columns = df.columns.astype(str)
    df.to_parquet(f"{output_dir}/f_vectors.parquet", engine="pyarrow")
    logger.info("Latent space representation saved")

    # Reconstructed images
    inference_result = einops.rearrange(inference_result, "n c x y -> n x y c")
    inference_result = inference_result.cpu().detach().numpy()

    # Define color mode according to number of channels in input images
    if temp_channels == 3:
        colormode = "RGB"
    else:
        colormode = "L"

    start = time.time()
    # Save reconstructed images
    for indx in range(inference_result.shape[0]):
        im = Image.fromarray(
            (np.squeeze(inference_result[indx]) * 255).astype(np.uint8)
        )
        im = im.convert(colormode)
        im.save(f"{output_dir}/reconstructed_{indx}.jpg")
    logger.info(f"Reconstructed images saved in {time.time() - start:.2f} seconds")
