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

from dataloaders import get_inference_dataloaders
from helper_utils import embed_imgs, write_results
from model import Autoencoder
from parameters import InferenceParameters, IOParameters

warnings.filterwarnings("ignore")
logger = logging.getLogger("pytorch_lightning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    # Parse parameters
    io_parameters = IOParameters.parse_obj(parameters["io_parameters"])
    inference_parameters = InferenceParameters.parse_obj(parameters["model_parameters"])

    # Set device
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("Device:" + str(device))

    # Set target size
    if inference_parameters.target_width * inference_parameters.target_height > 0:
        target_size = (
            inference_parameters.target_width,
            inference_parameters.target_height,
        )
    else:
        target_size = None

    # Get dataloaders
    inference_loader, (temp_channels, temp_w, temp_h) = get_inference_dataloaders(
        io_parameters.data_uris,
        io_parameters.root_uri,
        io_parameters.data_type,
        inference_parameters.batch_size,
        inference_parameters.num_workers,
        target_size=target_size,
        log=inference_parameters.log,
        data_tiled_api_key=io_parameters.data_tiled_api_key,
        detector_uri=io_parameters.detector_uri,
        detector_source=io_parameters.detector_source,
        detector_tiled_api_key=io_parameters.detector_tiled_api_key,
    )

    # Load model
    model = Autoencoder.load_from_checkpoint(
        f"{io_parameters.model_dir}/{io_parameters.uid_retrieve}/last.ckpt"
    )

    # Get latent space representation of inference images and reconstructed images
    start = time.time()
    inference_img_embeds, inference_result = embed_imgs(model, inference_loader)
    logger.info(f"Time taken to embed images: {time.time() - start:.2f} seconds")

    # Create output directory if it does not exist
    results_dir = Path(f"{io_parameters.results_dir}/{io_parameters.uid_save}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save latent space representation of inference images
    df = pd.DataFrame(inference_img_embeds.cpu().detach().numpy())
    df.columns = df.columns.astype(str)
    df.to_parquet(f"{results_dir}/f_vectors.parquet", engine="pyarrow")

    logger.info("Latent space representation saved")

    # Reconstructed images
    inference_result = einops.rearrange(inference_result, "n c x y -> n x y c")
    inference_result = inference_result.cpu().detach().numpy()

    # Define color mode according to number of channels in input images
    if temp_channels == 3:
        colormode = "RGB"
    else:
        colormode = "L"
        inference_result = np.squeeze(inference_result, axis=-1)

    # Save reconstructed images
    start = time.time()
    recons_size = inference_result.shape
    recons_filepaths = []
    for indx in range(recons_size[0]):
        filepath = f"{results_dir}/reconstructed_{indx}.jpg"
        im = Image.fromarray(
            (np.squeeze(inference_result[indx]) * 255).astype(np.uint8)
        )
        im = im.convert(colormode)
        im.save(filepath)
        recons_filepaths.append(filepath)
    logger.info(f"Reconstructed images saved in {time.time() - start:.2f} seconds")

    # Write results to Tiled
    write_results(
        df,
        io_parameters,
        f"{results_dir}/f_vectors.parquet",
        inference_result,
        parameters,
    )
    logger.info("Results written to tiled")
