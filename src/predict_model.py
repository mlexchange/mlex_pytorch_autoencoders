import argparse
import json
import logging
import warnings
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
from PIL import Image

from helper_utils import embed_imgs, get_dataloaders
from model import Autoencoder
from parameters import InferenceParameters

warnings.filterwarnings("ignore")
logger = logging.getLogger("pytorch_lightning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_info", help="path to dataframe of filepaths")
    parser.add_argument("-m", "--model_dir", help="input directory")
    parser.add_argument("-o", "--output_dir", help="output directory")
    parser.add_argument("-p", "--parameters", help="list of training parameters")
    args = parser.parse_args()
    inference_parameters = InferenceParameters(**json.loads(args.parameters))

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

    inference_loader, (temp_channels, temp_w, temp_h) = get_dataloaders(
        args.data_info,
        inference_parameters.batch_size,
        inference_parameters.num_workers,
        shuffle=False,
        target_size=target_size,
        log=inference_parameters.log,
        train=False,
    )

    model = Autoencoder.load_from_checkpoint(args.model_dir + "/last.ckpt")
    logger.info("Model loaded")

    # Get latent space representation of inference images and reconstructed images
    inference_img_embeds, inference_result = embed_imgs(model, inference_loader)
    logger.info("Inference images embedded")

    # Create output directory if it does not exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save latent space representation of inference images
    df = pd.DataFrame(inference_img_embeds.cpu().detach().numpy())
    df.columns = df.columns.astype(str)
    df.to_parquet(f"{args.output_dir}/f_vectors.parquet", engine="pyarrow")
    logger.info("Latent space representation saved")

    # Reconstructed images
    inference_result = einops.rearrange(inference_result, "n c x y -> n x y c")
    inference_result = inference_result.cpu().detach().numpy()

    # Define color mode according to number of channels in input images
    if temp_channels == 3:
        colormode = "RGB"
    else:
        colormode = "L"

    # Save reconstructed images
    for indx in range(inference_result.shape[0]):
        im = Image.fromarray(
            (np.squeeze(inference_result[indx]) * 255).astype(np.uint8)
        )
        im = im.convert(colormode)
        im.save(f"{args.output_dir}/reconstructed_{indx}.jpg")
    logger.info("Reconstructed images saved")
