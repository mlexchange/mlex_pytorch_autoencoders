import argparse
import json
import logging
import multiprocessing
import warnings
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
from PIL import Image

from helper_utils import embed_imgs, get_dataloaders
from model import Autoencoder, TestingParameters

num_cpus = multiprocessing.cpu_count()
if num_cpus > 6:
    NUM_WORKERS = round(num_cpus / 6)
else:
    NUM_WORKERS = num_cpus
if NUM_WORKERS % 2 != 0:
    NUM_WORKERS -= 1

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(
    logging.WARNING
)  # disable logs from pytorch lightning


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_info", help="path to dataframe of filepaths")
    parser.add_argument("-m", "--model_dir", help="input directory")
    parser.add_argument("-o", "--output_dir", help="output directory")
    parser.add_argument("-p", "--parameters", help="list of training parameters")
    args = parser.parse_args()
    test_parameters = TestingParameters(**json.loads(args.parameters))

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Device:", device)

    if test_parameters.target_width * test_parameters.target_height > 0:
        target_size = (test_parameters.target_width, test_parameters.target_height)
    else:
        target_size = None

    test_loader, (temp_channels, temp_w, temp_h), datasets_uris = get_dataloaders(
        args.data_info,
        test_parameters.batch_size,
        NUM_WORKERS,
        shuffle=False,
        target_size=target_size,
        log=test_parameters.log,
        train=False,
    )

    model = Autoencoder.load_from_checkpoint(args.model_dir + "/last.ckpt")

    # Get latent space representation of test images and reconstructed images
    test_img_embeds, test_result = embed_imgs(model, test_loader)

    # Create output directory if it does not exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save latent space representation of test images
    df = pd.DataFrame(test_img_embeds.cpu().detach().numpy())
    df.index = datasets_uris
    df.columns = df.columns.astype(str)
    df.to_parquet(f"{args.output_dir}/f_vectors.parquet", engine="pyarrow")

    # Reconstructed images
    test_result = einops.rearrange(test_result, "n c x y -> n x y c")
    test_result = test_result.cpu().detach().numpy()

    # Define color mode according to number of channels in input images
    if temp_channels == 3:
        colormode = "RGB"
    else:
        colormode = "L"

    # Save reconstructed images
    for indx, uri in enumerate(datasets_uris):
        im = Image.fromarray((np.squeeze(test_result[indx]) * 255).astype(np.uint8))
        im = im.convert(colormode)
        im.save(f"{args.output_dir}/reconstructed_{indx}.jpg")
