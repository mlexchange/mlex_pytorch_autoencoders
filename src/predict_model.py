import argparse, os
import json
import einops
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
import warnings
import torch

from model import Autoencoder, TestingParameters
from helper_utils import get_dataloaders, embed_imgs


NUM_WORKERS = 0
warnings.filterwarnings('ignore')
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)    # disable logs from pytorch lightning


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_info', help='path to dataframe of filepaths')
    parser.add_argument('-m', '--model_dir', help='input directory')
    parser.add_argument('-o', '--output_dir', help='output directory')
    parser.add_argument('-p', '--parameters', help='list of training parameters')
    args = parser.parse_args()
    test_parameters = TestingParameters(**json.loads(args.parameters))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    if test_parameters.target_width*test_parameters.target_height > 0:
        target_size = (test_parameters.target_width, test_parameters.target_height)
    else:
        target_size = None

    [test_loader, temp], (temp_channels, temp_w, temp_h), datasets_uris = \
        get_dataloaders(
            args.data_info,
            test_parameters.batch_size,
            NUM_WORKERS,
            False,
            target_size)

    model = Autoencoder.load_from_checkpoint(args.model_dir + '/last.ckpt')

    trainer = pl.Trainer(enable_progress_bar=False)
    test_img_embeds = embed_imgs(model, test_loader)  # test images in latent space

    # Create output directory if it does not exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(test_img_embeds.cpu().detach().numpy())
    df.index = datasets_uris
    df.columns = df.columns.astype(str)
    df.to_parquet(f'{args.output_dir}/f_vectors.parquet', engine='pyarrow')

    # Reconstructed images
    test_result = trainer.predict(model, dataloaders=test_loader)
    test_result = torch.cat(test_result)
    test_result = einops.rearrange(test_result, 'n c x y -> n x y c')
    test_result = test_result.cpu().detach().numpy()

    for indx, uri in enumerate(datasets_uris):
        filename = uri.split('/')[-1]
        im = Image.fromarray((((test_result[indx]-np.min(test_result[indx])) ) * 255)\
                                 .astype(np.uint8)).convert('L')
        im.save(f'{args.output_dir}/reconstructed_{filename}.jpg')