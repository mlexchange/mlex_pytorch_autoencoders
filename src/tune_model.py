import argparse
import json
import logging
import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint

from helper_utils import get_dataloaders
from model import Autoencoder
from parameters import TuningParameters

SEED = 0

warnings.filterwarnings("ignore")
logger = logging.getLogger("pytorch_lightning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_info", help="path to dataframe of filepaths")
    parser.add_argument("-m", "--model_dir", help="input directory")
    parser.add_argument("-o", "--output_dir", help="output directory")
    parser.add_argument("-p", "--parameters", help="list of training parameters")
    args = parser.parse_args()
    tune_parameters = TuningParameters(**json.loads(args.parameters))

    if tune_parameters.seed:
        seed = tune_parameters.seed  # Setting the user-defined seed
    else:
        seed = SEED  # Setting the pre-defined seed
    pl.seed_everything(seed)
    logger.info("Seed: " + str(seed))

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("Device:" + str(device))

    if tune_parameters.target_width * tune_parameters.target_height > 0:
        target_size = (tune_parameters.target_width, tune_parameters.target_height)
    else:
        target_size = None

    [train_loader, val_loader], (input_channels, width, height) = get_dataloaders(
        args.data_info,
        tune_parameters.batch_size,
        tune_parameters.num_workers,
        tune_parameters.shuffle,
        target_size,
        tune_parameters.horz_flip_prob,
        tune_parameters.vert_flip_prob,
        tune_parameters.brightness,
        tune_parameters.contrast,
        tune_parameters.saturation,
        tune_parameters.hue,
        tune_parameters.val_pct,
        tune_parameters.augm_invariant,
        tune_parameters.log,
        tune_parameters.percentiles,
    )

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=tune_parameters.num_epochs,
        enable_progress_bar=False,
        callbacks=[
            ModelCheckpoint(
                dirpath=args.output_dir,
                save_last=True,
                filename="checkpoint_file",
                save_weights_only=True,
            )
        ],
    )

    model = Autoencoder.load_from_checkpoint(args.model_dir + "/last.ckpt")
    model.define_save_loss_dir(args.output_dir)
    model.optimizer = getattr(optim, tune_parameters.optimizer.value)
    criterion = getattr(nn, tune_parameters.criterion.value)
    model.criterion = criterion()

    model.learning_rate = tune_parameters.learning_rate
    model.gamma = tune_parameters.gamma
    model.step_size = tune_parameters.step_size

    logger.info("epoch,train_loss,val_loss")
    trainer.fit(model, train_loader, val_loader)
