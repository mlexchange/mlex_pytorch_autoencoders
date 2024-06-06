import argparse
import logging
import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

from helper_utils import get_dataloaders
from parameters import IOParameters, TuningParameters
from src.model import Autoencoder, Decoder, Encoder  # noqa: F401

SEED = 0

warnings.filterwarnings("ignore")
logger = logging.getLogger("pytorch_lightning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    io_parameters = IOParameters.parse_obj(parameters["io_parameters"])
    tune_parameters = TuningParameters.parse_obj(parameters)

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
        io_parameters.data_uris,
        io_parameters.root_uri,
        io_parameters.data_type,
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
        api_key=io_parameters.data_tiled_api_key,
        detector_name=tune_parameters.detector_name,
    )

    output_dir = io_parameters.output_dir
    model_dir = io_parameters.model_dir

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=tune_parameters.num_epochs,
        enable_progress_bar=False,
        profiler=tune_parameters.profiler.value,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir,
                save_last=True,
                filename="checkpoint_file",
                save_weights_only=True,
            )
        ],
    )

    model = Autoencoder.load_from_checkpoint(model_dir + "/last.ckpt")
    model.define_save_loss_dir(output_dir)
    model.optimizer = getattr(optim, tune_parameters.optimizer.value)
    criterion = getattr(nn, tune_parameters.criterion.value)
    model.criterion = criterion()

    model.learning_rate = tune_parameters.learning_rate
    model.gamma = tune_parameters.gamma
    model.step_size = tune_parameters.step_size

    logger.info("epoch,train_loss,val_loss")
    trainer.fit(model, train_loader, val_loader)
