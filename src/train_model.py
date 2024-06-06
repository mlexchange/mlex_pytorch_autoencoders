import argparse
import logging
import time
import warnings

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint

from helper_utils import get_dataloaders
from parameters import IOParameters, TrainingParameters
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
    train_parameters = TrainingParameters.parse_obj(parameters)

    if train_parameters.seed:
        seed = train_parameters.seed  # Setting the user-defined seed
    else:
        seed = SEED  # Setting the pre-defined seed
    pl.seed_everything(seed)
    logger.info("Seed: " + str(seed))

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("Device:" + str(device))

    if train_parameters.target_width * train_parameters.target_height > 0:
        target_size = (train_parameters.target_width, train_parameters.target_height)
    else:
        target_size = None

    logger.info(f"Number of workers: {train_parameters.num_workers}")
    [train_loader, val_loader], (input_channels, width, height) = get_dataloaders(
        io_parameters.data_uris,
        io_parameters.root_uri,
        io_parameters.data_type,
        train_parameters.batch_size,
        train_parameters.num_workers,
        train_parameters.shuffle,
        target_size,
        train_parameters.horz_flip_prob,
        train_parameters.vert_flip_prob,
        train_parameters.brightness,
        train_parameters.contrast,
        train_parameters.saturation,
        train_parameters.hue,
        train_parameters.val_pct,
        train_parameters.augm_invariant,
        train_parameters.log,
        api_key=io_parameters.data_tiled_api_key,
        detector_name=train_parameters.detector_name,
    )

    output_dir = io_parameters.output_dir

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=train_parameters.num_epochs,
        enable_progress_bar=False,
        profiler=train_parameters.profiler.value,
        callbacks=[
            ModelCheckpoint(
                dirpath=output_dir,
                save_last=True,
                filename="checkpoint_file",
                save_weights_only=True,
            )
        ],
    )

    model = Autoencoder(
        base_channel_size=train_parameters.base_channel_size,
        depth=train_parameters.depth,
        latent_dim=train_parameters.latent_dim,
        num_input_channels=input_channels,
        optimizer=train_parameters.optimizer,
        criterion=train_parameters.criterion,
        learning_rate=train_parameters.learning_rate,
        step_size=train_parameters.step_size,
        gamma=train_parameters.gamma,
        width=width,
        height=height,
    )
    model.define_save_loss_dir(output_dir)

    start = time.time()
    logger.info("epoch,train_loss,val_loss")
    trainer.fit(model, train_loader, val_loader)
    logger.info(f"Training time: {time.time()-start}")
