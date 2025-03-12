import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
import yaml
from dvclive import Live
from dvclive.lightning import DVCLiveLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloaders import get_train_dataloaders
from model import Autoencoder
from parameters import IOParameters, TrainingParameters

SEED = 42

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    stream=sys.stdout,  # Force all logs to stdout
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    with open(args.yaml_path, "r") as file:
        parameters = yaml.safe_load(file)

    # Parse parameters
    io_parameters = IOParameters.parse_obj(parameters["io_parameters"])
    train_parameters = TrainingParameters.parse_obj(parameters["model_parameters"])

    # Setup MLflow
    mlflow.set_tracking_uri(io_parameters.mlflow_uri)
    logger.info(f"Setting MLflow tracking uir: {io_parameters.mlflow_uri}")

    mlflow.set_experiment(io_parameters.uid_save)
    logger.info(f"Setting MLflow experiment name: {io_parameters.uid_save}")

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Set seed
        if train_parameters.seed:
            seed = train_parameters.seed
        else:
            seed = SEED
        pl.seed_everything(seed)
        logger.info("Seed: " + str(seed))

        # Set device
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        logger.info("Device:" + str(device))

        # Set target size
        if train_parameters.target_width * train_parameters.target_height > 0:
            target_size = (
                train_parameters.target_width,
                train_parameters.target_height,
            )
        else:
            target_size = None

        # Get dataloaders
        logger.info(f"Number of workers: {train_parameters.num_workers}")
        [train_loader, val_loader], (input_channels, width, height) = (
            get_train_dataloaders(
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
                data_tiled_api_key=io_parameters.data_tiled_api_key,
                detector_uri=io_parameters.detector_uri,
                detector_source=io_parameters.detector_source,
                detector_tiled_api_key=io_parameters.detector_tiled_api_key,
            )
        )

        # Set up model directory
        model_dir = Path(f"{io_parameters.models_dir}/{io_parameters.uid_save}")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Set up dvclive
        with Live(model_dir, report="html") as live:
            trainer = pl.Trainer(
                default_root_dir=model_dir,
                gpus=1 if str(device).startswith("cuda") else 0,
                max_epochs=train_parameters.num_epochs,
                enable_progress_bar=False,
                profiler=train_parameters.profiler,
                callbacks=[
                    ModelCheckpoint(
                        dirpath=model_dir,
                        save_last=True,
                        filename="checkpoint_file",
                        save_weights_only=True,
                    )
                ],
                logger=DVCLiveLogger(experiment=live),
            )

            # Set up model
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
            model.define_save_loss_dir(model_dir)

            start = time.time()
            trainer.fit(model, train_loader, val_loader)
            logger.info(f"Training time: {time.time()-start}")

            # Save model to MLflow
            mlflow.pytorch.log_model(
                model, "model", registered_model_name=io_parameters.uid_save
            )
            logger.info(
                f"Training complete. Model saved to MLflow with model name: {io_parameters.uid_save}"
            )
