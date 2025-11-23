import argparse
import logging
import os
import sys
import tempfile
import time
import warnings

import mlflow
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from dvclive import Live
from dvclive.lightning import DVCLiveLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloaders import get_train_dataloaders
from model import Autoencoder
from parameters import IOParameters, TuningParameters

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
    tune_parameters = TuningParameters.parse_obj(parameters)

    # Setup MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = io_parameters.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = io_parameters.mlflow_tracking_password
    mlflow.set_tracking_uri(io_parameters.mlflow_uri)
    logger.info(f"Setting MLflow tracking uri: {io_parameters.mlflow_uri}")
    mlflow.set_experiment(io_parameters.uid_save)

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Set seed
        if tune_parameters.seed:
            seed = tune_parameters.seed
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
        if tune_parameters.target_width * tune_parameters.target_height > 0:
            target_size = (tune_parameters.target_width, tune_parameters.target_height)
        else:
            target_size = None

        # Get dataloaders
        [train_loader, val_loader], (input_channels, width, height) = (
            get_train_dataloaders(
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
                data_tiled_api_key=io_parameters.data_tiled_api_key,
                detector_uri=io_parameters.detector_uri,
                detector_source=io_parameters.detector_source,
                detector_tiled_api_key=io_parameters.detector_tiled_api_key,
            )
        )

        # Define model and results directory (changed to use temp directory)
        output_dir = tempfile.mkdtemp(prefix=f"{io_parameters.uid_save}_tune_")
        logger.info(f"Using temporary directory: {output_dir}")

        dvclive_savepath = f"{output_dir}/dvc_metrics"
        model_dir = output_dir

        # Set up dvclive
        with Live(dvclive_savepath, report="html") as live:
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
                logger=DVCLiveLogger(experiment=live),
            )

            # Load model
            model = Autoencoder.load_from_checkpoint(model_dir + "/last.ckpt")
            model.define_save_loss_dir(output_dir)
            model.optimizer = getattr(optim, tune_parameters.optimizer.value)
            criterion = getattr(nn, tune_parameters.criterion.value)
            model.criterion = criterion()

            model.learning_rate = tune_parameters.learning_rate
            model.gamma = tune_parameters.gamma
            model.step_size = tune_parameters.step_size

            start = time.time()
            logger.info("epoch,train_loss,val_loss")
            trainer.fit(model, train_loader, val_loader)
            logger.info(f"Tuning time: {time.time()-start}")

            # Log hyperparameters
            mlflow.log_params(tune_parameters.dict())

            # Log DVC metrics to MLflow
            if os.path.exists(dvclive_savepath):
                mlflow.log_artifacts(dvclive_savepath, artifact_path="dvc_metrics")
                logger.info(f"DVC metrics logged to MLflow from {dvclive_savepath}")
