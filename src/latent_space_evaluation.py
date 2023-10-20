import argparse
from IPython.utils import io
import json
import logging
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from model import Autoencoder, EvaluationParameters
from helper_utils import get_dataloaders


SEED = 42
NUM_WORKERS = 0
warnings.filterwarnings('ignore')
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)    # disable logs from pytorch lightning


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('parameters', help='list of training parameters')
    args = parser.parse_args()
    eval_parameters = EvaluationParameters(**json.loads(args.parameters))

    if eval_parameters.seed:
        seed = eval_parameters.seed     # Setting the user-defined seed
    else:
        seed = SEED                     # Setting the pre-defined seed
    pl.seed_everything(seed)
    print("Seed: ", seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    if eval_parameters.target_width*eval_parameters.target_height > 0:
        target_size = (eval_parameters.target_width, eval_parameters.target_height)
    else:
        target_size = None

    [train_loader, val_loader], (input_channels, width, height), tmp = get_dataloaders(args.input_dir,
                                                                                       eval_parameters.batch_size,
                                                                                       NUM_WORKERS,
                                                                                       eval_parameters.shuffle,
                                                                                       target_size,
                                                                                       'x_train',
                                                                                       eval_parameters.val_pct)

    [test_loader, temp], (temp_channels, temp_w, temp_h), tmp = get_dataloaders(args.input_dir,
                                                                                eval_parameters.batch_size,
                                                                                NUM_WORKERS,
                                                                                False,
                                                                                target_size,
                                                                                'x_test')

    val_result = [{'test_loss': 0}]
    for count, latent_dim in enumerate(eval_parameters.latent_dim):
        with io.capture_output() as captured:
            trainer = pl.Trainer(default_root_dir=args.output_dir + f"/model_{latent_dim}",
                                 gpus=1 if str(device).startswith("cuda") else 0,
                                 max_epochs=eval_parameters.num_epochs,
                                 progress_bar_refresh_rate=0,
                                 weights_summary=None,
                                 callbacks=[ModelCheckpoint(dirpath=args.output_dir + f"/model_{latent_dim}",
                                                            save_last=True,
                                                            filename='checkpoint_file',
                                                            save_weights_only=True)])
            model_ld = Autoencoder(base_channel_size=eval_parameters.base_channel_size,
                                   latent_dim=latent_dim,
                                   num_input_channels=input_channels,
                                   optimizer=eval_parameters.optimizer,
                                   criterion=eval_parameters.criterion,
                                   learning_rate=eval_parameters.learning_rate,
                                   width=width,
                                   height=height)
            trainer.fit(model_ld, train_loader, val_loader)
            # Test best model on validation and test set
            test_result = trainer.test(model_ld, test_dataloaders=test_loader, verbose=False)
            if val_loader:
                val_result = trainer.test(model_ld, test_dataloaders=val_loader, verbose=False)
        if count == 0:
            print('latent_space_dim test validation')
        print(latent_dim, ' ', test_result[0]['test_loss'], ' ', val_result[0]['test_loss'])
