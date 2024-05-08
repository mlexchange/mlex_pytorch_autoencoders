import csv
from collections import OrderedDict
from enum import Enum

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from parameters import Criterion, Optimizer


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        depth: int,
        base_channel_size: int,
        width: int,
        height: int,
        latent_dim: int,
        act_fn: object = nn.GELU,
    ):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter
              is 3
            - depth: Number of layers
            - base_channel_size : Number of channels we use in the first convolutional layers.
              Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - width, height: Dimensionality of the input image
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        auto = []
        for layer in range(depth):
            if layer == 0:
                auto.append(
                    (
                        f"conv{2 * layer}",
                        nn.Conv2d(
                            num_input_channels,
                            c_hid,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                        ),
                    )
                )
            else:
                auto.append(
                    (
                        f"conv{2 * layer}",
                        nn.Conv2d(
                            (2 ** (layer - 1)) * c_hid,
                            (2**layer) * c_hid,
                            kernel_size=3,
                            padding=1,
                            stride=2,
                        ),
                    )
                )
            auto.append((f"act{2 * layer}", act_fn()))
            if layer == depth - 1:
                auto.append(
                    ("flat", nn.Flatten())
                )  # Image grid to single feature vector
                auto.append(
                    (
                        "lin",
                        nn.Linear(
                            int(width * height * c_hid / (2 ** (layer + 2))), latent_dim
                        ),
                    )
                )
            else:
                auto.append(
                    (
                        f"conv{2 * layer + 1}",
                        nn.Conv2d(
                            (2**layer) * c_hid,
                            (2**layer) * c_hid,
                            kernel_size=3,
                            padding=1,
                        ),
                    )
                )
                auto.append((f"act{2 * layer + 1}", act_fn()))

        self.net = nn.Sequential(OrderedDict(auto))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        depth: int,
        base_channel_size: int,
        width: int,
        height: int,
        latent_dim: int,
        act_fn: object = nn.GELU,
    ):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR,
              this parameter is 3
            - depth: Number of layers
            - base_channel_size : Number of channels we use in the last convolutional layers.
              Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - width, height: Dimensionality of the input image
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.width = width
        self.height = height
        self.depth = depth
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, int(width * height * c_hid / (2 ** (depth + 1)))),
            act_fn(),
        )
        auto = []
        for layer in reversed(range(depth)):
            if layer == 0:
                auto.append(
                    (
                        f"tconv{2 * layer}",
                        nn.ConvTranspose2d(
                            c_hid,
                            num_input_channels,
                            kernel_size=3,
                            output_padding=1,
                            padding=1,
                            stride=2,
                        ),
                    )
                )
                auto.append(
                    ("sigmoid", nn.Sigmoid())
                )  # The input images is scaled between 0-1
            else:
                auto.append(
                    (
                        f"tconv{2 * layer}",
                        nn.ConvTranspose2d(
                            (2**layer) * c_hid,
                            (2 ** (layer - 1)) * c_hid,
                            kernel_size=3,
                            output_padding=1,
                            padding=1,
                            stride=2,
                        ),
                    )
                )
                auto.append((f"act{2 * layer}", act_fn()))
                auto.append(
                    (
                        f"conv{2 * layer + 1}",
                        nn.Conv2d(
                            (2 ** (layer - 1)) * c_hid,
                            (2 ** (layer - 1)) * c_hid,
                            kernel_size=3,
                            padding=1,
                        ),
                    )
                )
                auto.append((f"act{2 * layer + 1}", act_fn()))
        self.net = nn.Sequential(OrderedDict(auto))

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(
            x.shape[0],
            -1,
            int(self.width / (2 ** (self.depth))),
            int(self.height / (2 ** (self.depth))),
        )
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        depth: int,
        num_input_channels: int = 3,
        optimizer: object = Optimizer,
        criterion: object = Criterion,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        learning_rate: float = 1e-3,
        step_size: int = 30,
        gamma: float = 0.1,
        width: int = 32,
        height: int = 32,
        act_fn: object = nn.GELU,
    ):
        super().__init__()
        self.optimizer = getattr(optim, optimizer.value)
        self.learning_rate = learning_rate
        self.step_size = step_size
        self.gamma = gamma
        if isinstance(criterion, Enum):
            criterion = getattr(nn, criterion.value)
            self.criterion = criterion()
        else:
            self.criterion = criterion
        self.train_loss = 0
        self.validation_loss = 0
        self.train_loss_summary = []
        self.validation_loss_summary = []
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(
            num_input_channels,
            depth,
            base_channel_size,
            width,
            height,
            latent_dim,
            act_fn,
        )
        self.decoder = decoder_class(
            num_input_channels,
            depth,
            base_channel_size,
            width,
            height,
            latent_dim,
            act_fn,
        )
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def define_save_loss_dir(self, dir_save_loss):
        self.dir_save_loss = dir_save_loss
        with open(f"{self.dir_save_loss}/training_log.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
        pass

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def predict_step(self, batch, batch_idx):
        """
        The forward function takes in an image and returns the reconstructed image
        during the prediction step
        """
        x = batch[0]
        batch_hat = self.forward(x)
        return batch_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x = batch[0]
        x_hat = self.forward(x)
        if len(batch) == 1:
            loss = self.criterion(x_hat, x)
        else:
            loss = self.criterion(x_hat, batch[1])
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss, on_epoch=True)
        self.train_loss += float(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss, on_epoch=True)
        self.validation_loss += float(loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss, on_epoch=True)

    def on_train_epoch_end(self):
        current_epoch = self.current_epoch
        num_batches = self.trainer.num_training_batches
        train_loss = self.train_loss
        validation_loss = self.validation_loss
        self.train_loss_summary.append(train_loss / num_batches)
        self.validation_loss_summary.append(validation_loss)
        self.train_loss = 0
        self.validation_loss = 0
        print(
            f"{current_epoch},{train_loss / num_batches},{validation_loss}", flush=True
        )
        # Write to CSV
        with open(f"{self.dir_save_loss}/training_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([current_epoch, train_loss / num_batches, validation_loss])

    def on_validation_epoch_end(self):
        num_batches = self.trainer.num_val_batches[0]  # may be a list[int]
        self.validation_loss = self.validation_loss / num_batches

    def on_train_end(self):
        print("Train process completed", flush=True)
