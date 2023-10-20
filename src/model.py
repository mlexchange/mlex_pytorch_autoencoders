from enum import Enum
from collections import OrderedDict

from PIL import Image
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torchvision import transforms


class Optimizer(str, Enum):
    adadelta = "Adadelta"
    adagrad = "Adagrad"
    adam = "Adam"
    adamw = "AdamW"
    sparseadam = "SparseAdam"
    adamax = "Adamax"
    asgd = "ASGD"
    lbfgs = "LBFGS"
    rmsprop = "RMSprop"
    rprop = "Rprop"
    sgd = "SGD"


class Criterion(str, Enum):
    l1loss = "L1Loss"
    mseloss = "MSELoss"
    crossentropyloss = "CrossEntropyLoss"
    ctcloss = "CTCLoss"
    nllloss = "NLLLoss"
    poissonnllloss = "PoissonNLLLoss"
    gaussiannllloss = "GaussianNLLLoss"
    kldivloss = "KLDivLoss"
    bceloss = "BCELoss"
    bcewithlogitsloss = "BCEWithLogitsLoss"
    marginrankingloss = "MarginRankingLoss"
    hingeembeddingloss = "HingeEnbeddingLoss"
    multilabelmarginloss = "MultiLabelMarginLoss"
    huberloss = "HuberLoss"
    smoothl1loss = "SmoothL1Loss"
    softmarginloss = "SoftMarginLoss"
    multilabelsoftmarginloss = "MutiLabelSoftMarginLoss"
    cosineembeddingloss = "CosineEmbeddingLoss"
    multimarginloss = "MultiMarginLoss"
    tripletmarginloss = "TripletMarginLoss"
    tripletmarginwithdistanceloss = "TripletMarginWithDistanceLoss"


class DataAugmentation(BaseModel):
    target_width: int = Field(description='data target width')
    target_height: int = Field(description='data target height')
    horz_flip_prob: Optional[float] = Field(description='probability of the image being flipped \
                                            horizontally')
    vert_flip_prob: Optional[float] = Field(description='probability of the image being flipped \
                                            vertically')
    brightness: Optional[float] = Field(description='how much to jitter brightness. brightness_factor \
                                        is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]')
    contrast: Optional[float] = Field(description='how much to jitter contrast. contrast_factor is \
                                      chosen uniformly from [max(0, 1 - contrast), 1 + contrast].')
    saturation: Optional[float] = Field(description='how much to jitter saturation. saturation_factor \
                                        is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]. ')
    hue: Optional[float] = Field(description='how much to jitter hue. hue_factor is chosen uniformly \
                                 from [-hue, hue]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max \
                                 <= 0.5. To jitter hue, the pixel values of the input image has to \
                                 be non-negative for conversion to HSV space.')
    data_key: Optional[str] = Field(description='keyword for data in NPZ')


class TrainingParameters(DataAugmentation):
    shuffle: bool = Field(description='shuffle data')
    batch_size: int = Field(description='batch size')
    val_pct: int = Field(description='validation percentage')
    latent_dim: int = Field(description='latent space dimension')
    depth: int = Field(description='Network depth')
    base_channel_size: int = Field(description='number of base channels')
    num_epochs: int = Field(description='number of epochs')
    optimizer: Optimizer
    criterion: Criterion
    learning_rate: float = Field(description='learning rate')
    seed: Optional[int] = Field(description='random seed')


class EvaluationParameters(TrainingParameters):
    latent_dim: List[int] = Field(description='list of latent space dimensions')


class TestingParameters(DataAugmentation):
    batch_size: int = Field(description='batch size')
    seed: Optional[int] = Field(description='random seed')


class Encoder(nn.Module):
    def __init__(self,
                 num_input_channels: int,
                 depth: int,
                 base_channel_size: int,
                 width: int,
                 height: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
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
                    (f'conv{2 * layer}', nn.Conv2d(num_input_channels, c_hid, kernel_size=3, 
                                                   padding=1, stride=2)))
            else:
                auto.append((f'conv{2 * layer}',
                             nn.Conv2d((2 ** (layer - 1)) * c_hid, (2 ** layer) * c_hid, 
                                       kernel_size=3, padding=1, stride=2)))
            auto.append((f'act{2 * layer}', act_fn()))
            if layer == depth - 1:
                auto.append(('flat', nn.Flatten()))  # Image grid to single feature vector
                auto.append(('lin', nn.Linear(int(width * height * c_hid / (2 ** (layer + 2))), 
                                              latent_dim)))
            else:
                auto.append((f'conv{2 * layer + 1}',
                             nn.Conv2d((2 ** layer) * c_hid, (2 ** layer) * c_hid, kernel_size=3, 
                                       padding=1)))
                auto.append((f'act{2 * layer + 1}', act_fn()))

        self.net = nn.Sequential(OrderedDict(auto))

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self,
                 num_input_channels: int,
                 depth: int,
                 base_channel_size: int,
                 width: int,
                 height: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
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
            act_fn()
        )
        auto = []
        for layer in reversed(range(depth)):
            if layer == 0:
                auto.append((f'tconv{2 * layer}',
                             nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, 
                                                output_padding=1, padding=1, stride=2)))
                auto.append(('tan',
                             nn.Tanh()))  # The input images is scaled between -1 and 1, 
                                          # hence the output has to be bounded as well
            else:
                auto.append((f'tconv{2 * layer}',
                             nn.ConvTranspose2d((2 ** layer) * c_hid, (2 ** (layer - 1)) * c_hid, 
                                                kernel_size=3, output_padding=1, padding=1, stride=2)))
                auto.append((f'act{2 * layer}', act_fn()))
                auto.append((f'conv{2 * layer + 1}',
                             nn.Conv2d((2 ** (layer - 1)) * c_hid, (2 ** (layer - 1)) * c_hid, 
                                       kernel_size=3, padding=1)))
                auto.append((f'act{2 * layer + 1}', act_fn()))
        self.net = nn.Sequential(OrderedDict(auto))

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, int(self.width / (2 ** (self.depth))), 
                      int(self.height / (2 ** (self.depth))))
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):
    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 depth: int,
                 num_input_channels: int = 3,
                 optimizer: object = Optimizer,
                 criterion: object = Criterion,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 learning_rate: float = 1e-3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        self.optimizer = getattr(optim, optimizer.value)
        self.learning_rate = learning_rate
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
        self.encoder = encoder_class(num_input_channels, depth, base_channel_size, width, height, latent_dim)
        self.decoder = decoder_class(num_input_channels, depth, base_channel_size, width, height, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

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
        x, _ = batch  # We do not need the labels
        batch_hat = self.forward(x)
        return batch_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, on_epoch=True)
        self.train_loss += float(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss, on_epoch=True)
        self.validation_loss += float(loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss, on_epoch=True)

    def on_train_epoch_end(self):
        current_epoch = self.current_epoch
        num_batches = self.trainer.num_training_batches
        train_loss = self.train_loss
        validation_loss = self.validation_loss
        self.train_loss_summary.append(train_loss / num_batches)
        self.validation_loss_summary.append(validation_loss)
        self.train_loss = 0
        self.validation_loss = 0
        print(f'{current_epoch},{train_loss / num_batches},{validation_loss}', flush=True)

    def on_validation_epoch_end(self):
        num_batches = self.trainer.num_val_batches[0]  # may be a list[int]
        self.validation_loss = self.validation_loss / num_batches

    def on_train_end(self):
        print('Train process completed', flush=True)


class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class CustomDirectoryDataset(Dataset):
    def __init__(self, data, transform):
        self.transform = transform
        self.data = data
        self.simple_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert("RGB")
        tensor_image = self.transform(image)
        return (tensor_image, self.simple_transform(image))