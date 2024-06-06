from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DataType(str, Enum):
    tiled = "tiled"
    file = "file"


class IOParameters(BaseModel):
    data_uris: List[str] = Field(description="directory containing the data")
    data_type: DataType = Field(description="type of data")
    root_uri: str = Field(description="root URI containing the data")
    model_dir: str = Field(description="directory containing the model")
    output_dir: str = Field(
        description="directory to create the inference results folder"
    )
    uid_save: str = Field(description="uid to save models, metrics and etc")
    uid_retrieve: Optional[str] = Field(
        description="optional, uid to retrieve models for inference"
    )
    data_tiled_api_key: Optional[str] = Field(description="API key for data tiled")


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
    target_width: int = Field(description="data target width")
    target_height: int = Field(description="data target height")
    horz_flip_prob: Optional[float] = Field(
        description="probability of the image being flipped \
                                            horizontally"
    )
    vert_flip_prob: Optional[float] = Field(
        description="probability of the image being flipped \
                                            vertically"
    )
    brightness: Optional[float] = Field(
        description="how much to jitter brightness. brightness_factor \
                                        is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]"
    )
    contrast: Optional[float] = Field(
        description="how much to jitter contrast. contrast_factor is \
                                      chosen uniformly from [max(0, 1 - contrast), 1 + contrast]."
    )
    saturation: Optional[float] = Field(
        description="how much to jitter saturation. saturation_factor \
                                        is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]. "
    )
    hue: Optional[float] = Field(
        description="how much to jitter hue. hue_factor is chosen uniformly \
                                 from [-hue, hue]. Should have 0<= hue <= 0.5 or -0.5 <= min <= max \
                                 <= 0.5. To jitter hue, the pixel values of the input image has to \
                                 be non-negative for conversion to HSV space."
    )
    augm_invariant: Optional[bool] = Field(
        description="Ground truth changes (or not) according to \
                                           selected transformations"
    )
    log: Optional[bool] = Field(description="log information")
    detector_name: Optional[str] = Field(description="detector name")


class TuningParameters(DataAugmentation):
    shuffle: bool = Field(description="shuffle data")
    batch_size: int = Field(description="batch size")
    val_pct: int = Field(description="validation percentage")
    num_epochs: int = Field(description="number of epochs")
    optimizer: Optimizer
    criterion: Criterion
    gamma: float = Field(description="Multiplicative factor of learning rate decay")
    step_size: int = Field(description="Period of learning rate decay")
    learning_rate: float = Field(description="learning rate")
    seed: Optional[int] = Field(description="random seed")
    num_workers: Optional[int] = Field(description="number of workers")


class TrainingParameters(TuningParameters):
    latent_dim: int = Field(description="latent space dimension")
    depth: int = Field(description="Network depth")
    base_channel_size: int = Field(description="number of base channels")


class EvaluationParameters(TrainingParameters):
    latent_dim: List[int] = Field(description="list of latent space dimensions")


class InferenceParameters(DataAugmentation):
    batch_size: int = Field(description="batch size")
    seed: Optional[int] = Field(description="random seed")
    num_workers: Optional[int] = Field(description="number of workers")
