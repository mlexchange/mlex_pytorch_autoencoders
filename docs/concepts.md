# Concepts

## Training with pytorch_autoencoder
You can train an assortment of neural networks under different conditions according to the
definition of the following parameters:

### Data Augmentation
* Target width: Width in pixels to which the image will be resized.
* Target height: Height in pixels to which the image will be resized.
* Horizontal Flip Probability: Probability of random horizontal flip.
* Vertical Flip Probability: Probability of random vertical flip.
* Brightness: How much to jitter brightness.
* Contrast: How much to jitter contrast.
* Saturation: How much to jitter saturation.
* Hue: How much to jitter hue.

Further information can be found [here](https://pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html).

### Training setup
* Latent Dimension: Dimension size of latent space (Lx1).
* Shuffle: Shuffle dataset.
* Batch Size: The number of images in a batch.
* Validation Percentage: Percentage of training images that should be used for validation.
* Base Channel Size: Size of the base channel in the autoencoder.
* Depth: Number of instances where the image size is decreased and the number of channels is
increased per side in the network architecture.
* Optimizer: A specific implementation of the gradient descent algorithm.
* Criterion.
* Learning Rate: A scalar used to train a model via gradient descent.
* Number of Epochs: An epoch is a full training pass over the entire dataset such that 
each image has been seen once.
* Seed: Initialization reference for the pseudo-random number generator. Set up this value 
for the reproduction of the results.

## Output
The output of the training step is the trained autoencoder.

## Prediction with pytorch_autoencoder
To predict the reconstructed images of a given testing dataset, you can define the following 
parameters:

### Testing setup
* Batch Size: The number of images in a batch.
* Seed: Initialization reference for the pseudo-random number generator. Set up this value 
for the reproduction of the results.

Similarly to the training step, this approach will resize your dataset to the previously selected target width and height.

## Output
The output of the prediction step is `f_vectors.parquet` with the feature vectors extracted from the testing dataset, and their corresponding reconstructed images at the output of the autoencoder.