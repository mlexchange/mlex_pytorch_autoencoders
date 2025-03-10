<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">

# How To Guide

This repository provides an machine learning (ML)-framework to train and test convolutional autoencoders for latent space exploration.

## Data Format
Currently, this ML algorithm supports directory based data definition. The supported image formats are: TIFF, TIF, JPG, JPEG, and PNG. To feed in your data, please arrange a parquet file with the following information:

```
                         uri  type
0     /data/path/image_0.png  file
1     /data/path/image_1.png  file
2     /data/path/image_2.png  file
...                      ...   ...
x     /data/path/image_x.png  file
```
Saving command:

```
df.to_parquet('path/to/data_info.parquet', engine='pyarrow')
```

## Installation

### Using Docker 
You can use the docker image provided in this repository with the following steps:

1. Open a new terminal window, and download the docker image with the command:
   ```
   docker pull ghcr.io/mlexchange/mlex_pytorch_autoencoders:main
   ```

2. Once downloaded, you can start a docker container by running the following command in terminal:
   ```
   docker run -it --gpus all -v /path/to/data:/app/work/data ghcr.io/mlexchange/mlex_pytorch_autoencoders:main bash
   ```
   Make sure to update *-v /path/to/data:/app/work/data* according to the folder you'd like to mount to your docker container. Further information about volume mounting can be found [here](https://docs.docker.com/storage/volumes/).
   
   Please note that the [flag](https://docs.docker.com/config/containers/resource_constraints/#expose-gpus-for-use) *--gpus all* is only compatible with NVIDIA drivers. If your set up does not have this hardware, remove the flag before executing the command.

3. You have now set up your docker container! You should be able to see something like:
   ```bash
   root@container_id:/app/work#
   ```
   where *container_id* corresponds to the ID of your docker container.

4. To start a **training** process, execute the following:
   ```bash
   root@container_id:/app/work# python3 src/train_model.py -d /data/data_info.parquet -o /data/output_folder -p {"target_width": 32, "target_height": 32, "shuffle": true, "batch_size": 32, "val_pct": 20, "latent_dim": 16, "depth": 3, "base_channel_size": 32, "num_epochs": 5, "optimizer": "AdamW", "criterion": "MSELoss", "learning_rate": 0.01, "seed": 32548}
   ```
   Make sure to update your data path *-d data/data_info.parquet*, output path *-o /data/output_folder* and list of parameters *-p* accordingly.

5. To start a **prediction** process, execute the following:
   ```bash
   root@container_id:/app/work# python3 src/predict_model.py -d /data/data_info.parquet -m /data/model_folder -o /data/output_folder -p {"target_width": 32, "target_height": 32, "batch_size": 32, "seed": 32548}
   ```
   Similarly, make sure to update your data path *-d data/data_info.parquet*, model path *-m /data/model_folder*, output path *-o /data/output_folder* and list of parameters *-p* accordingly.

## Training
To train a model, please follow the following steps:

1. Choose your dataset.
   1. As a standalone application: Click on "Open File Manager", and choose your dataset.
   2. From [Label Maker](https://github.com/mlexchange/mlex_dash_labelmaker_demo): The 
   dataset you uploaded in Label Maker should be visible in Data Clinic by default at start-up.
   When selecting a different dataset in Label Maker after start-up, you can refresh the 
   dataset in Data Clinic by clicking "Refresh Project".
2. Choose "Model Training" in Actions.
3. Modify the [training parameters](./concepts.md) as needed.
4. Click Execute.
5. Choose the computing resources that should be used for this task. Please note that 
these should not exceed the constraints defined in the [computing API](https://github.com/mlexchange/mlex_computing_api).
Recommended values: CPU - 4 and GPU - 0. Click "Submit".
6. The training job has been successfully submitted! You can check the progress of this
job in the "List of Jobs", where you can select the corresponding row to display the loss
plot in real-time. Additionally, you can check the logs and parameters of each job by 
clicking on it's corresponding cells.

## Testing
To test a mode, please follow the following steps:

1. Choose your dataset.
   1. As a standalone application: Click on "Open File Manager", and choose your dataset.
   2. From [Label Maker](https://github.com/mlexchange/mlex_dash_labelmaker_demo): The 
   dataset you uploaded in Label Maker should be visible in Data Clinic by default at start-up.
   When selecting a different dataset in Label Maker after start-up, you can refresh the 
   dataset in Data Clinic by clicking "Refresh Project".
2. Choose "Test Prediction using Model" in Actions.
3. Modify the [testing parameters](./concepts.md) as needed.
4. Choose a trained model from the "List of Jobs".
5. Click Execute.
6. Choose the computing resources that should be used for this task. Please note that 
these should not exceed the constraints defined in the [computing API](https://github.com/mlexchange/mlex_computing_api).
Recommended values: CPU - 4 and GPU - 0. Click "Submit".
7. The testing job has been successfully submitted! You can check the progress of this
job in the "List of Jobs", where you can select the corresponding row to display the 
reconstructed images in real-time. Additionally, you can check the logs and parameters 
of each job by clicking on it's corresponding cells.
