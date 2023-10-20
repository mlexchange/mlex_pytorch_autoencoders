TAG 			:= 0.1
USER 			:= mlexchange1
PROJECT			:= autoencoder-pytorch

IMG_WEB_SVC    	:= ${USER}/${PROJECT}:${TAG}
IMG_WEB_SVC_JYP := ${USER}/${PROJECT_JYP}:${TAG}
ID_USER			:= ${shell id -u}
ID_GROUP		:= ${shell id -g}
DATA_PATH		:= ${PWD}/data

.PHONY:

test:
	echo ${IMG_WEB_SVC}
	echo ${TAG}
	echo ${PROJECT}
	echo ${PROJECT}:${TAG}
	echo ${ID_USER}

build_docker: 
	docker build -t ${IMG_WEB_SVC} -f ./docker/Dockerfile .

build_docker_arm64: 
	docker build -t ${IMG_WEB_SVC} -f ./docker/Dockerfile_arm64 .

run_docker:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} -it --gpus all -v ${PWD}:/app/work/ -v ${DATA_PATH}:/app/work/data -p 8888:8888 ${IMG_WEB_SVC}

train_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} -it --gpus all -v ${DATA_PATH}:/app/work/data ${IMG_WEB_SVC} python3 src/train_model.py data/mixed_small_32x32.npz data/out/single '{"target_width": 32, "target_height": 32, "shuffle": true, "batch_size": 32, "val_pct": 20, "latent_dim": 16,  "base_channel_size": 32, "num_epochs": 1, "optimizer": "Adam", "criterion": "MSELoss", "learning_rate": 0.0001, "seed": 32548}' 

evaluate_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} -it --gpus all -v ${DATA_PATH}:/app/work/data ${IMG_WEB_SVC} python3 src/latent_space_evaluation.py data/mixed_small_32x32.npz data/out/swipe '{"target_width": 32, "target_height": 32, "shuffle": true, "batch_size": 32, "val_pct": 20, "latent_dim": [16,20,32],  "base_channel_size": 32, "num_epochs": 1, "optimizer": "Adam", "criterion": "MSELoss", "learning_rate": 0.0001, "seed": 32548}'

predict_example:
	docker run -u ${ID_USER $USER}:${ID_GROUP $USER} -it --gpus all -v ${DATA_PATH}:/app/work/data ${IMG_WEB_SVC} python3 src/predict_model.py data/mixed_small_32x32.npz data/out/single data/out/results '{"target_width": 32, "target_height": 32, "batch_size": 32, "seed": 32548}'

clean: 
	find -name "*~" -delete
	-rm .python_history
	-rm -rf .config
	-rm -rf .cache

push_docker:
	docker push ${IMG_WEB_SVC}