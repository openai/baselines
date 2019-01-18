# Projection Network with Chainer
中枢神経系 (action) からPre-moter Nueronへの投射を行うネットワークの学習.


## Installation
### Requirements
+ chainer
+ cupy

### Docker Setup
Fist, build docker image with this command.

```
$ docker build -t synergy/chainer:v4.5.0 .
```

And then make container,

```
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${GPU} \
	-v ${DIR_code}:/root/work \
	-v ${DIR_DATA}:/root/dataStore \
	--name ${NAME}  \
	-p ${PORT_NOTE} -p ${PORT_TFB} \
	-it ${IMAGE} jupyter notebook --allow-root --ip 0.0.0.0
```

## Data Preparation
### Resampling
Use `utils/make_inputs.py`. After this, split data for`train/val/test` by yourself.


```
python3 make_inputs.py \
    --path-in  /root/dataStore/grasp_v1/episodes \
    --path-out /root/dataStore/grasp_v1/Inputs
```

## Training
note: Set correct paths!


```
python3 run.py TRAIN \
	--path-data-train /root/dataStore/grasp_v1/Inputs/train \
	--path-data-val   /root/dataStore/grasp_v1/Inputs/val \
	--path-model      /root/dataStore/grasp_v1/Log/ChainerDenseNet.model \
	--path-log        /root/dataStore/grasp_v1/Log/ \
	--gpu             0 \
	--batch-size      64 \
	--epoch           10

```



## Prediction (Generation)
TBA
