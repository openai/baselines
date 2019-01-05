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

## Training
TBA

## Prediction (Generation)
TBA
