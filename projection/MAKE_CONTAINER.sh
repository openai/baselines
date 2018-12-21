mode=$1

if [ ${mode} = 0 ];
then
    DIR_CODE="/home/yoshimura/code708/synergy"
    DIR_DATA="/home/yoshimura/code708/dataStore"
    NAME="synergy"
    IMAGE="yoshimura/synergy:v4.5.0"
    PORT_NOTE=7088
    PORT_TFB=7086
    
    docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
	   -v ${DIR_CODE}:/root/work \
	   -v ${DIR_DATA}:/root/dataStore \
	   --name ${NAME}  \
	   -p ${PORT_NOTE}:8888 -p ${PORT_TFB}:6006 \
	   -it ${IMAGE} jupyter notebook --allow-root --ip 0.0.0.0
fi
