DIR=`pwd`
nvidia-docker run \
	-v ${DIR}:/workspace \
	-it dncnn
