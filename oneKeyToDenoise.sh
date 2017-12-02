#!/usr/bin/env bash
DIR=`pwd`
nvidia-docker run \
	-v ${DIR}:/workspace \
	-it dncnn
