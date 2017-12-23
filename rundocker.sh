#!/usr/bin/env bash
DIR=`pwd`
nvidia-docker run \
    --network host \
	-v ${DIR}:/workspace \
	-it wenbodut/dncnn bash

