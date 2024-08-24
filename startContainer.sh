#!/bin/bash

sudo docker run -it --rm --gpus all --runtime=nvidia --name tf_container -p 8888:8888 -v "$PWD:/tmp" -w /tmp tf_image
