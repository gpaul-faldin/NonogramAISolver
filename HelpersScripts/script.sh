#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}--------------------------------------"
echo -e "      Docker and Tensorflow(2.10.1) Installer         "
echo -e "--------------------------------------${NC}"
echo ""

progress() {
  echo -e "${GREEN}[+]${NC} $1"
}

error() {
  echo -e "${RED}[-]${NC} Error: $1" >&2
  if [ "$2" = true ]; then
    exit 1
  fi
}

help() {
    echo "Usage: $0 [-noautostart]"
    echo ""
    echo "Arguments:"
    echo "  -noautostart       If specified, disable the start of a container at the end of the installation"
    echo "  -help            Well you are here arent you ?"
    echo ""
    exit 0
}

if [[ "$1" == "-help" ]]; then
    help
fi

if [[ $(basename "$PWD") =~ [A-Z] ]]; then
    error "Your current directory shouldn't contain uppercase"
    error "It will cause issue for linking the current directory to the docker container"
    exit 1
fi

progress "Installing Docker"

sudo apt-get update > /dev/null || error "Failed to update apt repositories"
sudo apt-get install -y ca-certificates curl > /dev/null || error "Failed to install prerequisites for Docker"
sudo install -m 0755 -d /etc/apt/keyrings > /dev/null || error "Failed to create directory for Docker GPG key"
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc > /dev/null || error "Failed to download Docker GPG key"
sudo chmod a+r /etc/apt/keyrings/docker.asc > /dev/null || error "Failed to change permission for Docker GPG key"
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null || error "Failed to add Docker repository"
sudo apt-get update > /dev/null || error "Failed to update apt repositories"
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin > /dev/null || error "Failed to install Docker"
sudo usermod -aG docker $USER > /dev/null || error "Failed to add current user to docker group"

progress "Docker installed!"
progress "Installing Nvidia Toolkit"

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list  > /dev/null
sudo apt-get update > /dev/null || error "Failed to update apt repositories"
sudo apt-get install -y nvidia-container-toolkit > /dev/null || error "Failed to install Nvidia Toolkit"
sudo nvidia-ctk runtime configure --runtime=docker > /dev/null || error "Failed to create the nvidia docker runtime"
sudo systemctl restart docker || error "Failed to restart Docker service"

progress "Nvidia Toolkit installed!"
progress "Pulling and building docker image (this step can take a long time depending on your connection)"
sudo docker build -t tf_image . > /dev/null
progress "Image retrieved"

if [[ "$1" == "-noautostart" ]]; then
  exit 0;
fi

echo -e "${GREEN}--------------------------------------"
echo -e "      Starting TensorFlow container         "
echo -e "      container name: tf_container         "
echo -e "      Your current repository will be linked inside the container         "
echo -e "--------------------------------------${NC}"
echo ""
sudo docker run -itd --rm --gpus all --runtime=nvidia --name tf_container -v $PWD:/tmp -w /tmp tensorflow/tensorflow:latest-gpu > /dev/null
sudo docker exec -it tf_container python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))" | grep 'Created device' | awk '{print $0}'
progress "If you have a GPU you should see it on the line above"

progress "Done!"
newgrp docker || error "Failed to apply group changes"