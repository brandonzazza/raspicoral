#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up dependencies for USB TPU..."

# Add Coral Edge TPU repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Install dependencies
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
    libnss3-dev libssl-dev libreadline-dev libffi-dev curl libbz2-dev \
    gdal-bin libgdal-dev python3-gdal libedgetpu1-std

echo "Dependencies installed."

# Setup Python environment
PYTHON_VERSION="3.9.16"
PYTHON_SRC="/usr/src/Python-${PYTHON_VERSION}"
PYTHON_TGZ="Python-${PYTHON_VERSION}.tgz"

echo "Setting up Python environment with Python ${PYTHON_VERSION}..."

cd /usr/src/
sudo curl -O "https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_TGZ}"
sudo tar -xzf "${PYTHON_TGZ}"
cd "Python-${PYTHON_VERSION}"

sudo ./configure --enable-optimizations
sudo make -j "$(nproc)"
sudo make altinstall

cd ~
python3.9 -m venv --system-site-packages coral-env
source coral-env/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install tflite-runtime==2.7.0
pip install "numpy<2"
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
pip install pillow
pip install opencv-python
pip install flask

echo "Python environment setup complete."
echo "USB TPU setup is complete. Activate the environment using: source coral-env/bin/activate"
