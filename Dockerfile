FROM tensorflow/tensorflow:1.4.0-gpu-py3

RUN apt update -y && apt install -y \
python3-dev \
libsm6 \
libxext6 \
libxrender-dev \
python3-tk

RUN pip3 install \
numpy \
opencv-python \
matplotlib
