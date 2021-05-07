FROM nvcr.io/nvidia/pytorch:21.04-py3

RUN apt-get update
RUN pip install tensorboardX