#FROM nvcr.io/nvidia/pytorch:20.12-py3

#WORKDIR /workspace


ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.12-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/source_code
RUN pip install --upgrade --no-cache-dir pip \
 && pip install --no-cache-dir tqdm tensorboardX