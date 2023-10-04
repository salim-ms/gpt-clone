FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel


ARG USERNAME=torchuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND noninteractive

RUN  apt-get update \
  && apt-get install -y wget ffmpeg libsm6 libxext6

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN chown -R $USERNAME:$USER_GID /workspace

USER $USERNAME

RUN export PYTHONPATH=/workspace




