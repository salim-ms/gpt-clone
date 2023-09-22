FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

ARG USERNAME=torchuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN  apt-get update \
  && apt-get install -y wget ffmpeg libsm6 libxext6

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

RUN chown -R $USERNAME:$USER_GID /workspace

RUN export PYTHONPATH=/workspace

USER $USERNAME




