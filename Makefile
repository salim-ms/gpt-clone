run-hello-world:
	echo "Running the world"

test-docker-torch:
	docker run -it --gpus all docker.io/pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime nvidia-smi

run-torch-no-dependencies:
	docker run -it --rm --name torch_docker -v $$(pwd):/workspace --gpus all docker.io/pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime bash

build-torch-image:
	docker build -t pytorch-gpu . -f torch.Dockerfile

run-my-torch:
	docker run -it --rm --name torch_docker -v $$(pwd):/workspace --gpus all pytorch-gpu bash