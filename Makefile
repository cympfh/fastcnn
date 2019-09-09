TAG=fastcnn
GPU_DEVICE=$(shell empty-gpu-device)

build:
	sudo docker build -t $(TAG) .

run-inner:
	python main.py

run: build
	nvidia-smi
	sudo docker run --gpus \"device=$(GPU_DEVICE)\" --rm $(TAG) make run-inner

test-inner:
	python -m isort -rc -c
	python -m mypy --ignore-missing-imports --no-strict-optional .
	python -m pytest -v --color=yes --disable-warnings tests

test: build
	sudo docker run --gpus \"device=$(GPU_DEVICE)\"  --rm $(TAG) make test-inner
