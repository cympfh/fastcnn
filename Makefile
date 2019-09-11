TAG=fastcnn

build:
	docker build -t $(TAG) .

test-inner:
	python -m isort -rc -c
	python -m mypy --ignore-missing-imports --no-strict-optional .
	python -m pytest -v --color=yes --disable-warnings tests

test: build
	sudo docker run --gpus \"device=0\"  --rm $(TAG) make test-inner
