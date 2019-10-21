TAG=fastcnn:cpu

build:
	docker build -t $(TAG) .

test:
	python -m isort -rc -c
	python -m mypy --ignore-missing-imports --no-strict-optional .
