# [WIP] fastcnn

`fastcnn` is another text classification tool (inspired by `fasttext`).

- using char-level CNN to classify
- not yet `fast`
- implemented with `Keras`

## Usage

We prepare `Dockerfile` and `bin/fastcnn`.
I recommend this style rather than directly running.

If you need `sudo` privilege to run docker, please use properly.

```bash
make build  # build docker image
export PATH=$PWD/bin:$PATH  # add ./bin/fastcnn

fastcnn --help

# sample: binary classification
fastcnn supervised \
    ./samples/en_ja/input \
    --validate ./samples/en_ja/validate \
    --verbose --maxlen 20 --epochs 10 --lr 0.2

# sample: categorical (3 classes) classification
fastcnn supervised \
    ./samples/python_bash_coq/train \
    --validate ./samples/python_bash_coq/valid \
    --verbose --epochs 300 --maxlen 10 --dim 8 --kernel-size 3 --lr 0.3 --clip-norm 2.0
```

## TODO

- [x] supervised command
    - [x] binary classification
    - [x] categorical classification
    - [x] multi-label classification
- [x] predict command
- [ ] test command
    - [ ] Acc
    - [ ] F1/Recall/Precision
- more features
    - [ ] improvement models
