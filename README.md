# fastcnn

`fastcnn` is another text classification tool (inspired by `fasttext`).

- using char-level CNN to classify
- not yet `fast`
- implemented with `tensorflow.keras`

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
    --maxlen 20 --epochs 10 --lr 0.2

# sample: categorical (3 classes) classification
fastcnn supervised \
    ./samples/python_bash_coq/train \
    --validate ./samples/python_bash_coq/valid \
    --epochs 300 \
    --dim 8 \
    --kernel-size 3 \
    --lr 0.3
```

