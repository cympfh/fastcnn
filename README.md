# fastcnn

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

# toy sample
fastcnn supervised \
    ./samples/en_ja/input \
    --validate ./samples/en_ja/validate \
    --verbose --maxlen 20 --epochs 20 --lr 0.1
```

## TODO

- [x] binary classification
- [ ] categorical classification
- [ ] multi-label classification
- [ ] improvement models
