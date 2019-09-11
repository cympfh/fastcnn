# fastcnn

`fastcnn` is another text classification tool (inspired by `fasttext`).

- using char-level CNN to classify
- not yet `fast`
- implemented with `Keras`

## Usage

You can run with docker (the latest docker can use GPU in default).

```bash
make build
make example
```

Or you can run directly

```bash
python ./main.py \
    supervised \
    ./samples/en_ja/input \
    --validate ./samples/en_ja/validate \
    --verbose --maxlen 20 --epochs 20 --lr 0.1
```


## TODO

- [x] binary classification
- [ ] categorical classification
- [ ] multi-label classification
- [ ] improvement models
