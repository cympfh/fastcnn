# fastcnn

`fastcnn` is another text classification CLI tool (inspired by `fasttext`).

- Char-level CNN to classify
    - supports binary/multi-class/multi-label classification
- Command line tool (you don't need to write Python)
- Not yet `fast`
- Implemented with `tensorflow.keras`

## Build, Install

We prepare `Dockerfile` and `bin/fastcnn`.
I recommend this style rather than directly running.

```bash
make build  # build docker image
export PATH=$PWD/bin:$PATH  # add ./bin/fastcnn
```

If you need `sudo` privilege to run docker, please use properly.

## Usage

### Supervised training

The task type (binary, multi-class or multi-labels classification) are automatically detected. Some parameters also are adjusted.

```bash
# sample: binary classification
$ fastcnn supervised \
    ./samples/en_ja/input \
    --validate ./samples/en_ja/validate \
    --maxlen 20 --epochs 100 --lr 0.2

# sample: categorical (3 classes) classification
$ fastcnn supervised \
    ./samples/python_bash_coq/train \
    --validate ./samples/python_bash_coq/valid \
    --epochs 300 \
    --dim 8 \
    --kernel-size 3 \
    --lr 0.3
```

`fastcnn supervised` generates `{out}.h5` and `{out}.meta.yml` as a model file.
`{out}` is `"out"` in default (this can be specified by `-o`).

### test and predict

`fastcnn test` tests the model.
`fastcnn predict` shows the prediction.

```bash
$ fastcnn test out ./samples/python_bash_coq/test

Acc@1: 0.8156
__label__bash
- Recall: 0.8590
- Prec: 0.9306
- F1: 0.8933
__label__coq
- Recall: 0.5227
- Prec: 0.9583
- F1: 0.6765
__label__python
- Recall: 0.9825
- Prec: 0.6747
- F1: 0.8000

$ fastcnn predict out ./samples/python_bash_coq/test --show-data
__label__python - discriminate.
__label__coq - done.
__label__coq - exact.
__label__python Example trans_eq_example :
__label__python From mathcomp Require Import all_ssreflect.
__label__python Import ListNotations.
__label__coq Proof.
__label__coq Qed.
__label__python Require Import Arith List Omega ZArith.
__label__coq S n = S m -> n = m.
  :
  :
```

### more detail

```bash
fastcnn --help
fastcnn [subcommand] --help
```
