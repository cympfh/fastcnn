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
    --maxlen 20 --epochs 10 --lr 0.2

# sample: categorical (3 classes) classification
$ fastcnn supervised \
    ./samples/python_bash_coq/train \
    --validate ./samples/python_bash_coq/valid \
    --epochs 300 \
    --dim 8 \
    --kernel-size 3 \
    --lr 0.3
```

After `fastcnn supervised`,
model parameter file `{out}.h5` and a meta info file `{out}.meta.yml` are generated.
Here, `{out}` is `"out"` by default (You can specify with `-o`).

### test and predict

You can test the trained model by `fastcnn test`.
And `fastcnn predict` shows the prediction.

```bash
$ fastcnn test out ./samples/python_bash_coq/test
Acc@1: 0.8883
__label__bash
- Recall: 0.9103
- Prec: 0.9595
- F1: 0.9342
__label__coq
- Recall: 0.7045
- Prec: 1.0000
- F1: 0.8267
__label__python
- Recall: 1.0000
- Prec: 0.7703
- F1: 0.8702

$ fastcnn predict out ./samples/python_bash_coq/test --show-data | head
__label__bash - discriminate.
__label__coq - done.
__label__coq - exact.
__label__python Example trans_eq_example :
__label__python From mathcomp Require Import all_ssreflect.
__label__python Import ListNotations.
__label__coq Proof.
__label__coq Qed.
__label__python Require Import Arith List Omega ZArith.
__label__coq S n = S m -> n = m.
```

### more detail

```bash
fastcnn --help
fastcnn [subcommand] --help
```
