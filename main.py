from typing import List, Optional

import click
import cowsay
import numpy
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.models import Model

from core import utils
from core.batch import BatchSequence
from core.entity import Dataset, Metadata, Task
from core.model import load_model, make_model
from core.optimizer import make_optimizer
from core.read import read

DEBUG = False


@click.group()
@click.option('--debug', is_flag=True, default=False)
def cli(debug):
    global DEBUG
    DEBUG = debug


@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--validate', type=click.Path(exists=True), default=None)
@click.option('--output', '-o', help='output file path')
@click.option('--verbose', is_flag=True, default=False, show_default=True)
@click.option('--stat', is_flag=True, default=False, show_default=True)
@click.option('-e', '--epochs', type=int, default=10, help='number of epochs', show_default=True)
@click.option('--batch-size', type=int, default=100, help='size of a batch', show_default=True)
@click.option('--dim', type=int, default=100,
              help='size of sentence vectors', show_default=True)
@click.option('--maxlen', type=int, default=-1,
              help='max length (chars) of a sentence; If -1, it is estimated',
              show_default=True)
@click.option('--kernel-size', type=int, default=5,
              help='size of a kernel (char window)', show_default=True)
@click.option('--opt', type=click.Choice(['sgd', 'adagrad', 'adam', 'adamax', 'nadam']),
              default='sgd',
              help='optimizer', show_default=True)
@click.option('--lr', type=float, default=None,
              help='learning rate (default value is selected for each optimizers)',
              show_default=True)
@click.option('--clip-norm', type=float, default=2.0,
              help='clip norm of optimizers', show_default=True)
@click.option('--stop-window', type=int, default=-1,
              help='number of epochs of stop window for early-stopping; If -1, disabled',
              show_default=True)
@click.option('--label-smoothing', type=float, default=0.0,
              help='increase values of negative labels')
def supervised(input: click.Path,
               validate: click.Path,
               output: str,
               verbose: bool,
               stat: bool,
               epochs: int,
               batch_size: int,
               dim: int,
               maxlen: int,
               kernel_size: int,
               opt: str,
               lr: Optional[float],
               clip_norm: float,
               stop_window: int,
               label_smoothing: float,
               ):
    """train a supervised classifier"""

    assert epochs > 0
    assert batch_size > 0
    assert maxlen > 2 or maxlen == -1
    assert kernel_size > 1
    assert lr is None or lr > 0.0
    assert clip_norm > 0.0
    assert dim > 0

    dataset = read(input, remove_no_labels=True)

    if maxlen < 0:
        maxlen = max(len(sample.data) for sample in dataset.samples) + 2

    metadata = Metadata(
        dataset.task,
        dataset.labels,
        dataset.chars,
        {
            'dim': dim,
            'maxlen': maxlen,
            'kernel_size': kernel_size,
            'opt': {
                'name': opt,
                'lr': lr,
                'clip_norm': clip_norm
            }
        })
    metadata_file = "out.meta.yml" if output is None else f"{output}.meta.yml"
    metadata.dump(metadata_file)
    cowsay.cow(f"Metadata file saved as {metadata_file}")

    if validate is not None:
        dataset_validate = read(validate, remove_no_labels=True)
        dataset_validate = Dataset(
            dataset.task,
            dataset.labels,
            dataset.chars,
            dataset_validate.samples
        )

    cowsay.cow(f"""
Task: {dataset.task.name}, #labels={len(dataset.labels)}, #chars={len(dataset.chars)}
- Training with {len(dataset.samples)} samples
- {"No validation"
   if validate is None else
   f"Validation with {len(dataset_validate.samples)} samples"}
""")

    if stat:
        cowsay.cow(f"Training data stat:\n {utils.stat(dataset)}")
        if validate:
            cowsay.cow(f"Validation data stat:\n {utils.stat(dataset_validate)}")

    optimizer = make_optimizer(opt, lr, clip_norm)
    model = make_model(dataset, dim, maxlen, kernel_size, optimizer, verbose=verbose)

    callbacks: List[Callback] = []
    if validate and stop_window > 0:
        callbacks += [EarlyStopping(monitor='val_acc', patience=stop_window)]

    model.fit_generator(
        BatchSequence(dataset, batch_size, maxlen, label_smoothing),
        validation_data=(None if validate is None
                         else BatchSequence(dataset_validate, batch_size, maxlen)),
        shuffle=True,
        callbacks=callbacks,
        epochs=epochs,
        verbose=2,
        workers=1,
        use_multiprocessing=True)

    output = output or 'out.h5'
    model.save(output)
    cowsay.cow(f"Model saved as {output}")


@cli.command()
@click.argument('model-file', type=str)
@click.argument('test-data', type=click.Path(exists=True))
@click.option('-k', type=int, default=1,
              help='predict top k labels (for class classification)',
              show_default=True)
@click.option('-t', type=float, default=0.0,
              help='probability threshold (for class classification)',
              show_default=True)
@click.option('--batch-size', type=int, default=100, help='size of a batch', show_default=True)
@click.option('--stat', is_flag=True, default=False, show_default=True)
def test(
        model_file: str,
        test_data: str,
        k: int,
        t: float,
        batch_size: int,
        stat: bool,
        ):
    """evaluate a supervised classifier"""
    # metadata
    metadata_file = f"{model_file}.meta.yml"
    metadata = Metadata.load(metadata_file)
    # load model
    model = load_model(model_file)
    # test data
    dataset_test = read(test_data, remove_no_labels=True)
    dataset_test = Dataset(
        metadata.task,
        metadata.labels,
        metadata.chars,
        dataset_test.samples,
    )

    if stat:
        cowsay.cow(f"Test data stat:\n {utils.stat(dataset_test)}")

    # prediction
    pred = model.predict_generator(
            BatchSequence(dataset_test, batch_size, metadata.params['maxlen']),
            workers=1,
            use_multiprocessing=True)
    indices = numpy.argsort(-pred)[:, :k]

    n_labels = len(dataset_test.labels)
    confusion_matrix = [
        [0] * n_labels
        for _ in range(n_labels)
    ]

    if metadata.task == Task.binary:
        for i in range(len(pred)):
            y_pred = 1 if pred[i, 0] > 0.5 else 0
            label_true = dataset_test.samples[i].labels[0]
            y_true = metadata.labels.index(label_true)
            confusion_matrix[y_true][y_pred] += 1

        n_sum = sum(confusion_matrix[i][j]
                    for i in range(n_labels)
                    for j in range(n_labels))
        n_acc = sum(confusion_matrix[i][i] for i in range(n_labels))
        print(f"Acc: {n_acc / n_sum:.4f}")

        performance = utils.labels_performance(confusion_matrix)
        for i, perf in enumerate(performance):
            print(metadata.labels[i])
            print(f"Recall: {perf['recall']:.4f}")
            print(f"Prec: {perf['prec']:.4f}")
            print(f"F1: {perf['f1']:.4f}")

    else:
        n_sum = 0
        n_acc = 0
        for i, ind in enumerate(indices):
            preds = [j for j in ind if pred[i, j] > t]
            truth = [metadata.labels.index(label) for label in dataset_test.samples[i].labels]
            n_sum += 1
            if all(j in preds for j in truth):
                n_acc += 1
            for y_pred in preds:
                for y_true in truth:
                    confusion_matrix[y_true][y_pred] += 1

        print(f"Acc@{k}: {n_acc / n_sum:.4f}")

        performance = utils.labels_performance(confusion_matrix)
        for i, perf in enumerate(performance):
            print(metadata.labels[i])
            print(f"- Recall: {perf['recall']:.4f}")
            print(f"- Prec: {perf['prec']:.4f}")
            print(f"- F1: {perf['f1']:.4f}")


@cli.command()
@click.argument('model-file', type=str)
@click.argument('test-data', type=click.Path(exists=True))
@click.option('-k', type=int, default=1,
              help='predict top k labels (for class classification)',
              show_default=True)
@click.option('-t', type=float, default=0.0,
              help='probability threshold', show_default=True)
@click.option('--batch-size', type=int, default=100, help='size of a batch', show_default=True)
@click.option('--stat', is_flag=True, default=False, show_default=True)
@click.option('--show-data', is_flag=True, default=False, show_default=True)
def predict(model_file: click.Path,
            test_data: click.Path,
            k: int,
            t: float,
            batch_size: int,
            stat: bool,
            show_data: bool,
            ):
    """predict most likely labels"""
    # metadata
    metadata_file = f"{model_file}.meta.yml"
    metadata = Metadata.load(metadata_file)
    # load model
    model = load_model(model_file)
    # test data
    dataset_test = read(test_data)
    dataset_test = Dataset(
        metadata.task,
        metadata.labels,
        metadata.chars,
        dataset_test.samples,
    )

    if stat:
        cowsay.cow(f"Test data stat:\n {utils.stat(dataset_test)}")

    # prediction
    pred = model.predict_generator(
            BatchSequence(dataset_test, batch_size, metadata.params['maxlen']),
            workers=1,
            use_multiprocessing=True)

    indices = numpy.argsort(-pred)[:, :k]
    for i, ind in enumerate(indices):
        if metadata.task == Task.binary:
            pred_labels = metadata.labels[0 if pred[i, 0] < 0.5 else 1]
        else:
            pred_labels = ' '.join(metadata.labels[j] for j in ind if pred[i, j] > t)
        if show_data:
            print(pred_labels, dataset_test.samples[i].data)
        else:
            print(pred_labels)


@cli.command()
@click.argument('model-file', type=str)
@click.argument('test-data', type=click.Path(exists=True))
@click.option('-k', type=int, default=1,
              help='predict top k labels', show_default=True)
@click.option('-t', type=float, default=0.0,
              help='probability threshold', show_default=True)
@click.option('--batch-size', type=int, default=100, help='size of a batch', show_default=True)
@click.option('--stat', is_flag=True, default=False, show_default=True)
@click.option('--show-data', is_flag=True, default=False, show_default=True)
def predict_prob(model_file: click.Path,
                 test_data: click.Path,
                 k: int,
                 t: float,
                 batch_size: int,
                 stat: bool,
                 show_data: bool,
                 ):
    """predict most likely labels with probabilities"""
    # metadata
    metadata_file = f"{model_file}.meta.yml"
    metadata = Metadata.load(metadata_file)
    # load model
    model = load_model(model_file)
    # test data
    dataset_test = read(test_data)
    dataset_test = Dataset(
        metadata.task,
        metadata.labels,
        metadata.chars,
        dataset_test.samples,
    )

    if stat:
        cowsay.cow(f"Test data stat:\n {utils.stat(dataset_test)}")

    # prediction
    pred = model.predict_generator(
            BatchSequence(dataset_test, batch_size, metadata.params['maxlen']),
            workers=1,
            use_multiprocessing=True)
    indices = numpy.argsort(-pred)[:, :k]
    for i, ind in enumerate(indices):
        if metadata.task == Task.binary:
            pred_label = metadata.labels[0 if pred[i, 0] < 0.5 else 1]
            pred_prob = utils.float4(max(pred[i, 0], 1.0 - pred[i, 0]))
            pred_msg = f"{pred_label} {pred_prob}"
        else:
            pred_labels = [metadata.labels[j] for j in ind if pred[i, j] > t]
            pred_probs = [utils.float4(pred[i, j]) for j in ind if pred[i, j] > t]
            pred_msg = ' '.join(str(x) for pair in zip(pred_labels, pred_probs) for x in pair)
        if show_data:
            print(pred_msg, dataset_test.samples[i].data)
        else:
            print(pred_msg)


@cli.command()
@click.argument('model-file', type=str)
@click.argument('data-file', type=click.Path(exists=True))
@click.option('--batch-size', type=int, default=100, help='size of a batch', show_default=True)
def print_sentence_vectors(
        model_file: str,
        data_file: str,
        batch_size: int,
        ):
    """print sentence vectors given a trained model"""
    # metadata
    metadata_file = f"{model_file}.meta.yml"
    metadata = Metadata.load(metadata_file)
    # load model
    model = load_model(model_file)
    feature_model = Model(inputs=model.input,
                          outputs=model.get_layer('dense_1').output)
    # data
    dataset = read(data_file)
    pred = feature_model.predict_generator(
            BatchSequence(dataset, batch_size, metadata.params['maxlen']),
            workers=1,
            use_multiprocessing=True)

    for v in pred:
        print(' '.join(str(utils.float4(x)) for x in v))


if __name__ == '__main__':
    cli()
