from typing import Optional

import click
import cowsay
import numpy
from keras.callbacks import EarlyStopping
from keras.models import load_model

import core.utils as utils
from core.batch import BatchSequence
from core.entity import Dataset, Metadata
from core.model import make_model
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
              help='max length (chars) of a sentence; If -1, it is decided by the maximum in training data',
              show_default=True)
@click.option('--kernel-size', type=int, default=5,
              help='size of a kernel (char window)', show_default=True)
@click.option('--opt', type=click.Choice(['sgd', 'adagrad', 'adam', 'adamax', 'nadam']),
              default='sgd',
              help='optimizer', show_default=True)
@click.option('--lr', type=float, default=None,
              help='learning rate (default value is selected for each optimizers)',
              show_default=True)
@click.option('--clip-norm', type=float, default=1.0,
              help='clip norm of optimizers', show_default=True)
@click.option('--stop-window', type=int, default=4,
              help='number of epochs of stop window for early-stopping', show_default=True)
@click.option('--dis-es', is_flag=True, default=False,
              help='disable early-stopping', show_default=True)
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
               dis_es: bool,
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
        dataset_validate = read(validate)
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

    callbacks = []
    if validate and not dis_es:
        callbacks += [EarlyStopping(monitor='val_acc', patience=stop_window)]

    model.fit_generator(
        BatchSequence(dataset, batch_size, maxlen),
        validation_data=(None if validate is None
                         else BatchSequence(dataset_validate, batch_size, maxlen)),
        shuffle=True,
        callbacks=callbacks,
        epochs=epochs,
        verbose=2 if verbose else 0,
        workers=1,
        use_multiprocessing=True)

    output = output or 'out.h5'
    model.save(output)
    cowsay.cow(f"Model saved as {output}")


@cli.command()
def test():
    """evaluate a supervised classifier"""
    raise NotImplementedError


@cli.command()
def test_label():
    """print labels with precision and recall scores"""
    raise NotImplementedError


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
def predict(model_file: click.Path,
            test_data: click.Path,
            k: int,
            t: float,
            batch_size: int,
            stat: bool,
            show_data: bool,
            ):
    """predict most likely labels"""
    metadata_file = f"{model_file}.meta.yml"
    metadata = Metadata.load(metadata_file)
    # load model
    model = load_model(f"{model_file}.h5")
    # test data
    dataset_test = read(test_data)
    dataset_test = Dataset(
        metadata.task,
        metadata.labels,
        metadata.chars,
        dataset_test.samples,
    )

    if stat:
        utils.stat(dataset_test)

    # prediction
    pred = model.predict_generator(
            BatchSequence(dataset_test, batch_size, metadata.params['maxlen']),
            workers=1,
            use_multiprocessing=True)
    indices = numpy.argsort(-pred)[:, :k]
    for i, ind in enumerate(indices):
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
    metadata_file = f"{model_file}.meta.yml"
    metadata = Metadata.load(metadata_file)
    # load model
    model = load_model(f"{model_file}.h5")
    # test data
    dataset_test = read(test_data)
    dataset_test = Dataset(
        metadata.task,
        metadata.labels,
        metadata.chars,
        dataset_test.samples,
    )

    if stat:
        utils.stat(dataset_test)

    def float4(x):
        return round(float(x) * 10000) / 10000

    # prediction
    pred = model.predict_generator(
            BatchSequence(dataset_test, batch_size, metadata.params['maxlen']),
            workers=1,
            use_multiprocessing=True)
    indices = numpy.argsort(-pred)[:, :k]
    for i, ind in enumerate(indices):
        pred_labels = [metadata.labels[j] for j in ind if pred[i, j] > t]
        pred_probs = [float4(pred[i, j]) for j in ind if pred[i, j] > t]
        pred_msg = ' '.join(str(x) for pair in zip(pred_labels, pred_probs) for x in pair)
        if show_data:
            print(pred_msg, dataset_test.samples[i].data)
        else:
            print(pred_msg)


@cli.command()
def print_sentence_vectors():
    """print sentence vectors given a trained model"""
    raise NotImplementedError


if __name__ == '__main__':
    cli()
