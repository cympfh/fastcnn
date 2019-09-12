from typing import Optional

import click

from core.optimizer import make_optimizer
from core.batch import BatchSequence
from core.model import make_model
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
@click.option('--epochs', type=int, default=10, help='number of epochs', show_default=True)
@click.option('--batch-size', type=int, default=100, help='size of a batch', show_default=True)
@click.option('--maxlen', type=int, default=100,
              help='The max length (chars) of a sentence', show_default=True)
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
@click.option('--dim', type=int, default=100,
              help='size of sentence vectors', show_default=True)
def supervised(input: click.Path,
               validate: click.Path,
               output: str,
               verbose: bool,
               epochs: int,
               batch_size: int,
               maxlen: int,
               kernel_size: int,
               opt: str,
               lr: Optional[float],
               clip_norm: float,
               dim: int,
               ):

    assert epochs > 0
    assert batch_size > 0
    assert maxlen > 2
    assert kernel_size > 1
    assert lr is None or lr > 0.0
    assert clip_norm > 0.0
    assert dim > 0

    dataset = read(input)
    click.echo(f"{dataset.task}", err=True)
    click.echo(f"#labels={len(dataset.labels)}, "
               f"#samples={len(dataset.samples)}, "
               f"#chars={len(dataset.chars)}", err=True)

    if validate is not None:
        dataset_validate = read(validate)
        click.echo(f"validation with {len(dataset_validate.samples)} samples")

    optimizer = make_optimizer(opt, lr, clip_norm)
    model = make_model(dataset, dim, maxlen, kernel_size, optimizer, verbose=verbose)

    model.fit_generator(
        BatchSequence(dataset, batch_size, maxlen),
        validation_data=(None if validate is None
                         else BatchSequence(dataset_validate, batch_size, maxlen)),
        shuffle=True,
        epochs=epochs,
        verbose=verbose,
        workers=1,
        use_multiprocessing=True)


@cli.command()
def test():
    raise NotImplementedError


@cli.command()
def test_label():
    raise NotImplementedError


@cli.command()
def predict():
    raise NotImplementedError


@cli.command()
def predict_prob():
    raise NotImplementedError


@cli.command()
def print_sentence_vectors():
    raise NotImplementedError


if __name__ == '__main__':
    cli()
