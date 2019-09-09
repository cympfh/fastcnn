import click

DEBUG = False


@click.group()
@click.option('--debug', is_flag=True, default=False)
def cli(debug):
    global DEBUG
    DEBUG = debug


@cli.command()
def supervised():
    raise NotImplementedError


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
