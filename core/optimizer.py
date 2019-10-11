from typing import Optional

from tensorflow.keras import optimizers


def make_optimizer(name: str, lr: Optional[float], clipnorm: float) -> optimizers.Optimizer:

    if name == 'sgd':
        lr = lr or 0.01
        return optimizers.SGD(lr=lr, clipnorm=clipnorm)
    elif name == 'adagrad':
        lr = lr or 0.01
        return optimizers.Adagrad(lr=lr, clipnorm=clipnorm)
    elif name == 'adam':
        lr = lr or 0.001
        return optimizers.Adam(lr=lr, clipnorm=clipnorm)
    elif name == 'adamax':
        lr = lr or 0.001
        return optimizers.Adamax(lr=lr, clipnorm=clipnorm)
    elif name == 'nadam':
        lr = lr or 0.001
        return optimizers.Nadam(lr=lr, clipnorm=clipnorm)
    else:
        raise NotImplementedError
