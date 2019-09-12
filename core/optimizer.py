from typing import Optional

import keras.optimizers


def make_optimizer(name: str, lr: Optional[float], clipnorm: float) -> keras.optimizers.Optimizer:

    if name == 'sgd':
        lr = lr or 0.01
        return keras.optimizers.SGD(lr=lr, clipnorm=clipnorm)
    elif name == 'adagrad':
        lr = lr or 0.01
        return keras.optimizers.Adagrad(lr=lr, clipnorm=clipnorm)
    elif name == 'adam':
        lr = lr or 0.001
        return keras.optimizers.Adam(lr=lr, clipnorm=clipnorm)
    elif name == 'adamax':
        lr = lr or 0.001
        return keras.optimizers.Adamax(lr=lr, clipnorm=clipnorm)
    elif name == 'nadam':
        lr = lr or 0.001
        return keras.optimizers.Nadam(lr=lr, clipnorm=clipnorm)
    else:
        raise NotImplementedError
