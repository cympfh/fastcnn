from keras import layers, models, optimizers

from core.entity import Dataset, Task


def make_model(dataset: Dataset,
               maxlen: int,
               lr: float,
               verbose: bool = False) -> models.Model:

    voc_size = len(dataset.chars) + 4
    num_labels = len(dataset.labels)

    def make_model_binary():
        model = models.Sequential()
        model.add(layers.Embedding(voc_size, 64, input_length=maxlen))
        model.add(layers.Conv1D(100, 5))
        model.add(layers.MaxPool1D(16))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        opt = optimizers.SGD(lr=lr)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
        return model

    def make_model_classify_single():
        model = models.Sequential()
        model.add(layers.Embedding(voc_size, 64, input_length=maxlen))
        model.add(layers.Conv1D(100, 5))
        model.add(layers.MaxPool1D(16))
        model.add(layers.Flatten())
        model.add(layers.Dense(num_labels, activation='softmax'))
        opt = optimizers.SGD(lr=lr)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
        return model

    def make_model_classify_multiple():
        raise NotImplementedError

    if dataset.task == Task.binary:
        model = make_model_binary()
    elif dataset.task == Task.classify_single:
        model = make_model_classify_single()
    else:
        model = make_model_classify_multiple()

    if verbose:
        model.summary()

    return model
