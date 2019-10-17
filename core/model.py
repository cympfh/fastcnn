from tensorflow.keras import layers, models, optimizers

from core.entity import Dataset, Task


def load_model(model_file: str) -> models.Model:
    return models.load_model(f"{model_file}.h5")


def make_model(dataset: Dataset,
               dim: int,
               maxlen: int,
               kernel_size: int,
               opt: optimizers.Optimizer,
               verbose: bool = False) -> models.Model:

    voc_size = len(dataset.chars) + 4
    num_labels = len(dataset.labels)
    feature_size = (maxlen - kernel_size + 1) // 2 - 1
    assert feature_size > 0, "kernel_size is too large, maxlen is too short"

    def make_conv_layers(name: str):
        """Input -> Conv -> MaxPool -> Conv -> MaxPool -> Flatten"""
        model = models.Sequential(name=name)
        model.add(layers.Embedding(voc_size, dim, input_length=maxlen))
        model.add(layers.Conv1D(dim * 2, kernel_size,
                                activation='relu'))
        model.add(layers.MaxPool1D(2))
        model.add(layers.Conv1D(dim * 2, 2,
                                activation='relu'))
        model.add(layers.MaxPool1D(feature_size))
        model.add(layers.Flatten())
        model.add(layers.GaussianNoise(0.1 / dim))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(dim))
        return model

    def make_model_binary():
        model = make_conv_layers('binary')
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
        return model

    def make_model_classify_single():
        model = make_conv_layers('classify_single')
        model.add(layers.Dense(num_labels, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
        return model

    def make_model_classify_multiple():
        model = make_conv_layers('classify_multiple')
        model.add(layers.Dense(num_labels, activation='softmax'))
        model.compile(loss='kullback_leibler_divergence', optimizer=opt)
        return model

    if dataset.task == Task.binary:
        model = make_model_binary()
    elif dataset.task == Task.classify_single:
        model = make_model_classify_single()
    else:
        model = make_model_classify_multiple()

    if verbose:
        model.summary()

    return model
