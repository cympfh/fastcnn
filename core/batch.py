import math
from typing import Tuple

import numpy
from tensorflow.keras import utils

from core.entity import Dataset, Task
from core.text import vectorize


def remove_unknown_label_instances(xs, ys):
    xs_ = []
    ys_ = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        xs_.append(x)
        ys_.append(y)
    return xs_, ys_


class BatchSequence(utils.Sequence):

    def __init__(self, dataset: Dataset, batch_size: int, maxlen: int, label_smoothing=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.label_size = len(dataset.labels)
        self.label_smoothing = label_smoothing or 0.0

    def __len__(self) -> int:
        return math.ceil(len(self.dataset.samples) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        index_begin = idx * self.batch_size
        index_end = min(len(self.dataset.samples), (idx + 1) * self.batch_size)
        X = [
            vectorize(self.dataset.samples[i].data, self.dataset.chars, self.maxlen)
            for i in range(index_begin, index_end)
        ]

        if self.dataset.task == Task.binary:
            Y = [
                self.dataset.labels.index(self.dataset.samples[i].labels[0])
                for i in range(index_begin, index_end)
            ]
            X, Y = remove_unknown_label_instances(X, Y)
            Y = numpy.array(Y, dtype='f')

        elif self.dataset.task == Task.classify_single:
            Y = [
                self.dataset.labels.index(self.dataset.samples[i].labels[0])
                for i in range(index_begin, index_end)
            ]
            X, Y = remove_unknown_label_instances(X, Y)
            Y = numpy.array(Y, dtype='i')

        elif self.dataset.task == Task.classify_multiple:
            minibatch_size = len(range(index_begin, index_end))
            smoothing = self.label_smoothing / self.label_size
            Y = numpy.zeros((minibatch_size, self.label_size), dtype='f') + smoothing
            for i in range(index_begin, index_end):
                m = len(self.dataset.samples[i].labels)
                for label in self.dataset.samples[i].labels:
                    j = self.dataset.labels.index(label)
                    if j is None:
                        continue
                    Y[i - index_begin, j] = 1.0 / m

        X = numpy.array(X, dtype='i')
        return X, Y
