import math
from typing import Tuple

import keras.utils
import numpy

from core.entity import Dataset, Task
from core.text import vectorize


class BatchSequence(keras.utils.Sequence):

    def __init__(self, dataset: Dataset, batch_size: int, maxlen: int):
        self.dataset = dataset
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.label_size = len(dataset.labels)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset.samples) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        index_begin = idx * self.batch_size
        index_end = min(len(self.dataset.samples), (idx + 1) * self.batch_size)
        X = [
            vectorize(self.dataset.samples[i].data, self.dataset.chars, self.maxlen)
            for i in range(index_begin, index_end)
        ]
        X = numpy.array(X, dtype='i')
        if self.dataset.task == Task.binary:
            Y = [
                self.dataset.labels.index(self.dataset.samples[i].labels[0])
                for i in range(index_begin, index_end)
            ]
            Y = numpy.array(Y, dtype='f')
        elif self.dataset.task == Task.classify_single:
            Y = [
                self.dataset.labels.index(self.dataset.samples[i].labels[0])
                for i in range(index_begin, index_end)
            ]
            Y = numpy.array(Y, dtype='i')
        else:
            minibatch_size = len(range(index_begin, index_end))
            Y = numpy.zeros((minibatch_size, self.label_size), dtype='f')
            for i in range(index_begin, index_end):
                for label in self.dataset.samples[i].labels:
                    j = self.dataset.labels.index(label)
                    Y[i - index_begin, j] = 1.0
        return X, Y
