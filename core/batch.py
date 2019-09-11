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

    def __len__(self) -> int:
        return math.ceil(len(self.dataset.samples) / self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[numpy.ndarray, numpy.ndarray]:
        index_begin = idx * self.batch_size
        index_end = min(len(self.dataset.samples), (idx + 1) * self.batch_size)
        X = [
            vectorize(self.dataset.samples[i].data, self.dataset.chars, self.maxlen)
            for i in range(index_begin, index_end)
        ]
        Y = [
            self.dataset.labels.index(self.dataset.samples[i].labels[0])
            for i in range(index_begin, index_end)
        ]
        X = numpy.array(X, dtype='i')
        if self.dataset.task == Task.binary:
            Y = numpy.array(Y, dtype='f')
        else:
            Y = numpy.array(Y, dtype='i')
        return X, Y
