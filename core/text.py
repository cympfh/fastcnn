import numpy

from core.entity import Index

EMPTY = 0
UNK = 1
BOS = 2
EOS = 3


def vectorize(sentence: str,
              chars: Index,
              maxlen: int,
              use_bos: bool = False,
              use_eos: bool = False,
              ) -> numpy.ndarray:
    offset = 2  # unk + empty
    if use_bos:
        offset += 1
    if use_eos:
        offset += 1
    indices = [chars.index(ord(c)) for c in sentence]

    # add offset + (None -> unk)
    indices = [
        i + offset if i is not None else UNK
        for i in indices
    ]

    # add bos + eos
    if use_bos:
        indices = [BOS] + indices
    if use_eos:
        indices = indices + [EOS]

    # padding
    if len(indices) > maxlen:
        indices = indices[:maxlen]
    else:
        indices = indices + [EMPTY] * (maxlen - len(indices))

    return numpy.array(indices, 'i')
