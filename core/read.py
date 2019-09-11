from typing import List, Set

from core.entity import Dataset, Sample, Task, Index


def make_vocabulary(samples: List[Sample]) -> Index:
    charset = set()
    for sample in samples:
        charset.update(ord(c) for c in sample.data)
    vocab = Index(list(sorted(charset)))
    return vocab


def read(path: str, label_suffix: str = '__label__') -> Dataset:

    # read samples
    samples: List[Sample] = []
    for line in open(path, 'r'):
        line = line.strip()

        # ignore line
        if line == '' or line[0] == '#':
            continue

        fs = line.replace('\t', ' ').split(' ')
        labels: List[str] = []
        data = ''
        for i, f in enumerate(fs):
            if f.startswith(label_suffix):
                labels.append(f)
            else:
                data = ' '.join(fs[i:])
        if len(labels) == 0:
            continue
        samples.append(Sample(labels, data))

    # label set
    labels: Set[str] = set()
    for sample in samples:
        labels.update(sample.labels)

    # label index
    label_index = Index(list(sorted(labels)))

    # char vocabulary
    chars = make_vocabulary(samples)

    # estimate task type
    multiple = False
    for sample in samples:
        if len(sample.labels) > 1:
            multiple = True
            break
    if multiple:
        task = Task.classify_multiple
    else:
        if len(labels) == 2:
            task = Task.binary
        else:
            task = Task.classify_single

    return Dataset(task, label_index, chars, samples)
