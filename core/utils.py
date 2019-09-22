from collections import Counter

from core.entity import Dataset


def stat(dataset: Dataset) -> str:

    ret = ""

    ret += "Labels:\n"
    count = Counter(label
                    for sample in dataset.samples
                    for label in sample.labels)
    for label in count:
        ret += f"- {label} : {count[label]}\n"

    ret += "Sentence Length:\n"
    lens = [len(sample.data) for sample in dataset.samples]
    ret += f"- max : {max(lens)}\n"
    ret += f"- min : {min(lens)}\n"
    ret += f"- avg : {sum(lens) / len(lens)}\n"

    return ret.strip()


def float4(x):
    return round(float(x) * 10000) / 10000
