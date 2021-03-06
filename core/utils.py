from collections import Counter
from typing import Dict, List

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


def div(a: float, b: float, default: float = 0) -> float:
    if b != 0 and b != 0.0:
        return a / b
    return default


def f1(prec: float, recall: float) -> float:
    return div(2 * prec * recall, (prec + recall))


def labels_performance(confusion_matrix: List[List[int]]) -> List[Dict[str, float]]:
    n_labels = len(confusion_matrix)
    recalls = [
        div(confusion_matrix[i][i], sum(confusion_matrix[i][j] for j in range(n_labels)))
        for i in range(n_labels)
    ]
    precs = [
        div(confusion_matrix[i][i], sum(confusion_matrix[j][i] for j in range(n_labels)))
        for i in range(n_labels)
    ]
    f1s = [f1(p, r) for p, r in zip(precs, recalls)]
    return [
        {
            'recall': r,
            'prec': p,
            'f1': f1,
        }
        for r, p, f1 in zip(recalls, precs, f1s)
    ]
