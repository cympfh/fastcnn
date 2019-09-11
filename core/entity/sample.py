from typing import List, NamedTuple


class Sample(NamedTuple):
    """A sample (or instance) for training"""
    labels: List[str]
    data: str
