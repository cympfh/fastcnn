from typing import NamedTuple, List


class Sample(NamedTuple):
    """A sample (or instance) for training"""
    labels: List[str]
    data: str
