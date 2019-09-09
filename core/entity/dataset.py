from typing import List, NamedTuple

from core.entity.index import Index
from core.entity.sample import Sample
from core.entity.task import Task


class Dataset(NamedTuple):
    task: Task
    labels: Index
    chars: Index
    samples: List[Sample]

    def __repr__(self) -> str:
        return (f"Dataset({self.task}, "
                f"#labels={len(self.labels)}, "
                f"#chars={len(self.chars)}, "
                f"#samples={len(self.samples)})")
