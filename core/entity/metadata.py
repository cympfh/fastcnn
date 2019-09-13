import yaml

from core.entity.index import Index
from core.entity.task import Task


class Metadata:
    """Task type, Label Index, Chars Index and other Model Parameters"""

    def __init__(self,
                 task: Task,
                 labels: Index,
                 chars: Index,
                 params: dict):
        self.task = task
        self.labels = labels
        self.chars = chars
        self.params = params

    def dump(self, file_path: str):
        with open(file_path, 'w') as f:
            obj = {
                'task': self.task.name,
                'labels': self.labels.data,
                'chars': self.chars.data,
                'params': self.params,
            }
            yaml.dump(obj, f)

    @classmethod
    def load(cls, file_path: str) -> 'Metadata':
        with open(file_path, 'r') as f:
            obj = yaml.load(f)
            return Metadata(
                Task[obj['task']],
                Index(obj['labels']),
                Index(obj['chars']),
                obj['params'])
