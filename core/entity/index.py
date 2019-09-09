from collections import UserList
from typing import Any, List


class Index(UserList):
    """Indexed List"""

    def __init__(self, data: List):
        super().__init__(data)
        self.item2index = dict((item, idx) for idx, item in enumerate(data))

    def index(self, item: Any) -> int:
        return self.item2index[item]
