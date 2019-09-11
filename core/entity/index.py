from collections import UserList
from typing import Any, List, Optional


class Index(UserList):
    """Indexed List"""

    def __init__(self, data: List):
        super().__init__(data)
        self.item2index = dict((item, idx) for idx, item in enumerate(data))

    def index(self, item: Any) -> Optional[int]:
        if item not in self.item2index:
            return None
        return self.item2index[item]
