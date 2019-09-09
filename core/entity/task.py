import enum


class Task(enum.Enum):
    """Task types"""
    binary = enum.auto()
    classify_single = enum.auto()
    classify_multiple = enum.auto()
