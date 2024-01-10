from enum import Enum


class Process(str, Enum):
    """
    Class representing the different processes that can be used to tackle tasks
    """

    sequential = "sequential"
    # TODO: consensual = 'consensual'
    # TODO: hierarchical = 'hierarchical'
