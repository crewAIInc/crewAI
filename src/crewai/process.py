from enum import Enum, auto


class Process(str, Enum):
    """
    Class representing the different processes that can be used to tackle tasks
    """

    sequential = "sequential"
    hierarchical = "hierarchical"
    parallel = "parallel"  # Yeni eklenen process tipi
    hybrid = "hybrid"      # Yeni eklenen process tipi
    # TODO: consensual = 'consensual'

