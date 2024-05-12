from typing import TYPE_CHECKING, Any, Callable, Union
from abc import ABC


if TYPE_CHECKING:
    from crewai.blocks.block_group import BlockGroup


class Block(ABC):
    def __init__(self, process_func: Callable[[Any], Any]):
        self.process_func = process_func
        self.next_block: Union['Block', 'BlockGroup', None] = None

    def process(self, input: Any) -> Any:
        return self.process_func(input)

    def add_next(self, block: Union['Block', 'BlockGroup', None]) -> None:
        self.next_block = block
