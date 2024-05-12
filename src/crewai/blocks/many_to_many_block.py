from typing import Callable, List, Dict, Any

from crewai.blocks.base import Block
from crewai.blocks.block_group import BlockGroup


class ManyToManyBlock(Block):
    def __init__(self, process_func: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]):
        super().__init__(process_func)

    def process(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.process_func(input) for input in inputs]

    def add_next(self, blockGroup: BlockGroup) -> None:
        return super().add_next(blockGroup)
