from typing import Callable, Dict, Any, List
from crewai.blocks.base import Block
from crewai.blocks.block_group import BlockGroup


class OneToManyBlock(Block):
    def __init__(self, process_func: Callable[[Dict[str, Any]], List[Dict[str, Any]]]):
        super().__init__(process_func)

    def process(self, input: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self.process_func(input)

    def add_next(self, block: BlockGroup) -> None:
        return super().add_next(block=block)
