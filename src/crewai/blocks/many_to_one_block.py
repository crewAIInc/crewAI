from typing import Callable, List, Dict, Any

from crewai.blocks.base import Block


class ManyToOneBlock(Block):
    def __init__(self, process_func: Callable[[List[Dict[str, Any]]],  Dict[str, Any]]):
        super().__init__(process_func)

    def process(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.process_func(inputs)

    def add_next(self, block: Block) -> None:
        return super().add_next(block)
