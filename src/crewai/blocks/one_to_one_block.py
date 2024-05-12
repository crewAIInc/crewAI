from typing import Dict, Any, Callable,  Sequence
from crewai.blocks.base import Block


class OneToOneBlock(Block):
    def __init__(self, process_func: Callable[[Dict[str, Any]], Dict[str, Any]]):
        super().__init__(process_func)

    def process(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.process_func(input)

    def add_next(self, block: Block) -> None:
        return super().add_next(block)
