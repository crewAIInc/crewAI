from typing import TYPE_CHECKING, List, Union
from crewai.blocks.base import Block

if TYPE_CHECKING:
    from .base import Block


class BlockGroup:
    def __init__(self, blocks: List[Block] = []):
        self.blocks = blocks
        self.next_block: Union[Block, 'BlockGroup', None] = None

    def add_next(self, block: Union[Block, 'BlockGroup', None]) -> None:
        self.next_block = block
