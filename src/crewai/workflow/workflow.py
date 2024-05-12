from typing import Any, Dict, List, Sequence, Union

from crewai.blocks.base import Block
from crewai.blocks.block_group import BlockGroup

BlockState = Dict[str, Any]


class Workflow:
    def __init__(self, starting_block: Union[Block, BlockGroup, None]):
        self.starting_block: Union[Block,
                                   BlockGroup, None] = starting_block

    def run(self, initial_state: Union[BlockState, List[BlockState]] = {}) -> Union[BlockState, List[BlockState]]:
        current_block = self.starting_block
        state = initial_state

        while current_block:
            # Handle multiple blocks
            if isinstance(current_block, BlockGroup) and not isinstance(current_block, str):
                # Handle multiple states
                if isinstance(state, list):
                    # Case 1: Equal number of blocks and states
                    if len(state) == len(current_block.blocks):
                        output = []
                        for idx, block in enumerate(current_block.blocks):
                            temp_state = state[idx]

                            output.append(block.process(temp_state))
                        state = output
                    # Case 2: 1 State to Pass to All Blocks
                    elif len(state) == 1:
                        # Many blocks with one state
                        output = []
                        for block in current_block.blocks:
                            output.append(block.process(state[0]))
                        state = output

                    # Case 3: 1 Block with Many States. Need to spin up a new block for each state
                    elif len(current_block.blocks) == 1:
                        output = []
                        for temp_state in state:
                            output.append(
                                current_block.blocks[0].process(temp_state))
                        state = output

                    else:
                        raise ValueError(
                            "Number of blocks and states do not match")

                else:
                    output = []
                    for block in current_block.blocks:
                        output.append(block.process(state))
                    state = output

            # Handle single block
            elif isinstance(current_block, Block):
                if isinstance(state, list):
                    # ManyToOneBlock
                    state = current_block.process(state)
                else:
                    # OneToOneBlock
                    print("I think I'm i a one to one block")
                    state = current_block.process(state)

            else:
                raise ValueError("Invalid block type")

            next_block = current_block.next_block

            if isinstance(next_block, Block) or isinstance(next_block, BlockGroup) or not next_block:
                current_block = next_block
            else:
                raise ValueError("Next Block Type Invalid")

        return state
