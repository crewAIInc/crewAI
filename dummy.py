from typing import List
from crewai.blocks.block_group import BlockGroup
from crewai.blocks.many_to_one_block import ManyToOneBlock
from crewai.blocks.one_to_many_block import OneToManyBlock
from crewai.blocks.one_to_one_block import OneToOneBlock
from crewai.workflow.workflow import BlockState, Workflow


def generateRandomNumber(state: BlockState) -> List[BlockState]:
    new_state = []
    for i in range(3):
        new_state.append({"number": i})

    print("New State: ", new_state)
    return new_state


def printNumber(state: BlockState) -> BlockState:
    print(f"Hello World, the number is {state['number']}")
    return state


def sumNumbers(states: List[BlockState]) -> BlockState:
    sum = 0
    for state in states:
        sum += state["number"]
    return {"sum": sum}


random_block = OneToManyBlock(generateRandomNumber)
print_block = OneToOneBlock(printNumber)
print_blocks = BlockGroup([print_block])
sum_block = ManyToOneBlock(sumNumbers)


random_block.add_next(print_blocks)
print_blocks.add_next(sum_block)

workflow = Workflow(random_block)

final_output = workflow.run()

print("Final Output: ", final_output)
