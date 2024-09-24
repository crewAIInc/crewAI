import asyncio

from crewai.flow.flow import Flow, listen, start


class BinaryBranchingFlow(Flow):
    @start()
    def start_method(self):
        print("Starting the binary branching flow")
        self.state["counter"] = 1
        return "Start result"

    # Level 1
    @listen(start_method)
    def first_branch_left(self):
        print("First branch (left)")
        self.state["counter"] += 1
        return "First branch left result"

    @listen(start_method)
    def first_branch_right(self):
        print("First branch (right)")
        self.state["counter"] += 1
        return "First branch right result"

    # Level 2 - Left Branch
    @listen(first_branch_left)
    def second_branch_left_left(self):
        print("Second branch from first left (left)")
        self.state["counter"] += 1
        return "Second branch left left result"

    @listen(first_branch_left)
    def second_branch_left_right(self):
        print("Second branch from first left (right)")
        self.state["counter"] += 1
        return "Second branch left right result"

    # Level 2 - Right Branch
    @listen(first_branch_right)
    def second_branch_right_left(self):
        print("Second branch from first right (left)")
        self.state["counter"] += 1
        return "Second branch right left result"

    @listen(first_branch_right)
    def second_branch_right_right(self):
        print("Second branch from first right (right)")
        self.state["counter"] += 1
        return "Second branch right right result"

    # Level 3 - Left Left Branch
    @listen(second_branch_left_left)
    def third_branch_left_left_left(self):
        print("Third branch from second left left (left)")
        self.state["counter"] += 1
        return "Third branch left left left result"

    @listen(second_branch_left_left)
    def third_branch_left_left_right(self):
        print("Third branch from second left left (right)")
        self.state["counter"] += 1
        return "Third branch left left right result"

    # Level 3 - Left Right Branch
    @listen(second_branch_left_right)
    def third_branch_left_right_left(self):
        print("Third branch from second left right (left)")
        self.state["counter"] += 1
        return "Third branch left right left result"

    @listen(second_branch_left_right)
    def third_branch_left_right_right(self):
        print("Third branch from second left right (right)")
        self.state["counter"] += 1
        return "Third branch left right right result"

    # Level 3 - Right Left Branch
    @listen(second_branch_right_left)
    def third_branch_right_left_left(self):
        print("Third branch from second right left (left)")
        self.state["counter"] += 1
        return "Third branch right left left result"

    @listen(second_branch_right_left)
    def third_branch_right_left_right(self):
        print("Third branch from second right left (right)")
        self.state["counter"] += 1
        return "Third branch right left right result"

    # Level 3 - Right Right Branch
    @listen(second_branch_right_right)
    def third_branch_right_right_left(self):
        print("Third branch from second right right (left)")
        self.state["counter"] += 1
        return "Third branch right right left result"

    @listen(second_branch_right_right)
    def third_branch_right_right_right(self):
        print("Third branch from second right right (right)")
        self.state["counter"] += 1
        return "Third branch right right right result"

    # Final method for visualization
    @listen(third_branch_left_left_left)  # This is the deepest branch in the tree
    def final_method(self):
        print("Final method reached!")
        print(f"Final counter value: {self.state['counter']}")
        return "Final result"


async def main():
    flow = BinaryBranchingFlow()
    # Uncomment this if you want to run the flow with kickoff
    # await flow.kickoff()
    flow.visualize()


asyncio.run(main())
