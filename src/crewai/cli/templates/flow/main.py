#!/usr/bin/env python
import asyncio

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start

# TODO: THERE SHOULD BE 3 FLOWS IN HERE: SIMPLE, ASYNC, BRANCHING (with router)

class ExampleState(BaseModel):
    counter: int = 0
    message: str = ""

class ExampleFlow(Flow[ExampleState]):
    initial_state = ExampleState

    @start()
    def start_method(self):
        print("Starting the structured flow")
        self.state.message = "Hello from structured flow"

    @listen(start_method)
    def second_method(self, result):
        print(f"Second method, received: {result}")
        print(f"State before increment: {self.state}")
        self.state.counter += 1
        self.state.message += " - updated"
        print(f"State after second_method: {self.state}")
        return "Second result"

async def run(): 
    """
    Run the flow.
    """
    example_flow = ExampleFlow()
    await example_flow.run()


if __name__ == "__main__":
    asyncio.run(run())
