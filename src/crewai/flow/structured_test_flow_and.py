import asyncio

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel


class ExampleState(BaseModel):
    counter: int = 0
    message: str = ""


class StructuredExampleFlow(Flow[ExampleState]):
    initial_state = ExampleState

    @start()
    async def start_method(self):
        print("Starting the structured flow")
        print(f"State in start_method: {self.state}")
        self.state.message = "Hello from structured flow"
        print(f"State after start_method: {self.state}")
        return "Start result"

    @listen(start_method)
    async def second_method(self, result):
        print(f"Second method, received: {result}")
        print(f"State before increment: {self.state}")
        self.state.counter += 1
        self.state.message += " - updated"
        print(f"State after second_method: {self.state}")
        return "Second result"

    @listen(start_method)
    async def third_method(self, result):
        print(f"Third method, received: {result}")
        print(f"State before increment: {self.state}")
        self.state.counter += 1
        self.state.message += " - updated"
        print(f"State after third_method: {self.state}")
        return "Third result"


async def main():
    flow = StructuredExampleFlow()
    await flow.run()


asyncio.run(main())
