import asyncio

from crewai.flow.flow import Flow, and_, listen, or_, start
from pydantic import BaseModel


class ExampleState(BaseModel):
    counter: int = 0
    message: str = ""


class StructuredExampleFlow(Flow[ExampleState]):

    @start()
    async def start_method(self):
        print("Starting the structured flow")
        print(f"State in start_method: {self.state}")
        self.state.message = "Hello from structured flow"
        print(f"State after start_method: {self.state}")
        return "Start result"

    @listen(start_method)
    async def second_method(self):
        print(f"State before increment: {self.state}")
        self.state.counter += 1
        self.state.message += " - updated"
        print(f"State after second_method: {self.state}")
        return "Second result"

    @listen(or_(start_method, second_method))
    async def logger(self):
        print("LOGGER METHOD RUNNING")
        print("CURRENT STATE FROM LOGGER: ", self.state)

    @listen(and_(start_method, second_method))
    async def and_logger(self):
        print("AND LOGGER METHOD RUNNING")
        print("CURRENT STATE FROM AND LOGGER: ", self.state)


async def main():
    flow = StructuredExampleFlow()
    await flow.kickoff()


asyncio.run(main())
