import asyncio
import random

from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel


class ExampleState(BaseModel):
    success_flag: bool = False


class StructuredExampleFlow(Flow):
    initial_state = ExampleState

    @start()
    async def start_method(self):
        print("Starting the structured flow")
        random_boolean = random.choice([True, False])
        self.state.success_flag = random_boolean

    @router(start_method)
    async def second_method(self):
        if self.state.success_flag:
            return "success"
        else:
            return "failed"

    @listen("success")
    async def third_method(self):
        print("Third method running")

    @listen("failed")
    async def fourth_method(self):
        print("Fourth method running")


async def main():
    flow = StructuredExampleFlow()
    await flow.kickoff()


asyncio.run(main())
