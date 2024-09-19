import asyncio

from crewai.flow.flow import Flow, listen, start


class FlexibleExampleFlow(Flow):
    @start()
    def start_method(self):
        print("Starting the flexible flow")
        self.state["counter"] = 1
        return "Start result"

    @listen(start_method)
    def second_method(self):
        print("Second method")
        self.state["counter"] += 1
        self.state["message"] = "Hello from flexible flow"
        return "Second result"

    @listen(second_method)
    def third_method(self):
        print("Third method")
        print(f"Final counter value: {self.state["counter"]}")
        print(f"Final message: {self.state["message"]}")
        return "Third result"


async def main():
    flow = FlexibleExampleFlow()
    await flow.kickoff()


asyncio.run(main())
