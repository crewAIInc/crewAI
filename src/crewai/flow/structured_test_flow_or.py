import asyncio

from crewai.flow.flow import Flow, listen, or_, start


class OrExampleFlow(Flow):

    @start()
    def start_method(self):
        return "Hello from the start method"

    @listen(start_method)
    def second_method(self):
        return "Hello from the second method"

    @listen(or_(start_method, second_method))
    def logger(self, result):
        print(f"Logger: {result}")


async def main():
    flow = OrExampleFlow()
    flow.visualize()
    # await flow.kickoff()


asyncio.run(main())
