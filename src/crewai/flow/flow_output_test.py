import asyncio

from crewai.flow.flow import Flow, listen, start


class OutputExampleFlow(Flow):
    @start()
    async def first_method(self):
        return "Output from first_method"

    @listen(first_method)
    async def second_method(self, first_output):
        return f"Second method received: {first_output}"


async def main():
    flow = OutputExampleFlow()
    outputs = await flow.kickoff()
    print("---- Flow Outputs ----")
    print(outputs)

    print(" FLOW STATE POST RUN")
    print(flow.state)


asyncio.run(main())
