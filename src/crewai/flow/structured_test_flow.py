from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel


class ExampleState(BaseModel):
    counter: int = 0
    message: str = ""


class StructuredExampleFlow(Flow[ExampleState]):
    @start()
    def start_method(self):
        print("Starting the structured flow")
        self.state.message = "Hello from structured flow"
        return "Start result"

    @listen(start_method)
    def second_method(self, result):
        print(f"Second method, received: {result}")
        self.state.counter += 1
        self.state.message = "Hello from structured flow"
        return "Second result"


# Run the flow
structured_flow = StructuredExampleFlow()
structured_flow.run()
