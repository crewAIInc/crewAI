from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel


class ExampleState(BaseModel):
    counter: int = 0
    message: str = ""


class StructuredExampleFlow(Flow[ExampleState]):
    initial_state = ExampleState

    @start()
    def start_method(self):
        print("Starting the structured flow")
        print(f"State in start_method: {self.state}")
        self.state.message = "Hello from structured flow"
        print(f"State after start_method: {self.state}")
        return "Start result"

    @listen(start_method)
    def second_method(self, result):
        print(f"Second method, received: {result}")
        print(f"State before increment: {self.state}")
        self.state.counter += 1
        self.state.message += " - updated"
        print(f"State after second_method: {self.state}")
        return "Second result"


# Instantiate and run the flow
structured_flow = StructuredExampleFlow()
structured_flow.run()
