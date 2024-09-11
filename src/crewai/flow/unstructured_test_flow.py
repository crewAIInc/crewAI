from crewai.flow.flow import Flow, listen, start


class FlexibleExampleFlow(Flow):
    @start()
    def start_method(self):
        print("Starting the flexible flow")
        self.state["counter"] = 1
        return "Start result"

    @listen(start_method)
    def second_method(self, result):
        print(f"Second method, received: {result}")
        self.state["counter"] += 1
        self.state["message"] = "Hello from flexible flow"
        return "Second result"

    @listen(second_method)
    def third_method(self, result):
        print(f"Third method, received: {result}")
        print(f"Final counter value: {self.state["counter"]}")
        print(f"Final message: {self.state["message"]}")
        return "Third result"


# Run the flows
flexible_flow = FlexibleExampleFlow()
flexible_flow.run()
