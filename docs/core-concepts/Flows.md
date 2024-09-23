# CrewAI Flows

## Introduction

CrewAI Flows is a powerful feature designed to streamline the creation and management of AI workflows. Flows allow developers to combine and coordinate coding tasks and Crews efficiently, providing a robust framework for building sophisticated AI automations.

Flows allow you to create structured, event-driven workflows. They provide a seamless way to connect multiple tasks, manage state, and control the flow of execution in your AI applications. With Flows, you can easily design and implement multi-step processes that leverage the full potential of CrewAI's capabilities.

1. **Simplified Workflow Creation**: Easily chain together multiple Crews and tasks to create complex AI workflows.

2. **State Management**: Flows make it super easy to manage and share state between different tasks in your workflow.

3. **Event-Driven Architecture**: Built on an event-driven model, allowing for dynamic and responsive workflows.

4. **Flexible Control Flow**: Implement conditional logic, loops, and branching within your workflows.

## Getting Started

Let's create a simple Flow where you will use OpenAI to generate a random city in one task and then use that city to generate a fun fact in another task.

```python
import asyncio

from crewai.flow.flow import Flow, listen, start
from litellm import completion


class ExampleFlow(Flow):
    model = "gpt-4o-mini"

    @start()
    def generate_city(self):
        print("Starting flow")

        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Return the name of a random city in the world.",
                },
            ],
        )

        random_city = response["choices"][0]["message"]["content"]
        print(f"Random City: {random_city}")

        return random_city

    @listen(generate_city)
    def generate_fun_fact(self, random_city):
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Tell me a fun fact about {random_city}",
                },
            ],
        )

        fun_fact = response["choices"][0]["message"]["content"]
        return fun_fact


async def main():
    flow = ExampleFlow()
    result = await flow.kickoff()

    print(f"Generated fun fact: {result}")

asyncio.run(main())
```

In the above example, we have created a simple Flow that generates a random city using OpenAI and then generates a fun fact about that city. The Flow consists of two tasks: `generate_city` and `generate_fun_fact`. The `generate_city` task is the starting point of the Flow, and the `generate_fun_fact` task listens for the output of the `generate_city` task.

When you run the Flow, it will generate a random city and then generate a fun fact about that city. The output will be printed to the console.

### @start()

The `@start()` decorator is used to mark a method as the starting point of a Flow. When a Flow is started, all the methods decorated with `@start()` are executed in parallel. You can have multiple start methods in a Flow, and they will all be executed when the Flow is started.

### @listen()

The `@listen()` decorator is used to mark a method as a listener for the output of another task in the Flow. The method decorated with `@listen()` will be executed when the specified task emits an output. The method can access the output of the task it is listening to as an argument.

#### Usage

The `@listen()` decorator can be used in several ways:

1. **Listening to a Method by Name**: You can pass the name of the method you want to listen to as a string. When that method completes, the listener method will be triggered.

   ```python
   @listen("generate_city")
   def generate_fun_fact(self, random_city):
       # Implementation
   ```

2. **Listening to a Method Directly**: You can pass the method itself. When that method completes, the listener method will be triggered.
   ```python
   @listen(generate_city)
   def generate_fun_fact(self, random_city):
       # Implementation
   ```

### Flow Output

Accessing and handling the output of a Flow is essential for integrating your AI workflows into larger applications or systems. CrewAI Flows provide straightforward mechanisms to retrieve the final output, access intermediate results, and manage the overall state of your Flow.

#### Retrieving the Final Output

When you run a Flow, the final output is determined by the last method that completes. The `kickoff()` method returns the output of this final method.

Here's how you can access the final output:

```python
import asyncio
from crewai.flow.flow import Flow, listen, start

class OutputExampleFlow(Flow):
    @start()
    def first_method(self):
        return "Output from first_method"

    @listen(first_method)
    def second_method(self, first_output):
        return f"Second method received: {first_output}"

async def main():
    flow = OutputExampleFlow()
    final_output = await flow.kickoff()
    print("---- Final Output ----")
    print(final_output)

asyncio.run(main())
```

In this example, the `second_method` is the last method to complete, so its output will be the final output of the Flow. The `kickoff()` method will return this final output, which is then printed to the console.

The output of the Flow will be:

```
---- Final Output ----
Second method received: Output from first_method
```

#### Accessing and Updating State

In addition to retrieving the final output, you can also access and update the state within your Flow. The state can be used to store and share data between different methods in the Flow. After the Flow has run, you can access the state to retrieve any information that was added or updated during the execution.

Here's an example of how to update and access the state:

```python
import asyncio
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

class ExampleState(BaseModel):
    counter: int = 0
    message: str = ""

class StateExampleFlow(Flow[ExampleState]):

    @start()
    def first_method(self):
        self.state.message = "Hello from first_method"
        self.state.counter += 1

    @listen(first_method)
    def second_method(self):
        self.state.message += " - updated by second_method"
        self.state.counter += 1
        return self.state.message

async def main():
    flow = StateExampleFlow()
    final_output = await flow.kickoff()
    print("---- Final Output ----")
    print(final_output)
    print("---- Final State ----")
    print(flow.state)

asyncio.run(main())
```

In this example, the state is updated by both `first_method` and `second_method`. After the Flow has run, you can access the final state to see the updates made by these methods.

The output of the Flow will be:

```
---- Final Output ----
Hello from first_method - updated by second_method
---- Final State ----
counter=2 message='Hello from first_method - updated by second_method'
```

By ensuring that the final method's output is returned and providing access to the state, CrewAI Flows make it easy to integrate the results of your AI workflows into larger applications or systems, while also maintaining and accessing the state throughout the Flow's execution.

## Flow State Management

Managing state effectively is crucial for building reliable and maintainable AI workflows. CrewAI Flows provides robust mechanisms for both unstructured and structured state management, allowing developers to choose the approach that best fits their application's needs.

### Unstructured State Management

In unstructured state management, all state is stored in the `state` attribute of the `Flow` class. This approach offers flexibility, enabling developers to add or modify state attributes on the fly without defining a strict schema.

```python
import asyncio

from crewai.flow.flow import Flow, listen, start

class UntructuredExampleFlow(Flow):

    @start()
    def first_method(self):
        self.state.message = "Hello from structured flow"
        self.state.counter = 0

    @listen(first_method)
    def second_method(self):
        self.state.counter += 1
        self.state.message += " - updated"

    @listen(second_method)
    def third_method(self):
        self.state.counter += 1
        self.state.message += " - updated again"

        print(f"State after third_method: {self.state}")


async def main():
    flow = UntructuredExampleFlow()
    await flow.kickoff()


asyncio.run(main())
```

**Key Points:**

- **Flexibility:** You can dynamically add attributes to `self.state` without predefined constraints.
- **Simplicity:** Ideal for straightforward workflows where state structure is minimal or varies significantly.

### Structured State Management

Structured state management leverages predefined schemas to ensure consistency and type safety across the workflow. By using models like Pydantic's `BaseModel`, developers can define the exact shape of the state, enabling better validation and auto-completion in development environments.

```python
import asyncio

from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel


class ExampleState(BaseModel):
    counter: int = 0
    message: str = ""


class StructuredExampleFlow(Flow[ExampleState]):

    @start()
    def first_method(self):
        self.state.message = "Hello from structured flow"

    @listen(first_method)
    def second_method(self):
        self.state.counter += 1
        self.state.message += " - updated"

    @listen(second_method)
    def third_method(self):
        self.state.counter += 1
        self.state.message += " - updated again"

        print(f"State after third_method: {self.state}")


async def main():
    flow = StructuredExampleFlow()
    await flow.kickoff()


asyncio.run(main())
```

**Key Points:**

- **Defined Schema:** `ExampleState` clearly outlines the state structure, enhancing code readability and maintainability.
- **Type Safety:** Leveraging Pydantic ensures that state attributes adhere to the specified types, reducing runtime errors.
- **Auto-Completion:** IDEs can provide better auto-completion and error checking based on the defined state model.

### Choosing Between Unstructured and Structured State Management

- **Use Unstructured State Management when:**

  - The workflow's state is simple or highly dynamic.
  - Flexibility is prioritized over strict state definitions.
  - Rapid prototyping is required without the overhead of defining schemas.

- **Use Structured State Management when:**
  - The workflow requires a well-defined and consistent state structure.
  - Type safety and validation are important for your application's reliability.
  - You want to leverage IDE features like auto-completion and type checking for better developer experience.

By providing both unstructured and structured state management options, CrewAI Flows empowers developers to build AI workflows that are both flexible and robust, catering to a wide range of application requirements.

## Flow Control

### Conditional Logic

#### or

The `or_` function in Flows allows you to listen to multiple methods and trigger the listener method when any of the specified methods emit an output.

```python
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
    await flow.kickoff()


asyncio.run(main())
```

When you run this Flow, the `logger` method will be triggered by the output of either the `start_method` or the `second_method`. The `or_` function is to listen to multiple methods and trigger the listener method when any of the specified methods emit an output.

The output of the Flow will be:

```
Logger: Hello from the start method
Logger: Hello from the second method
```

#### and

The `and_` function in Flows allows you to listen to multiple methods and trigger the listener method only when all the specified methods emit an output.

```python
import asyncio
from crewai.flow.flow import Flow, and_, listen, start

class AndExampleFlow(Flow):

    @start()
    def start_method(self):
        self.state["greeting"] = "Hello from the start method"

    @listen(start_method)
    def second_method(self):
        self.state["joke"] = "What do computers eat? Microchips."

    @listen(and_(start_method, second_method))
    def logger(self):
        print("---- Logger ----")
        print(self.state)


async def main():
    flow = AndExampleFlow()
    await flow.kickoff()


asyncio.run(main())
```

When you run this Flow, the `logger` method will be triggered only when both the `start_method` and the `second_method` emit an output. The `and_` function is used to listen to multiple methods and trigger the listener method only when all the specified methods emit an output.

The output of the Flow will be:

```
---- Logger ----
{'greeting': 'Hello from the start method', 'joke': 'What do computers eat? Microchips.'}
```

### Router

The `@router()` decorator in Flows allows you to define conditional routing logic based on the output of a method. You can specify different routes based on the output of the method, allowing you to control the flow of execution dynamically.

```python
import asyncio
import random
from crewai.flow.flow import Flow, listen, router, start
from pydantic import BaseModel

class ExampleState(BaseModel):
    success_flag: bool = False

class RouterFlow(Flow[ExampleState]):

    @start()
    def start_method(self):
        print("Starting the structured flow")
        random_boolean = random.choice([True, False])
        self.state.success_flag = random_boolean

    @router(start_method)
    def second_method(self):
        if self.state.success_flag:
            return "success"
        else:
            return "failed"

    @listen("success")
    def third_method(self):
        print("Third method running")

    @listen("failed")
    def fourth_method(self):
        print("Fourth method running")


async def main():
    flow = RouterFlow()
    await flow.kickoff()


asyncio.run(main())
```

In the above example, the `start_method` generates a random boolean value and sets it in the state. The `second_method` uses the `@router()` decorator to define conditional routing logic based on the value of the boolean. If the boolean is `True`, the method returns `"success"`, and if it is `False`, the method returns `"failed"`. The `third_method` and `fourth_method` listen to the output of the `second_method` and execute based on the returned value.

When you run this Flow, the output will change based on the random boolean value generated by the `start_method`, but you should see an output similar to the following:

```
Starting the structured flow
Third method running
```

## Adding Crews to Flows

- The easiest way to create flows with Crews is to use the `crewai create flow <name_of_flow>` command. This will create a new CrewAI project for you that includes a folders for your Crews.

```

```

## Next Steps

- Recommend checking out our flow examples in the CrewAI Examples repository to see more use cases.
- Currently, there are 4 flow examples:
  - email auto responder flow
  - lead score flow
  - Write a book flow
  - Meeting assistant flow
