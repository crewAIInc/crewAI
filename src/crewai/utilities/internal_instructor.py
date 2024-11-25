import time
from typing import Any, Optional, Type


class InternalInstructor:
    """Class that wraps an agent llm with instructor."""

    def __init__(
        self,
        content: str,
        model: Type,
        agent: Optional[Any] = None,
        llm: Optional[str] = None,
        instructions: Optional[str] = None,
    ):
        self.content = content
        self.agent = agent
        self.llm = llm
        self.instructions = instructions
        self.model = model
        self._client = None
        self.set_instructor()

    def set_instructor(self):
        """Set instructor."""
        if self.agent and not self.llm:
            self.llm = self.agent.function_calling_llm or self.agent.llm

        start_time = time.time()
        # Lazy import
        import instructor
        from litellm import completion

        end_time = time.time()

        print(
            f"Time taken to import instructor and completion: {end_time - start_time:.6f} seconds"
        )

        start_time = time.time()
        self._client = instructor.from_litellm(
            completion,
            mode=instructor.Mode.TOOLS,
        )
        end_time = time.time()

        print(f"Time taken to create the client: {end_time - start_time:.6f} seconds")

    def to_json(self):
        model = self.to_pydantic()
        return model.model_dump_json(indent=2)

    def to_pydantic(self):
        print("INSTRUCTIONS: ", self.instructions)
        messages = [{"role": "user", "content": self.content}]
        if self.instructions:
            messages.append({"role": "system", "content": self.instructions})

        import time  # Import the time module

        start_time = time.time()  # Record the start time for chat completion
        model = self._client.chat.completions.create(
            model=self.llm.model, response_model=self.model, messages=messages
        )
        end_time = time.time()  # Record the end time for chat completion

        # Log the time taken for the chat completion
        print(f"Time taken for chat completion: {end_time - start_time:.6f} seconds")

        return model
