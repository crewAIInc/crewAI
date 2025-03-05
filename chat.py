#!/usr/bin/env python
import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from crewai.flow import Flow, start
from crewai.flow.persistence.decorators import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from crewai.llm import LLM


class Message(BaseModel):
    role: str = Field(
        description="The role of the message sender (e.g., 'user', 'assistant')"
    )
    content: str = Field(description="The actual content/text of the message")


class ChatState(BaseModel):
    message: Optional[Message] = None
    history: list[Message] = Field(default_factory=list)


@persist(SQLiteFlowPersistence(), verbose=True)
class PersonalAssistantFlow(Flow[ChatState]):
    @start()
    def chat(self):
        user_message_pydantic = self.state.message

        # Safety check for None message
        if not user_message_pydantic:
            return "No message provided"

        # Format history for prompt
        history_formatted = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in self.state.history]
        )

        prompt = f"""
        You are a helpful assistant.
        Answer the user's question: {user_message_pydantic.content}

        Just for the sake of being context-aware, this is the entire conversation history:
        {history_formatted}

        Be friendly and helpful, yet to the point.
        """

        response = LLM(model="gemini/gemini-2.0-flash", response_format=Message).call(
            prompt
        )

        # Parse the response
        if isinstance(response, str):
            try:
                llm_response_json = json.loads(response)
                llm_response_pydantic = Message(**llm_response_json)
            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON
                llm_response_pydantic = Message(
                    role="assistant",
                    content="I'm sorry, I encountered an error processing your request.",
                )
        else:
            # If response is already a Message object
            llm_response_pydantic = response

        # Update history - with type safety
        if user_message_pydantic:  # Ensure message is not None before adding to history
            self.state.history.append(user_message_pydantic)
        self.state.history.append(llm_response_pydantic)

        print("History", self.state.history)
        return llm_response_pydantic.content


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = input("> ")

    flow = PersonalAssistantFlow()
    flow.state.message = Message(role="user", content=user_input)

    response = flow.kickoff()
    print(response)
