#!/usr/bin/env python
from pathlib import Path

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from {{folder_name}}.crews.content_crew.content_crew import ContentCrew


class ContentState(BaseModel):
    topic: str = ""
    outline: str = ""
    draft: str = ""
    final_post: str = ""


class ContentFlow(Flow[ContentState]):

    @start()
    def plan_content(self, crewai_trigger_payload: dict = None):
        print("Planning content")

        if crewai_trigger_payload:
            self.state.topic = crewai_trigger_payload.get("topic", "AI Agents")
            print(f"Using trigger payload: {crewai_trigger_payload}")
        else:
            self.state.topic = "AI Agents"

        print(f"Topic: {self.state.topic}")

    @listen(plan_content)
    def generate_content(self):
        print(f"Generating content on: {self.state.topic}")
        result = (
            ContentCrew()
            .crew()
            .kickoff(inputs={"topic": self.state.topic})
        )

        print("Content generated")
        self.state.final_post = result.raw

    @listen(generate_content)
    def save_content(self):
        print("Saving content")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "post.md", "w") as f:
            f.write(self.state.final_post)
        print("Post saved to output/post.md")


def kickoff():
    content_flow = ContentFlow()
    content_flow.kickoff()


def plot():
    content_flow = ContentFlow()
    content_flow.plot()


def run_with_trigger():
    """
    Run the flow with trigger payload.
    """
    import json
    import sys

    # Get trigger payload from command line argument
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    # Create flow and kickoff with trigger payload
    # The @start() methods will automatically receive crewai_trigger_payload parameter
    content_flow = ContentFlow()

    try:
        result = content_flow.kickoff({"crewai_trigger_payload": trigger_payload})
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the flow with trigger: {e}")


if __name__ == "__main__":
    kickoff()
