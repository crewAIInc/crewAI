#!/usr/bin/env python
import sys
from {{folder_name}}.crew import {{crew_name}}Crew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'AI LLMs'
    }
    {{crew_name}}Crew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        {{crew_name}}Crew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay_from_task():
    """
    Replay the crew execution from a specific task.
    """
    try:
        {{crew_name}}Crew().crew().replay_from_task(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
