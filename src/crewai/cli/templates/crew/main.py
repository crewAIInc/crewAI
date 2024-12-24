#!/usr/bin/env python
import sys
import json
import warnings

from {{folder_name}}.crew import {{crew_name}}
from crewai.utilities.llm_utils import create_llm

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': 'AI LLMs'
    }
    {{crew_name}}().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        {{crew_name}}().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        {{crew_name}}().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        {{crew_name}}().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def fetch_inputs():
    """
    Command that gathers required placeholders/inputs from the Crew, then
    prints them as JSON to stdout so external scripts can parse them easily.
    """
    try:
        crew = {{crew_name}}().crew()
        crew_inputs = crew.fetch_inputs()
        json_string = json.dumps(list(crew_inputs))
        print(json_string)
    except Exception as e:
        raise Exception(f"An error occurred while fetching inputs: {e}")

def fetch_chat_llm():
    """
    Command that fetches the 'chat_llm' property from the Crew,
    instantiates it via create_llm(),
    and prints the resulting LLM as JSON (using LLM.to_dict()) to stdout.
    """
    try:
        crew = {{crew_name}}().crew()
        raw_chat_llm = getattr(crew, "chat_llm", None)

        if not raw_chat_llm:
            # If the crew doesn't have chat_llm, fallback to create_llm(None)
            final_llm = create_llm(None)
        else:
            # raw_chat_llm might be a dict, or an LLM, or something else
            final_llm = create_llm(raw_chat_llm)

        if final_llm:
            # Print the final LLM as JSON, so fetch_chat_llm.py can parse it
            from crewai.llm import LLM  # Import here to avoid circular references

            # Make sure it's an instance of the LLM class:
            if isinstance(final_llm, LLM):
                print(json.dumps(final_llm.to_dict()))
            else:
                # If somehow it's not an LLM, try to interpret as a dict
                # or revert to an empty fallback
                if isinstance(final_llm, dict):
                    print(json.dumps(final_llm))
                else:
                    print(json.dumps({}))
        else:
            print(json.dumps({}))
    except Exception as e:
        raise Exception(f"An error occurred while fetching chat LLM: {e}")
